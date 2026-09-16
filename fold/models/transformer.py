"""The sequential branch: a Transformer encoder over the packet sequence, and
FOLD, which fuses it with the random forest's output.

Both models read the same encoded sequence:

  1. batch normalisation of the per-packet features
  2. concatenation of the learned packet-size embedding
  3. linear projection to ``d_model``
  4. sinusoidal positional encoding, scaled by a per-trace coefficient
  5. ``num_layers`` pre-norm encoder blocks (multi-head attention + feed-forward)
  6. final layer norm and masked average pooling over the real packets

TR stops there and classifies the pooled representation.  FOLD additionally
takes the forest's class-probability vector, builds a cross-attention query out
of it, and reads the encoded sequence with it, so that the packets supporting
the forest's prediction weigh more.

Pre-norm residuals (layer norm before each sublayer, not after) are used
because they train more stably from scratch on a dataset this size.
"""
from typing import Optional

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


class PositionalEncoding(layers.Layer):
    """Sinusoidal positional encoding, scaled by a per-trace coefficient.

    The encoding table is precomputed once and never trained.  What is trained
    is ``alpha``: a small MLP over the mean of the projected packets produces a
    scalar in (0, 1) per trace, and the encoding is added as ``x + alpha * pe``.

    Absolute position is not equally informative in every trace: a defense that
    injects or fragments packets shifts every position after the one it
    touched.  A per-trace coefficient lets one trained model lean on position
    where it is reliable and ignore it where it is not, instead of committing
    to one amount for the whole dataset.
    """

    def __init__(self, max_len: int, d_model: int, gate_hidden_dim: int = 16, **kwargs):
        super().__init__(**kwargs)
        positions = np.arange(max_len, dtype=np.float32)[:, np.newaxis]
        dims = np.arange(d_model, dtype=np.float32)[np.newaxis, :]
        angles = positions / np.power(10000.0, (2 * (dims // 2)) / d_model)
        angles[:, 0::2] = np.sin(angles[:, 0::2])
        angles[:, 1::2] = np.cos(angles[:, 1::2])
        self._pe = tf.constant(angles[np.newaxis, :, :], dtype=tf.float32)
        self.gate_hidden_dim = gate_hidden_dim
        self.gate_dense1 = layers.Dense(gate_hidden_dim, activation="gelu",
                                        name="pos_gate_dense1")
        self.gate_dense2 = layers.Dense(1, activation="sigmoid", name="pos_gate_dense2")

    def call(self, x: tf.Tensor) -> tf.Tensor:
        seq_len = tf.shape(x)[1]
        pe = self._pe[:, :seq_len, :]
        # A cheap per-trace summary. Padded positions are a minority and this
        # only steers a soft coefficient, so the mean is left unmasked.
        summary = tf.reduce_mean(x, axis=1)
        alpha = self.gate_dense2(self.gate_dense1(summary))
        return x + alpha[:, :, tf.newaxis] * pe

    def get_config(self):
        cfg = super().get_config()
        cfg.update({"max_len": int(self._pe.shape[1]), "d_model": int(self._pe.shape[2]),
                    "gate_hidden_dim": self.gate_hidden_dim})
        return cfg


class TransformerEncoderBlock(layers.Layer):
    """One pre-norm encoder block: multi-head attention, then feed-forward."""

    def __init__(self, d_model: int, num_heads: int, d_ff: int,
                 dropout_rate: float = 0.1, **kwargs):
        super().__init__(**kwargs)
        if d_model % num_heads != 0:
            raise ValueError(
                f"d_model ({d_model}) must be divisible by num_heads ({num_heads}).")

        self.attn = layers.MultiHeadAttention(
            num_heads=num_heads,
            key_dim=d_model // num_heads,   # standard split: total QK dim == d_model
            dropout=dropout_rate,
        )
        self.ffn = keras.Sequential(
            [layers.Dense(d_ff, activation="gelu"), layers.Dense(d_model)], name="ffn")
        self.norm1 = layers.LayerNormalization(epsilon=1e-6)
        self.norm2 = layers.LayerNormalization(epsilon=1e-6)
        self.drop1 = layers.Dropout(dropout_rate)
        self.drop2 = layers.Dropout(dropout_rate)

    def call(self, x: tf.Tensor, training: bool = False,
             attention_mask: Optional[tf.Tensor] = None) -> tf.Tensor:
        x_norm = self.norm1(x)
        attn_out = self.attn(x_norm, x_norm, attention_mask=attention_mask,
                             training=training)
        x = x + self.drop1(attn_out, training=training)

        x_norm = self.norm2(x)
        x = x + self.drop2(self.ffn(x_norm, training=training), training=training)
        return x

    def get_config(self):
        cfg = super().get_config()
        cfg.update({"num_heads": self.attn.num_heads, "key_dim": self.attn.key_dim,
                    "d_ff": self.ffn.layers[0].units, "dropout_rate": self.drop1.rate})
        return cfg


# ---------------------------------------------------------------------------
# Shared pieces of the two models
# ---------------------------------------------------------------------------

def _masks(seq_input):
    """The two padding masks the encoder needs.

    ``attn_mask`` is boolean of shape (batch, 1, seq_len): True where a key is
    a real packet that attention may read.  It broadcasts over query positions
    and heads.  ``padding_mask`` is float of shape (batch, seq_len), 1.0 on
    real packets, and weighs the average pooling.
    """
    attn_mask = layers.Lambda(
        lambda x: tf.reduce_any(tf.not_equal(x, 0.0), axis=-1)[:, tf.newaxis, :],
        name="attn_mask")(seq_input)
    padding_mask = layers.Lambda(
        lambda x: tf.cast(tf.reduce_any(tf.not_equal(x, 0.0), axis=-1), tf.float32),
        name="padding_mask")(seq_input)
    return attn_mask, padding_mask


def _encode(seq_input, size_vocab_input, attn_mask, *, sequence_length,
            size_vocab_size, size_embed_dim, d_model, num_heads, d_ff,
            num_layers, dropout_rate, pos_gate_hidden_dim,
            unigram_agg_gate_hidden_dim):
    """Encode the packet sequence; return the encoded sequence, the pooled
    representation and the coefficient gating the size-histogram branch."""
    size_emb = layers.Embedding(size_vocab_size, size_embed_dim,
                                name="size_vocab_embedding")(size_vocab_input)

    # Batch normalisation over each feature: raw timestamps (~0.001 s), sizes
    # (54-1500 B) and roles (0-3) live on very different scales.
    x = layers.BatchNormalization(name="input_bn")(seq_input)
    x = layers.Concatenate(name="concat_vocab_embs")([x, size_emb])
    x = layers.Dense(d_model, activation="gelu", name="input_projection")(x)

    # The coefficient of the size-histogram branch, from the same per-trace
    # summary the positional coefficient uses. The two coefficients get their
    # own small MLP rather than a shared trunk: sharing was tried and was worse.
    seq_summary = layers.Lambda(lambda t: tf.reduce_mean(t, axis=1),
                                name="seq_summary")(x)
    uni_gate_h = layers.Dense(unigram_agg_gate_hidden_dim, activation="gelu",
                              name="unigram_gate_hidden")(seq_summary)
    alpha_unigram = layers.Dense(1, activation="sigmoid",
                                 name="unigram_gate_out")(uni_gate_h)

    x = PositionalEncoding(max_len=sequence_length, d_model=d_model,
                           gate_hidden_dim=pos_gate_hidden_dim,
                           name="pos_encoding")(x)

    for i in range(num_layers):
        x = TransformerEncoderBlock(d_model=d_model, num_heads=num_heads, d_ff=d_ff,
                                    dropout_rate=dropout_rate,
                                    name=f"transformer_block_{i + 1}")(x, attention_mask=attn_mask)

    x_seq = layers.LayerNormalization(epsilon=1e-6, name="final_norm")(x)
    return x_seq, alpha_unigram


def _masked_avg_pool(x_seq, padding_mask):
    """Average the encoded packets, ignoring padding."""
    return layers.Lambda(
        lambda args: tf.reduce_sum(args[0] * args[1][:, :, tf.newaxis], axis=1)
                     / (tf.reduce_sum(args[1][:, :, tf.newaxis], axis=1) + 1e-8),
        name="masked_avg_pool")([x_seq, padding_mask])


def _size_histogram_branch(unigram_agg_dim, alpha_unigram):
    """The packet-size histogram branch, scaled by its own per-trace coefficient.

    The coefficient scales the branch's *contribution*, so that a coefficient
    near zero makes it a no-op instead of corrupting its batch statistics.
    """
    unigram_input = keras.Input(shape=(unigram_agg_dim,), name="unigram_agg_input")
    h = layers.BatchNormalization(name="unigram_agg_bn")(unigram_input)
    h = layers.Dense(32, activation="gelu", name="unigram_agg_dense")(h)
    h = layers.Multiply(name="unigram_agg_gated")([h, alpha_unigram])
    return unigram_input, h


def _aux_head(pooled, num_classes, dropout_rate):
    """A softmax placed directly on the pooled sequence, before any fusion.

    Its loss gives the encoder a gradient that the aggregate branches cannot
    satisfy on their behalf, so the sequence has to stay informative by itself.
    """
    aux_x = layers.Dropout(dropout_rate, name="aux_dropout")(pooled)
    return layers.Dense(num_classes, activation="softmax", name="aux_multi_output")(aux_x)


# ---------------------------------------------------------------------------
# TR: the sequential branch alone
# ---------------------------------------------------------------------------

def build_transformer(
    sequence_length: int,
    num_features: int,
    num_classes: int,
    num_agg_features: int = 3,
    *,
    size_vocab_size: int = 65,
    size_embed_dim: int = 8,
    pos_gate_hidden_dim: int = 16,
    unigram_agg_dim: int = 64,
    unigram_agg_gate_hidden_dim: int = 16,
    d_model: int = 64,
    num_heads: int = 4,
    d_ff: int = 128,
    num_layers: int = 5,
    dropout_rate: float = 0.2,
) -> keras.Model:
    """The Transformer baseline: sequence, aggregate statistics, size histogram.

    Inputs: ``transformer_input`` (batch, seq_len, num_features),
    ``agg_input`` (batch, num_agg_features), ``size_vocab_input``
    (batch, seq_len) and ``unigram_agg_input`` (batch, unigram_agg_dim).

    Outputs: ``multi_output`` (the website), ``binary_output`` (monitored or
    not) and ``aux_multi_output`` (the auxiliary head).
    """
    seq_input = keras.Input(shape=(sequence_length, num_features), name="transformer_input")
    size_vocab_input = keras.Input(shape=(sequence_length,), dtype="int32",
                                   name="size_vocab_input")
    attn_mask, padding_mask = _masks(seq_input)

    x_seq, alpha_unigram = _encode(
        seq_input, size_vocab_input, attn_mask,
        sequence_length=sequence_length, size_vocab_size=size_vocab_size,
        size_embed_dim=size_embed_dim, d_model=d_model, num_heads=num_heads,
        d_ff=d_ff, num_layers=num_layers, dropout_rate=dropout_rate,
        pos_gate_hidden_dim=pos_gate_hidden_dim,
        unigram_agg_gate_hidden_dim=unigram_agg_gate_hidden_dim)

    pooled = _masked_avg_pool(x_seq, padding_mask)
    aux_multi_output = _aux_head(pooled, num_classes, dropout_rate)

    agg_input = keras.Input(shape=(num_agg_features,), name="agg_input")
    agg = layers.BatchNormalization(name="agg_bn")(agg_input)
    agg = layers.Dense(32, activation="gelu", name="agg_dense")(agg)

    unigram_input, unigram_agg = _size_histogram_branch(unigram_agg_dim, alpha_unigram)

    fused = layers.Concatenate(name="fuse")([pooled, agg, unigram_agg])

    shared = layers.Dropout(dropout_rate, name="pre_cls_dropout")(fused)
    shared = layers.Dense(128, activation="gelu", name="shared_dense")(shared)
    shared = layers.LayerNormalization(epsilon=1e-6, name="shared_norm")(shared)

    bin_x = layers.Dropout(dropout_rate, name="binary_dropout")(shared)
    binary_output = layers.Dense(1, activation="sigmoid", name="binary_output")(bin_x)

    multi_x = layers.Dropout(dropout_rate, name="multi_dropout")(shared)
    multi_output = layers.Dense(num_classes, activation="softmax", name="multi_output")(multi_x)

    return keras.Model(
        inputs={"transformer_input": seq_input, "agg_input": agg_input,
                "size_vocab_input": size_vocab_input,
                "unigram_agg_input": unigram_input},
        outputs={"binary_output": binary_output, "multi_output": multi_output,
                 "aux_multi_output": aux_multi_output},
        name="transformer_attack_model",
    )


# ---------------------------------------------------------------------------
# FOLD
# ---------------------------------------------------------------------------

def split_n1(p: np.ndarray, num_classes: int):
    """Turn one softmax over N+1 classes into the two views the rest expects.

    Returns ``(multi, binary)``: the N monitored columns renormalised, and
    1 - P(background).  Computed outside the graph on purpose, see the note in
    :func:`build_fold`.
    """
    mon = p[:, :num_classes]
    return (mon / np.maximum(mon.sum(axis=1, keepdims=True), 1e-9),
            1.0 - p[:, num_classes])


def build_fold(
    sequence_length: int,
    num_features: int,
    num_classes: int,
    num_agg_features: int = 3,
    *,
    rf_proj_dim: int = 32,
    rf_branch_dropout: float = 0.3,
    gate_hidden_dim: int = 16,
    n1_head: bool = False,
    size_vocab_size: int = 65,
    size_embed_dim: int = 8,
    pos_gate_hidden_dim: int = 16,
    unigram_agg_dim: int = 64,
    unigram_agg_gate_hidden_dim: int = 16,
    d_model: int = 64,
    num_heads: int = 4,
    d_ff: int = 128,
    num_layers: int = 5,
    dropout_rate: float = 0.2,
) -> keras.Model:
    """FOLD: the Transformer encoder fused with the random forest's output.

    The forest's probability vector reaches the two heads by three different
    routes, because the two heads need different things from it:

    * as the query of a cross-attention over the encoded sequence, so that the
      packets supporting the forest's prediction weigh more in the website
      decision;
    * as a projected vector concatenated into the fused representation;
    * as a two-number confidence summary (maximum probability and normalised
      entropy), which is what the monitored/background decision needs.

    ``n1_head`` replaces the two heads with a single softmax over N+1 classes,
    the standard open-world formulation, for the ablation of Fig. 8.

    Inputs: ``transformer_input``, ``agg_input``, ``rf_input``,
    ``size_vocab_input``, ``unigram_agg_input``.
    Outputs: ``multi_output`` + ``binary_output`` + ``aux_multi_output``, or
    ``n1_output`` + ``aux_multi_output`` when ``n1_head`` is set.
    """
    seq_input = keras.Input(shape=(sequence_length, num_features), name="transformer_input")
    rf_input = keras.Input(shape=(num_classes,), name="rf_input")
    size_vocab_input = keras.Input(shape=(sequence_length,), dtype="int32",
                                   name="size_vocab_input")
    attn_mask, padding_mask = _masks(seq_input)

    x_seq, alpha_unigram = _encode(
        seq_input, size_vocab_input, attn_mask,
        sequence_length=sequence_length, size_vocab_size=size_vocab_size,
        size_embed_dim=size_embed_dim, d_model=d_model, num_heads=num_heads,
        d_ff=d_ff, num_layers=num_layers, dropout_rate=dropout_rate,
        pos_gate_hidden_dim=pos_gate_hidden_dim,
        unigram_agg_gate_hidden_dim=unigram_agg_gate_hidden_dim)

    pooled = _masked_avg_pool(x_seq, padding_mask)
    aux_multi_output = _aux_head(pooled, num_classes, dropout_rate)

    agg_input = keras.Input(shape=(num_agg_features,), name="agg_input")
    agg_bn = layers.BatchNormalization(name="agg_bn")(agg_input)
    agg_proj = layers.Dense(32, activation="gelu", name="agg_projection")(agg_bn)

    unigram_input, unigram_agg = _size_histogram_branch(unigram_agg_dim, alpha_unigram)

    # ---- how confident is the forest, regardless of what it predicted -------
    # Everything else the forest feeds this model goes through `rf_bn`, a
    # per-dimension batch normalisation over the class probabilities followed
    # by one dense layer. But the statistic that decides monitored versus
    # background is max_j p_j: an order statistic ACROSS dimensions, which
    # per-dimension normalisation scrambles and a dense layer approximates
    # poorly. So it is computed explicitly and handed over unnormalised,
    # together with the normalised entropy; both are already in [0, 1].
    def _conf_feats(p):
        eps = 1e-7
        conf = tf.reduce_max(p, axis=-1, keepdims=True)
        ent = -tf.reduce_sum(p * tf.math.log(p + eps), axis=-1, keepdims=True)
        ent = ent / tf.math.log(tf.cast(tf.shape(p)[-1], tf.float32))
        return tf.concat([conf, ent], axis=-1)

    rf_conf = layers.Lambda(_conf_feats, name="rf_conf_feats")(rf_input)

    rf_bn = layers.BatchNormalization(name="rf_bn")(rf_input)
    rf_proj = layers.Dense(rf_proj_dim, activation="gelu", name="rf_projection")(rf_bn)

    # ---- how much to trust the forest, per trace ---------------------------
    # A small MLP over the forest's own output scales both the cross-attention
    # contribution and the projected vector. At 0 the model falls back exactly
    # to the sequence-only representation for that trace; at 1 the fusion is
    # unattenuated. The bias starts at +2 (sigmoid(2) ~ 0.88, "trust the forest
    # by default") so training does not have to travel there from a neutral
    # 0.5 before early stopping ends it.
    gate_in = layers.Concatenate(name="gate_input")([rf_bn, agg_proj, rf_conf])
    gate_h = layers.Dense(gate_hidden_dim, activation="gelu", name="gate_hidden")(gate_in)
    rf_trust_gate = layers.Dense(
        1, activation="sigmoid", name="rf_trust_gate",
        bias_initializer=keras.initializers.Constant(2.0))(gate_h)

    # ---- cross-attention fusion --------------------------------------------
    # Query from the forest and the aggregate statistics, keys and values from
    # the encoded sequence. Dropout on the query simulates a degraded forest.
    # The skip connection to `pooled` guarantees graceful degradation to plain
    # average pooling when the query carries nothing.
    q_cat = layers.Concatenate(name="cross_q_cat")([agg_proj, rf_proj])
    q_proj = layers.Dense(d_model, use_bias=False, name="cross_q_proj")(q_cat)
    q_proj = layers.Dropout(rf_branch_dropout, name="cross_q_dropout")(q_proj)
    q_seq = layers.Lambda(lambda t: tf.expand_dims(t, axis=1), name="cross_q_expand")(q_proj)
    cross_out_seq = layers.MultiHeadAttention(
        num_heads=num_heads, key_dim=d_model // num_heads, dropout=dropout_rate,
        name="cross_attn_fusion")(query=q_seq, value=x_seq, key=x_seq,
                                  attention_mask=attn_mask)
    cross_out = layers.Lambda(lambda t: tf.squeeze(t, axis=1),
                              name="cross_attn_squeeze")(cross_out_seq)
    # The trust coefficient scales the cross-attention delta, not the residual:
    # at 0, cross_out is exactly `pooled`.
    cross_out = layers.Multiply(name="cross_attn_gated")([cross_out, rf_trust_gate])
    cross_out = layers.Add(name="cross_skip")([cross_out, pooled])
    cross_out = layers.LayerNormalization(epsilon=1e-6, name="cross_skip_norm")(cross_out)
    rf_proj_gated = layers.Multiply(name="rf_proj_gated")([rf_proj, rf_trust_gate])

    # rf_conf is deliberately NOT scaled by the trust coefficient: the
    # coefficient attenuates the forest when the forest looks unreliable, and a
    # background trace is exactly when it looks unreliable. Scaling it would
    # suppress the evidence at the one moment it matters.
    fused = layers.Concatenate(name="fused_repr")(
        [cross_out, agg_proj, rf_proj_gated, unigram_agg, rf_conf])

    shared = layers.Dropout(dropout_rate, name="pre_cls_dropout")(fused)
    shared = layers.Dense(128, activation="gelu", name="shared_dense")(shared)
    shared = layers.LayerNormalization(epsilon=1e-6, name="shared_norm")(shared)

    # ---- a separate, forest-free stem for the monitored decision ------------
    # The forest must name one of the N monitored websites, so a background
    # trace that resembles a monitored site gets a confident wrong answer, and
    # a fused binary head inherits that tail. Detection therefore gets its own
    # stem, fed only by what the forest never touched: `pooled` (not
    # `cross_out`, whose query is built from the forest), the aggregate
    # statistics and the size histogram -- plus `rf_conf`. That distinction is
    # the point: the detection head wants "how sure is the forest", a scalar,
    # and specifically not "which website", the vector that carries the
    # confidently-wrong tail.
    bin_fused = layers.Concatenate(name="fused_repr_binary")(
        [pooled, agg_proj, unigram_agg, rf_conf])
    bin_shared = layers.Dropout(dropout_rate, name="pre_cls_dropout_binary")(bin_fused)
    bin_shared = layers.Dense(128, activation="gelu", name="shared_dense_binary")(bin_shared)
    bin_shared = layers.LayerNormalization(epsilon=1e-6, name="shared_norm_binary")(bin_shared)

    if n1_head:
        # One softmax over N+1 classes, background as class N: the standard
        # formulation, where detection and identification are coupled because
        # probability mass given to background is mass taken from the monitored
        # websites. Decoupling them is exactly what the two heads do, and this
        # arm measures whether that pays.
        #
        # multi_output and binary_output are left out of the graph: as derived
        # views with zero loss weight they would still get their gradient paths
        # built, making a training step about 4x slower. They are recovered
        # from n1_output after predicting, with split_n1.
        multi_x = layers.Dropout(dropout_rate, name="multi_dropout")(shared)
        n1_output = layers.Dense(num_classes + 1, activation="softmax",
                                 name="n1_output")(multi_x)
        outputs = {"n1_output": n1_output, "aux_multi_output": aux_multi_output}
    else:
        bin_x = layers.Dropout(dropout_rate, name="binary_dropout")(bin_shared)
        binary_output = layers.Dense(1, activation="sigmoid", name="binary_output")(bin_x)

        multi_x = layers.Dropout(dropout_rate, name="multi_dropout")(shared)
        multi_output = layers.Dense(num_classes, activation="softmax",
                                    name="multi_output")(multi_x)
        outputs = {"binary_output": binary_output, "multi_output": multi_output,
                   "aux_multi_output": aux_multi_output}

    return keras.Model(
        inputs={"transformer_input": seq_input, "agg_input": agg_input,
                "rf_input": rf_input, "size_vocab_input": size_vocab_input,
                "unigram_agg_input": unigram_input},
        outputs=outputs,
        name="fold",
    )
