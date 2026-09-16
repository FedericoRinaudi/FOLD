"""Deep Fingerprinting (Sirinam et al., CCS 2018), as adapted to ODoH.

The architecture follows the paper's Appendix A and the official
implementation (https://github.com/deep-fingerprinting/df,
src/Model_NoDef.py): four blocks of [Conv1D -> BN -> activation] x2 ->
MaxPooling1D -> Dropout, ELU in the first block and ReLU in the others, then
two Dense(512) layers and a softmax.  Filters, kernel size, strides and
dropout rates are the published ones.

Two deviations are deliberate, and both are consequences of trace length.  DF
was designed for Tor traces of 5000 fixed-size cells; an ODoH page load is tens
to a few hundred packets.

* Pooling is reduced from window 8 / stride 4 to 2 / 2.  The official blocks
  compress by stride^4 = 256x, which collapses a short ODoH trace to a couple
  of positions after four blocks.
* The classifier head averages over positions instead of flattening them.
  With so few surviving positions, a Flatten + Dense head gives every position
  its own weights and does not generalise; shrinking the width instead left
  accuracy flat across a 20x parameter range, which is why the head's
  inductive bias is what changes here, not its size.
"""
from tensorflow import keras
from tensorflow.keras import layers

_FILTERS = (32, 64, 128, 256)
_KERNEL_SIZE = 8
_CONV_STRIDE = 1
_POOL_SIZE = 2
_POOL_STRIDE = 2
_BLOCK_DROPOUT = 0.1
_FC_UNITS = 512
_FC1_DROPOUT = 0.7
_FC2_DROPOUT = 0.5


def build_df(sequence_length: int, num_classes: int) -> keras.Model:
    """Inputs ``df_input`` (batch, seq_len, 1), direction only; one softmax head."""
    inputs = keras.Input(shape=(sequence_length, 1), name="df_input")
    x = inputs

    for i, block_filters in enumerate(_FILTERS, start=1):
        activation = "elu" if i == 1 else "relu"
        x = layers.Conv1D(block_filters, _KERNEL_SIZE, strides=_CONV_STRIDE,
                          padding="same", name=f"block{i}_conv1")(x)
        x = layers.BatchNormalization(name=f"block{i}_bn1")(x)
        x = layers.Activation(activation, name=f"block{i}_act1")(x)
        x = layers.Conv1D(block_filters, _KERNEL_SIZE, strides=_CONV_STRIDE,
                          padding="same", name=f"block{i}_conv2")(x)
        x = layers.BatchNormalization(name=f"block{i}_bn2")(x)
        x = layers.Activation(activation, name=f"block{i}_act2")(x)
        x = layers.MaxPooling1D(pool_size=_POOL_SIZE, strides=_POOL_STRIDE,
                                padding="same", name=f"block{i}_pool")(x)
        x = layers.Dropout(_BLOCK_DROPOUT, name=f"block{i}_dropout")(x)

    x = layers.GlobalAveragePooling1D(name="gap")(x)

    x = layers.Dense(_FC_UNITS, kernel_initializer="glorot_uniform", name="fc1")(x)
    x = layers.BatchNormalization(name="fc1_bn")(x)
    x = layers.Activation("relu", name="fc1_act")(x)
    x = layers.Dropout(_FC1_DROPOUT, name="fc1_dropout")(x)

    x = layers.Dense(_FC_UNITS, kernel_initializer="glorot_uniform", name="fc2")(x)
    x = layers.BatchNormalization(name="fc2_bn")(x)
    x = layers.Activation("relu", name="fc2_act")(x)
    x = layers.Dropout(_FC2_DROPOUT, name="fc2_dropout")(x)

    output = layers.Dense(num_classes, activation="softmax", name="multi_output")(x)

    return keras.Model(inputs={"df_input": inputs},
                       outputs={"multi_output": output},
                       name="deep_fingerprinting")
