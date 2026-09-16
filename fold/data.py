"""Loading, splitting, defending and representing ODoH traces.

A *trace* is one page load: a variable-length list of packets, each of them a
``[timestamp, size, role]`` triple.  The four roles are the two ODoH segments
times the two directions.  Everything the attack sees is derived from these
three quantities, which are what encryption leaves exposed.
"""
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple
import csv
import random

import numpy as np


# ---------------------------------------------------------------------------
# Packet roles
# ---------------------------------------------------------------------------
# The role names follow the structure measured on the 54155 queries of the
# dataset, not the column names of the CSV:
#
#   Input-U - Out-U = 89 or 90, always (correlation 1.0000)
#       -> the same message on the two segments, up to the ODoH encapsulation:
#          the QUERY, client->proxy (Input-U) and proxy->target (Out-U).
#   Input-D <-> Out-D correlate 0.833, with a variable difference (-147..+701)
#       -> the RESPONSE on the two segments, with variable padding/re-encoding.
#   Query/response pairs on the same segment correlate 0.03-0.06.
#
# The sizes confirm the direction: query 135 -> response 249 on the client
# segment, 46 -> 140 on the target one (DNS responses are the larger message).
PROXY_RESPONSE = 0   # Input-D : proxy  -> client, the response delivered to the client
CLIENT_QUERY = 1     # Input-U : client -> proxy,  the query
PROXY_QUERY = 2      # Out-U   : proxy  -> target, the same query, minus 89/90 of header
TARGET_RESPONSE = 3  # Out-D   : target -> proxy,  the response

#: Each packet carries a timestamp, a size and a role.
PACKET_FEATURES = 3

#: What a packet becomes in the sequential representation: those three, plus
#: the five derived in :func:`_add_derived_features`.
SEQUENCE_FEATURES = 8

#: The 4 -> 2 semantic collapse, by measured structure rather than by name.
QUERY_ROLES = (CLIENT_QUERY, PROXY_QUERY)
RESPONSE_ROLES = (PROXY_RESPONSE, TARGET_RESPONSE)

ROLES = (CLIENT_QUERY, PROXY_RESPONSE, PROXY_QUERY, TARGET_RESPONSE)


def _one_hot(label: int, num_classes: int) -> List[int]:
    out = [0] * num_classes
    out[label] = 1
    return out


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def load_traces(csv_path: Path) -> Tuple[List[List[List[float]]], List[int], List[str]]:
    """Read the dataset CSV into per-visit traces.

    One CSV row is one ODoH resolution and expands to four packets; rows
    sharing a ``trace_id`` (``location|round|subcollection|site``) form one
    page load.  Every sample therefore comes from a single capture and so from
    a single client location, which the leave-one-location-out experiment
    relies on.

    Returns ``(traces, labels, locations)`` as parallel lists.
    """
    groups: dict = {}
    order: List[str] = []
    with Path(csv_path).open("r", newline="") as f:
        for row in csv.DictReader(f):
            tid = row["trace_id"]
            packets = [
                [float(row["Input-D-Time"]), float(row["Input-D-Size"]), PROXY_RESPONSE],
                [float(row["Input-U-Time"]), float(row["Input-U-Size"]), CLIENT_QUERY],
                [float(row["Out-U-Time"]), float(row["Out-U-Size"]), PROXY_QUERY],
                [float(row["Out-D-Time"]), float(row["Out-D-Size"]), TARGET_RESPONSE],
            ]
            if tid not in groups:
                groups[tid] = {"packets": [],
                               "label": int(float(row["categorical_label"])),
                               "location": row["location"]}
                order.append(tid)
            groups[tid]["packets"].extend(packets)

    traces = [groups[t]["packets"] for t in order]
    labels = [groups[t]["label"] for t in order]
    locations = [groups[t]["location"] for t in order]
    return traces, labels, locations


def load_balanced(csv_path: Path) -> Tuple[List[List[List[float]]], List[int], int]:
    """Load the dataset and interleave the classes.

    Traces are shuffled within each class and then interleaved, so that any
    contiguous slice of the result is balanced across websites.  The shuffle
    uses a fixed seed: the balanced ordering must be identical in every process
    of a run, otherwise two jobs of the same experiment would train and test on
    different trace subsets.  Per-experiment variation comes from the seeded
    permutation in :func:`fold.experiments.load_and_split`, not from here.
    """
    traces, labels, _ = load_traces(csv_path)
    num_classes = max(labels) + 1

    labels_array = np.array(labels)
    indices_by_class = [np.where(labels_array == c)[0] for c in range(num_classes)]

    rng = np.random.default_rng(0)
    for class_indices in indices_by_class:
        rng.shuffle(class_indices)

    balanced_indices = []
    for i in range(max(len(idx) for idx in indices_by_class)):
        for class_indices in indices_by_class:
            if i < len(class_indices):
                balanced_indices.append(class_indices[i])

    traces = [traces[i] for i in balanced_indices]
    labels = [labels[i] for i in balanced_indices]
    return traces, labels, num_classes


def split_data(*arrays, val_ratio: float = 0.10, test_ratio: float = 0.25):
    """Split arrays into (train, val, test) with the paper's 65/10/25 ratio."""
    n = len(arrays[0])
    test_idx = int(n * (1 - test_ratio))
    val_idx = int(n * (1 - test_ratio - val_ratio))
    return [(arr[:val_idx], arr[val_idx:test_idx], arr[test_idx:]) for arr in arrays]


# ---------------------------------------------------------------------------
# WFSafeTLS defense
# ---------------------------------------------------------------------------
# The defense operates on TLS records, above TCP/IP.  The dataset stores
# application-layer payload sizes with no TLS/TCP/IP framing, so:
#   * padding adds N x 32B (HMAC-derived blocks) to a record's payload;
#   * fragmentation splits a record's payload over several TLS records
#     (recursive, two-sided, depth <= 4, each fragment >= MESSAGE_SIZE);
#   * dummy injection appends a fake TLS record (content + 32B HMAC mark).
# Every record created anew (each fragment beyond the first, and every dummy)
# pays the TLS record framing seen on the wire: 5B record header + 1B inner
# content-type + 16B AEAD tag = 22B.  Lower-layer headers are not represented
# in this dataset and are therefore not added.
_PADDING_UNIT = 32    # bytes per padding block (HMAC-derived)
_MESSAGE_SIZE = 32    # minimum fragment payload / HMAC message window
_MARK_SIZE = 32       # dummy-record HMAC mark length
_TLS_OVERHEAD = 22    # 5B record header + 1B content-type + 16B AEAD tag
_MAX_DEPTH = 4        # recursive fragmentation depth (both sides)
_SIZE_CAP = 1349      # per-record payload budget, to stay under the MSS


def _cut_points(size: int, p_frag: float) -> List[int]:
    """Two-sided recursive fragmentation cut points over ``[0, size]``.

    At each node, with probability ``p_frag``, split the interval in two (each
    side >= MESSAGE_SIZE) and recurse on both halves, up to MAX_DEPTH.  Returns
    the interior cut positions in sorted order.
    """
    cuts: List[int] = []

    def build(start: int, end: int, depth: int) -> None:
        if depth >= _MAX_DEPTH or end - start < 2 * _MESSAGE_SIZE:
            return
        if random.random() >= p_frag:
            return
        cut = start + _MESSAGE_SIZE + random.randint(0, end - start - 2 * _MESSAGE_SIZE)
        build(start, cut, depth + 1)      # left first, so the cuts stay sorted
        cuts.append(cut)
        build(cut, end, depth + 1)

    build(0, size, 0)
    return cuts


def _padding_bytes(payload: int) -> int:
    """Random padding in whole units, capped toward SIZE_CAP; 0 when no room."""
    max_pad = _SIZE_CAP - payload - _PADDING_UNIT
    if max_pad <= 0:
        return 0
    padding = random.randint(0, max_pad - 1) + _PADDING_UNIT
    return (padding // _PADDING_UNIT) * _PADDING_UNIT


def apply_defense(traces: List[List[List[float]]], config: dict) -> List[List[List[float]]]:
    """Simulate WFSafeTLS on recorded traces.

    Each mechanism is applied independently to every record with its own
    probability, whatever the record's role.  Config keys are ``padding``,
    ``dummy`` and ``fragmentation``; a falsy config returns the traces
    unchanged.

    Simulating the defense offline reproduces the packet-level changes a live
    deployment would produce, because these mechanisms only alter packet sizes
    and the structure of the sequence, both of which the dataset records.
    """
    if not config:
        return list(traces)

    p_padd = config.get("padding", 0.0)
    p_dummy = config.get("dummy", 0.0)
    p_frag = config.get("fragmentation", 0.0)

    defended = []
    for trace in traces:
        new_trace = []
        for timestamp, size, role in trace:
            size_i = int(size)
            role_f = float(role)

            # fragmentation: split the payload into contiguous TLS records
            cuts = _cut_points(size_i, p_frag) if p_frag > 0 else []
            bounds = [0] + cuts + [size_i]
            payloads = [bounds[i + 1] - bounds[i] for i in range(len(bounds) - 1)]

            ts = timestamp
            for idx, payload in enumerate(payloads):
                if idx > 0:
                    ts += random.uniform(0, 0.0002)   # a new record is a new packet
                rec_size = payload
                if idx > 0:
                    rec_size += _TLS_OVERHEAD         # each new record pays the framing
                if p_padd > 0 and random.random() < p_padd:
                    rec_size += _padding_bytes(payload)
                new_trace.append([ts, float(rec_size), role_f])

            # dummy record after the data (one per write)
            if p_dummy > 0 and random.random() < p_dummy:
                content = random.randint(_MESSAGE_SIZE, _SIZE_CAP - _MARK_SIZE)
                ts += random.uniform(0, 0.0002)
                new_trace.append([ts, float(content + _MARK_SIZE + _TLS_OVERHEAD), role_f])

        defended.append(new_trace)
    return defended


# ---------------------------------------------------------------------------
# Representations
# ---------------------------------------------------------------------------
# Fixed normalisation constants for the three packet-level features, applied
# only to the sequential representation:
#   feature[0] = timestamp    - batch normalisation inside the model handles it
#   feature[1] = size (bytes) - typical range 54-1500 B, divided by 1500
#   feature[2] = role         - values 0-3, divided by 3
# Padding zeros survive unchanged, because 0 / c == 0.
_SIZE_SCALE = 1500.0
_TYPE_SCALE = 3.0

_F_TIME, _F_SIZE, _F_TYPE = 0, 1, 2

#: Quantile buckets of the packet-size vocabulary, shared by the forest's
#: histogram and the transformer's size embedding.
VOCAB_BUCKETS = 64


def pad_traces(traces: List[List[List[float]]], target_length: int) -> np.ndarray:
    """Pad (or truncate) traces with zeros so they share a common length."""
    padded = []
    for trace in traces:
        if len(trace) > target_length:
            padded.append(trace[:target_length])
        else:
            padded.append(trace + [[0.0] * PACKET_FEATURES] * (target_length - len(trace)))
    return np.asarray(padded, dtype=np.float32)


def one_hot(labels: List[int], num_classes: int) -> np.ndarray:
    """Website labels, one-hot."""
    return np.array([_one_hot(l, num_classes) for l in labels], dtype=np.float32)


def _global_stats(padded: np.ndarray) -> np.ndarray:
    """The three whole-trace quantities late-fused after pooling:
    number of real packets, sum of timestamps, total size."""
    n_packets = np.sum(np.any(padded != 0, axis=2), axis=1, keepdims=True)
    total_time = np.sum(padded[:, :, 0], axis=1, keepdims=True)
    total_size = np.sum(padded[:, :, 1], axis=1, keepdims=True)
    return np.concatenate([n_packets, total_time, total_size], axis=1).astype(np.float32)


def _add_derived_features(seq: np.ndarray) -> np.ndarray:
    """Append five per-packet features derived from the neighbouring packets.

    Input  (batch, seq_len, 3): [timestamp, size/1500, role/3]
    Output (batch, seq_len, 8), with
      [3] delta_t     - inter-packet time (0.0 for the first real packet)
      [4] cum_size    - cumulative normalised size up to this packet
      [5] delta_size  - size change against the previous packet
      [6] type_change - 1.0 when the role differs from the previous packet; a
                        boundary marker that survives dummy injection and
                        fragmentation breaking the fixed [0,1,2,3] cycle
      [7] is_response - 1.0 for PROXY_RESPONSE or TARGET_RESPONSE

    Padding rows are all-zero and their derived features are forced back to
    zero, so the masking inside the model stays valid.
    """
    real = np.any(seq != 0.0, axis=-1)          # (batch, seq_len)

    t = seq[:, :, _F_TIME]
    s = seq[:, :, _F_SIZE]
    p = seq[:, :, _F_TYPE]

    delta_t = np.zeros_like(t)
    delta_t[:, 1:] = np.diff(t, axis=1)
    delta_t = np.clip(delta_t, 0.0, None)       # never negative after defense jitter

    cum_size = np.cumsum(s, axis=1)
    max_cum = cum_size[:, -1:].copy()
    max_cum[max_cum == 0.0] = 1.0
    cum_size = cum_size / max_cum

    delta_size = np.zeros_like(s)
    delta_size[:, 1:] = np.diff(s, axis=1)

    type_change = np.zeros_like(p)
    type_change[:, 1:] = (np.diff(p, axis=1) != 0.0).astype(np.float32)

    raw_type = np.round(p * _TYPE_SCALE).astype(int)
    is_response = ((raw_type == PROXY_RESPONSE) | (raw_type == TARGET_RESPONSE)).astype(np.float32)

    derived = np.stack([delta_t, cum_size, delta_size, type_change, is_response], axis=-1)
    derived[~real] = 0.0
    return np.concatenate([seq, derived], axis=-1).astype(np.float32)


def sequence_features(traces, target_length: int) -> Tuple[np.ndarray, np.ndarray]:
    """The ordered view of a trace: one feature vector per packet.

    Returns the (batch, target_length, 8) sequence and the three whole-trace
    statistics that are fused back in after pooling.
    """
    padded = pad_traces(traces, target_length)
    stats = _global_stats(padded)

    seq = padded.copy()
    seq[..., _F_SIZE] /= _SIZE_SCALE
    seq[..., _F_TYPE] /= _TYPE_SCALE
    return _add_derived_features(seq), stats


def direction_channel(traces, target_length: int) -> np.ndarray:
    """Deep Fingerprinting's view: packet direction only, +1/-1, 0 for padding.

    DF was designed around a two-symbol alphabet, and porting it means keeping
    that alphabet.  ODoH has four roles rather than two directions, so both
    segments are collapsed onto one signed channel by the measured structure:
    Input-U and Out-U are the same query on the two segments and get +1,
    Input-D and Out-D are the response and get -1.
    """
    out = np.zeros((len(traces), target_length, 1), dtype=np.float32)
    for i, trace in enumerate(traces):
        for j, (_ts, _size, role) in enumerate(trace[:target_length]):
            out[i, j, 0] = 1.0 if int(role) in QUERY_ROLES else -1.0
    return out


def aggregate_features(traces) -> np.ndarray:
    """The order-independent view: 33 aggregate statistics per trace.

      global (5)        : n_packets, total_size, time_span, mean_size, std_size
      per role (4 x 6)  : count, mean_size, std_size, sum_size, mean_ts, std_ts
      consecutive (4)   : mean_delta_t, std_delta_t, mean|delta_size|, std|delta_size|
    """
    rows = []
    for trace in traces:
        if not trace:
            rows.append(np.zeros(33, dtype=np.float32))
            continue

        arr = np.array(trace, dtype=np.float32)
        timestamps, sizes, types = arr[:, 0], arr[:, 1], arr[:, 2].astype(int)

        feats: List[float] = [
            float(len(arr)),
            float(np.sum(sizes)),
            float(timestamps[-1] - timestamps[0]) if len(arr) > 1 else 0.0,
            float(np.mean(sizes)),
            float(np.std(sizes)),
        ]

        for role in ROLES:
            mask = types == role
            if mask.sum() == 0:
                feats.extend([0.0] * 6)
            else:
                r_sizes, r_ts = sizes[mask], timestamps[mask]
                feats.extend([
                    float(mask.sum()),
                    float(np.mean(r_sizes)),
                    float(np.std(r_sizes)),
                    float(np.sum(r_sizes)),
                    float(np.mean(r_ts)),
                    float(np.std(r_ts)),
                ])

        if len(arr) > 1:
            delta_t = np.diff(timestamps)
            delta_size = np.abs(np.diff(sizes))
            feats.extend([float(np.mean(delta_t)), float(np.std(delta_t)),
                          float(np.mean(delta_size)), float(np.std(delta_size))])
        else:
            feats.extend([0.0, 0.0, 0.0, 0.0])

        rows.append(feats)

    return np.array(rows, dtype=np.float32)


# ---------------------------------------------------------------------------
# Packet-size quantisation
# ---------------------------------------------------------------------------

def fit_size_vocab(train_traces: List[List[List[float]]], n_buckets: int) -> np.ndarray:
    """Fit packet-size quantile bucket edges on the training traces only.

    Returns ``n_buckets - 1`` edges, so that ``np.digitize(size, edges)`` maps a
    size to a bucket in ``0..n_buckets-1``.  The same edges must be reused for
    validation and test: refitting them there would leak the test distribution.
    """
    sizes = [float(p[1]) for trace in train_traces for p in trace]
    quantiles = np.linspace(0, 1, n_buckets + 1)[1:-1]
    return np.quantile(sizes, quantiles)


def size_vocab_indices(traces, bucket_edges: np.ndarray, target_length: int) -> np.ndarray:
    """Per-packet bucket index, padded to ``target_length``.

    Index 0 is reserved for padding, so a real packet lands in ``1..n_buckets``.
    """
    out = np.zeros((len(traces), target_length), dtype=np.int32)
    for i, trace in enumerate(traces):
        sizes = np.array([p[1] for p in trace[:target_length]], dtype=np.float32)
        if len(sizes) == 0:
            continue
        out[i, :len(sizes)] = np.digitize(sizes, bucket_edges) + 1
    return out


def size_histogram(traces, bucket_edges: np.ndarray) -> np.ndarray:
    """How many packets of a trace fall into each size bucket.

    This is the order-independent packet-size view: the same sizes in any order
    give the same histogram, which is what keeps it usable when a defense
    perturbs the packet order.
    """
    n_buckets = len(bucket_edges) + 1
    out = np.zeros((len(traces), n_buckets), dtype=np.float32)
    for i, trace in enumerate(traces):
        if not trace:
            continue
        sizes = np.array([p[1] for p in trace], dtype=np.float32)
        out[i] = np.bincount(np.digitize(sizes, bucket_edges),
                             minlength=n_buckets).astype(np.float32)
    return out


# ---------------------------------------------------------------------------
# Turning traces into model inputs
# ---------------------------------------------------------------------------

@dataclass
class Encoder:
    """Turns traces into the inputs each model reads.

    Everything it needs beyond the traces themselves is fixed once, on the
    training traces of the run: how many websites there are, how long a padded
    sequence is, and where the packet-size buckets fall.  Reusing the same
    encoder for validation and test is the point -- refitting the buckets on
    either would leak their size distribution into the attack.

    The methods return inputs only.  Labels are built separately, by
    :meth:`labels`, because background traces have inputs but no website.
    """
    num_classes: int
    target_length: int
    size_edges: np.ndarray

    @classmethod
    def fit(cls, train_traces, num_classes: int, target_length: int,
            n_buckets: int = VOCAB_BUCKETS) -> "Encoder":
        return cls(num_classes, target_length, fit_size_vocab(train_traces, n_buckets))

    @property
    def vocab_size(self) -> int:
        """Size of the packet-size vocabulary, padding index included."""
        return len(self.size_edges) + 2

    @property
    def histogram_bins(self) -> int:
        return len(self.size_edges) + 1

    def sequence(self, traces, forest_probs=None) -> dict:
        """The inputs of TR and FOLD; FOLD also reads the forest's output."""
        seq, stats = sequence_features(traces, self.target_length)
        inputs = {
            "transformer_input": seq,
            "agg_input": stats,
            "size_vocab_input": size_vocab_indices(traces, self.size_edges,
                                                   self.target_length),
            "unigram_agg_input": size_histogram(traces, self.size_edges),
        }
        if forest_probs is not None:
            inputs["rf_input"] = np.asarray(forest_probs, dtype=np.float32)
        return inputs

    def forest(self, traces) -> np.ndarray:
        """The forest's features: the aggregate statistics and the histogram."""
        return np.concatenate([aggregate_features(traces),
                               size_histogram(traces, self.size_edges)], axis=1)

    def cnn(self, traces) -> dict:
        """The convolutional baseline reads the raw padded packets."""
        return {"cnn_input": pad_traces(traces, self.target_length)[..., np.newaxis]}

    def df(self, traces) -> dict:
        return {"df_input": direction_channel(traces, self.target_length)}

    def labels(self, labels: List[int], num_classes: Optional[int] = None):
        """``(website one-hot, monitored)`` for a set of monitored traces."""
        n = self.num_classes if num_classes is None else num_classes
        return one_hot(labels, n), np.ones(len(labels), dtype=np.float32)
