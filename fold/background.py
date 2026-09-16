"""Synthetic background traffic for the open-world experiments.

The open world needs traces from outside the monitored set.  They are
synthetic: a Gaussian copula (Synthetic Data Vault) is fit on the statistics of
the real traces -- how many queries a visit makes, and the location and scale of
the packet sizes and times of each role -- and then sampled to invent new
"sites".  Only real traces are used to *fit* the generator; no real trace ever
enters the pool or is labelled background.

Statistically close background is the harder case for the attack, so this is
the worst case for the adversary rather than a convenient one.

Reconstruction
--------------
The synthesiser decides how many queries a fake site makes and how large and
how slow its packets are.  Turning those statistics back into a trace is where
the structure of ODoH has to be respected, and three properties of the real
traffic drive the reconstruction:

* **Order.** A real trace is exactly n_queries repetitions of the four roles in
  order, so the trace is emitted as cycles rather than role by role.
* **Values.** Real packet sizes are whole bytes drawn from a small per-role
  vocabulary -- the proxy query takes only 42 distinct values in the whole
  dataset.  Sizes are therefore snapped to sizes actually observed for that
  role.  Per role, a real site draws a median of 100% of its own sizes from the
  vocabulary of the other sites: sites do not own private byte values, what
  identifies them is which sizes they use and how often.
* **The proxy invariant.** size(proxy response) - size(proxy query) is 89 or 90
  on every one of the 54155 real queries: the response carries the query's
  payload plus a fixed framing.  The response size is therefore derived rather
  than modelled, so the invariant holds by construction.

Within a trace, queries are drawn from a small pool of real query tuples rather
than independently, which reproduces the way a visit re-resolves a handful of
domains.  The proxy column gets its own, smaller vocabulary and a sticky walk
over it: in real traces the lag-1 size autocorrelation lives almost entirely in
the proxy query (+0.41, against ~0 for the other two roles), which is to say
proxy sizes arrive in runs.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

#: Roles in the order in which they appear within a query.
ROLES = (0, 1, 2, 3)

#: Roles the synthesiser models. The proxy response is derived from the proxy
#: query (see PROXY_OFFSET); handing a Gaussian copula two perfectly correlated
#: columns is both redundant and numerically bad.
GENERATED_ROLES = (0, 2, 3)

#: The two values of size(proxy response) - size(proxy query), measured on
#: every real query.
PROXY_OFFSET = (89.0, 90.0)

#: Longest real trace, in queries.
MAX_QUERIES = 148


def trace_features(trace) -> dict:
    """Summarise a trace for the synthesiser.

    The number of queries, plus the location and scale of size and time for
    each modelled role.  Statistics of the time *values* are used rather than
    of the inter-arrivals: the time column of the dataset is a per-packet
    duration, not a clock, so differencing it telescopes to nearly zero and
    would carry no information.
    """
    arr = np.asarray(trace, dtype=float)
    row = {"n_queries": float(len(arr) // 4)}
    for r in GENERATED_ROLES:
        m = arr[:, 2] == r
        t, s = arr[m, 0], arr[m, 1]
        row[f"r{r}_mean_t"] = float(t.mean()) if len(t) else 0.0
        row[f"r{r}_std_t"] = float(t.std()) if len(t) > 1 else 0.0
        row[f"r{r}_mean_size"] = float(s.mean()) if len(s) else 100.0
        row[f"r{r}_std_size"] = float(s.std()) if len(s) > 1 else 0.0
    m1 = arr[:, 2] == 1
    row["r1_mean_t"] = float(arr[m1, 0].mean()) if m1.any() else 0.0
    row["r1_std_t"] = float(arr[m1, 0].std()) if m1.sum() > 1 else 0.0
    return row


class RealPools:
    """Pools of real values that give the synthetic traces a real shape.

    ``tuples`` has shape (queries, 4, 2): for every real query, the time and
    size of its four packets in role order.  Resampling whole queries preserves
    the correlation within a query that per-role sampling would destroy.
    """

    def __init__(self, traces):
        tuples = []
        for trace in traces:
            a = np.asarray(trace, dtype=float)
            if len(a) % 4:
                continue
            tuples.append(a.reshape(-1, 4, 3)[:, :, :2])   # drop the role column
        self.tuples = np.concatenate(tuples, axis=0)

        self.t_mean = self.tuples[:, :, 0].mean(axis=0)
        self.t_std = self.tuples[:, :, 0].std(axis=0)
        self.s_mean = self.tuples[:, :, 1].mean(axis=0)
        self.s_std = self.tuples[:, :, 1].std(axis=0)
        self.t_lo = self.tuples[:, :, 0].min(axis=0)
        self.t_hi = self.tuples[:, :, 0].max(axis=0)
        self.sizes = [np.unique(self.tuples[:, r, 1]) for r in ROLES]

        # The empirical mix of the two proxy offsets, measured on this pool.
        d = self.tuples[:, 1, 1] - self.tuples[:, 2, 1]
        p = np.array([float(np.mean(np.abs(d - o) < 1e-9)) for o in PROXY_OFFSET])
        self.offset_p = p / p.sum() if p.sum() > 0 else np.array([0.5, 0.5])

    def snap(self, role: int, x):
        """Snap sizes to the nearest size actually observed for this role."""
        pool = self.sizes[role]
        return pool[np.abs(pool[:, None] - np.asarray(x)[None, :]).argmin(axis=0)]


def _remap(x, src_mu, src_sd, dst_mu, dst_sd):
    """Move real values to a target mean and standard deviation, keeping shape."""
    if src_sd < 1e-12:
        return np.full_like(x, dst_mu)
    return dst_mu + (x - src_mu) * (dst_sd / src_sd)


def features_to_trace(row, pools: RealPools, rng: np.random.Generator,
                      vocab_ratio: float, proxy_vocab_ratio: float,
                      seq_repeat: float) -> list:
    """Rebuild one trace from the statistics the synthesiser produced."""
    n_q = max(1, min(int(round(float(row["n_queries"]))), MAX_QUERIES))

    # The handful of "domains" this fake site resolves, and the queries drawn
    # from them. Without this the trace would use a distinct size per query.
    k = max(1, int(round(vocab_ratio * n_q)))
    vocab = rng.integers(0, len(pools.tuples), size=k)
    tup = pools.tuples[vocab[rng.integers(0, k, size=n_q)]].copy()

    # The proxy gets its own, usually smaller vocabulary. Within a query the
    # correlation is +0.833 between client and target but only +0.058 and
    # +0.033 for the proxy against either, so a single shared vocabulary does
    # useful work for the first pair and none for the proxy -- while forcing
    # all three to the same vocabulary size, where a real trace uses a median
    # of 6 distinct sizes for client and target and 2 for the proxy.
    k_p = max(1, int(round(proxy_vocab_ratio * n_q)))
    proxy_vocab = rng.integers(0, len(pools.tuples), size=k_p)

    if seq_repeat > 0.0 and k_p > 1:
        # A sticky walk, so proxy sizes recur in runs without touching the
        # order of the other two roles.
        seq = np.empty(n_q, dtype=int)
        fresh = rng.integers(0, k_p, size=n_q)
        stay = rng.random(n_q) < seq_repeat
        seq[0] = fresh[0]
        for j in range(1, n_q):
            seq[j] = seq[j - 1] if stay[j] else fresh[j]
        tup[:, 2, :] = pools.tuples[proxy_vocab[seq]][:, 2, :]
    else:
        tup[:, 2, :] = pools.tuples[proxy_vocab[rng.integers(0, k_p, size=n_q)]][:, 2, :]

    for r in GENERATED_ROLES:
        t = _remap(tup[:, r, 0], pools.t_mean[r], pools.t_std[r],
                   max(0.0, float(row[f"r{r}_mean_t"])),
                   max(0.0, float(row[f"r{r}_std_t"])))
        s = _remap(tup[:, r, 1], pools.s_mean[r], pools.s_std[r],
                   float(row[f"r{r}_mean_size"]), max(0.0, float(row[f"r{r}_std_size"])))
        tup[:, r, 0] = np.clip(t, pools.t_lo[r], pools.t_hi[r])
        tup[:, r, 1] = pools.snap(r, s)

    # The proxy response: its timing is modelled, its size derived.
    t1 = _remap(tup[:, 1, 0], pools.t_mean[1], pools.t_std[1],
                max(0.0, float(row["r1_mean_t"])), max(0.0, float(row["r1_std_t"])))
    tup[:, 1, 0] = np.clip(t1, pools.t_lo[1], pools.t_hi[1])
    tup[:, 1, 1] = tup[:, 2, 1] + rng.choice(PROXY_OFFSET, size=n_q, p=pools.offset_p)

    return [[float(tup[q, r, 0]), float(tup[q, r, 1]), float(r)]
            for q in range(n_q) for r in ROLES]


def build_pool(fit_traces, size: int, seed: int, vocab_ratio: float,
               proxy_vocab_ratio: float, seq_repeat: float) -> list:
    """Fit the synthesiser on real traces and sample ``size`` background traces."""
    from sdv.metadata import Metadata
    from sdv.single_table import GaussianCopulaSynthesizer

    pools = RealPools(fit_traces)
    df = pd.DataFrame([trace_features(t) for t in fit_traces])
    synth = GaussianCopulaSynthesizer(Metadata.detect_from_dataframe(df))
    synth.fit(df)
    sampled = synth.sample(num_rows=size, output_file_path=None)

    rng = np.random.default_rng(seed=seed)
    return [features_to_trace(row, pools, rng, vocab_ratio, proxy_vocab_ratio, seq_repeat)
            for _, row in sampled.iterrows()]


def pool_to_frame(traces) -> pd.DataFrame:
    """One row per packet: trace_id, pkt_idx, t, size, role.

    The pool is written flat, without a train/validation/test split: each
    repetition reshuffles it with its own seed and carves its own disjoint
    slices, so different repetitions draw different background traces.
    """
    rows = [(tid, k, float(p[0]), float(p[1]), int(round(float(p[2]))))
            for tid, trace in enumerate(traces)
            for k, p in enumerate(trace)]
    return pd.DataFrame(rows, columns=["trace_id", "pkt_idx", "t", "size", "role"])
