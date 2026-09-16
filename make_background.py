#!/usr/bin/env python
"""Generate the synthetic background pool used by the open-world experiments.

The pool is built once and reused by every repetition: background traffic
depends neither on the defense, which only touches monitored traces, nor on the
monitored split, since a background trace never comes from a monitored site.
Regenerating it per experiment would only add noise and time.

The defaults below are the ones behind the published results, and the seeds are
pinned, so the command needed to rebuild that pool is

    python make_background.py

Every parameter is also written into the header of the CSV, which makes an
existing pool self-describing.

The pool is written flat, one row per packet:

    trace_id,pkt_idx,t,size,role
"""
import argparse
import time
from pathlib import Path

import numpy as np

from fold.background import build_pool, pool_to_frame
from fold.data import load_balanced

HERE = Path(__file__).resolve().parent

#: Seed of the fit subset. Pinned and independent of any experiment split, so
#: the generator never sees a selection correlated with a repetition's data.
FIT_SEED = 12345


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset", type=Path, default=HERE / "data" / "odoh_traces.csv",
                   help="the real traces the generator is fit on")
    p.add_argument("--out", type=Path, default=HERE / "data" / "background_pool.csv")
    p.add_argument("--size", type=int, default=40000,
                   help="background traces to generate; enough for a background "
                        "ratio up to 93%%")
    p.add_argument("--n-fit", type=int, default=1800,
                   help="real traces the generator is fit on")
    p.add_argument("--seed", type=int, default=10,
                   help="seed of the trace reconstruction")
    p.add_argument("--vocab-ratio", type=float, default=0.95,
                   help="size vocabulary of a fake site, as a fraction of its "
                        "number of queries")
    p.add_argument("--proxy-vocab-ratio", type=float, default=0.35,
                   help="the same, for the proxy role alone, which in real "
                        "traces reuses far fewer sizes")
    p.add_argument("--seq-repeat", type=float, default=0.55,
                   help="probability that a proxy size repeats the previous one; "
                        "0.55 reproduces the +0.41 lag-1 autocorrelation of the "
                        "real traces")
    return p.parse_args()


def main():
    args = parse_args()

    print(f"[background] loading {args.dataset.name} ...")
    traces, _, _ = load_balanced(args.dataset)
    idx = np.random.default_rng(seed=FIT_SEED).permutation(len(traces))[:min(args.n_fit, len(traces))]
    fit_traces = [traces[i] for i in idx]
    print(f"[background] {len(traces)} real traces, fitting on {len(fit_traces)}")

    t0 = time.time()
    pool = build_pool(fit_traces, args.size, args.seed, args.vocab_ratio,
                      args.proxy_vocab_ratio, args.seq_repeat)
    print(f"[background] generated {len(pool)} traces in {time.time() - t0:.0f}s")

    frame = pool_to_frame(pool)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    header = [
        "# ODoH website-fingerprinting: synthetic background pool",
        f"# generated_at       : {time.strftime('%Y-%m-%dT%H:%M:%S%z')}",
        f"# dataset            : {args.dataset.name}",
        f"# size               : {len(pool)}",
        f"# n_fit              : {args.n_fit}  (drawn with the pinned seed {FIT_SEED})",
        f"# seed               : {args.seed}",
        f"# vocab_ratio        : {args.vocab_ratio}",
        f"# proxy_vocab_ratio  : {args.proxy_vocab_ratio}",
        f"# seq_repeat         : {args.seq_repeat}",
    ]
    with args.out.open("w") as fh:
        fh.write("\n".join(header) + "\n")
        frame.to_csv(fh, index=False)

    print(f"[background] wrote {len(frame)} packet rows -> {args.out} "
          f"({args.out.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
