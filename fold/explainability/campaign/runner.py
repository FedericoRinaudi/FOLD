"""Five-run, 26-cell explainability campaign; dry-run never trains."""
import argparse
import importlib.util
import json
from pathlib import Path
import shlex
import shutil
import subprocess
import sys
import uuid

import numpy as np

from ..core.artifacts import _digest, _metadata
from ..core.configuration import scenario_fields, scenario_tag
from .preflight import (audit, cross_run_availability, require_cross_run_available,
                        write_audit, write_cross_run_audit)
from .selection import prepare_selection, save_selection, validate_selection
from ..train import class_names


def configurations():
    yield scenario_fields("closed-world"), []
    for ratio in (.30, .40, .50, .60, .70):
        yield scenario_fields("open-world", ratio), ["--open-world", "--background-ratio", f"{ratio:.2f}"]
    for family in ("padding", "dummy", "fragmentation", "combined"):
        for p in (.10, .30, .50, .70, 1.):
            mechanisms = ("padding", "dummy", "fragmentation") if family == "combined" else (family,)
            defense = {name: p for name in mechanisms}
            flags = [v for name in mechanisms for v in (f"--defense-{name}", f"{p:.2f}")]
            yield scenario_fields("defense", defense=defense), flags


def jobs(root, dataset, pool, epochs=500, batch_size=16):
    for run in range(1, 6):
        for fields, flags in configurations():
            tag = scenario_tag(fields)
            common = ["--model", "fold", "--experiment-id", str(run), "--dataset", str(dataset),
                      "--background-pool", str(pool), "--epochs", str(epochs), "--batch-size", str(batch_size)]
            train = [sys.executable, "-m", "fold.explainability.train", *common, "--out-dir", str(root), *flags]
            evaluate = [sys.executable, "run_experiment.py", *common, "--out-dir", str(root / tag), *flags]
            evaluate[evaluate.index("fold")] = "evaluate"
            yield {"configuration": tag, "experiment_id": run, "fields": fields,
                   "training": {"epochs": epochs, "batch_size": batch_size},
                   "path": root / tag / f"exp_{run:03d}", "train": train, "evaluate": evaluate}


def verify_run(path, job, split, *, need_metrics=True):
    """Check provenance, every bundle hash and prediction dimensions before skip."""
    manifest = json.loads((path / "bundle/manifest.json").read_text(encoding="utf-8"))
    if manifest.get("format") != "fold-explainability" or manifest.get("version") != 1:
        raise ValueError("Unsupported manifest")
    metadata = _metadata(manifest["metadata"])
    from .preflight import original_seed
    if (metadata.get("provenance_version") != 1 or metadata["seed"] != original_seed(job["experiment_id"])
            or metadata["experiment_id"] != job["experiment_id"] or metadata["split_ids"] != split):
        raise ValueError("Run split or experiment_id mismatch")
    if metadata["training"] != job["training"]:
        raise ValueError("Existing run uses different requested training parameters")
    for key in ("scenario", "defense", "background_ratio"):
        if metadata.get(key) != job["fields"].get(key):
            raise ValueError(f"Run configuration mismatch: {key}")
    if set(manifest["sha256"]) != {"encoder.npz", "forest.joblib", "fold.weights.h5"}:
        raise ValueError("Incomplete manifest hashes")
    for name, digest in manifest["sha256"].items():
        if _digest(path / "bundle" / name) != digest:
            raise ValueError(f"Bundle integrity failure: {name}")
    for model in ("forest", "fold"):
        with np.load(path / f"{model}.npz", allow_pickle=False) as data:
            p = data["test_monitored_probs"]
            if (p.shape != (len(split["test"]), 100) or not np.isfinite(p).all()
                    or np.any((p < 0) | (p > 1)) or not np.allclose(p.sum(1), 1, atol=1e-5)):
                raise ValueError(f"Invalid {model} monitored predictions")
            if job["fields"]["scenario"] == "open-world":
                back = metadata["background_split_ids"]["test"]
                bp = data["test_background_probs"]
                if (bp.shape != (len(back), 100) or not np.isfinite(bp).all()
                        or np.any((bp < 0) | (bp > 1)) or not np.allclose(bp.sum(1), 1, atol=1e-5)):
                    raise ValueError(f"Invalid {model} background predictions")
                if model == "fold":
                    for population, count in (("monitored", len(split["test"])), ("background", len(back))):
                        score = data[f"test_{population}_score"]
                        if score.size != count or not np.isfinite(score).all() or np.any((score < 0) | (score > 1)):
                            raise ValueError(f"Invalid binary {population} scores")
    if need_metrics:
        report = json.loads((path / "result.json").read_text())
        if report["experiment_id"] != job["experiment_id"] or report["scenario"] != job["fields"]["scenario"]:
            raise ValueError("Metrics provenance mismatch")
        for model in ("forest", "fold"):
            if not all(np.isfinite(report[model][metric]) for metric in ("accuracy", "macro_f1")):
                raise ValueError("Invalid performance metrics")
        labels = np.load(path / "y_test.npy", allow_pickle=False)
        if labels.shape != (len(split["test"]),) or np.any((labels < 0) | (labels >= 100)):
            raise ValueError("Invalid evaluation labels")
    return manifest


def persist_selection(path, selection):
    """Immutable selection: existing different content is a hard error."""
    path = Path(path)
    if path.exists():
        if json.loads(path.read_text()) != selection:
            raise ValueError(f"Existing selection differs: {path}")
    else:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("x", encoding="utf-8") as stream:
            json.dump(selection, stream, indent=2)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--stage", choices=("train", "explain", "all"), default="all")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--dataset", type=Path, default=Path("data/odoh_traces.csv"))
    p.add_argument("--background-pool", type=Path, default=Path("data/background_pool.csv"))
    p.add_argument("--artifacts", type=Path, default=Path("artifacts"))
    p.add_argument("--output", type=Path, default=Path("explanations/all_sites"))
    p.add_argument("--epochs", type=int, default=500)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--trusted", action="store_true", help="Explicitly trust bundle joblib files for inference")
    p.add_argument("--selection", type=Path, default=Path("analysis/cross_run/selections/all_sites_cross_run.json"))
    args = p.parse_args()
    rows, splits, insufficient = audit(args.dataset)
    write_audit("analysis/cross_run/preflight", rows, insufficient)
    names = class_names(args.dataset)
    cross_rows, cross_insufficient = cross_run_availability(splits, names)
    write_cross_run_audit("analysis/cross_run/preflight", cross_rows)
    selection = None
    if not cross_insufficient:
        if args.selection.exists():
            selection = json.loads(args.selection.read_text(encoding="utf-8"))
            validate_selection(selection, splits, names)
        else:
            selection = prepare_selection(splits, names)
        save_selection(selection, path=args.selection)
    plan = list(jobs(args.artifacts, args.dataset, args.background_pool, args.epochs, args.batch_size))
    for job in plan:
        run, tag = job["experiment_id"], job["configuration"]
        path = job["path"]
        state = "NEW"
        if path.exists():
            try:
                verify_run(path, job, splits[run][0])
                state = "VERIFIED_SKIP"
            except (ValueError, KeyError, OSError, TypeError) as error:
                state = f"EXISTING_REQUIRES_CHECK ({error})"
        print(f"{tag} exp_{run:03d} {state}")
        if args.stage in ("train", "all"):
            print(" TRAIN " + shlex.join(job["train"]))
            print(" EVAL  " + shlex.join(job["evaluate"]))
        if args.stage in ("explain", "all"):
            command = [sys.executable, "-m", "fold.explainability.campaign.worker", "--bundle", str(path / "bundle"),
                       "--dataset", str(args.dataset), "--background-pool", str(args.background_pool),
                       "--selection", str(args.selection), "--experiment-id", str(run), "--out-dir",
                       str(args.output / tag / f"exp_{run:03d}"), "--trusted"]
            job["explain"] = command
            print(" XAI   " + shlex.join(command) + (" [FOLD multiclass + RF + binary]" if tag.startswith("openworld") else " [FOLD multiclass + RF]"))
    print(f"TRAINING/EVALUATION JOBS: {len(plan)} = 26 configurations x 5 repetitions")
    print("EXPLANATION JOBS: 130 bundle loads; 130 FOLD multiclass + 130 RF + 25 binary")
    print(f"COHORT GATE: {'BLOCKED' if cross_insufficient else 'PASS'}; insufficient={len(cross_insufficient)}/100; protocol=sample-run observations")
    print(f"Historical per-run shortage={len(insufficient)}/500 (informational; no longer blocks).")
    if selection:
        from collections import Counter
        print(f"SELECTION: {selection['n_observations']} observations; assigned runs={dict(sorted(Counter(r['assigned_experiment_id'] for r in selection['records']).items()))}; reused rows={sum(r['is_reused_trace_id'] for r in selection['records'])}")
    print(f"Environment: TensorFlow={'available' if importlib.util.find_spec('tensorflow') else 'MISSING'}; background pool={'available' if args.background_pool.is_file() else 'MISSING'}")
    existing_bytes = sum(f.stat().st_size for f in args.artifacts.rglob("*") if f.is_file()) if args.artifacts.exists() else 0
    print(f"Disk free={shutil.disk_usage(Path.cwd()).free / 2**30:.2f} GiB; existing artifacts={existing_bytes / 2**30:.2f} GiB")
    print("Raw attribution values lower bound: 30.1 MiB = 26*1500*101*8 bytes; other arrays/CSV/models add overhead.")
    if args.dry_run:
        return
    require_cross_run_available(cross_insufficient)  # BEFORE dependencies or any training subprocess
    if args.stage in ("explain", "all") and not args.trusted:
        raise ValueError("Inference requires --trusted for locally trusted joblib bundles")
    if not args.background_pool.is_file():
        raise ValueError("Missing background pool; generate it with make_background.py before campaign")
    if importlib.util.find_spec("tensorflow") is None:
        raise ValueError("TensorFlow unavailable")
    for job in plan:
        path, run = job["path"], job["experiment_id"]
        if args.stage in ("train", "all"):
            if path.exists():
                verify_run(path, job, splits[run][0], need_metrics=False)
                if not (path / "result.json").exists():
                    subprocess.run(job["evaluate"], check=True)
                verify_run(path, job, splits[run][0])
            else:
                # Training runner reserves its directory. Use an isolated staging
                # root and publish ONLY a fully verified cell; failures preserve it.
                staging = args.artifacts / ".pending" / uuid.uuid4().hex
                command = job["train"].copy()
                command[command.index("--out-dir") + 1] = str(staging)
                staged = staging / job["configuration"] / f"exp_{run:03d}"
                evaluate = job["evaluate"].copy()
                evaluate[evaluate.index("--out-dir") + 1] = str(staged.parent)
                log = Path("logs/all_sites") / job["configuration"] / f"exp_{run:03d}.log"
                log.parent.mkdir(parents=True, exist_ok=True)
                print(f"Executing {job['configuration']} exp_{run:03d}; staging={staged}; log={log}", flush=True)
                with log.open("a", encoding="utf-8") as stream:
                    subprocess.run(command, stdout=stream, stderr=subprocess.STDOUT, check=True)
                    subprocess.run(evaluate, stdout=stream, stderr=subprocess.STDOUT, check=True)
                verify_run(staged, job, splits[run][0])
                path.parent.mkdir(parents=True, exist_ok=True)
                if path.exists():
                    raise ValueError(f"Output appeared concurrently: {path}; staged artifacts preserved")
                staged.rename(path)
        if args.stage in ("explain", "all"):
            verify_run(path, job, splits[run][0])
            subprocess.run(job["explain"], check=True)
