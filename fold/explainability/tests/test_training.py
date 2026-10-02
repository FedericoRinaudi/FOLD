"""Training provenance checks without fitting models."""
import ast
import importlib.util
import json
from pathlib import Path
import random
import re
import subprocess
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from fold.data import apply_defense, load_balanced, load_traces, split_data
from ..core.artifacts import _digest, _metadata, load_forest_bundle
from ..core.configuration import experiment_metadata, scenario_fields, scenario_tag, validate_split
from ..core.data import select_traces
from ..train import class_names, options_from_args, split_record

ROOT = Path(__file__).resolve().parents[3]
DATASET = ROOT / "data" / "odoh_traces.csv"


def arguments(*flags):
    from run_experiment import parse_args

    argv = ["--model", "fold", "--experiment-id", "1", "--out-dir", "artifacts", *flags]
    with patch.object(sys, "argv", ["train", *argv]):
        return options_from_args(parse_args(), argv=argv)


class ConfigurationTests(unittest.TestCase):
    def test_campaign_cells(self):
        campaign = (ROOT / "run_campaign.sh").read_text()
        def values(name):
            return [float(v) for v in re.search(rf"^{name}=\(([^)]+)\)", campaign, re.M)[1].split()]

        cells = [arguments()]
        for ratio in values("RATIOS"):
            cells.append(arguments("--open-world", "--background-ratio", str(ratio)))
        for intensity in values("PROBS"):
            for mechanisms in (("padding",), ("dummy",), ("fragmentation",),
                               ("padding", "dummy", "fragmentation")):
                flags = [arg for name in mechanisms for arg in (f"--defense-{name}", str(intensity))]
                cells.append(arguments(*flags))
        self.assertEqual(len(cells), 26)
        self.assertEqual(len({str(c["out_dir"]) for c in cells}), 26)

    def test_invalid_combinations(self):
        for flags in (("--open-world", "--defense-padding", "0.3"),
                      ("--defense-dummy", "0.3", "--background-ratio", "0.5"),
                      ("--defense-dummy", "0.3", "--background-r", "0.5"),
                      ("--defense-padding", "nan"), ("--defense-padding", "1.1"),
                      ("--open-world", "--background-ratio", "1"),
                      ("--open-world", "--background-ratio", "nan"),
                      ("--defense-padding", "0.3", "--defense-dummy", "0.3"),
                      ("--n1-head",), ("--epochs", "0"), ("--experiment-id", "-1")):
            with self.subTest(flags=flags), self.assertRaises(ValueError):
                arguments(*flags)
        with self.assertRaises(ValueError):
            scenario_fields("defense", defense={"padding": 0.1, "dummy": 0.3, "fragmentation": 0.3})

    def test_deterministic_metadata(self):
        cfg = SimpleNamespace(**arguments(), seed=59, is_open_world=False, is_defended=False)
        first = experiment_metadata(cfg)
        self.assertEqual(json.dumps(first, sort_keys=True), json.dumps(experiment_metadata(cfg), sort_keys=True))
        self.assertEqual(first["seeds"]["forest_final"], 1)
        self.assertEqual(first["seeds"]["oof_forests"], [1, 2, 3, 4, 5])
        self.assertEqual(scenario_tag(first), "baseline")
        self.assertNotEqual(scenario_tag(scenario_fields("open-world", 0.301)),
                            scenario_tag(scenario_fields("open-world", 0.304)))

    def test_legacy_metadata_and_trust(self):
        legacy = {"scenario": "closed-world", "forest_role": "final"}
        self.assertEqual(_metadata(legacy), legacy)
        with self.assertRaises(ValueError):
            _metadata({**legacy, "forest_role": "out-of-fold"})
        with self.assertRaises(ValueError):
            _metadata({**legacy, "scenario": "open-world", "background_ratio": 0.5})
        with self.assertRaises(ValueError):
            load_forest_bundle("unused", trusted=False)

    def test_no_neural_imports(self):
        script = ("import sys; import fold.explainability; "
                  "import fold.explainability.train; import fold.explainability.__main__; "
                  "assert not any(n.split('.')[0] in ('tensorflow', 'keras', 'shap', 'matplotlib') "
                  "for n in sys.modules)")
        subprocess.run([sys.executable, "-c", script], cwd=ROOT, check=True)

    def test_static_save_interface(self):
        from ..core.artifacts import save_bundle
        import inspect

        tree = ast.parse((ROOT / "fold/explainability/train.py").read_text())
        calls = [node for node in ast.walk(tree) if isinstance(node, ast.Call)
                 and isinstance(node.func, ast.Name) and node.func.id == "save_bundle"]
        self.assertEqual(len(calls), 1)
        inspect.signature(save_bundle).bind("bundle", **{key.arg: None for key in calls[0].keywords})


class DataTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.loaded = load_traces(DATASET, return_ids=True)
        cls.digest = _digest(DATASET)
        traces, labels, _, ids = cls.loaded
        cls.by_id = dict(zip(ids, traces))
        cls.labels = dict(zip(ids, labels))
        # Ordered real-data subsets exercise selection and defense replay, not a trained run.
        parts = split_data(ids[:12])[0]
        cls.split = {"dataset_sha256": cls.digest, **dict(zip(("train", "val", "test"), parts))}

    def test_loader_preserves_ids_and_legacy_outputs(self):
        self.assertEqual(load_traces(DATASET), self.loaded[:3])
        traces, labels, count, ids = load_balanced(DATASET, return_ids=True)
        self.assertEqual(load_balanced(DATASET), (traces, labels, count))
        self.assertEqual(len(ids), len(set(ids)))
        for trace, label, tid in zip(traces, labels, ids):
            self.assertEqual(trace, self.by_id[tid])
            self.assertEqual(label, self.labels[tid])
        self.assertEqual(len(class_names(DATASET)), count)

    def test_split_selection_and_rejections(self):
        validate_split(self.split)
        metadata = {"scenario": "closed-world", "split_ids": self.split,
                    "dataset_sha256": self.digest}
        first = select_traces(DATASET, metadata, background_size=2, max_samples=2)
        self.assertEqual(first, select_traces(DATASET, metadata, background_size=2, max_samples=2))
        self.assertTrue(set(first[2]) <= set(self.split["train"]))
        self.assertTrue(set(first[3]) <= set(self.split["test"]))
        for invalid in ({**self.split, "test": self.split["train"]},
                        {**self.split, "dataset_sha256": "0" * 64}):
            with self.assertRaises(ValueError):
                select_traces(DATASET, {**metadata, "split_ids": invalid}, background_size=1, max_samples=1)

    def test_versioned_manifest_metadata(self):
        for flags in ((), ("--open-world", "--background-ratio", "0.3"),
                      ("--defense-padding", "0.3")):
            options = arguments(*flags)
            cfg = SimpleNamespace(**options, seed=59,
                                  is_open_world=options["scenario"] == "open-world",
                                  is_defended=bool(options["defense"]))
            metadata = {**experiment_metadata(cfg), "dataset_sha256": self.digest,
                        "split_ids": self.split}
            if cfg.is_open_world:
                metadata["background_split_ids"] = self.split
            if cfg.is_defended:
                metadata["defense_replay"] = {"order": ["train", "val", "test"],
                                               "python_random_state": random.getstate(),
                                               "source_sha256": self.digest}
            self.assertEqual(_metadata(metadata), metadata)
            _metadata(json.loads(json.dumps(metadata)))
            with self.assertRaises(ValueError):
                _metadata({**metadata, "dataset_sha256": "0" * 64})
            with self.assertRaises(ValueError):
                _metadata({**metadata, "provenance_version": 2})

    def test_defense_replay_uses_whole_split_order(self):
        from fold import data
        import platform

        previous = random.getstate()
        try:
            random.seed(59)
            state = random.getstate()
            defense = {"padding": 0.3, "dummy": 0.3, "fragmentation": 0.3}
            expected = {}
            for name in ("train", "val", "test"):
                ids = self.split[name]
                expected.update(zip(ids, apply_defense([self.by_id[i] for i in ids], defense)))
        finally:
            random.setstate(previous)
        metadata = {"scenario": "defense", "defense": defense, "split_ids": self.split,
                    "defense_replay": {"order": ["train", "val", "test"],
                                       "python_random_state": state,
                                       "python_version": platform.python_version(),
                                       "source_sha256": _digest(Path(data.__file__))}}
        metadata = json.loads(json.dumps(metadata))
        result = select_traces(DATASET, metadata, background_size=2, max_samples=2)
        self.assertEqual(random.getstate(), previous)
        for traces, ids in ((result[0], result[2]), (result[1], result[3])):
            self.assertEqual(traces, [expected[tid] for tid in ids])
        metadata["defense_replay"]["source_sha256"] = "0" * 64
        with self.assertRaises(ValueError):
            select_traces(DATASET, metadata, background_size=2, max_samples=2)

    @unittest.skipUnless(importlib.util.find_spec("tensorflow"), "TensorFlow is not installed")
    def test_original_split_opt_in(self):
        from fold.experiments import Config, load_and_split

        cfg = Config(1, Path("unused"), DATASET)
        original = load_and_split(cfg)
        captured = load_and_split(cfg, capture_ids=True)
        record = split_record(captured, self.digest)
        for name in ("train", "val", "test"):
            left, right = getattr(original, name), getattr(captured, name)
            self.assertEqual((left.traces, left.labels), (right.traces, right.labels))
            self.assertEqual(right.traces, [self.by_id[tid] for tid in record[name]])


if __name__ == "__main__":
    unittest.main()
