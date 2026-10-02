"""Binary XAI tests with tiny CSV fixtures and mock inference; never train."""
import copy
import csv
from dataclasses import replace
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import numpy as np

from fold.data import Encoder
from fold.models.forest import RandomForest
from ..__main__ import parse_args
from ..core.artifacts import _digest
from ..end_to_end.explainer import group_bounds, hybrid_trace, resolution_edges
from ..open_world.data import load_background_pool, qualified_ids, select_open_world
from ..open_world.explainer import BinaryGroupExplainer
from ..open_world.predictor import BinaryFoldPredictor
from ..open_world.results import BinaryExplanation


def trace(units, size=10):
    return [[0.001 * (unit + 1), float(size + role), float(role)]
            for unit in range(units) for role in range(4)]


class OpenWorldTests(unittest.TestCase):
    def setUp(self):
        temporary = TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.directory = Path(temporary.name)
        self.dataset = self.directory / "monitored.csv"
        self.pool = self.directory / "pool.csv"
        self.samples = self.directory / "samples.json"
        self.reference_file = self.directory / "references.json"
        self.monitored = {"0": trace(3, 10), "m1": trace(4, 20), "m2": trace(3, 30),
                          "mshort": trace(1), "val": trace(2),
                          "test": trace(2, 40), "test2": trace(3, 50)}
        self.background = {"0": trace(3, 3), "1": trace(4, 4), "2": trace(3, 5),
                           "3": trace(1), "8": trace(2), "10": trace(2, 6), "11": trace(3, 7)}
        with self.dataset.open("w", newline="") as stream:
            fields = ["trace_id", "categorical_label", "location"]
            for prefix in ("Input-D", "Input-U", "Out-U", "Out-D"):
                fields += [f"{prefix}-Time", f"{prefix}-Size"]
            writer = csv.DictWriter(stream, fields)
            writer.writeheader()
            for tid, packets in self.monitored.items():
                for start in range(0, len(packets), 4):
                    row = {"trace_id": tid, "categorical_label": 0, "location": "fixture"}
                    for prefix, packet in zip(("Input-D", "Input-U", "Out-U", "Out-D"), packets[start:start+4]):
                        row[f"{prefix}-Time"], row[f"{prefix}-Size"] = packet[:2]
                    writer.writerow(row)
        with self.pool.open("w", newline="") as stream:
            stream.write("# Original pool layout; packet rows intentionally reversed\n")
            writer = csv.writer(stream)
            writer.writerow(["trace_id", "pkt_idx", "t", "size", "role"])
            for tid, packets in reversed(list(self.background.items())):
                writer.writerows([tid, i, *packet] for i, packet in reversed(list(enumerate(packets))))
        self.metadata = {
            "scenario": "open-world", "background_ratio": 0.3, "defense": {},
            "provenance_version": 1, "forest_role": "final", "experiment_id": 1,
            "seed": 59, "seeds": {}, "training": {}, "dataset_sha256": _digest(self.dataset),
            "split_ids": {"dataset_sha256": _digest(self.dataset),
                          "train": ["0", "m1", "mshort", "m2"], "val": ["val"],
                          "test": ["test", "test2"]},
            "background_split_ids": {"dataset_sha256": _digest(self.pool),
                                     "train": ["0", "1", "3", "2"], "val": ["8"],
                                     "test": ["10", "11"]}}
        self.write_samples(["test2", "test"], ["10"])

    def write_samples(self, monitored, background):
        self.samples.write_text(json.dumps({"output": "binary_output",
                                           "monitored_sample_ids": monitored,
                                           "background_sample_ids": background}))

    def select(self, **kwargs):
        return select_open_world(self.dataset, self.pool, self.metadata,
                                 sample_ids_file=self.samples,
                                 **{"num_groups": 2, "reference_size": 4, "seed": 7, **kwargs})

    def binary_mock(self):
        predictor = Mock()
        predictor.predict_target.side_effect = lambda traces, target: np.asarray([
            0.1 + (sum(packet[1] for packet in t) / 1000)**2 for t in traces])
        return predictor

    def result(self):
        return BinaryGroupExplainer(self.binary_mock(), self.select()).explain()

    def test_background_loader_matches_original_order_and_values(self):
        actual = load_background_pool(self.pool)
        self.assertEqual(actual, self.background)
        self.assertEqual(list(actual), ["0", "1", "2", "3", "8", "10", "11"])

    def test_pool_duplicate_or_missing_packet_rejected(self):
        text = self.pool.read_text()
        rows = text.splitlines()
        self.pool.write_text(text + rows[2] + "\n")
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            load_background_pool(self.pool)
        self.pool.write_text("\n".join(rows[:3] + rows[4:]) + "\n")
        with self.assertRaisesRegex(ValueError, "consecutive"):
            load_background_pool(self.pool)

    def test_mixed_selection_order_and_source_preserved(self):
        before = copy.deepcopy(self.metadata)
        selected = self.select()
        self.assertEqual(selected.sample_ids, ("test2", "test", "10"))
        self.assertEqual(selected.sample_types, ("monitored", "monitored", "background"))
        self.assertEqual(selected.samples, [self.monitored["test2"], self.monitored["test"], self.background["10"]])
        self.assertEqual(before, self.metadata)
        self.assertEqual(selected.metadata["selection"]["background_pool_sha256"], _digest(self.pool))

    def test_single_population_samples_supported(self):
        for monitored, background, expected in [(["test"], [], "monitored"), ([], ["11", "10"], "background")]:
            with self.subTest(expected=expected):
                self.write_samples(monitored, background)
                selected = self.select()
                self.assertEqual(selected.sample_ids, tuple(monitored + background))
                self.assertEqual(set(selected.sample_types), {expected})
                self.assertEqual(set(selected.reference_types), {"monitored", "background"})

    def test_wrong_split_population_duplicates_and_empty_rejected(self):
        for monitored, background in [(["0"], []), (["val"], []), ([], ["0"]), ([], ["8"]),
                                      (["10"], []), ([], ["test"]), (["test", "test"], []),
                                      ([], ["10", "10"]), ([], [])]:
            with self.subTest(monitored=monitored, background=background):
                self.write_samples(monitored, background)
                with self.assertRaises(ValueError):
                    self.select()

    def test_wrong_output_and_unknown_selection_fields_rejected(self):
        for payload in [{"output": "multi_output", "monitored_sample_ids": ["test"]},
                        {"output": "binary_output", "sample_ids": ["test"]}]:
            self.samples.write_text(json.dumps(payload))
            with self.assertRaises(ValueError):
                self.select()

    def test_hash_and_split_integrity(self):
        for key in ("split_ids", "background_split_ids"):
            original = copy.deepcopy(self.metadata[key])
            self.metadata[key]["dataset_sha256"] = "0" * 64
            with self.assertRaises(ValueError):
                self.select()
            self.metadata[key] = copy.deepcopy(original)
            self.metadata[key]["test"] = self.metadata[key]["train"][:1]
            with self.assertRaises(ValueError):
                self.select()
            self.metadata[key] = original
        self.metadata["background_split_ids"]["val"] = ["999"]
        with self.assertRaisesRegex(ValueError, "absent"):
            self.select()

    def test_balanced_deterministic_eligible_references(self):
        first = self.select()
        second = self.select()
        self.assertEqual(first.reference_ids, second.reference_ids)
        self.assertEqual(first.references, second.references)
        self.assertEqual(first.reference_types, ("monitored", "monitored", "background", "background"))
        info = first.metadata["selection"]
        self.assertEqual(info["eligible_reference_counts"], {"monitored": 3, "background": 3})
        self.assertNotIn("mshort", first.reference_ids)
        self.assertNotIn("3", first.reference_ids)
        self.assertEqual(info["effective_num_groups"], 2)
        for p, candidates in [("monitored", ["0", "m1", "m2"]), ("background", ["0", "1", "2"])]:
            expected = np.random.default_rng(7).choice(candidates, 2, replace=False).tolist()
            self.assertEqual(info[f"{p}_reference_ids"], expected)

    def test_odd_reference_size_and_insufficient_pool(self):
        selected = self.select(reference_size=5)
        self.assertEqual(selected.reference_types.count("monitored"), 3)
        self.assertEqual(selected.reference_types.count("background"), 2)
        with self.assertRaisesRegex(ValueError, "monitored references: Insufficient"):
            self.select(reference_size=8)
        with self.assertRaises(ValueError):
            self.select(reference_size=1)

    def test_effective_groups_fixed_by_samples_not_short_references(self):
        selected = self.select(num_groups=8)
        self.assertEqual(selected.metadata["selection"]["effective_num_groups"], 2)
        self.assertEqual(selected.metadata["selection"]["requested_num_groups"], 8)
        result = BinaryGroupExplainer(self.binary_mock(), selected).explain()
        self.assertEqual(result.values.shape, (3, 8))
        np.testing.assert_array_equal(result.values[:, 2:], 0)
        np.testing.assert_array_equal(result.bounds[:, 2:], -1)

    def test_explicit_references_preserve_order_and_namespaces(self):
        payload = {"output": "binary_output", "monitored_reference_ids": ["m2", "0"],
                   "background_reference_ids": ["2", "0"]}
        self.reference_file.write_text(json.dumps(payload))
        selected = self.select(reference_ids_file=self.reference_file)
        self.assertEqual(selected.reference_ids, ("m2", "0", "2", "0"))
        result = BinaryGroupExplainer(self.binary_mock(), selected).explain()
        self.assertEqual(len(set(qualified_ids(result.reference_ids, result.reference_types))), 4)
        for ids in (["10", "0"], ["3", "0"], ["0", "0"], ["0"]):
            payload["background_reference_ids"] = ids
            self.reference_file.write_text(json.dumps(payload))
            with self.assertRaises(ValueError):
                self.select(reference_ids_file=self.reference_file)

    def test_exact_binary_game_with_interaction(self):
        selected = self.select()
        predictor = self.binary_mock()
        result = BinaryGroupExplainer(predictor, selected).explain()
        def probability(t):
            return 0.1 + (sum(p[1] for p in t) / 1000)**2
        base = np.mean([probability(t) for t in selected.references])
        for row, sample in enumerate(selected.samples):
            bounds = group_bounds(resolution_edges(sample), 2)[0]
            coalition = []
            for mask in range(4):
                coalition.append(np.mean([
                    probability(hybrid_trace(sample, ref, bounds, group_bounds(resolution_edges(ref), 2)[0], mask))
                    for ref in selected.references]))
            expected = [(coalition[1] - coalition[0] + coalition[3] - coalition[2]) / 2,
                        (coalition[2] - coalition[0] + coalition[3] - coalition[1]) / 2]
            np.testing.assert_allclose(result.values[row], expected)
            self.assertAlmostEqual(result.base_values[row], base)
            self.assertAlmostEqual(result.probabilities[row], probability(sample))
        self.assertEqual(sum(len(c.args[0]) for c in predictor.predict_target.call_args_list), 31)
        self.assertTrue(all(c.args[1] == 0 for c in predictor.predict_target.call_args_list))
        result.check_additivity()
        self.assertEqual(result.output, "binary_output")
        self.assertEqual(result.positive_class, "monitored")
        self.assertEqual(result.metadata["experiment"]["split_ids"], self.metadata["split_ids"])

    def test_additivity_checked_at_construction_save_and_load(self):
        result = self.result()
        invalid = result.values.copy()
        invalid[0, 0] += 0.1
        with self.assertRaisesRegex(ValueError, "additivity"):
            replace(result, values=invalid)
        valid_path = self.directory / "valid.npz"
        result.save(valid_path)
        result.values[0, 0] += 0.1
        with self.assertRaisesRegex(ValueError, "additivity"):
            result.save(self.directory / "invalid.npz")
        with np.load(valid_path, allow_pickle=False) as archive:
            stored = {key: archive[key] for key in archive.files}
        stored["values"][0, 0] += 0.1
        np.savez(self.directory / "tampered.npz", **stored)
        with self.assertRaisesRegex(ValueError, "additivity"):
            BinaryExplanation.load(self.directory / "tampered.npz")

    def test_npz_round_trip_and_distinct_format(self):
        result = self.result()
        path = self.directory / "binary.npz"
        result.save(path)
        with np.load(path, allow_pickle=False) as stored:
            self.assertTrue(all(stored[key].dtype.kind != "O" for key in stored.files))
            header = json.loads(stored["header"].item())
            self.assertEqual(header["format"], "fold-open-world-binary-shap")
            self.assertEqual(header["positive_class"], "monitored")
            self.assertNotIn("target_class", header)
            arrays = {key: stored[key] for key in stored.files}
        loaded = BinaryExplanation.load(path)
        for key in vars(result):
            a, b = getattr(result, key), getattr(loaded, key)
            if isinstance(a, np.ndarray):
                np.testing.assert_array_equal(a, b)
            else:
                self.assertEqual(a, b)
        with self.assertRaises(FileExistsError):
            loaded.save(path)
        header["output"] = "multi_output"
        arrays["header"] = json.dumps(header)
        np.savez(self.directory / "wrong.npz", **arrays)
        with self.assertRaisesRegex(ValueError, "format"):
            BinaryExplanation.load(self.directory / "wrong.npz")

    def test_population_metadata_cannot_be_silently_changed(self):
        result = self.result()
        with self.assertRaises(ValueError):
            replace(result, sample_types=("background",) * 3)
        metadata = copy.deepcopy(result.metadata)
        metadata["experiment"]["selection"]["monitored_sample_ids"] = ["test"]
        with self.assertRaisesRegex(ValueError, "provenance"):
            replace(result, metadata=metadata)

    def make_bundle(self):
        encoder = Encoder(num_classes=2, target_length=20, size_edges=np.array([12.0]))
        forest = RandomForest(2)
        # Populate fitted-state attributes solely to test validation and input
        # plumbing. predict_proba is mocked; no trees are fitted or evaluated.
        forest.clf.estimators_ = [object()]
        forest.clf.classes_ = np.array([0, 1])
        forest.clf.n_features_in_ = 35
        forest.clf.n_outputs_ = 1
        forest.predict_proba = Mock(side_effect=lambda x: np.column_stack([x[:, 0] / 100, 1 - x[:, 0] / 100]))
        shapes = {"transformer_input": (None, 20, 8), "agg_input": (None, 3),
                  "size_vocab_input": (None, 20), "unigram_agg_input": (None, 2), "rf_input": (None, 2)}
        model = SimpleNamespace(
            input={k: SimpleNamespace(shape=v, dtype="int32" if k == "size_vocab_input" else "float32")
                   for k, v in shapes.items()},
            output={k: SimpleNamespace(shape=(None, n)) for k, n in
                    (("multi_output", 2), ("aux_multi_output", 2), ("binary_output", 1))})
        model.predict = Mock(side_effect=lambda inputs, **kwargs: {
            "multi_output": inputs["rf_input"], "binary_output": 0.2 + 0.4 * inputs["rf_input"][:, :1]})
        return SimpleNamespace(encoder=encoder, forest=forest, model=model, metadata=copy.deepcopy(self.metadata))

    def test_binary_predictor_chunks_and_recomputes_rf(self):
        bundle = self.make_bundle()
        predictor = BinaryFoldPredictor(bundle, batch_size=2)
        batch = [trace(1), trace(2), trace(3)]
        np.testing.assert_allclose(predictor.predict_monitored(batch), [0.216, 0.232, 0.248])
        self.assertEqual(bundle.forest.predict_proba.call_count, 2)
        self.assertEqual(bundle.model.predict.call_count, 2)
        for call, expected in zip(bundle.model.predict.call_args_list, ([0.04, 0.08], [0.12])):
            np.testing.assert_allclose(call.args[0]["rf_input"][:, 0], expected)
        self.assertEqual(predictor.predict_monitored([]).shape, (0,))
        self.assertEqual(predictor.predict_target(trace(1), 0).shape, (1,))
        with self.assertRaises(ValueError):
            predictor.predict_target(trace(1), 1)

    def test_binary_predictor_shape_range_and_architecture_guards(self):
        bundle = self.make_bundle()
        predictor = BinaryFoldPredictor(bundle)
        for output in (np.array([0.5]), np.array([[1.1]]), np.array([[np.nan]]), np.array([[0.2, 0.8]])):
            bundle.model.predict.side_effect = None
            bundle.model.predict.return_value = {"binary_output": output}
            with self.assertRaisesRegex(ValueError, "binary_output"):
                predictor.predict_monitored(trace(1))
        bundle.model.output = {"n1_output": SimpleNamespace(shape=(None, 3))}
        with self.assertRaisesRegex(ValueError, "two-head"):
            BinaryFoldPredictor(bundle)
        bundle = self.make_bundle()
        bundle.metadata["scenario"] = "closed-world"
        with self.assertRaisesRegex(ValueError, "open-world"):
            BinaryFoldPredictor(bundle)

    def test_parser_and_mock_cli_save(self):
        from ..open_world import cli

        output = self.directory / "output"
        argv = ["xai", "explain-open-world", "--bundle", str(self.directory), "--trusted",
                "--dataset", str(self.dataset), "--background-pool", str(self.pool),
                "--sample-ids-file", str(self.samples), "--num-groups", "2", "--reference-size", "4",
                "--skip-plots", "--out-dir", str(output)]
        with patch.object(sys, "argv", argv):
            _, args = parse_args()
        self.assertEqual((args.seed, args.batch_size), (0, 16))
        (self.directory / "manifest.json").write_text("{}")
        bundle = self.make_bundle()
        with patch.object(cli, "load_bundle", return_value=bundle), patch.object(cli, "_versions", return_value={}):
            cli.run(args)
        result = BinaryExplanation.load(output / "binary_monitored_shap.npz")
        self.assertEqual(result.metadata["experiment"]["bundle_manifest_sha256"], _digest(self.directory / "manifest.json"))
        self.assertEqual(result.sample_types, ("monitored", "monitored", "background"))
        with patch.object(sys, "argv", ["xai", "plot-open-world", "--attributions", "a.npz", "--out-dir", "out"]):
            _, args = parse_args()
        self.assertEqual(args.command, "plot-open-world")

    def test_lazy_imports(self):
        code = ("import sys; import fold.explainability; pandas_before = 'pandas' in sys.modules; "
                "import fold.explainability.__main__; "
                "import fold.explainability.open_world; import fold.explainability.open_world.cli; "
                "import fold.explainability.open_world.plots; "
                "assert ('pandas' in sys.modules) == pandas_before; "
                "assert not any(n.split('.')[0] in ('tensorflow','keras','shap','matplotlib') "
                "for n in sys.modules)")
        subprocess.run([sys.executable, "-B", "-c", code], cwd=Path(__file__).resolve().parents[3], check=True)

    @unittest.skipUnless(importlib.util.find_spec("matplotlib"), "matplotlib is not installed")
    def test_binary_plot_smoke(self):
        from ..open_world.plots import plot_binary

        path = self.directory / "binary.npz"
        self.result().save(path)
        files = plot_binary(BinaryExplanation.load(path), self.directory, sample_index=2)
        self.assertEqual(len(files), 4)
        self.assertTrue(all(p.name.startswith("binary_monitored_") and p.stat().st_size > 0 for p in files))


if __name__ == "__main__":
    unittest.main()
