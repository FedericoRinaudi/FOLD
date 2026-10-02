"""Small exact-game tests; no trained model or SHAP library is needed."""
import importlib.util
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import Mock, patch

import numpy as np

from fold.data import load_traces
from ..__main__ import parse_args
from ..end_to_end.results import FoldExplanation
from ..end_to_end.explainer import (FoldGroupExplainer, group_bounds, hybrid_trace,
                      resolution_edges, select_references, shapley_values)
from ..end_to_end.predictor import FoldPredictor


def trace(units, size=10):
    return [[0.001 * (i + 1), float(size + i), role]
            for i in range(units) for role in range(4)]


class GroupedTests(unittest.TestCase):
    def setUp(self):
        self.references = [trace(9, 2), trace(11, 4)]
        self.samples = [trace(10, 15), trace(3, 20)]
        self.metadata = {"split_ids": {"train": ["r0", "r1"], "val": ["v0"],
                                       "test": ["s0", "s1"]}}
        self.predictor = Mock(spec=FoldPredictor)
        self.predictor.predict_target.side_effect = lambda traces, target: np.asarray([
            0.1 + sum(p[1] for p in t) / 10000 for t in traces])

    def explainer(self, **kwargs):
        return FoldGroupExplainer(self.predictor, self.references, reference_ids=["r0", "r1"],
                                  target_class=1, class_names=("a", "b"),
                                  metadata=self.metadata, eligible_reference_count=2, **kwargs)

    def result(self):
        return self.explainer().explain(self.samples, sample_ids=["s0", "s1"])

    def test_complete_balanced_groups_and_loader_units(self):
        dataset = Path(__file__).resolve().parents[3] / "data/odoh_traces.csv"
        actual = load_traces(dataset)[0][0]
        np.testing.assert_array_equal(resolution_edges(actual), np.arange(0, len(actual) + 1, 4))
        for units in (1, 3, 8, 9, 17):
            edges = resolution_edges(trace(units))
            bounds, unit_bounds = group_bounds(edges, min(8, units))
            self.assertEqual(bounds[0, 0], 0)
            self.assertEqual(bounds[-1, 1], units * 4)
            np.testing.assert_array_equal(bounds[1:, 0], bounds[:-1, 1])
            self.assertTrue(np.all(bounds % 4 == 0))
            widths = np.diff(unit_bounds, axis=1).ravel()
            self.assertGreater(widths.min(), 0)
            self.assertLessEqual(widths.max() - widths.min(), 1)
        defended = [p for i, p in enumerate(trace(3)) for _ in range(1 + i % 3)]
        edges = resolution_edges(defended)
        self.assertEqual(len(edges) - 1, 3)
        for start, end in zip(edges[:-1], edges[1:]):
            self.assertEqual(list(dict.fromkeys(p[2] for p in defended[start:end])), [0, 1, 2, 3])
        with self.assertRaisesRegex(ValueError, "ODoH"):
            resolution_edges(trace(2)[:-1])

    def test_replacement_deterministic_and_endpoints(self):
        sample, reference = self.samples[0], self.references[0]
        bounds = group_bounds(resolution_edges(sample), 8)[0]
        ref_bounds = group_bounds(resolution_edges(reference), 8)[0]
        self.assertEqual(hybrid_trace(sample, reference, bounds, ref_bounds, 0), reference)
        self.assertEqual(hybrid_trace(sample, reference, bounds, ref_bounds, 255), sample)
        first = hybrid_trace(sample, reference, bounds, ref_bounds, 1)
        self.assertEqual(first, sample[:bounds[0, 1]] + reference[ref_bounds[0, 1]:])
        self.assertEqual(first, hybrid_trace(sample, reference, bounds, ref_bounds, 1))
        first[0][1] = -1
        self.assertGreater(sample[0][1], 0)

    def test_split_membership(self):
        with self.assertRaisesRegex(ValueError, "training split"):
            FoldGroupExplainer(self.predictor, self.references, reference_ids=["r0", "s0"],
                               target_class=1, class_names=("a", "b"), metadata=self.metadata,
                               eligible_reference_count=2)
        with self.assertRaisesRegex(ValueError, "recorded test ID"):
            self.explainer().explain(self.samples, sample_ids=["s0", "v0"])
        self.predictor.predict_target.assert_not_called()

    def test_wrapper_receives_hybrids_and_exact_values(self):
        result = self.result()
        np.testing.assert_array_equal(result.group_counts, [3, 3])
        self.assertEqual(result.values.shape, (2, 8))
        np.testing.assert_array_equal(result.values[1, 3:], 0)
        expected_evaluations = 2 + 2 + 2 * 2 * (2**3 - 2)
        calls = self.predictor.predict_target.call_args_list
        self.assertEqual(sum(len(c.args[0]) for c in calls), expected_evaluations)
        self.assertTrue(all(c.args[1] == 1 for c in calls))
        expected_hybrid = hybrid_trace(self.samples[0], self.references[0], result.bounds[0, :3],
                                       result.reference_bounds[0, 0, :3], 1)
        self.assertEqual(calls[2].args[0][0], expected_hybrid)
        for row, count in enumerate(result.group_counts):
            for group in range(count):
                lo, hi = result.bounds[row, group]
                sample_sum = sum(p[1] for p in self.samples[row][lo:hi])
                reference_sum = np.mean([sum(p[1] for p in reference[start:end])
                                         for reference, (start, end) in zip(
                                             self.references, result.reference_bounds[row, :, group])])
                self.assertAlmostEqual(result.values[row, group], (sample_sum - reference_sum) / 10000)
        result.check_additivity()

    def test_short_selected_reference_rejected_without_inference(self):
        self.references[0] = trace(1)
        with self.assertRaisesRegex(ValueError, "shorter than g_run=3"):
            self.result()
        self.predictor.predict_target.assert_not_called()

    def test_short_training_reference_excluded_and_deterministic(self):
        pool = [trace(1), *self.references, trace(12)]
        ids = ["short", "r0", "r1", "r2"]
        samples = [trace(10), trace(9)]
        first = select_references(samples, pool, ids, num_groups=8, reference_size=2, seed=7)
        self.assertNotIn("short", first[1])
        self.assertEqual(first[2], {"requested_num_groups": 8, "effective_num_groups": 8,
                                    "eligible_reference_count": 3})
        self.assertEqual(first, select_references(samples, pool, ids, num_groups=8,
                                                  reference_size=2, seed=7))
        expected = np.random.default_rng(7).choice([1, 2, 3], 2, replace=False)
        self.assertEqual(first[1], [ids[i] for i in expected])

    def test_insufficient_eligible_references(self):
        with self.assertRaisesRegex(ValueError, "g_run=8, requested=2, eligible=1"):
            select_references([trace(9)], [trace(1), trace(8)], ["short", "long"],
                              num_groups=8, reference_size=2, seed=0)

    def test_interaction_and_additivity_failure(self):
        np.testing.assert_allclose(shapley_values([0, 0.2, 0.3, 0.9]), [0.4, 0.5])
        result = self.result()
        result.values[0, 0] += 0.1
        with self.assertRaisesRegex(ValueError, "additivity failed"):
            result.check_additivity()
        with TemporaryDirectory() as directory:
            with self.assertRaisesRegex(ValueError, "additivity failed"):
                result.save(Path(directory) / "invalid.npz")

    def test_safe_npz_round_trip(self):
        result = self.result()
        with TemporaryDirectory() as directory:
            path = Path(directory) / "result.npz"
            result.save(path)
            with np.load(path, allow_pickle=False) as stored:
                self.assertTrue(all(stored[key].dtype.kind != "O" for key in stored.files))
            restored = FoldExplanation.load(path)
        for name in ("values", "base_values", "probabilities", "group_counts", "bounds",
                     "unit_bounds", "reference_bounds"):
            np.testing.assert_array_equal(getattr(result, name), getattr(restored, name))
        for name in ("sample_ids", "reference_ids", "target_class", "class_name", "seed", "metadata"):
            self.assertEqual(getattr(result, name), getattr(restored, name))
        self.assertEqual(restored.metadata["requested_num_groups"], 8)
        self.assertEqual(restored.metadata["effective_num_groups"], 3)
        self.assertEqual(restored.metadata["eligible_reference_count"], 2)

    def test_cli_and_lazy_imports(self):
        args = ["fold", "explain-fold", "--bundle", "bundle", "--trusted", "--dataset", "data.csv",
                "--target-class", "8", "--sample-ids-file", "selection.json", "--out-dir", "out"]
        with patch.object(sys, "argv", args):
            _, options = parse_args()
        self.assertEqual((options.num_groups, options.reference_size, options.seed), (8, 16, 0))
        self.assertEqual(options.sample_ids_file, Path("selection.json"))
        self.assertIsNone(options.max_samples)
        self.assertNotIn("tensorflow", sys.modules)
        self.assertNotIn("shap", sys.modules)

    @unittest.skipUnless(importlib.util.find_spec("matplotlib"), "matplotlib is not installed")
    def test_figures_from_npz(self):
        from ..end_to_end.plots import plot_fold

        with TemporaryDirectory() as directory:
            path = Path(directory) / "result.npz"
            self.result().save(path)
            saved = plot_fold(FoldExplanation.load(path), directory, sample_index=1)
            self.assertEqual(len(saved), 4)
            self.assertTrue(all(p.stat().st_size > 0 for p in saved))


if __name__ == "__main__":
    unittest.main()
