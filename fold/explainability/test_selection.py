"""Explicit XAI selection from recorded test IDs, without fitting or SHAP."""
import json
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import numpy as np

from fold.data import load_traces
from .artifacts import _digest
from .data import select_traces
from .__main__ import parse_args


class SelectionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.dataset = Path(__file__).resolve().parents[2] / "data/odoh_traces.csv"
        traces, _, _, ids = load_traces(cls.dataset, return_ids=True)
        cls.by_id = dict(zip(ids, traces))
        cls.split = {"dataset_sha256": _digest(cls.dataset),
                     "train": ids[:20], "val": ids[20:25], "test": ids[25:45]}
        cls.metadata = {"scenario": "closed-world", "split_ids": cls.split}

    def setUp(self):
        directory = TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.path = Path(directory.name) / "selection.json"

    def select(self, ids, **kwargs):
        payload = {"target_class": 8, "class_name": "informational", "sample_ids": ids}
        self.path.write_text(json.dumps(payload), encoding="utf-8")
        return select_traces(self.dataset, self.metadata, sample_ids_file=self.path,
                             target_class=kwargs.pop("target_class", 8),
                             background_size=3, seed=7, **kwargs)

    def test_valid_subset(self):
        before = json.dumps(self.metadata, sort_keys=True)
        ids = self.split["test"][:12]
        selected = self.select(ids)
        self.assertEqual(selected[3], ids)
        self.assertEqual(selected[1], [self.by_id[tid] for tid in ids])
        previous = select_traces(self.dataset, self.metadata, background_size=3, seed=7)
        self.assertEqual((selected[0], selected[2]), (previous[0], previous[2]))
        self.assertEqual(json.dumps(self.metadata, sort_keys=True), before)
        for invalid in ([], "not a list", [""]):
            with self.subTest(ids=invalid), self.assertRaises(ValueError):
                self.select(invalid)
        argv = ["explain", "--bundle", "bundle", "--dataset", str(self.dataset),
                "--target-class", "8", "--sample-ids-file", str(self.path), "--out-dir", "out"]
        with patch.object(sys, "argv", ["explainability", *argv]):
            _, args = parse_args()
        self.assertEqual(args.sample_ids_file, self.path)
        self.assertIsNone(args.max_samples)

    def test_non_test_id_rejected(self):
        for tid in (self.split["train"][0], self.split["val"][0], "unknown"):
            with self.subTest(tid=tid), self.assertRaisesRegex(ValueError, "recorded test split"):
                self.select([self.split["test"][0], tid], max_samples=1)
        split_file = self.path.parent / "split.json"
        split_file.write_text(json.dumps({**self.split, "test": self.split["test"][:2]}))
        with self.assertRaisesRegex(ValueError, "Explicit split differs from bundle split_ids"):
            self.select(self.split["test"][:2], split_file=split_file)

    def test_duplicates_rejected(self):
        tid = self.split["test"][0]
        with self.assertRaisesRegex(ValueError, "unique"):
            self.select([tid, tid], max_samples=1)

    def test_target_class_mismatch_rejected(self):
        with self.assertRaisesRegex(ValueError, "target_class differs"):
            self.select(self.split["test"][:2], target_class=9)

    def test_sample_order_preserved(self):
        ids = self.split["test"][::-1][:12]
        for limit in (None, 3, 100):
            with self.subTest(limit=limit):
                selected = self.select(ids, max_samples=limit)
                self.assertEqual(selected[3], ids[:limit])
                self.assertEqual(selected[1], [self.by_id[tid] for tid in ids[:limit]])
        self.path.write_text(json.dumps({"sample_ids": ids}), encoding="utf-8")
        selected = select_traces(self.dataset, self.metadata, sample_ids_file=self.path,
                                 target_class=8, background_size=3)
        self.assertEqual(selected[3], ids)

    def test_previous_sampling_unchanged(self):
        rng = np.random.default_rng(7)
        train, test = self.split["train"], self.split["test"]
        expected_background = [train[i] for i in rng.choice(len(train), 3, replace=False)]
        expected_samples = [test[i] for i in rng.choice(len(test), 10, replace=False)]
        selected = select_traces(self.dataset, self.metadata, background_size=3, seed=7)
        self.assertEqual(selected[2], expected_background)
        self.assertEqual(selected[3], expected_samples)
        self.assertEqual(selected[1], [self.by_id[tid] for tid in expected_samples])
        self.assertEqual(selected[4]["selection"], "uniform_without_replacement_from_explicit_train_test_ids")


if __name__ == "__main__":
    unittest.main()
