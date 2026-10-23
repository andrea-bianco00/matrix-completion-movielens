"""Offline isolation regression checks with synthetic ratings, never MovieLens data.

These test the *load_fit* interface and directory boundary, not OS-level
sandboxing, nor the historical split/export integration (which requires MovieLens).
"""
import csv
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
from isolation_36 import load_fit


class SyntheticIsolationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.fit = self.root / "fit"
        self.fit.mkdir()
        (self.root / "validation").mkdir()
        (self.root / "sealed_test").mkdir()
        (self.fit / "catalog.json").write_text(
            json.dumps({"users": [10, 20], "items": [100, 200]}),
            encoding="utf-8",
        )
        # The published loader deliberately enforces the historical 70,585
        # training observations. Build a tiny-valued *synthetic* fixture of
        # that length rather than weakening the production invariant.
        with (self.fit / "train.csv").open("w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["userId", "movieId", "rating", "timestamp"])
            for i in range(70585):
                writer.writerow([10 if i % 2 == 0 else 20,
                                 100 if i % 3 == 0 else 200,
                                 2.0 if i % 2 == 0 else 4.0, 0])
        for folder, name in [("validation", "validation.csv"),
                             ("sealed_test", "test.csv")]:
            (self.root / folder / name).write_text(
                "userId,movieId,rating,timestamp\n10,100,5.0,0\n",
                encoding="utf-8",
            )

    def test_fit_reads_only_fit_directory(self):
        before = load_fit(self.fit)
        self.assertEqual(before[3], (2, 2))
        self.assertEqual(len(before[2]), 70585)
        self.assertTrue(np.all(np.isin(before[2], [2.0, 4.0])))
        for folder, name in [("validation", "validation.csv"),
                             ("sealed_test", "test.csv")]:
            (self.root / folder / name).write_text(
                "userId,movieId,rating,timestamp\n20,200,999.0,0\n",
                encoding="utf-8",
            )
        after = load_fit(self.fit)
        for left, right in zip(before[:3], after[:3]):
            np.testing.assert_array_equal(left, right)
        self.assertEqual(before[3], after[3])

    def test_fit_directory_contains_no_holdout_files(self):
        self.assertEqual(sorted(p.name for p in self.fit.iterdir()),
                         ["catalog.json", "train.csv"])

    def test_corrupted_training_labels_fail_closed(self):
        with (self.fit / "train.csv").open("a") as f:
            f.write("10,100,nan,0\n")
        with self.assertRaises(ValueError):
            load_fit(self.fit)

    def test_invalid_catalog_rejected(self):
        (self.fit / "catalog.json").write_text(
            json.dumps({"users": [20, 10], "items": [100, 200]}),
            encoding="utf-8",
        )
        with self.assertRaises(ValueError):
            load_fit(self.fit)


if __name__ == "__main__":
    unittest.main()
