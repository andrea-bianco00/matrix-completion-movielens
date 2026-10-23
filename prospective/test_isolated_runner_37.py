"""Self-contained tests: generated ratings are synthetic fixtures, not MovieLens."""
import csv
import json
import tempfile
import unittest
from pathlib import Path
import numpy as np
from isolation_36 import load_fit
from isolated_runner_37 import fit_predict, run_child


class IsolatedRunnerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory()
        cls.fit = Path(cls.temp.name) / 'fit'
        cls.fit.mkdir()
        (cls.fit / 'catalog.json').write_text(json.dumps({'users': [1, 2], 'items': [10, 20]}))
        # The production loader intentionally checks the frozen train size.
        # These synthetic records exercise that contract without private data.
        with (cls.fit / 'train.csv').open('w', newline='') as f:
            w = csv.writer(f)
            w.writerow(['userId', 'movieId', 'rating', 'timestamp'])
            for i in range(70585):
                w.writerow([1 + i % 2, 10 + 10 * (i % 2), 3.0 + i % 2, 0])

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    def query(self, directory, rows, cols):
        q = Path(directory) / 'query.npz'
        np.savez(q, rows=np.array(rows, dtype=np.int64), cols=np.array(cols, dtype=np.int64))
        return q

    def test_mean_matches_train(self):
        r, c, y, shape = load_fit(self.fit)
        with tempfile.TemporaryDirectory() as t:
            q = self.query(t, r[:3], c[:3])
            out = fit_predict('global_mean', {}, self.fit, q)
            self.assertEqual(out['training_n'], 70585)
            np.testing.assert_allclose(out['predictions'], np.full(3, y.mean()), rtol=0, atol=1e-12)

    def test_worker_subprocess(self):
        with tempfile.TemporaryDirectory() as t:
            q = self.query(t, [0, 1], [0, 1])
            out = run_child('global_mean', {}, self.fit, q, seconds=20)
            self.assertEqual(out['status'], 'ok', out)
            self.assertEqual(out['query_n'], 2)

    def test_invalid_coordinates(self):
        with tempfile.TemporaryDirectory() as t:
            q = self.query(t, [-1], [0])
            with self.assertRaises(ValueError):
                fit_predict('global_mean', {}, self.fit, q)

    def test_heldout_mutation_no_effect(self):
        with tempfile.TemporaryDirectory() as t:
            q = self.query(t, [0], [0])
            a = fit_predict('global_mean', {}, self.fit, q)
            (Path(t) / 'test.csv').write_text('malicious,not,a,rating\n')
            b = fit_predict('global_mean', {}, self.fit, q)
            self.assertEqual(a['predictions'], b['predictions'])


if __name__ == '__main__':
    unittest.main()
