import unittest
import numpy as np
from matrix_completion import svt_proximal, admira_lab
from scalable_19 import svt_matrix_free, admira_matrix_free

class Equivalence(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(19)
        self.shape = (13, 11)
        self.rows, self.cols = np.indices(self.shape)
        self.rows, self.cols = self.rows.ravel(), self.cols.ravel()
        self.values = rng.normal(size=len(self.rows))
        self.y = self.values.reshape(self.shape)

    def test_svt_one_step(self):
        dense = svt_proximal(self.y, (self.rows, self.cols), threshold=2.0, max_iter=1)
        sparse = svt_matrix_free(self.shape, (self.rows, self.cols), self.values, threshold=2.0, max_iter=1, initial_rank=4)
        np.testing.assert_allclose(sparse.to_dense(), dense.matrix, atol=1e-8)

    def test_svt_multiple_steps(self):
        dense = svt_proximal(self.y, (self.rows, self.cols), threshold=1.5, max_iter=4, tol=0)
        sparse = svt_matrix_free(self.shape, (self.rows, self.cols), self.values, threshold=1.5, max_iter=4, tol=0, initial_rank=4)
        np.testing.assert_allclose(sparse.to_dense(), dense.matrix, atol=1e-8)

    def test_svt_no_cutoff_raises(self):
        with self.assertRaises(RuntimeError):
            svt_matrix_free(self.shape, (self.rows, self.cols), self.values, threshold=1e-12, max_iter=1, initial_rank=3)

    def test_admira_first_step(self):
        dense = admira_lab(self.y, (self.rows, self.cols), rank=2, max_iter=1)
        sparse = admira_matrix_free(self.shape, (self.rows, self.cols), self.values, rank=2, max_iter=1)
        np.testing.assert_allclose(sparse.to_dense(), dense.matrix, atol=1e-8)

    def test_admira_multiple_steps(self):
        dense = admira_lab(self.y, (self.rows, self.cols), rank=2, max_iter=3, tol=0)
        sparse = admira_matrix_free(self.shape, (self.rows, self.cols), self.values, rank=2, max_iter=3, tol=0)
        np.testing.assert_allclose(sparse.to_dense(), dense.matrix, atol=1e-7)

    def test_admira_partial_observations(self):
        rng = np.random.default_rng(91)
        selected = rng.choice(len(self.values), size=80, replace=False)
        rows, cols, vals = self.rows[selected], self.cols[selected], self.values[selected]
        y = np.zeros(self.shape)
        y[rows, cols] = vals
        dense = admira_lab(y, (rows, cols), rank=2, max_iter=3, tol=0)
        sparse = admira_matrix_free(self.shape, (rows, cols), vals, rank=2, max_iter=3, tol=0)
        np.testing.assert_allclose(sparse.to_dense(), dense.matrix, atol=1e-7)

    def test_svt_partial_observations(self):
        rng = np.random.default_rng(92)
        selected = rng.choice(len(self.values), size=80, replace=False)
        rows, cols, vals = self.rows[selected], self.cols[selected], self.values[selected]
        y = np.zeros(self.shape)
        y[rows, cols] = vals
        dense = svt_proximal(y, (rows, cols), threshold=3., max_iter=3, tol=0)
        sparse = svt_matrix_free(self.shape, (rows, cols), vals, threshold=3., max_iter=3, tol=0, initial_rank=4)
        np.testing.assert_allclose(sparse.to_dense(), dense.matrix, atol=1e-7)

    def test_zero_admira(self):
        sparse = admira_matrix_free(self.shape, (self.rows, self.cols), np.zeros_like(self.values), rank=2, max_iter=2)
        np.testing.assert_allclose(sparse.to_dense(), 0, atol=1e-10)

if __name__ == '__main__': unittest.main()
