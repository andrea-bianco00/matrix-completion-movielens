import unittest
import numpy as np
from matrix_completion import svp
from sparse_svp import sparse_svp

class TestSparseSVP(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(20261023)
        self.shape = (14, 11)
        self.y = rng.normal(size=self.shape)
        self.rows, self.cols = np.where(rng.random(self.shape) < .6)
        self.values = self.y[self.rows, self.cols]

    def test_matches_dense_reference(self):
        dense = svp(self.y, (self.rows, self.cols), rank=2, max_iter=5, tol=0)
        sparse = sparse_svp(self.shape, (self.rows, self.cols), self.values, rank=2, max_iter=5, tol=0)
        np.testing.assert_allclose(sparse.to_dense(), dense.matrix, atol=1e-7, rtol=1e-7)
        np.testing.assert_allclose(sparse.train_rmse, dense.train_errors, atol=1e-8)

    def test_predict_matches_materialized_matrix(self):
        result = sparse_svp(self.shape, (self.rows, self.cols), self.values, rank=2, max_iter=3)
        np.testing.assert_allclose(result.predict(self.rows, self.cols), result.to_dense()[self.rows, self.cols], atol=1e-10)

    def test_zero_values(self):
        result = sparse_svp(self.shape, (self.rows, self.cols), np.zeros(len(self.rows)), rank=2)
        self.assertAlmostEqual(result.train_rmse[-1], 0)
        np.testing.assert_allclose(result.to_dense(), 0, atol=1e-12)

    def test_duplicate_indices_rejected(self):
        with self.assertRaises(ValueError):
            sparse_svp((4, 4), ([0, 0], [1, 1]), [1., 2.], rank=1)

    def test_invalid_rank_rejected(self):
        with self.assertRaises(ValueError):
            sparse_svp((4, 4), ([0], [1]), [1.], rank=4)

    def test_invalid_values_rejected(self):
        with self.assertRaises(ValueError):
            sparse_svp((4, 4), ([0], [1]), [np.nan], rank=1)

    def test_full_observation_rank_one(self):
        a = np.outer(np.arange(1., 9.), np.arange(1., 7.))
        rows, cols = np.indices(a.shape)
        result = sparse_svp(a.shape, (rows.ravel(), cols.ravel()), a.ravel(), rank=1, max_iter=4)
        np.testing.assert_allclose(result.to_dense(), a, atol=1e-8)

if __name__ == '__main__': unittest.main()
