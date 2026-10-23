import unittest
import numpy as np
from matrix_completion import relative_error, truncated_svd, svp, svt_proximal, admira_lab, evaluate

class TestMatrixCompletion(unittest.TestCase):
    def setUp(self):
        self.y = np.outer([1., 2., 3., 4.], [2., 1., 3.])
        self.all = tuple(a.ravel() for a in np.indices(self.y.shape))

    def test_zero_denominator_raises(self):
        with self.assertRaises(ValueError):
            relative_error(np.ones((1, 1)), np.zeros((1, 1)), (np.array([0]), np.array([0])))

    def test_duplicate_observations_rejected(self):
        with self.assertRaises(ValueError):
            svp(self.y, (np.array([0,0]), np.array([0,0])), rank=1)

    def test_svd_eckart_young(self):
        rng = np.random.default_rng(20261023)
        a = rng.normal(size=(6, 4))
        u, s, vt = truncated_svd(a, 2)
        self.assertLess(np.linalg.norm((u*s)@vt - (np.linalg.svd(a, full_matrices=False)[0][:,:2]*np.linalg.svd(a, full_matrices=False)[1][:2])@np.linalg.svd(a, full_matrices=False)[2][:2]), 1e-10)

    def test_svp_exact_full_observation(self):
        fit = svp(self.y, self.all, rank=1)
        self.assertLess(np.linalg.norm(fit.matrix-self.y), 1e-9)

    def test_svt_soft_threshold_full_observation(self):
        fit = svt_proximal(self.y, self.all, threshold=0.5)
        u, s, vt = np.linalg.svd(self.y, full_matrices=False)
        expected = (u*np.maximum(s-0.5,0))@vt
        np.testing.assert_allclose(fit.matrix, expected, atol=1e-9)

    def test_admira_exact_full_observation(self):
        fit = admira_lab(self.y, self.all, rank=1)
        self.assertLess(np.linalg.norm(fit.matrix-self.y), 1e-9)

    def test_metrics(self):
        metrics = evaluate(np.zeros_like(self.y), self.y, self.all)
        self.assertAlmostEqual(metrics['rmse'], np.sqrt(np.mean(self.y**2)))
        self.assertAlmostEqual(metrics['mae'], np.mean(np.abs(self.y)))

    def test_sparse_case_finite(self):
        indices = (np.array([0,0,1,1,2,2,3,3]),np.array([0,1,0,2,1,2,0,2]))
        for fit in (svp(self.y, indices, rank=1, max_iter=12),svt_proximal(self.y, indices, threshold=0.5, max_iter=12),admira_lab(self.y, indices, rank=1, max_iter=12)):
            self.assertTrue(np.isfinite(fit.matrix).all())
            self.assertTrue(np.isfinite(fit.train_errors).all())

if __name__ == '__main__':
    unittest.main()
