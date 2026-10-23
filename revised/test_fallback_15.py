"""Failure-policy tests; deterministic fault injection, no random solver failures."""
import unittest
from unittest.mock import patch
import numpy as np
from scipy.sparse.linalg import ArpackNoConvergence
from sparse_svp import sparse_svp


def fail(*args, **kwargs):
    raise ArpackNoConvergence('injected failure', np.array([]), np.empty((6, 0)))


class TestFallback15(unittest.TestCase):
    def setUp(self):
        self.a = np.diag([5., 3., 1., 0., 0., 0.])
        r, c = np.indices(self.a.shape)
        self.indices = (r.ravel(), c.ravel())

    def test_default_raises(self):
        with patch('sparse_svp.svds', side_effect=fail):
            with self.assertRaises(ArpackNoConvergence):
                sparse_svp(self.a.shape, self.indices, self.a.ravel(), rank=2, max_iter=1)

    def test_explicit_dense_fallback_reconstructs_optimum(self):
        with patch('sparse_svp.svds', side_effect=fail):
            result = sparse_svp(self.a.shape, self.indices, self.a.ravel(), rank=2,
                                max_iter=1, on_arpack_failure='dense')
        np.testing.assert_allclose(result.to_dense(), np.diag([5.,3.,0.,0.,0.,0.]), atol=1e-12)
        self.assertEqual(result.dense_fallbacks, 1)
        self.assertEqual(result.stop_reason, 'max_iter')

    def test_dense_fallback_rejected_if_size_exceeds_limit(self):
        with patch('sparse_svp.svds', side_effect=fail):
            with self.assertRaises(ArpackNoConvergence):
                sparse_svp(self.a.shape, self.indices, self.a.ravel(), rank=2,
                           max_iter=1, on_arpack_failure='dense', max_dense_elements=35)

    def test_invalid_policy_and_budget(self):
        for opts in ({'on_arpack_failure':'ignore'}, {'max_dense_elements':0},
                     {'svd_maxiter':0}):
            with self.subTest(opts=opts), self.assertRaises(ValueError):
                sparse_svp(self.a.shape, self.indices, self.a.ravel(), rank=2, **opts)

    def test_no_fallback_when_arpack_succeeds(self):
        result = sparse_svp(self.a.shape, self.indices, self.a.ravel(), rank=2,
                            max_iter=1, on_arpack_failure='dense')
        self.assertEqual(result.dense_fallbacks, 0)
        np.testing.assert_allclose(result.to_dense(), np.diag([5.,3.,0.,0.,0.,0.]), atol=1e-7)

if __name__ == '__main__':
    unittest.main()
