"""Deterministic numerical stress tests for the course ADMiRA-style implementation."""
import unittest
import numpy as np
from matrix_completion import admira_lab, svp, svt_proximal, _validate

class Stability09(unittest.TestCase):
    def test_duplicate_indices_rejected(self):
        y=np.ones((3,3)); ix=(np.array([0,0]),np.array([1,1]))
        with self.assertRaises(ValueError): admira_lab(y,ix,rank=1)

    def test_zero_observations_fit_exactly(self):
        y=np.zeros((4,4)); ix=np.indices(y.shape)
        idx=(ix[0].ravel(),ix[1].ravel())
        for f,kwargs in [(svp,{'rank':1}),(svt_proximal,{'threshold':1.}),(admira_lab,{'rank':1})]:
            with self.subTest(method=f.__name__):
                fit = f(y,idx,max_iter=2,**kwargs)
                np.testing.assert_array_equal(fit.matrix, y)
                self.assertEqual(fit.train_errors, [0.0])

    def test_zero_residual_repeated_atoms_does_not_crash(self):
        # Exact rank-one matrix: second iteration uses zero/near-zero residual,
        # where residual singular vectors are not uniquely identified.
        u=np.array([1.,2.,3.,4.]); v=np.array([2.,-1.,3.]); y=np.outer(u,v)
        ix=np.indices(y.shape); idx=(ix[0].ravel(),ix[1].ravel())
        fit=admira_lab(y,idx,rank=1,max_iter=6,tol=0)
        self.assertTrue(np.all(np.isfinite(fit.matrix)))
        self.assertLess(np.linalg.norm(fit.matrix-y),1e-9)

    def test_underdetermined_sparse_observation(self):
        y=np.zeros((8,8)); idx=(np.array([0,1,2]),np.array([0,1,2]));y[idx]=[5.,3.,4.]
        fit=admira_lab(y,idx,rank=2,max_iter=6)
        self.assertTrue(np.all(np.isfinite(fit.matrix)))
        self.assertTrue(np.isfinite(fit.train_errors).all())

    def test_rank_projection_on_near_degenerate_spectrum(self):
        y=np.diag([2.,2.-1e-12,0.,0.]); ix=np.indices(y.shape);idx=(ix[0].ravel(),ix[1].ravel())
        fit=svp(y,idx,rank=1,max_iter=2)
        self.assertTrue(np.isfinite(fit.matrix).all())
        self.assertAlmostEqual(np.linalg.norm(y-fit.matrix),2.-1e-12,places=8)

if __name__=='__main__':unittest.main()
