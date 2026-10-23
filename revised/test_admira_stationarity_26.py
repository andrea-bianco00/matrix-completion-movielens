"""Independent checks of first-order stationarity for the ADMiRA lab variant."""
import unittest
import numpy as np
from matrix_completion import admira_lab
from nonconvex_diagnostics import tangent_gradient_norm, observed_objective

class TestAdmiraStationarity(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(2601)
        self.y = rng.normal(size=(8, 7))
        self.idx = np.indices(self.y.shape).reshape(2,-1)
        self.indices = (self.idx[0],self.idx[1])

    def test_exact_rank_one_full_observation_stationary(self):
        a=np.arange(1.,9.)[:,None] @ np.arange(1.,8.)[None,:]
        fit=admira_lab(a,self.indices,rank=1,max_iter=15,tol=1e-12)
        self.assertLess(tangent_gradient_norm(fit.matrix,a,self.indices,1),1e-8)

    def test_gradient_tangent_matches_factor_directional_derivative(self):
        # For X=UV^T, dX = dU V^T is tangent. Check finite differences.
        rng=np.random.default_rng(7)
        u=rng.normal(size=(8,2)); v=rng.normal(size=(7,2)); du=rng.normal(size=(8,2))
        x=u@v.T; direction=du@v.T
        grad=np.zeros_like(x); grad[self.indices]=x[self.indices]-self.y[self.indices]
        expected=float(np.sum(grad*direction))
        h=1e-6
        actual=(observed_objective(x+h*direction,self.y,self.indices)-observed_objective(x-h*direction,self.y,self.indices))/(2*h)
        self.assertAlmostEqual(actual,expected,places=6)

    def test_nonzero_tangent_gradient_not_mistaken_for_convergence(self):
        selected=np.random.default_rng(99).choice(self.indices[0].size, size=35, replace=False)
        partial=(self.indices[0][selected], self.indices[1][selected])
        fit=admira_lab(self.y,partial,rank=2,max_iter=1,tol=0.)
        residual=tangent_gradient_norm(fit.matrix,self.y,partial,2)
        self.assertTrue(np.isfinite(residual))
        self.assertGreater(residual,1e-6)
        self.assertEqual(fit.stop_reason,'max_iter')

if __name__=='__main__': unittest.main()
