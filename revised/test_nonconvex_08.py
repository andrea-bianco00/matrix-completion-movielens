import unittest
import numpy as np
from matrix_completion import svp, admira_lab
from nonconvex_diagnostics import observed_objective, tangent_gradient_norm, svp_projected_step_norm

class NonconvexDiagnostics(unittest.TestCase):
    def setUp(self):
        self.rng = np.random.default_rng(8008)
        self.y = self.rng.normal(size=(7, 5))
        self.full = np.indices(self.y.shape)
        self.idx = (self.full[0].ravel(), self.full[1].ravel())

    def test_objective_nonnegative_and_exact(self):
        self.assertEqual(observed_objective(self.y, self.y, self.idx), 0.)
        self.assertGreater(observed_objective(np.zeros_like(self.y), self.y, self.idx), 0.)

    def test_full_observation_svp_stationarity(self):
        x = svp(self.y, self.idx, rank=2, max_iter=5).matrix
        self.assertLess(svp_projected_step_norm(x, self.y, self.idx, 2), 1e-10)
        self.assertLess(tangent_gradient_norm(x, self.y, self.idx, 2), 1e-10)

    def test_full_observation_admira_tangent_stationarity(self):
        x = admira_lab(self.y, self.idx, rank=2, max_iter=15).matrix
        self.assertLess(tangent_gradient_norm(x, self.y, self.idx, 2), 1e-8)

    def test_tangent_gradient_matches_directional_derivative(self):
        u, _, vt = np.linalg.svd(self.y, full_matrices=False)
        x = (u[:, :2] * np.array([4., 2.])) @ vt[:2]
        g = x - self.y
        ur, vr = u[:, :2], vt[:2].T
        p = ur @ (ur.T @ g) + (g @ vr) @ vr.T - ur @ (ur.T @ g @ vr) @ vr.T
        t = p / np.linalg.norm(p)
        h = 1e-6
        fd = (observed_objective(x+h*t,self.y,self.idx)-observed_objective(x-h*t,self.y,self.idx))/(2*h)
        self.assertAlmostEqual(fd, np.linalg.norm(p), places=7)
        self.assertAlmostEqual(tangent_gradient_norm(x,self.y,self.idx,2),np.linalg.norm(p),places=10)

    def test_reject_rank_deficient_manifold_point(self):
        x = np.zeros_like(self.y)
        with self.assertRaises(ValueError):tangent_gradient_norm(x,self.y,self.idx,2)

    def test_reject_nonfinite(self):
        x=np.zeros_like(self.y);x[0,0]=np.nan
        with self.assertRaises(ValueError):observed_objective(x,self.y,self.idx)
        with self.assertRaises(ValueError):svp_projected_step_norm(x,self.y,self.idx,2)

if __name__=='__main__':unittest.main()
