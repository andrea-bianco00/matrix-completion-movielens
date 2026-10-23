"""Independent mathematical properties and input-validation regression checks."""
import unittest
import numpy as np
from matrix_completion import svp, svt_proximal, admira_lab, _project_rank, evaluate

class Properties(unittest.TestCase):
    def setUp(self):
        rng=np.random.default_rng(601)
        self.y=rng.normal(size=(7,5))
        self.idx=tuple(np.indices(self.y.shape)[k].ravel() for k in range(2))
    def test_rank_projection_is_optimal_vs_random_rank_candidates(self):
        rng=np.random.default_rng(602)
        x=_project_rank(self.y,2)
        err=np.linalg.norm(self.y-x)
        s=np.linalg.svd(self.y,compute_uv=False)
        self.assertAlmostEqual(err,np.linalg.norm(s[2:]),places=11)
        for _ in range(100):
            candidate=rng.normal(size=(7,2))@rng.normal(size=(2,5))
            self.assertLessEqual(err,np.linalg.norm(self.y-candidate)+1e-10)
    def test_svt_proximal_objective_full_observation(self):
        lam=0.8
        fit=svt_proximal(self.y,self.idx,threshold=lam,max_iter=4)
        s=np.linalg.svd(self.y,compute_uv=False)
        optimum=np.sum(0.5*np.minimum(s,lam)**2+lam*np.maximum(s-lam,0))
        def objective(x):return 0.5*np.linalg.norm(x-self.y)**2+lam*np.linalg.svd(x,compute_uv=False).sum()
        self.assertAlmostEqual(objective(fit.matrix),optimum,places=9)
    def test_svp_full_observation_best_rank2(self):
        fit=svp(self.y,self.idx,rank=2,max_iter=5)
        self.assertLess(np.linalg.norm(fit.matrix-_project_rank(self.y,2)),1e-10)
    def test_admira_full_observation_rank2(self):
        fit=admira_lab(self.y,self.idx,rank=2,max_iter=15)
        self.assertLess(np.linalg.norm(fit.matrix-_project_rank(self.y,2)),1e-8)
    def test_invalid_rank_svp(self):
        for rank in (0,6,1.5):
            with self.subTest(rank=rank):
                with self.assertRaises(ValueError):svp(self.y,self.idx,rank=rank,max_iter=2)
    def test_invalid_prediction_nan(self):
        x=np.zeros_like(self.y);x[0,0]=np.nan
        with self.assertRaises(ValueError):evaluate(x,self.y,self.idx)

if __name__=='__main__':unittest.main()
