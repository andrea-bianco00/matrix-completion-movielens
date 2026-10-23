import unittest
import numpy as np
from scalable_19 import svt_matrix_free, LowRankFit
from stationarity_25 import matrix_free_pg_residual
from optimality import proximal_gradient_mapping_norm

class TestMatrixFreeStationarity(unittest.TestCase):
    def test_matches_dense_reference_partial_observations(self):
        rng=np.random.default_rng(25)
        shape=(9,7); rows,cols=np.where(rng.random(shape)<0.7)
        values=rng.normal(size=len(rows))
        fit=svt_matrix_free(shape,(rows,cols),values,threshold=3.5,max_iter=4,initial_rank=3)
        observed=np.zeros(shape); observed[rows,cols]=values
        mask=np.zeros(shape,dtype=bool); mask[rows,cols]=True
        dense=proximal_gradient_mapping_norm(fit.to_dense(),observed,(rows,cols),3.5)
        actual=matrix_free_pg_residual(fit,shape,(rows,cols),values,threshold=3.5,initial_rank=3)
        self.assertAlmostEqual(actual['pg_mapping_norm'],dense,places=6)

    def test_zero_target_has_zero_mapping(self):
        shape=(8,6); rows=np.array([0,1,2]); cols=np.array([1,2,3]); values=np.zeros(3)
        fit=LowRankFit(np.zeros((8,0)),np.empty(0),np.zeros((0,6)),[], 'test')
        out=matrix_free_pg_residual(fit,shape,(rows,cols),values,threshold=1,initial_rank=2)
        self.assertAlmostEqual(out['pg_mapping_norm'],0,places=10)

    def test_nonoptimal_zero_matrix_has_positive_mapping(self):
        shape=(8,6); rows=np.array([0,1,2]); cols=np.array([1,2,3]); values=np.array([3.,3.,3.])
        fit=LowRankFit(np.zeros((8,0)),np.empty(0),np.zeros((0,6)),[], 'test')
        out=matrix_free_pg_residual(fit,shape,(rows,cols),values,threshold=1,initial_rank=2)
        self.assertGreater(out['pg_mapping_norm'],0)

    def test_rejects_invalid_penalty(self):
        shape=(8,6); rows=np.array([0]); cols=np.array([0]); values=np.array([1.])
        fit=LowRankFit(np.zeros((8,0)),np.empty(0),np.zeros((0,6)),[], 'test')
        with self.assertRaises(ValueError):
            matrix_free_pg_residual(fit,shape,(rows,cols),values,threshold=-1)

if __name__=='__main__': unittest.main()
