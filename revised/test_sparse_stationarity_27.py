import unittest
import numpy as np
from scipy.linalg import svd
from scalable_19 import LowRankFit
from sparse_stationarity_27 import tangent_gradient_sparse
from nonconvex_diagnostics import tangent_gradient_norm

class TestSparseStationarity(unittest.TestCase):
    def test_matches_independent_dense_reference(self):
        rng=np.random.default_rng(27)
        for m,n,r in [(8,7,1),(10,9,2),(12,11,3)]:
            a=rng.normal(size=(m,n)); u,s,vt=svd(a,full_matrices=False)
            fit=LowRankFit(u[:,:r],s[:r],vt[:r],[], 'test')
            y=rng.normal(size=(m,n)); mask=rng.random((m,n))<0.55
            rows,cols=np.where(mask); vals=y[rows,cols]
            result=tangent_gradient_sparse(fit,rows,cols,vals,(m,n),r)
            dense=tangent_gradient_norm(fit.to_dense(),np.where(mask,y,0.),(rows,cols),r)
            self.assertAlmostEqual(result['tangent_gradient_norm'],dense,places=9)
    def test_exact_solution(self):
        u=np.array([[1.],[0.],[0.]]); vt=np.array([[1.,0.,0.,0.]])
        fit=LowRankFit(u,np.array([2.]),vt,[],'test')
        rows=np.array([0,1,2]); cols=np.array([0,1,2]); vals=fit.predict(rows,cols)
        self.assertEqual(tangent_gradient_sparse(fit,rows,cols,vals,(3,4),1)['tangent_gradient_norm'],0.)
    def test_rejects_rank_deficiency(self):
        fit=LowRankFit(np.eye(3,1),np.array([0.]),np.eye(1,4),[],'test')
        with self.assertRaises(ValueError):
            tangent_gradient_sparse(fit,np.array([0]),np.array([0]),np.array([1.]),(3,4),1)
    def test_rejects_duplicates(self):
        fit=LowRankFit(np.eye(3,1),np.array([1.]),np.eye(1,4),[],'test')
        with self.assertRaises(ValueError):
            tangent_gradient_sparse(fit,np.array([0,0]),np.array([0,0]),np.array([1.,2.]),(3,4),1)
if __name__=='__main__':unittest.main()
