import unittest
import numpy as np
from matrix_completion import svt_proximal
from optimality import nuclear_objective,proximal_gradient_mapping_norm

class OptimalityTests(unittest.TestCase):
 def test_known_full_observation_optimum(self):
  y=np.diag([5.,2.,0.5]); idx=np.indices(y.shape).reshape(2,-1)
  idx=tuple(idx)
  x=np.diag([4.,1.,0.])
  self.assertLess(proximal_gradient_mapping_norm(x,y,idx,1.),1e-12)
  self.assertAlmostEqual(nuclear_objective(x,y,idx,1.),6.125)
  self.assertGreater(proximal_gradient_mapping_norm(np.zeros_like(y),y,idx,1.),1.)
 def test_svt_objective_nonincreasing(self):
  rng=np.random.default_rng(710)
  y=rng.normal(size=(8,6)); idx=np.where(rng.uniform(size=y.shape)<.55)
  fit=svt_proximal(y,idx,threshold=.7,step=1,max_iter=50,tol=0)
  # Independent iteration reimplementation solely for testing objective descent.
  x=np.zeros_like(y); previous=nuclear_objective(x,y,idx,.7)
  for _ in range(50):
   z=x.copy();z[idx]=y[idx]
   u,s,vt=np.linalg.svd(z,full_matrices=False)
   x=(u*np.maximum(s-.7,0))@vt
   now=nuclear_objective(x,y,idx,.7)
   self.assertLessEqual(now,previous+1e-9)
   previous=now
  self.assertLess(np.linalg.norm(fit.matrix-x),1e-10)
 def test_sparse_fixed_point_mapping(self):
  y=np.array([[3.,0.],[0.,2.]])
  idx=(np.array([0,1]),np.array([0,1]))
  fit=svt_proximal(y,idx,threshold=1.,max_iter=100,tol=1e-10)
  self.assertLess(proximal_gradient_mapping_norm(fit.matrix,y,idx,1.),1e-8)
 def test_invalid_values(self):
  y=np.eye(2);idx=(np.array([0,1]),np.array([0,1]))
  with self.assertRaises(ValueError):proximal_gradient_mapping_norm(y,y,idx,-1)
  with self.assertRaises(ValueError):nuclear_objective(np.full((2,2),np.nan),y,idx,1)
if __name__=='__main__':unittest.main()
