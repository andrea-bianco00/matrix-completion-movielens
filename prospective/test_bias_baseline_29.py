import unittest
import numpy as np
from bias_baseline_29 import fit_bias,predict_bias
class TestBias(unittest.TestCase):
    def test_stationary_coordinate_updates(self):
        r=np.array([0,0,1,1,2]);c=np.array([0,1,0,2,2]);y=np.array([1.,2.,3.,4.,5.]);lam=5.
        m=fit_bias(r,c,y,(4,4),regularization=lam)
        residual=y-predict_bias(m,r,c)
        np.testing.assert_allclose(np.bincount(r,weights=residual,minlength=4),lam*m['user_bias'],atol=1e-7)
        np.testing.assert_allclose(np.bincount(c,weights=residual,minlength=4),lam*m['item_bias'],atol=1e-7)
        self.assertTrue(all(b<=a+1e-8 for a,b in zip(m['objective_history'],m['objective_history'][1:])))
    def test_unseen(self):
        m=fit_bias(np.array([0,1]),np.array([0,1]),np.array([2.,4.]),(3,3),regularization=20)
        self.assertEqual(predict_bias(m,np.array([2]),np.array([2]))[0],3.)
        self.assertEqual(m['user_bias'][2],0.)
    def test_reject(self):
        with self.assertRaises(ValueError): fit_bias(np.array([0]),np.array([0]),np.array([1.]),(1,1),regularization=0)
        with self.assertRaises(ValueError): fit_bias(np.array([0]),np.array([0]),np.array([np.nan]),(1,1),regularization=5)
    def test_reproducible(self):
        args=(np.array([0,1,0]),np.array([0,1,1]),np.array([1.,3.,4.]),(2,2))
        a=fit_bias(*args,regularization=5);b=fit_bias(*args,regularization=5)
        np.testing.assert_array_equal(a['user_bias'],b['user_bias'])
if __name__=='__main__': unittest.main()
