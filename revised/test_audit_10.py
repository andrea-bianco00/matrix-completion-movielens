import unittest
import numpy as np
from matrix_completion import svp, svt_proximal, admira_lab, observed_rmse, relative_error
from conditioning import design_diagnostics

class Audit10(unittest.TestCase):
    def test_zero_targets_fitting(self):
        y=np.zeros((4,3)); idx=tuple(a.ravel() for a in np.indices(y.shape))
        for method, kw in [(svp, {'rank':1}), (svt_proximal, {'threshold':1}), (admira_lab, {'rank':1})]:
            with self.subTest(method=method.__name__):
                fit=method(y,idx,max_iter=3,**kw)
                np.testing.assert_array_equal(fit.matrix,y)
                self.assertEqual(fit.train_errors,[0.0])
                self.assertEqual(fit.iterations,1)
                self.assertEqual(fit.stop_reason,'relative iterate change')
                with self.assertRaises(ValueError): relative_error(fit.matrix,y,idx)

    def test_observed_rmse_known_value(self):
        y=np.zeros((2,2)); x=np.array([[3.,99.],[99.,4.]])
        self.assertAlmostEqual(observed_rmse(x,y,(np.array([0,1]),np.array([0,1]))),np.sqrt(12.5))

    def test_full_column_rank(self):
        d=design_diagnostics(np.eye(3))
        self.assertEqual(d['rank'],3)
        self.assertFalse(d['rank_deficient'])
        self.assertAlmostEqual(d['condition_number'],1)

    def test_dependent_columns(self):
        d=design_diagnostics(np.array([[1.,2.],[2.,4.],[3.,6.]]))
        self.assertEqual(d['rank'],1)
        self.assertTrue(d['rank_deficient'])

    def test_nearly_dependent_cutoff(self):
        a=np.diag([1.,1e-10])
        self.assertEqual(design_diagnostics(a,rcond=1e-8)['rank'],1)
        self.assertEqual(design_diagnostics(a,rcond=1e-12)['rank'],2)

    def test_unobserved_row_is_not_identifiable(self):
        # Two rank-one matrices agree on observed row but differ elsewhere.
        a=np.array([[1.,2.],[3.,6.]])
        b=np.array([[1.,2.],[7.,14.]])
        idx=(np.array([0,0]),np.array([0,1]))
        np.testing.assert_array_equal(a[idx],b[idx])
        self.assertEqual(np.linalg.matrix_rank(a),1)
        self.assertEqual(np.linalg.matrix_rank(b),1)
        self.assertGreater(np.linalg.norm(a-b),0)

if __name__=='__main__': unittest.main()
