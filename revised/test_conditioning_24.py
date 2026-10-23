import unittest
import numpy as np
from scalable_19 import admira_matrix_free

class TestConditioning24(unittest.TestCase):
    def test_diagnostics_do_not_change_fit(self):
        rng=np.random.default_rng(241)
        m,n=12,11
        rows,cols=np.indices((m,n)); rows=rows.ravel(); cols=cols.ravel()
        vals=(rng.normal(size=(m,2))@rng.normal(size=(2,n))).ravel()
        log=[]
        a=admira_matrix_free((m,n),(rows,cols),vals,rank=2,max_iter=4,tol=0,conditioning_log=log)
        b=admira_matrix_free((m,n),(rows,cols),vals,rank=2,max_iter=4,tol=0)
        np.testing.assert_allclose(a.to_dense(),b.to_dense(),atol=1e-8,rtol=1e-8)
        self.assertEqual(len(log),len(a.history))
        for x in log:
            self.assertGreaterEqual(x['condition_2'],1)
            self.assertLessEqual(x['numerical_rank'],x['columns'])
    def test_zero_target_log_empty(self):
        log=[]
        out=admira_matrix_free((5,6),(np.array([0]),np.array([1])),np.array([0.]),rank=1,conditioning_log=log)
        self.assertEqual(log,[])
        self.assertEqual(out.stop_reason,'zero observed target')

if __name__=='__main__': unittest.main()
