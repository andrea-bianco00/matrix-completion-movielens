import unittest
import numpy as np
from scalable_19 import svt_matrix_free

class TestSVTDiagnostics24(unittest.TestCase):
    def test_objective_and_fit_invariant(self):
        rng=np.random.default_rng(24)
        rows,cols=np.indices((8,7)); rows=rows.ravel(); cols=cols.ravel()
        vals=(rng.normal(size=(8,2))@rng.normal(size=(2,7))).ravel()
        log=[]
        a=svt_matrix_free((8,7),(rows,cols),vals,threshold=0.5,initial_rank=3,max_iter=6,tol=0,diagnostic_log=log)
        b=svt_matrix_free((8,7),(rows,cols),vals,threshold=0.5,initial_rank=3,max_iter=6,tol=0)
        np.testing.assert_allclose(a.to_dense(),b.to_dense(),atol=1e-8)
        self.assertEqual(len(log),len(a.history))
        for x in log:
            self.assertTrue(np.isfinite(x['objective']))
            self.assertGreaterEqual(x['relative_change'],0)
            self.assertGreaterEqual(x['nuclear_norm'],0)
        self.assertTrue(all(log[i+1]['objective']<=log[i]['objective']+1e-8 for i in range(len(log)-1)))

if __name__=='__main__': unittest.main()
