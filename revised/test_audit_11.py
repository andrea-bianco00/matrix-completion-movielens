import unittest
import numpy as np
from matrix_completion import admira_lab

class Audit11(unittest.TestCase):
    def setUp(self):
        rng=np.random.default_rng(20261023)
        self.y=rng.normal(size=(7,6))
        self.idx=np.where(np.ones_like(self.y,dtype=bool))

    def test_diagnostics_match_iterations(self):
        fit=admira_lab(self.y,self.idx,rank=2,max_iter=4,track_conditioning=True)
        self.assertEqual(len(fit.diagnostics),fit.iterations)
        for j,d in enumerate(fit.diagnostics,1):
            self.assertEqual(d['iteration'],j)
            self.assertLessEqual(d['rank'],d['columns'])
            self.assertTrue(np.isfinite(d['least_squares_residual_norm']))
            self.assertTrue(np.isfinite(d['coefficient_norm']))

    def test_diagnostics_do_not_change_solution(self):
        a=admira_lab(self.y,self.idx,rank=2,max_iter=4)
        b=admira_lab(self.y,self.idx,rank=2,max_iter=4,track_conditioning=True)
        np.testing.assert_allclose(a.matrix,b.matrix,rtol=1e-12,atol=1e-12)
        self.assertIsNone(a.diagnostics)

    def test_rcond_zero_vs_default_full_rank(self):
        a=admira_lab(self.y,self.idx,rank=2,max_iter=3,track_conditioning=True)
        b=admira_lab(self.y,self.idx,rank=2,max_iter=3,lstsq_rcond=0.0,track_conditioning=True)
        self.assertEqual(a.matrix.shape,b.matrix.shape)
        self.assertTrue(np.isfinite(b.matrix).all())

    def test_invalid_rcond(self):
        for v in [-1.,float('nan'),float('inf')]:
            with self.subTest(v=v),self.assertRaises(ValueError):
                admira_lab(self.y,self.idx,rank=2,lstsq_rcond=v)

if __name__=='__main__': unittest.main()
