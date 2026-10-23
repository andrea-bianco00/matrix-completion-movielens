"""Audit 14: spectral degeneracy and solver failure tests."""
import unittest
from unittest.mock import patch
import numpy as np
from scipy.sparse.linalg import ArpackNoConvergence
from matrix_completion import svp
from sparse_svp import sparse_svp

class TestSpectral14(unittest.TestCase):
    def test_repeated_singular_values_compare_matrices_not_vectors(self):
        # The top-2 subspace is unambiguous despite an exact tie inside it.
        a=np.diag([5.,5.,1.,.2,0.,0.])
        rows,cols=np.indices(a.shape)
        out=sparse_svp(a.shape,(rows.ravel(),cols.ravel()),a.ravel(),rank=2,max_iter=2,tol=0)
        expected=np.diag([5.,5.,0.,0.,0.,0.])
        np.testing.assert_allclose(out.to_dense(),expected,atol=1e-7,rtol=1e-7)

    def test_boundary_tie_nonunique_truncation(self):
        # A repeated singular value at the truncation boundary makes the
        # rank-1 projection nonunique: verify optimal error, not matrix equality.
        a=np.diag([4.,4.,1.,0.,0.,0.])
        rows,cols=np.indices(a.shape)
        out=sparse_svp(a.shape,(rows.ravel(),cols.ravel()),a.ravel(),rank=1,max_iter=1)
        self.assertAlmostEqual(np.linalg.norm(a-out.to_dense(),'fro'),np.sqrt(4**2+1),places=6)
        self.assertAlmostEqual(np.linalg.norm(out.to_dense(),'fro'),4.,places=6)

    def test_close_singular_values(self):
        a=np.diag([5.,5.-1e-7,1.,.2,0.,0.])
        rows,cols=np.indices(a.shape)
        out=sparse_svp(a.shape,(rows.ravel(),cols.ravel()),a.ravel(),rank=2,max_iter=2,tol=0)
        np.testing.assert_allclose(out.to_dense(),np.diag([5.,5.-1e-7,0.,0.,0.,0.]),atol=1e-6,rtol=1e-6)

    def test_arpack_failure_is_not_silently_treated_as_convergence(self):
        a=np.eye(6); rows,cols=np.indices(a.shape)
        with patch('sparse_svp.svds',side_effect=ArpackNoConvergence('deliberate test',np.array([]),np.empty((6,0)))):
            with self.assertRaises(ArpackNoConvergence):
                sparse_svp(a.shape,(rows.ravel(),cols.ravel()),a.ravel(),rank=2,max_iter=1)

if __name__=='__main__': unittest.main()
