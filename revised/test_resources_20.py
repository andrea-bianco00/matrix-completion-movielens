import unittest
import numpy as np
from resource_guard_20 import design_memory_estimate, enforce_design_budget
from scalable_19 import admira_matrix_free

class ResourceTests(unittest.TestCase):
    def test_exact_estimate(self):
        self.assertEqual(design_memory_estimate(100, 2, 3), 6400)

    def test_invalid_counts(self):
        with self.assertRaises(ValueError): design_memory_estimate(2.5, 0, 2)

    def test_budget_boundary(self):
        self.assertEqual(enforce_design_budget(100, 2, 3, 6400), 6400)
        with self.assertRaises(MemoryError): enforce_design_budget(100, 2, 3, 6399)

    def test_budget_rejects_before_ls(self):
        shape=(8,9); rows,cols=np.indices(shape); rows=rows.ravel(); cols=cols.ravel()
        with self.assertRaises(MemoryError):
            admira_matrix_free(shape,(rows,cols),np.ones(len(rows)),rank=1,max_iter=1,max_design_bytes=1)

    def test_budget_does_not_change_solution(self):
        shape=(8,9); rows,cols=np.indices(shape); rows=rows.ravel(); cols=cols.ravel()
        values=(rows+cols).astype(float)
        a=admira_matrix_free(shape,(rows,cols),values,rank=1,max_iter=2)
        b=admira_matrix_free(shape,(rows,cols),values,rank=1,max_iter=2,max_design_bytes=10_000)
        np.testing.assert_allclose(a.to_dense(),b.to_dense(),atol=1e-7)

if __name__=='__main__': unittest.main()
