import unittest
from pathlib import Path
import numpy as np
from runner_33 import fit_model,run_child
import os
SOURCE=Path(os.environ.get('MOVIELENS_SOURCE_ZIP', '/mnt/data/Dimension Reduction and Matrix Completion.zip'))
requires_original_dataset = unittest.skipUnless(SOURCE.is_file(), 'Original MovieLens source ZIP unavailable; set MOVIELENS_SOURCE_ZIP to run this integration test')
class BoundaryTests(unittest.TestCase):
    def test_fit_interface_training_only(self):
        p,n,s=fit_model('global_mean',{},np.array([0,1]),np.array([0,1]),np.array([2.,4.]),(2,2))
        np.testing.assert_array_equal(p(np.array([0,1]),np.array([1,0])),[3.,3.])
        self.assertEqual(s,'closed_form')
    def test_reject_unplanned_grid(self):
        with self.assertRaises(ValueError):
            fit_model('user_item_bias',{'regularization':999},np.array([0]),np.array([0]),np.array([2.]),(1,1))
    @requires_original_dataset
    def test_training_worker_does_not_score_validation(self):
        x=run_child('global_mean',{},SOURCE,20,evaluate=False)
        self.assertEqual(x['status'],'ok',x)
        self.assertNotIn('validation_rmse',x)
    @requires_original_dataset
    def test_validation_scoring_matches_independent_global_mean(self):
        from runner_30 import training_data
        from validation_32 import validation_data,score_validation
        _,_,train,_=training_data(SOURCE)
        _,_,truth,warm,_=validation_data(SOURCE)
        expected=score_validation(np.full(len(truth),train.mean()),truth,warm)
        x=run_child('global_mean',{},SOURCE,20,evaluate=True)
        self.assertEqual(x['status'],'ok',x)
        self.assertAlmostEqual(x['validation_rmse'],expected['validation_rmse'],places=12)
        self.assertEqual(x['validation_n'],15125)
    @requires_original_dataset
    def test_failed_candidate_not_scored(self):
        x=run_child('SVP',{'rank':999},SOURCE,20,evaluate=True)
        self.assertEqual(x['status'],'error')
        self.assertNotIn('validation_rmse',x)
if __name__=='__main__':unittest.main()
