import json, unittest
from pathlib import Path
import numpy as np
from validation_32 import score_validation,select_by_family,validation_data
from runner_30 import training_data

class TestValidation(unittest.TestCase):
    def test_rmse_and_strata(self):
        r=score_validation([1,4,5],[2,2,4],[True,False,True])
        self.assertAlmostEqual(r['validation_rmse'],np.sqrt(2))
        self.assertEqual(r['warm_item_n'],2)
        self.assertEqual(r['cold_item_n'],1)
        self.assertAlmostEqual(r['cold_item_rmse'],2)
    def test_empty_stratum(self):
        self.assertIsNone(score_validation([2],[3],[True])['cold_item_rmse'])
    def test_invalid_values(self):
        with self.assertRaises(ValueError): score_validation([float('nan')],[1],[True])
        with self.assertRaises(ValueError): score_validation([1,2],[1],[True])
    def test_selection_ties_failures(self):
        p={'candidates':{'a':[{'x':1},{'x':2}],'b':[{}]}}
        rows=[{'family':'a','params':{'x':1},'status':'ok','validation_rmse':1.},
              {'family':'a','params':{'x':2},'status':'ok','validation_rmse':1.},
              {'family':'b','params':{},'status':'timeout'}]
        selected=select_by_family(rows,p)
        self.assertEqual(selected['a']['params'],{'x':1})
        self.assertIsNone(selected['b'])
    def test_duplicate_candidate_rejected(self):
        p={'candidates':{'a':[{}]}}
        with self.assertRaises(ValueError): select_by_family([{'family':'a','params':{},'status':'timeout'}]*2,p)
    def test_frozen_real_split(self):
        source=Path(__file__).resolve().parents[2]/'Dimension Reduction and Matrix Completion.zip'
        if not source.exists(): source=Path('/mnt/data/Dimension Reduction and Matrix Completion.zip')
        tr,tc,ty,shape=training_data(source)
        vr,vc,vy,warm,vshape=validation_data(source)
        self.assertEqual((len(ty),len(vy)),(70585,15125))
        self.assertEqual(shape,vshape)
        self.assertTrue(np.all(np.isfinite(vy)))
        self.assertEqual(len(warm),len(vy))
        self.assertTrue(np.all(np.isin(vy,np.arange(.5,5.1,.5))))

if __name__=='__main__': unittest.main()
