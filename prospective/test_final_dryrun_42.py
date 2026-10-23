import json, tempfile, unittest
from pathlib import Path
import numpy as np
from final_dryrun_42 import verify_lock, score_sealed_predictions, synthetic_dryrun, LOCK
class TestFinalDryrun(unittest.TestCase):
    def test_synthetic_preflight(self):
        self.assertEqual(synthetic_dryrun()['status'],'synthetic_only_preflight_passed')
    def test_lock_tampering(self):
        with tempfile.TemporaryDirectory() as d:
            data=json.loads(LOCK.read_text());data['source_hashes']['plan']='0'*64
            p=Path(d)/'lock.json';p.write_text(json.dumps(data))
            with self.assertRaisesRegex(ValueError,'plan changed'): verify_lock(p)
    def test_inconsistent_winner(self):
        with tempfile.TemporaryDirectory() as d:
            data=json.loads(LOCK.read_text());data['global_winner']['params']={'regularization':20.}
            p=Path(d)/'lock.json';p.write_text(json.dumps(data))
            with self.assertRaisesRegex(ValueError,'inconsistent'): verify_lock(p)
    def test_reject_invalid_mask(self):
        with self.assertRaisesRegex(ValueError,'boolean'):
            score_sealed_predictions([1.,2.],[1.,2.],[1,0])
    def test_reject_nan(self):
        with self.assertRaisesRegex(ValueError,'nonfinite'):
            score_sealed_predictions([np.nan],[1.],[True])
    def test_reject_shape_mismatch(self):
        with self.assertRaisesRegex(ValueError,'shape mismatch'):
            score_sealed_predictions([1.],[1.,2.],[True,False])
    def test_warm_cold_scoring(self):
        r=score_sealed_predictions([1.,3.,1.,5.],[2.,3.,3.,3.],[True,True,False,False])
        self.assertAlmostEqual(r['warm_item_rmse'],np.sqrt(.5))
        self.assertAlmostEqual(r['cold_item_rmse'],2.)
        self.assertAlmostEqual(r['validation_rmse'],1.5)
if __name__=='__main__': unittest.main()
