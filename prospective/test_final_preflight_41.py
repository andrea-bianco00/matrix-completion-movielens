import json, tempfile, unittest
from pathlib import Path
from final_preflight_41 import freeze_selection, HERE
class TestFinalPreflight(unittest.TestCase):
    def setUp(self):
        self.plan=HERE/'benchmark_plan_28.json'; self.results=HERE/'validation_results_34.json'; self.freeze=HERE/'freeze_34.json'
    def test_actual_lock(self):
        with tempfile.TemporaryDirectory() as d:
            out=Path(d)/'lock.json'; result=freeze_selection(self.plan,self.results,self.freeze,out)
            self.assertEqual(result['global_winner']['family'],'user_item_bias')
            self.assertEqual(result['global_winner']['params'],{'regularization':5.0})
            self.assertEqual(len(result['family_winners']),5)
            self.assertEqual(json.loads(out.read_text()),result)
    def test_tampered_plan_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'plan.json'; p.write_text(self.plan.read_text()+' ')
            with self.assertRaisesRegex(ValueError,'plan differs'):
                freeze_selection(p,self.results,self.freeze,Path(d)/'out.json')
    def test_wrong_freeze_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'results.json'; data=json.loads(self.results.read_text());data['freeze_sha256']='0'*64;p.write_text(json.dumps(data))
            with self.assertRaisesRegex(ValueError,'different freeze'):
                freeze_selection(self.plan,p,self.freeze,Path(d)/'out.json')
    def test_duplicate_candidate_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'results.json'; data=json.loads(self.results.read_text());data['records'][1]=data['records'][0].copy();p.write_text(json.dumps(data))
            with self.assertRaises(ValueError):
                freeze_selection(self.plan,p,self.freeze,Path(d)/'out.json')
if __name__=='__main__': unittest.main()
