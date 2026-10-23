import json
import unittest
from pathlib import Path
ROOT = Path(__file__).resolve().parent

class BenchmarkPlanTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.plan = json.loads((ROOT / 'benchmark_plan_28.json').read_text())
        cls.manifest = json.loads((ROOT / 'split_manifest_18.json').read_text())

    def test_split_is_existing_frozen_manifest(self):
        self.assertEqual(self.plan['split_manifest'], 'split_manifest_18.json')
        self.assertEqual(self.manifest['counts'], {'train':70585,'validation':15125,'test':15126})
        self.assertEqual(sum(self.manifest['counts'].values()), self.manifest['source_rows'])

    def test_grid_unique_and_complete(self):
        expected = {'global_mean':1,'user_item_bias':2,'SVP':3,'SVT':3,'ADMiRA_lab_variant':2}
        self.assertEqual({k:len(v) for k,v in self.plan['candidates'].items()}, expected)
        for configurations in self.plan['candidates'].values():
            enc = [json.dumps(c,sort_keys=True) for c in configurations]
            self.assertEqual(len(enc),len(set(enc)))

    def test_solvers_within_valid_ranges(self):
        p = self.plan
        self.assertTrue(all(0 < c['rank'] < 610 for c in p['candidates']['SVP']))
        self.assertTrue(all(0 < 2*c['rank'] < 610 for c in p['candidates']['ADMiRA_lab_variant']))
        self.assertTrue(all(c['threshold'] > 0 for c in p['candidates']['SVT']))
        self.assertLessEqual(p['solver_settings']['SVT']['step'], 1)
        self.assertGreater(p['resource_policy'].find('120 second'),-1)

    def test_selection_uses_no_test(self):
        self.assertEqual(self.plan['selection_metric'],'validation_raw_rmse')
        self.assertIn('no additional grid expansion',self.plan['evaluation_policy'])
        self.assertIn('all failures',self.plan['failed_fit_policy'])

if __name__ == '__main__': unittest.main()
