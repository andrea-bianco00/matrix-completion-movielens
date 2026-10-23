import json
import shutil
import tempfile
import unittest
from pathlib import Path
from verify_release import ROOT, verify

class ReleaseIntegrityTests(unittest.TestCase):
    def test_actual_release(self):
        self.assertTrue(verify()['verified'])

    def test_modified_lock_is_rejected(self):
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / 'prospective'; p.mkdir()
            for name in ('selection_lock_41.json', 'final_winner_results_43.json',
                         'validation_results_34.json', 'split_manifest_18.json',
                         'benchmark_plan_28.json', 'freeze_34.json'):
                shutil.copy2(ROOT / 'prospective' / name, p / name)
            lock_path = p / 'selection_lock_41.json'
            lock = json.loads(lock_path.read_text())
            lock['global_winner']['params']['regularization'] = 20.0
            lock_path.write_text(json.dumps(lock))
            with self.assertRaises(AssertionError):
                verify(td)

    def test_modified_test_rmse_is_rejected(self):
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / 'prospective'; p.mkdir()
            for name in ('selection_lock_41.json', 'final_winner_results_43.json',
                         'validation_results_34.json', 'split_manifest_18.json',
                         'benchmark_plan_28.json', 'freeze_34.json'):
                shutil.copy2(ROOT / 'prospective' / name, p / name)
            result_path = p / 'final_winner_results_43.json'
            result = json.loads(result_path.read_text())
            result['results'][0]['test_rmse'] = 0.1
            result_path.write_text(json.dumps(result))
            with self.assertRaises(AssertionError):
                verify(td)

if __name__ == '__main__':
    unittest.main()
