import json, tempfile, unittest
from pathlib import Path
from runner_30 import run_child, training_data
SOURCE=Path('/mnt/data/Dimension Reduction and Matrix Completion.zip')
class RunnerTests(unittest.TestCase):
    def test_verified_training_count(self):
        r,c,y,shape=training_data(SOURCE)
        self.assertEqual(len(y),70585)
        self.assertEqual(shape,(610,9724))
        self.assertEqual(len(r),len(c))
    def test_global_mean_worker(self):
        out=run_child('global_mean',{},SOURCE,15)
        self.assertEqual(out['status'],'ok')
        self.assertEqual(out['evaluated_split'],'train_only')
        self.assertAlmostEqual(out['training_rmse'],1.0442424689329852,places=9)
    def test_bad_source_fails_closed(self):
        with tempfile.TemporaryDirectory() as d:
            bad=Path(d)/'not_dataset.zip';bad.write_bytes(b'bad')
            out=run_child('global_mean',{},bad,15)
            self.assertEqual(out['status'],'error')
    def test_timeout_is_reported(self):
        out=run_child('global_mean',{},SOURCE,0.00001)
        self.assertEqual(out['status'],'timeout')
    def test_unknown_family_not_silently_replaced(self):
        out=run_child('made_up_model',{},SOURCE,15)
        self.assertEqual(out['status'],'error')
if __name__=='__main__':unittest.main()
