import tempfile, unittest
from pathlib import Path
import numpy as np
from isolation_36 import export,load_fit
from runner_30 import training_data
SOURCE=Path('/mnt/data/Dimension Reduction and Matrix Completion.zip')
class IsolationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp=tempfile.TemporaryDirectory();cls.root=Path(cls.tmp.name)/'export';cls.meta=export(SOURCE,cls.root)
    @classmethod
    def tearDownClass(cls):cls.tmp.cleanup()
    def test_partition(self):
        self.assertEqual(sum(p['count'] for p in self.meta['partitions'].values()),100836)
    def test_equal_legacy_training(self):
        a=training_data(SOURCE);b=load_fit(self.root/'fit')
        self.assertEqual(a[3],b[3])
        for x,y in zip(a[:3],b[:3]):np.testing.assert_array_equal(x,y)
    def test_no_labels_from_holdouts_in_fit_directory(self):
        self.assertEqual(sorted(p.name for p in (self.root/'fit').iterdir()),['catalog.json','train.csv'])
    def test_heldout_mutation_does_not_change_fit(self):
        baseline=load_fit(self.root/'fit')
        with (self.root/'validation'/'validation.csv').open('a') as f:f.write('999,999,1.0,0\n')
        with (self.root/'sealed_test'/'test.csv').open('a') as f:f.write('999,999,1.0,0\n')
        changed=load_fit(self.root/'fit')
        for a,b in zip(baseline[:3],changed[:3]):np.testing.assert_array_equal(a,b)
    def test_reject_nonempty_export(self):
        with self.assertRaises(ValueError):export(SOURCE,self.root)
if __name__=='__main__':unittest.main()
