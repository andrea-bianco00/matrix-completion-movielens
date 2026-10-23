import hashlib, json, unittest
from pathlib import Path
import numpy as np
BASE=Path(__file__).resolve().parent
class ProtocolTests(unittest.TestCase):
 def setUp(self):
  self.m=json.loads((BASE/'split_manifest_18.json').read_text())
  self.z=np.load(BASE/'split_indices_18.npz')
 def test_partition(self):
  p=[self.z[k] for k in ('train','validation','test')]
  self.assertTrue(np.array_equal(np.sort(np.concatenate(p)),np.arange(self.m['source_rows'])))
 def test_sizes(self):
  for k,v in self.m['counts'].items(): self.assertEqual(len(self.z[k]),v)
 def test_fingerprints(self):
  for k,h in self.m['index_sha256'].items():
   self.assertEqual(hashlib.sha256(self.z[k].astype('<i8').tobytes()).hexdigest(),h)
 def test_recreate(self):
  n=self.m['source_rows']; perm=np.random.default_rng(self.m['seed']).permutation(n)
  a=int(.7*n); b=int(.85*n)
  for k,v in zip(('train','validation','test'),(perm[:a],perm[a:b],perm[b:])):
   np.testing.assert_array_equal(self.z[k],v)
if __name__=='__main__': unittest.main(verbosity=2)
