import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from docker_isolation_38 import docker_command, execute, stage_reviewed_code

class DockerIsolationTests(unittest.TestCase):
    def setUp(self):
        self.t=tempfile.TemporaryDirectory(); self.addCleanup(self.t.cleanup)
        self.base=Path(self.t.name)
        self.fit=self.base/'fit';self.fit.mkdir()
        (self.fit/'train.csv').write_text('train only')
        (self.fit/'catalog.json').write_text('{}')
        self.q=self.base/'coordinates.npz';self.q.write_bytes(b'coordinates')
        self.stage=stage_reviewed_code(self.base/'reviewed_code')
        self.secret=self.base/'sealed_test';self.secret.mkdir()
        (self.secret/'test.csv').write_text('secret')
    def test_no_secret_mount_and_restrictions(self):
        cmd=docker_command('global_mean',{},self.fit,self.q,stage_dir=self.stage,image="reviewed@sha256:aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa")
        s=' '.join(cmd)
        self.assertNotIn(str(self.secret),s)
        for part in ('--network none','--read-only','--cap-drop ALL',
                     '--security-opt no-new-privileges','--memory 2g','--user 65534:65534'):
            self.assertIn(part,s)
    def test_reject_unknown_family(self):
        with self.assertRaises(ValueError):docker_command('unknown',{},self.fit,self.q,stage_dir=self.stage,image="reviewed@sha256:aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa")
    def test_reject_missing_training(self):
        (self.fit/'train.csv').unlink()
        with self.assertRaises(ValueError):docker_command('global_mean',{},self.fit,self.q,stage_dir=self.stage,image="reviewed@sha256:aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa")
    def test_source_allowlist_excludes_unreviewed_files(self):
        from docker_isolation_38 import STAGE_FILES, REVISED_FILES
        staged={str(p.relative_to(self.stage)) for p in self.stage.rglob('*') if p.is_file()}
        expected={'prospective/'+p for p in STAGE_FILES}|{'revised/'+p for p in REVISED_FILES}
        self.assertEqual(staged,expected)
        self.assertNotIn('revised/test_matrix_completion.py',staged)

    def test_reject_mutable_image(self):
        with self.assertRaisesRegex(ValueError,'pinned'):
            docker_command('global_mean',{},self.fit,self.q,stage_dir=self.stage,image='python:latest')

    def test_reject_secret_in_fit_mount(self):
        (self.fit/'test.csv').write_text('hidden labels')
        with self.assertRaisesRegex(ValueError,'exactly'):
            docker_command('global_mean',{},self.fit,self.q,stage_dir=self.stage,image='reviewed@sha256:'+ 'a'*64)

    def test_fail_closed_without_docker(self):
        with patch('docker_isolation_38.shutil.which',return_value=None):
            with self.assertRaisesRegex(RuntimeError,'refusing unisolated fallback'):
                execute('global_mean',{},self.fit,self.q,image='example@sha256:aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa')

if __name__=='__main__': unittest.main()
