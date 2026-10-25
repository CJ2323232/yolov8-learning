import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from workflow import ROOT

spec=importlib.util.spec_from_file_location('cbam_update',ROOT/'tools/windows_cbam/install_update.py')
up=importlib.util.module_from_spec(spec)
spec.loader.exec_module(up)


class UpdateTests(unittest.TestCase):
    def setUp(self):
        (ROOT/'project_work/p2_cbam_training_2026-10-25').mkdir(parents=True,exist_ok=True)
        self.tmp=tempfile.TemporaryDirectory(dir=ROOT/'project_work/p2_cbam_training_2026-10-25')
        self.root=Path(self.tmp.name)/'old project 中文'
        self.bundle=Path(self.tmp.name)/'update'
        self.data=self.root/'datasets/helmet_scene_isolated_v1/split_manifest.csv'
        self.data.parent.mkdir(parents=True)
        self.data.write_text('original data identity')
        (self.root/'train_helmet.py').write_text('old code')
        (self.bundle/'payload').mkdir(parents=True)
        (self.bundle/'payload/train_helmet.py').write_text('new code')
        self.manifest={'dataset_manifest_sha256':up.digest(self.data),
                       'files':{'train_helmet.py':up.digest(self.bundle/'payload/train_helmet.py')}}
        self.save_manifest()

    def save_manifest(self):
        (self.bundle/'update_manifest.json').write_text(json.dumps(self.manifest))

    def tearDown(self): self.tmp.cleanup()

    def test_check_writes_nothing_and_install_backs_up_then_is_idempotent(self):
        self.assertEqual(up.install(self.bundle,self.root,True)['status'],'check_only')
        self.assertEqual((self.root/'train_helmet.py').read_text(),'old code')
        report=up.install(self.bundle,self.root)
        self.assertEqual((Path(report['backup'])/'before/train_helmet.py').read_text(),'old code')
        self.assertEqual((self.root/'train_helmet.py').read_text(),'new code')
        self.assertEqual(up.install(self.bundle,self.root)['status'],'already_installed')
        self.assertEqual(self.data.read_text(),'original data identity')

    def test_corrupt_payload_stops_before_overwriting(self):
        (self.bundle/'payload/train_helmet.py').write_text('corrupt')
        with self.assertRaises(ValueError): up.install(self.bundle,self.root)
        self.assertEqual((self.root/'train_helmet.py').read_text(),'old code')

    def test_mismatched_dataset_rejected(self):
        self.data.write_text('different data')
        with self.assertRaises(ValueError): up.install(self.bundle,self.root)

    def test_paths_cannot_touch_data_weights_or_outside_project(self):
        for path in ('../outside.py','/tmp/out.py','runs/a.json','datasets/a.yaml','model.pt','C:/x','x\\y'):
            with self.subTest(path=path), self.assertRaises(ValueError):
                up.safe_relative(self.root,path)


if __name__=='__main__': unittest.main()
