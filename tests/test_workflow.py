import csv
import importlib
import json
from pathlib import Path
import shutil
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from PIL import Image

from split_dataset import prepare
from voc_to_yolo import convert_annotation
from workflow import latest_training, read_labels, register_training, select_model, sha256, verify_dataset


class WorkflowTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.source = self.root/'VOC'
        for kind in ('JPEGImages', 'Annotations', 'labels'):
            (self.source/kind).mkdir(parents=True)
        for i in range(30):
            self.sample(f'{i:02d}', (i*7, i*3, i), i % 2)
    def tearDown(self):
        self.temp.cleanup()

    def sample(self, name, color, cls=0, box=(2, 2, 10, 10), copy_image=None):
        image = self.source/'JPEGImages'/f'{name}.png'
        if copy_image:
            shutil.copyfile(self.source/'JPEGImages'/f'{copy_image}.png', image)
        else:
            Image.new('RGB', (20, 20), color).save(image)
        x1, y1, x2, y2 = box
        xml = self.source/'Annotations'/f'{name}.xml'
        xml.write_text(f'<annotation><size><width>20</width><height>20</height></size><object>'
                       f'<name>{"hat" if cls == 0 else "person"}</name><difficult>1</difficult><truncated>0</truncated>'
                       f'<bndbox><xmin>{x1}</xmin><ymin>{y1}</ymin><xmax>{x2}</xmax><ymax>{y2}</ymax>'
                       '</bndbox></object></annotation>')
        lines, _ = convert_annotation(xml)
        (self.source/'labels'/f'{name}.txt').write_text('\n'.join(lines))

    def test_conflicts_quarantined_identical_deduplicated_and_replay_stable(self):
        self.sample('conflict', None, copy_image='00', box=(2, 2, 12, 15))
        self.sample('duplicate', None, cls=1, copy_image='01')
        a = prepare(self.source, self.root/'a')
        b = prepare(self.source, self.root/'b')
        self.assertEqual(a['quarantined_images'], 2)
        self.assertEqual(a['accepted_images'], 29)
        self.assertEqual(len(a['identical_annotation_duplicate_groups']), 1)
        self.assertEqual(a['manifest_sha256'], b['manifest_sha256'])
        self.assertEqual(verify_dataset(self.root/'a/data.yaml')['images'], 29)
        self.assertTrue((self.root/'a/quarantine/images/00.png').exists())
        with (self.root/'a/split_manifest.csv').open() as f:
            rows = list(csv.DictReader(f))
        hashes = {s: {r['image_sha256'] for r in rows if r['split'] == s} for s in ('train','val','test')}
        self.assertFalse(hashes['train'] & hashes['test'])
        self.assertFalse(hashes['train'] & hashes['val'])

    def test_changed_label_or_extra_image_cannot_reuse_audit(self):
        prepare(self.source, self.root/'a')
        label = next((self.root/'a/labels/train').glob('*.txt'))
        label.write_text('0 0.5 0.5 0.2 0.2')
        with self.assertRaisesRegex(ValueError, '变化'):
            verify_dataset(self.root/'a/data.yaml')

    def test_scene_group_is_never_split(self):
        group_file = self.root/'groups.csv'
        group_file.write_text('image,group_id\n'+'\n'.join(f'{i:02d}.png,scene_{i//3}' for i in range(30)))
        prepare(self.source, self.root/'a', groups_file=group_file)
        verify_dataset(self.root/'a/data.yaml')
        with (self.root/'a/split_manifest.csv').open() as f:
            rows = list(csv.DictReader(f))
        group_splits = {}
        for row in rows:
            self.assertEqual(group_splits.setdefault(row['group_id'], row['split']), row['split'])

    def test_invalid_edges_nonfinite_and_duplicate_rows_rejected(self):
        label = self.root/'invalid.txt'
        for text in ('0 0.95 0.5 0.4 0.1', '0 nan 0.5 0.2 0.2',
                     '0 0.5 0.5 0.2 0.2\n0 0.5 0.5 0.2 0.2'):
            label.write_text(text)
            with self.assertRaises(ValueError):
                read_labels(label)

    def test_artifact_binds_incremented_run_and_detects_model_tampering(self):
        prepare(self.source, self.root/'a')
        identity = verify_dataset(self.root/'a/data.yaml')
        best = self.root/'runs/helmet_clean2/weights/best.pt'
        best.parent.mkdir(parents=True)
        best.write_bytes(b'new-trained-model')
        ledger = self.root/'runs/latest.json'
        register_training(best, self.root/'a/data.yaml', identity, {}, {}, ledger)
        model, origin = select_model(require_training=True, record_path=ledger)
        self.assertEqual(model, best.resolve())
        self.assertEqual(origin, 'registered_training')
        best.write_bytes(b'different-model')
        with self.assertRaisesRegex(ValueError, '变化'):
            latest_training(ledger)

    def test_no_training_reference_never_silently_evaluates_legacy_model(self):
        with self.assertRaises(FileNotFoundError):
            select_model(require_training=True, record_path=self.root/'missing.json')

    def test_unverified_legacy_data_rejected_by_default(self):
        import yaml
        path = self.root/'legacy.yaml'
        path.write_text(yaml.safe_dump({'path': str(self.root), 'names': {0:'hat',1:'person'}}))
        with self.assertRaises(ValueError):
            verify_dataset(path)
        self.assertIn('legacy', verify_dataset(path, allow_legacy=True)['status'])

    def test_swapped_class_mapping_rejected(self):
        import yaml
        prepare(self.source, self.root/'a')
        config = self.root/'a/data.yaml'
        data = yaml.safe_load(config.read_text())
        data['names'] = {0:'person', 1:'hat'}
        config.write_text(yaml.safe_dump(data))
        with self.assertRaisesRegex(ValueError, '类别映射'):
            verify_dataset(config)

    def test_legacy_model_cannot_establish_new_split_baseline(self):
        import run
        prepare(self.source, self.root/'a')
        model = self.root/'legacy.pt'; model.write_bytes(b'legacy')
        with patch.object(run, 'RELEASE_SHA256', sha256(model)):
            with self.assertRaises(SystemExit):
                run.main(['val', '--model', str(model), '--data', str(self.root/'a/data.yaml')])

    def test_prediction_batch_is_effective_and_fingerprinted(self):
        import run
        checkpoint = self.root/'custom.pt'
        checkpoint.write_bytes(b'checkpoint')
        out = self.root/'prediction'; out.mkdir()
        calls = {}
        class FakeYOLO:
            ckpt_path = str(checkpoint)
            def __init__(self, model): pass
            def predict(self, **kwargs):
                calls.update(kwargs)
                self.predictor = SimpleNamespace(save_dir=out, args=SimpleNamespace(**kwargs))
                return iter([None])
        torch = SimpleNamespace(__version__='test', cuda=SimpleNamespace(is_available=lambda:False),
                                backends=SimpleNamespace(mps=SimpleNamespace(is_available=lambda:False)))
        with patch.dict(sys.modules, {'torch':torch, 'ultralytics':SimpleNamespace(__version__='test', YOLO=FakeYOLO)}):
            run.main(['predict','--model',str(checkpoint),'--source','images','--batch','7'])
        self.assertEqual(calls['batch'], 7)
        self.assertEqual(json.loads((out/'input_identity.json').read_text())['model_sha256'], sha256(checkpoint))

    def test_train_module_import_does_not_start_training(self):
        with patch.dict(sys.modules, {'ultralytics':None}):
            import train_helmet
            importlib.reload(train_helmet)

    def test_difficult_metadata_preserved_and_unknown_class_fails(self):
        xml = self.source/'Annotations/00.xml'
        _, metadata = convert_annotation(xml)
        self.assertEqual(metadata[0]['difficult'], 1)
        xml.write_text(xml.read_text().replace('<name>hat</name>', '<name>unexpected</name>'))
        with self.assertRaisesRegex(ValueError, '未知类别'):
            convert_annotation(xml)

    def test_jpeg_eoi_repaired_in_copy_before_fingerprinting(self):
        path = self.source/'JPEGImages/00.png'
        jpg = path.with_suffix('.jpg')
        with Image.open(path) as im:
            im.save(jpg)
        path.unlink()
        original = jpg.read_bytes()[:-2]
        jpg.write_bytes(original)
        audit = prepare(self.source, self.root/'a')
        self.assertIn('00.jpg', audit['source_jpeg_eoi_repairs'])
        self.assertEqual(jpg.read_bytes(), original)
        verify_dataset(self.root/'a/data.yaml')
        copied = next((self.root/'a/images').rglob('00.jpg'))
        self.assertTrue(copied.read_bytes().endswith(b'\xff\xd9'))


if __name__ == '__main__':
    unittest.main()
