"""Portable path contract and output tests; no dataset downloads or real weights."""
import csv
from contextlib import contextmanager
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from PIL import Image
import yaml

import cli
from workflow import sha256


@contextmanager
def working_directory(path):
    import os
    prior = Path.cwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(prior)


class PortableCLITests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name).resolve()
        self.dataset = self.root / 'dataset'
        rows = []
        for i, split in enumerate(('train', 'val', 'test')):
            image = self.dataset / 'images' / split / (split + '.png')
            label = self.dataset / 'labels' / split / (split + '.txt')
            image.parent.mkdir(parents=True)
            label.parent.mkdir(parents=True)
            Image.new('RGB', (20, 20), (30 * i, 10, 5)).save(image)
            label.write_text('0 0.5 0.5 0.2 0.2\n')
            rows.append({'split': split, 'image': image.relative_to(self.dataset).as_posix(),
                         'label': label.relative_to(self.dataset).as_posix(),
                         'image_sha256': sha256(image), 'label_sha256': sha256(label), 'group_id': split})
        manifest = self.dataset / 'split_manifest.csv'
        with manifest.open('w', newline='') as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        (self.dataset / 'dataset_audit.json').write_text(json.dumps({
            'manifest_sha256': sha256(manifest), 'cross_split_exact_duplicates': 0,
            'difficult_policy': 'include_as_ordinary', 'grouping': 'hash_only'}))
        self.config = self.root / 'data.yaml'
        self.config.write_text(yaml.safe_dump({'path': 'dataset', 'train': 'images/train',
            'val': 'images/val', 'test': 'images/test', 'names': {0: 'hat', 1: 'person'},
            'download': 'must never execute'}))

    def test_yaml_root_is_independent_of_cwd_and_source_remains_unchanged(self):
        before = self.config.read_bytes()
        elsewhere = self.root / 'elsewhere'
        elsewhere.mkdir()
        with working_directory(elsewhere):
            with cli.verified_data(self.config) as (runtime, report, cfg):
                self.assertEqual(cfg['path'], str(self.dataset.resolve()))
                self.assertEqual(report['dataset']['images'], 3)
                self.assertNotIn('download', cfg)
                self.assertEqual(yaml.safe_load(runtime.read_text())['path'], str(self.dataset))
            self.assertFalse(runtime.exists())
        self.assertEqual(self.config.read_bytes(), before)

    def test_explicit_root_is_relative_to_cwd_and_relocates_foreign_yaml(self):
        cfg = yaml.safe_load(self.config.read_text())
        cfg['path'] = 'C:\\old-computer\\dataset'
        self.config.write_text(yaml.safe_dump(cfg))
        with self.assertRaisesRegex(ValueError, 'foreign Windows path'):
            cli.canonical_data(self.config)
        with working_directory(self.root):
            _, resolved, _ = cli.canonical_data(self.config, Path('dataset'))
        self.assertEqual(resolved, self.dataset)

    def test_missing_split_and_wrong_classes_are_rejected(self):
        cfg = yaml.safe_load(self.config.read_text())
        cfg['names'] = {0: 'person', 1: 'hat'}
        self.config.write_text(yaml.safe_dump(cfg))
        with self.assertRaisesRegex(ValueError, 'class mapping'):
            cli.canonical_data(self.config)
        cfg['names'] = {0: 'hat', 1: 'person'}
        cfg['test'] = 'other/images'
        self.config.write_text(yaml.safe_dump(cfg))
        with self.assertRaisesRegex(ValueError, 'test must be'):
            cli.canonical_data(self.config)

    def test_unverified_dataset_needs_explicit_diagnostic_flag(self):
        (self.dataset / 'dataset_audit.json').unlink()
        with self.assertRaisesRegex(ValueError, '审计清单'):
            with cli.verified_data(self.config):
                pass
        with cli.verified_data(self.config, allow_legacy=True) as (_, report, _):
            self.assertIn('legacy_diagnostic', report['dataset']['status'])

    def test_save_config_refuses_to_replace_original(self):
        with self.assertRaises(SystemExit) as error:
            cli.main(['check-data', '--data', str(self.config), '--write-config', str(self.config)])
        self.assertEqual(error.exception.code, 2)

    def prediction_args(self, **options):
        weights = self.root / 'model.pt'
        weights.write_bytes(b'test fixture; never deserialize')
        args = cli.parser().parse_args(['predict', '--weights', str(weights),
            '--source', str(self.dataset / 'images'), '--output', str(self.root / 'prediction')])
        for name, value in options.items():
            setattr(args, name, value)
        return args

    def test_missing_weights_hash_and_output_errors_precede_model_load(self):
        args = self.prediction_args()
        loader = unittest.mock.Mock()
        args.weights.unlink()
        with self.assertRaisesRegex(FileNotFoundError, 'checkpoint not found'):
            cli.prediction(args, loader)
        args.weights.write_bytes(b'fixture')
        args.sha256 = '0' * 64
        with self.assertRaisesRegex(ValueError, 'SHA256 mismatch'):
            cli.prediction(args, loader)
        args.sha256 = None
        args.output.mkdir()
        with self.assertRaises(FileExistsError):
            cli.prediction(args, loader)
        loader.assert_not_called()

    def test_dry_run_never_loads_model_or_writes_prediction_output(self):
        args = self.prediction_args(dry_run=True)
        loader = unittest.mock.Mock()
        plan = cli.prediction(args, loader)
        self.assertEqual(plan['images'], 3)
        self.assertFalse(args.output.exists())
        loader.assert_not_called()

    def test_prediction_forwards_thresholds_loads_once_and_writes_box_records(self):
        import numpy as np
        class Tensor:
            def __init__(self, values): self.values = values
            def cpu(self): return self
            def tolist(self): return self.values
        result = SimpleNamespace(names={0: 'hat', 1: 'person'}, orig_shape=(20, 20),
            boxes=SimpleNamespace(cls=Tensor([1]), conf=Tensor([0.7]), xyxy=Tensor([[1, 2, 10, 12]])),
            plot=lambda: np.zeros((20, 20, 3), dtype=np.uint8))
        model = unittest.mock.Mock()
        model.predict.return_value = [result]
        loader = unittest.mock.Mock(return_value=model)
        args = self.prediction_args(conf=0.4, iou=0.6, device='cpu')
        cli.prediction(args, loader)
        loader.assert_called_once()
        self.assertEqual(model.predict.call_count, 3)
        for call in model.predict.call_args_list:
            self.assertEqual(call.kwargs['conf'], 0.4)
            self.assertEqual(call.kwargs['iou'], 0.6)
        records = json.loads((args.output / 'detections.json').read_text())
        self.assertEqual(records[0]['detections'][0], {'class_id': 1, 'class_name': 'person',
            'confidence': 0.7, 'xyxy': [1.0, 2.0, 10.0, 12.0]})
        self.assertTrue(all((args.output / record['annotated_image']).is_file() for record in records))

    def test_val_delegates_canonical_yaml_without_training(self):
        args = self.prediction_args()
        def inspect(options):
            self.assertEqual(options[0], 'val')
            data = Path(options[options.index('--data') + 1])
            self.assertEqual(yaml.safe_load(data.read_text())['path'], str(self.dataset))
            self.assertIn('--dry-run', options)
            return {'preflight': True}
        with patch('run.main', side_effect=inspect) as runner:
            cli.main(['val', '--weights', str(args.weights), '--data', str(self.config),
                      '--output', str(args.output), '--dry-run'])
        runner.assert_called_once()
        self.assertFalse(args.output.exists())


if __name__ == '__main__':
    unittest.main()
