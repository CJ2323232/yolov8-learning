"""Exercise delivery integrity and overwrite guards with tiny synthetic archives."""
import importlib.util
import json
from pathlib import Path
import pickle
import tempfile
import unittest
import zipfile

SPEC = importlib.util.spec_from_file_location('prepare_model_delivery',
    Path(__file__).resolve().parents[1] / 'tools/prepare_model_delivery.py')
delivery = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(delivery)


class ModelDeliveryTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name).resolve()
        models = self.root / 'models'
        models.mkdir()
        self.manifest_path = models / 'release_manifest.json'
        (models / 'MODEL_CARD.md').write_text('Synthetic test card, not a real checkpoint.\n')
        entries = []
        for index, key in enumerate(sorted(delivery.EXPECTED_IDS)):
            weight = self.root / 'run' / key / 'weights/best.pt'
            weight.parent.mkdir(parents=True)
            with zipfile.ZipFile(weight, 'w') as archive:
                archive.writestr('test/data.pkl', pickle.dumps({'note': key, 'path': '/Users/example/train'}))
            config = models / (key + '.yaml')
            config.write_text('nc: 2\n')
            entries.append({'id': key, 'source': weight.relative_to(self.root).as_posix(),
                'asset_filename': f'model_{index}.pt', 'bytes': weight.stat().st_size,
                'sha256': delivery.digest(weight), 'configuration': config.relative_to(self.root).as_posix(),
                'configuration_sha256': delivery.digest(config), 'download_url': None})
        self.manifest = {'status': 'staged_not_published', 'published': False,
            'release_url': None, 'release_tag': None, 'models': entries,
            'release_gates': {'rights': 'pending'}}
        self.manifest_path.write_text(json.dumps(self.manifest))

    def test_check_is_readonly_and_metadata_is_not_unpickled(self):
        before = {p.relative_to(self.root): p.read_bytes() for p in self.root.rglob('*') if p.is_file()}
        report = delivery.verify(self.root, self.manifest)
        self.assertEqual(len(report['models']), 5)
        self.assertFalse(report['model_execution'])
        self.assertTrue(all(x['checkpoint_metadata']['personal_absolute_path_string_count'] == 1
                            for x in report['models']))
        self.assertEqual(before, {p.relative_to(self.root): p.read_bytes()
                                 for p in self.root.rglob('*') if p.is_file()})

    def test_hash_tampering_and_missing_configuration_are_rejected(self):
        self.manifest['models'][0]['sha256'] = '0' * 64
        with self.assertRaisesRegex(delivery.DeliveryError, 'SHA-256 mismatch'):
            delivery.verify(self.root, self.manifest)
        self.manifest = json.loads(self.manifest_path.read_text())
        (self.root / self.manifest['models'][0]['configuration']).unlink()
        with self.assertRaisesRegex(delivery.DeliveryError, 'missing'):
            delivery.verify(self.root, self.manifest)

    def test_nonportable_paths_and_duplicate_assets_are_rejected(self):
        for value in ('../escape.pt', '/tmp/escape.pt', 'C:\\Users\\example\\best.pt'):
            with self.assertRaises(delivery.DeliveryError):
                delivery.source_path(self.root, value)
        self.manifest['models'][1]['asset_filename'] = self.manifest['models'][0]['asset_filename']
        with self.assertRaisesRegex(delivery.DeliveryError, 'Asset names'):
            delivery.verify(self.root, self.manifest)

    def test_refuses_published_claim_and_download_url(self):
        self.manifest['release_url'] = 'https://example.invalid/not-a-real-release'
        with self.assertRaisesRegex(delivery.DeliveryError, 'published release'):
            delivery.verify(self.root, self.manifest)
        self.manifest['release_url'] = None
        self.manifest['models'][0]['download_url'] = 'https://example.invalid/not-a-real-model'
        with self.assertRaisesRegex(delivery.DeliveryError, 'download URLs'):
            delivery.verify(self.root, self.manifest)

    def test_existing_or_original_run_output_cannot_be_overwritten(self):
        for output in (self.root / 'models', self.root / 'run/new_delivery'):
            with self.assertRaises(delivery.DeliveryError):
                delivery.prepare(self.root, self.manifest_path, self.manifest, output)

    def test_prepare_has_checksums_and_independent_copies(self):
        output = self.root / 'delivery'
        report = delivery.prepare(self.root, self.manifest_path, self.manifest, output)
        self.assertTrue(report['prepared'])
        self.assertFalse(report['publication'])
        self.assertFalse((output / '.INCOMPLETE').exists())
        for line in (output / 'SHA256SUMS').read_text().splitlines():
            digest, name = line.split('  ', 1)
            self.assertEqual(delivery.digest(output / name), digest)
        original = self.root / self.manifest['models'][0]['source']
        original_digest = delivery.digest(original)
        (output / self.manifest['models'][0]['asset_filename']).write_bytes(b'staged edit')
        self.assertEqual(delivery.digest(original), original_digest)


if __name__ == '__main__':
    unittest.main()
