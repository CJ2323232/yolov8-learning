"""Deployment guardrails; no network, Docker, datasets or real weights required."""
import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from deployment.hf.start import arguments, launch_settings
from tools.prepare_space import FILES, RENAMES, prepare
from workflow import sha256

ROOT = Path(__file__).resolve().parents[1]


class SpaceDeploymentTests(unittest.TestCase):
    def setUp(self):
        base = ROOT/'project_work/online_deployment_2026-10-31/test_temp'
        base.mkdir(parents=True, exist_ok=True)
        self.temp = tempfile.TemporaryDirectory(dir=base)
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        for name in (*FILES, *RENAMES):
            file = self.root/name
            file.parent.mkdir(parents=True, exist_ok=True)
            file.write_text('synthetic fixture\n')
        self.weight = self.root/'run/reviewed/weights/best.pt'
        self.weight.parent.mkdir(parents=True)
        self.weight.write_bytes(b'synthetic checkpoint, never unpickled')
        (self.root/'models/release_manifest.json').write_text(json.dumps({'models':[
            {'id': 'fixture', 'source': 'run/reviewed/weights/best.pt', 'sha256': sha256(self.weight)}]}))
        self.output = self.root/'project_work/package'

    def test_allowlist_excludes_data_secrets_weights_and_logs_by_default(self):
        for name in ['.env', 'datasets/private.jpg', 'runs/latest_helmet_training.json']:
            p = self.root/name; p.parent.mkdir(parents=True, exist_ok=True); p.write_text('private')
        result = prepare(self.output, root=self.root)
        self.assertFalse(result['weights_included'])
        for name in ['.env', 'datasets', 'runs', 'run']:
            self.assertFalse((self.output/name).exists())
        for entry in result['files']:
            self.assertEqual(sha256(self.output/entry['file']), entry['sha256'])

    def test_existing_or_non_staging_output_is_rejected(self):
        self.output.mkdir(parents=True)
        marker = self.output/'keep.txt'; marker.write_text('keep')
        with self.assertRaises(FileExistsError): prepare(self.output, root=self.root)
        with self.assertRaises(ValueError): prepare(self.root/'run/output', root=self.root)
        self.assertEqual(marker.read_text(), 'keep')

    def test_checked_weights_copied_without_changes(self):
        result = prepare(self.output, with_weights=True, root=self.root)
        self.assertTrue(result['weights_included'])
        self.assertEqual((self.output/'run/reviewed/weights/best.pt').read_bytes(), self.weight.read_bytes())

    def test_corrupt_weights_rejected_before_output_created(self):
        self.weight.write_bytes(b'changed')
        with self.assertRaises(ValueError): prepare(self.output, with_weights=True, root=self.root)
        self.assertFalse(self.output.exists())

    def test_launch_blocks_model_downloads_and_defaults_to_no_tunnel(self):
        settings = launch_settings(arguments(['--host', '0.0.0.0', '--port', '7881']))
        self.assertFalse(settings['share'])
        self.assertFalse(settings['inbrowser'])
        self.assertEqual(settings['server_port'], 7881)
        self.assertIn(str(ROOT/'run'), settings['blocked_paths'])
        self.assertIn(str(ROOT/'models'), settings['blocked_paths'])

    def test_public_share_requires_explicit_flag(self):
        settings = launch_settings(arguments(['--share']))
        self.assertTrue(settings['share'])
        self.assertEqual(settings['server_name'], '127.0.0.1')
        self.assertIn(str(ROOT/'datasets'), settings['blocked_paths'])

    def test_invalid_or_environment_port(self):
        with patch.dict('os.environ', {'PORT': '7883'}):
            self.assertEqual(arguments([]).port, 7883)
        for port in ['0', '65536', 'not-a-port']:
            with self.subTest(port=port), contextlib.redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit): arguments(['--port', port])
