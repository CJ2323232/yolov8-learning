"""Read-only requirements profile checks: no torch import, downloads or data."""
import json
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[1]


def requirements(relative):
    pins = {}
    for line in (ROOT / relative).read_text(encoding='utf-8-sig').splitlines():
        line = line.strip()
        if line and not line.startswith(('#', '-')):
            name, version = line.split('==', 1)
            pins[name.lower()] = version
    return pins


class EnvironmentRequirementsTests(unittest.TestCase):
    def test_framework_versions_stay_fixed(self):
        app = requirements('requirements.txt')
        train = requirements('requirements-training-windows.txt')
        self.assertEqual(app['ultralytics'], train['ultralytics'])
        self.assertEqual(app['ultralytics'], '8.4.164')
        self.assertEqual(app['torch'], '2.14.0')
        self.assertEqual(app['torchvision'], '0.29.0')
        self.assertEqual(train['torch'], '2.14.0+cu126')
        self.assertEqual(train['torchvision'], '0.29.0+cu126')

    def test_numpy_profiles_do_not_mix(self):
        self.assertEqual(requirements('requirements.txt')['numpy'], '2.5.3')
        self.assertEqual(requirements('requirements-training-windows.txt')['numpy'], '2.2.6')

    def test_training_pins_match_saved_actual_environments(self):
        expected = requirements('requirements-training-windows.txt')
        for folder in ['02_standard_P3_CBAM', '04_P2_P3_CBAM', '05_P2_P2_CBAM']:
            with self.subTest(model=folder):
                data = json.loads((ROOT / 'reports/final_evaluation/training' / folder / 'environment.json').read_text())
                self.assertEqual(data['python'], '3.10.10')
                installed = {k.lower(): v for k, v in data['packages'].items()}
                for package, version in expected.items():
                    self.assertEqual(installed[package], version)

    def test_demo_extras_reuse_application_profile(self):
        text = (ROOT / 'requirements-gradio.txt').read_text()
        self.assertIn('-r requirements.txt', text.splitlines())
        self.assertEqual(requirements('requirements-gradio.txt'), {'gradio': '6.30.0', 'sahi': '0.12.8'})

    def test_historical_migration_snapshot_is_identified(self):
        text = (ROOT / 'tools/windows_migration/requirements-windows.txt').read_text()
        self.assertIn('Historical migration-package snapshot', text)
        self.assertIn('requirements-training-windows.txt', text)
        self.assertEqual(requirements('tools/windows_migration/requirements-windows.txt')['numpy'], '2.5.3')

    def test_profiles_declare_their_python_scope(self):
        self.assertIn('Python >=3.12', (ROOT / 'requirements.txt').read_text())
        self.assertIn('Python3.10.10', (ROOT / 'requirements-training-windows.txt').read_text())


if __name__ == '__main__':
    unittest.main()
