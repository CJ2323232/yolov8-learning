import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
from PIL import Image

from helmet_detection import HelmetDetectionService, STANDARD, P2, NMS, WBF
from helmet_fusion import fuse_nms, calibrated_wbf
from helmet_inference import NORMAL_MODE, SLICED_MODE, sahi_to_result
from workflow import sha256

ROOT = Path(__file__).resolve().parents[1]


class FusionAppTests(unittest.TestCase):
    def setUp(self):
        work = ROOT / 'project_work/fusion_app_integration_2026-10-24'
        work.mkdir(parents=True, exist_ok=True)
        self.tmp = tempfile.TemporaryDirectory(dir=work)
        self.root = Path(self.tmp.name)
        (self.root/'models').mkdir()
        config = json.loads((ROOT/'models/fusion_config.json').read_text())
        for key in ['standard', 'p2']:
            p = self.root/f'{key}.pt';p.write_bytes(key.encode())
            config['models'][key] = {'path': p.name, 'sha256': sha256(p)}
        (self.root/'models/fusion_config.json').write_text(json.dumps(config))
        self.image = Image.new('RGB', (100, 80), (200, 10, 20))
        self.names = {0: 'hat', 1: 'person'}
        self.models = {}
        for key, score in [('standard', .8), ('p2', .6)]:
            box = SimpleNamespace(bbox=SimpleNamespace(to_xyxy=lambda: [10, 10, 30, 30]),
                                  score=SimpleNamespace(value=score), category=SimpleNamespace(id=0))
            model = SimpleNamespace(task='detect', names=self.names,
                                    predict=Mock(return_value=[sahi_to_result([box], self.image, self.names)]))
            self.models[key] = model
        self.loader = Mock(side_effect=lambda path: self.models[Path(path).stem])
        self.service = HelmetDetectionService(self.root, loader=self.loader)

    def tearDown(self): self.tmp.cleanup()

    def test_lazy_loading_reuse_and_original_pixels_unchanged(self):
        self.assertEqual(self.service.models, {})
        original = np.array(self.image)
        for choice in [NMS, WBF, P2, STANDARD]:
            result, counts, details = self.service.predict(self.image, choice)
            self.assertEqual(counts, {'hat': 1})
            self.assertEqual(result.size, self.image.size)
            self.assertTrue(details['calibrated_settings'])
        self.assertEqual(self.loader.call_count, 2)
        np.testing.assert_array_equal(np.array(self.image), original)

    def test_fusion_uses_low_candidate_floor_and_separate_output_threshold(self):
        for choice in [NMS, WBF]:
            self.service.predict(self.image, choice)
            for model in self.models.values():
                kw = model.predict.call_args.kwargs
                self.assertEqual(kw['conf'], .01)
                self.assertEqual(kw['imgsz'], 640)
                self.assertEqual(kw['iou'], .7)
                self.assertEqual(kw['max_det'], 300)
                self.assertFalse(kw['augment'])
        a = [[0, 0, 10, 10, .4, 0]]
        self.assertEqual(len(calibrated_wbf([a, []], 100, 80, output_conf=.17)), 1)
        self.assertEqual(len(calibrated_wbf([a, []], 100, 80, output_conf=.25)), 0)

    def test_hash_mismatch_rejected_before_model_load(self):
        (self.root/'p2.pt').write_bytes(b'wrong model')
        with self.assertRaisesRegex(ValueError, '不一致'):
            self.service.predict(self.image, P2)
        self.loader.assert_not_called()

    def test_invalid_combo_and_custom_threshold_flag(self):
        with self.assertRaises(ValueError): self.service.predict(self.image, WBF, mode=SLICED_MODE)
        with self.assertRaises(ValueError): self.service.predict(self.image, NMS, ordinary_imgsz=960)
        with self.assertRaises(ValueError): self.service.predict(self.image, WBF, conf=float('nan'))
        _, _, details = self.service.predict(self.image, NMS, conf=.4)
        self.assertFalse(details['calibrated_settings'])

    def test_class_aware_nms_and_empty_fusion(self):
        a = [[0, 0, 10, 10, .9, 0]]
        b = [[0, 0, 10, 10, .8, 0], [0, 0, 10, 10, .7, 1]]
        self.assertEqual(fuse_nms([a, b])[:, 5].tolist(), [0, 1])
        self.assertEqual(calibrated_wbf([[], []], 100, 80).shape, (0, 6))


if __name__ == '__main__': unittest.main()
