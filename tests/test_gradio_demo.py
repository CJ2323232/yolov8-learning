"""Demo contract checks with synthetic images and checkpoints; no model downloads."""
import csv
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import numpy as np
from PIL import Image
import torch
from ultralytics.engine.results import Results

from helmet_detection import (HelmetDetectionService, SINGLE_MODELS, MANIFEST_KEYS,
    STANDARD, P2, STANDARD_P3_CBAM, P2_P3_CBAM, NMS, WBF)
from helmet_inference import NORMAL_MODE, SLICED_MODE, HelmetPredictor, detection_records, render_result, sahi_to_result
from workflow import sha256
try:
    import app
except ModuleNotFoundError as exc:
    if exc.name != 'gradio':
        raise
    app = None

ROOT = Path(__file__).resolve().parents[1]


@unittest.skipIf(app is None, 'Install requirements-gradio.txt for demo tests')
class GradioDemoTests(unittest.TestCase):
    def setUp(self):
        directory = ROOT/'project_work/gradio_upgrade_2026-10-30/test_temp'
        directory.mkdir(parents=True, exist_ok=True)
        self.temp = tempfile.TemporaryDirectory(dir=directory)
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        (self.root/'models').mkdir()
        (self.root/'weights').mkdir()
        self.image = Image.new('RGB', (100, 80), (200, 10, 20))
        self.names = {0: 'hat', 1: 'person'}
        config = json.loads((ROOT/'models/fusion_config.json').read_text())
        manifest = []
        self.models = {}
        for key_id, key in MANIFEST_KEYS.items():
            file = self.root/'weights'/f'{key}.pt'
            file.write_bytes(key.encode())
            manifest.append({'id': key_id, 'source': file.relative_to(self.root).as_posix(), 'sha256': sha256(file)})
            if key in config['models']:
                config['models'][key] = {'path': str(file), 'sha256': sha256(file)}
            box = SimpleNamespace(bbox=SimpleNamespace(to_xyxy=lambda: [10, 12, 30, 35]),
                                  score=SimpleNamespace(value=.83), category=SimpleNamespace(id=0))
            result = sahi_to_result([box], self.image, self.names)
            self.models[key] = SimpleNamespace(task='detect', names=self.names, predict=Mock(return_value=[result]))
        (self.root/'models/fusion_config.json').write_text(json.dumps(config))
        (self.root/'models/release_manifest.json').write_text(json.dumps({'models': manifest}))
        self.loader = Mock(side_effect=lambda path: self.models[Path(path).stem])
        self.service = HelmetDetectionService(self.root, loader=self.loader)

    def test_five_model_choices_are_lazy_and_invalid_weights_are_excluded(self):
        self.assertEqual(len(self.service.available_presets()), 7)
        self.assertEqual(self.service.models, {})
        self.loader.assert_not_called()
        (self.root/'weights/standard_p3_cbam.pt').write_bytes(b'wrong')
        (self.root/'weights/p2_p3_cbam.pt').unlink()
        self.assertNotIn(STANDARD_P3_CBAM, self.service.available_presets())
        self.assertNotIn(P2_P3_CBAM, self.service.available_presets())
        states = str(self.service.model_status())
        self.assertIn('哈希不匹配', states)
        self.assertIn('缺少权重', states)

    def test_cbam_comparison_default_is_not_claimed_calibrated(self):
        _, counts, details = self.service.predict(self.image, STANDARD_P3_CBAM)
        self.assertEqual(counts, {'hat': 1})
        self.assertEqual(details['confidence'], .25)
        self.assertFalse(details['calibrated_settings'])
        self.assertEqual(details['models'][0]['key'], 'standard_p3_cbam')
        self.assertEqual(details['models'][0]['sha256'], sha256(self.root/'weights/standard_p3_cbam.pt'))

    def test_custom_nms_is_effective_and_cannot_leak_into_frozen_modes(self):
        _, _, details = self.service.predict(self.image, P2, nms_iou=.43)
        self.assertEqual(self.models['p2'].predict.call_args.kwargs['iou'], .43)
        self.assertEqual(details['nms_iou'], .43)
        self.assertFalse(details['calibrated_settings'])
        for preset, mode in [(P2, SLICED_MODE), (NMS, NORMAL_MODE), (WBF, NORMAL_MODE)]:
            with self.assertRaisesRegex(ValueError, '0.7'):
                self.service.predict(self.image, preset, mode=mode, nms_iou=.43)

    def test_sahi_restores_override_state_after_forcing_its_original_nms(self):
        model = SimpleNamespace(names=self.names, overrides={'iou': .43, 'max_det': 17})
        predictor = HelmetPredictor(model, 'cpu')
        seen = []
        def sliced(**kwargs):
            seen.append(dict(kwargs['detection_model'].model.overrides))
            return SimpleNamespace(object_prediction_list=[])
        with patch('sahi.predict.get_sliced_prediction', side_effect=sliced):
            predictor.predict(Image.new('RGB', (641, 640)), .25, SLICED_MODE)
        self.assertEqual(seen, [{'iou': .7, 'max_det': 300}])
        self.assertEqual(model.overrides, {'iou': .43, 'max_det': 17})

    def test_numpy_and_tensor_boxes_have_the_same_export_schema(self):
        boxes = [[10, 12, 30, 35, .83, 0]]
        for matrix in [np.asarray(boxes, dtype=np.float32), torch.tensor(boxes)]:
            result = Results(orig_img=np.asarray(self.image)[:, :, ::-1].copy(), path='fixture', names=self.names, boxes=matrix)
            record = detection_records(result)[0]
            self.assertEqual(record['xyxy'], [10., 12., 30., 35.])
            self.assertEqual((record['id'], record['class_id'], record['class_name']), (1, 0, 'hat'))
            self.assertAlmostEqual(record['confidence'], .83, places=6)
            before = matrix.copy() if isinstance(matrix, np.ndarray) else matrix.clone()
            image = render_result(result, show_labels=False, show_conf=False, show_box_ids=True)
            self.assertEqual(image.size, self.image.size)
            np.testing.assert_array_equal(result.boxes.data, before)

    def test_mode_controls_match_the_real_pipeline(self):
        fixed, note, iou = app.change_mode(SLICED_MODE, P2)
        self.assertEqual((fixed['value'], fixed['interactive'], iou['value'], iou['interactive']), (640, False, .7, False))
        self.assertIn('20%', note)
        self.assertIn('IOS=0.5', note)
        self.assertTrue(app.change_mode(NORMAL_MODE, P2)[0]['interactive'])
        self.assertFalse(app.change_mode(NORMAL_MODE, NMS)[0]['interactive'])
        for preset, threshold in [(STANDARD, .29), (P2, .25), (NMS, .3), (WBF, .17), (STANDARD_P3_CBAM, .25)]:
            self.assertEqual(app.change_preset(preset)[0]['value'], threshold)

    def test_csv_and_json_keep_original_coordinates_precision_and_parameters(self):
        _, _, details = self.service.predict(self.image, P2, nms_iou=.63)
        with patch.object(app, 'BASE_DIR', self.root):
            csv_path, json_path = app.write_exports(details)
        with Path(csv_path).open(encoding='utf-8-sig', newline='') as handle:
            record = next(csv.DictReader(handle))
        payload = json.loads(Path(json_path).read_text())
        self.assertEqual(payload['detections'], details['detections'])
        self.assertEqual(float(record['confidence']), details['detections'][0]['confidence'])
        self.assertEqual(payload['image'], {'width': 100, 'height': 80})
        self.assertEqual(payload['run']['nms_iou'], .63)
        self.assertNotIn(str(self.root), Path(json_path).read_text())

    def test_comparison_uses_one_original_and_shared_parameters_without_ground_truth(self):
        before = np.asarray(self.image).copy()
        with patch.object(app, 'BASE_DIR', self.root), patch.object(app, 'service', self.service), \
             patch.object(app, 'COMPARISON_CHOICES', list(SINGLE_MODELS)):
            output = app.compare(self.image, STANDARD, P2, .25, .63, False, False, True)
        self.assertEqual((output[0].size, output[1].size), (self.image.size, self.image.size))
        for key in ['standard', 'p2']:
            kwargs = self.models[key].predict.call_args.kwargs
            self.assertEqual((kwargs['conf'], kwargs['iou'], kwargs['imgsz']), (.25, .63, 640))
            np.testing.assert_array_equal(np.asarray(kwargs['source']), before)
        np.testing.assert_array_equal(np.asarray(self.image), before)
        payload = json.loads(Path(output[-1]).read_text())
        self.assertNotEqual(payload['A']['run']['models'], payload['B']['run']['models'])
        self.assertFalse(any(name in payload for name in ['accuracy', 'recall', 'f1', 'ground_truth']))
        self.assertNotIn(str(self.root), Path(output[-1]).read_text())

    def test_stale_registration_isolated_and_error_messages_hide_runtime_paths(self):
        with patch.object(app, 'BASE_DIR', self.root), patch.dict(app.os.environ, {'YOLO_MODEL_PATH': ''}):
            path, origin, warning = app.legacy_selection(Mock(side_effect=ValueError('/Users/private/old-model.pt')))
        self.assertIsNone(path)
        self.assertEqual(origin, 'missing')
        self.assertNotIn('/Users/', warning)
        self.assertIn('显存不足', app.error_message(RuntimeError('CUDA out of memory')))
        self.assertNotIn('/Users/', app.error_message(RuntimeError('/Users/private/trace')))

    def test_bad_and_oversized_images_fail_before_loading(self):
        with self.assertRaisesRegex(ValueError, '图片'):
            self.service.predict(None, P2)
        with patch('helmet_detection.MAX_INPUT_PIXELS', 1), self.assertRaisesRegex(ValueError, '4000万'):
            self.service.predict(self.image, P2)
        self.loader.assert_not_called()

    def test_sahi_candidate_limit_is_distinct_from_final_merged_output(self):
        _, _, single = self.service.predict(self.image, P2, mode=SLICED_MODE)
        self.assertEqual(single['per_model_max_det'], 300)
        self.assertEqual(single['final_max_det'], 300)
        with patch('sahi.predict.get_sliced_prediction', return_value=SimpleNamespace(object_prediction_list=[])):
            _, _, sliced = self.service.predict(Image.new('RGB', (641, 640)), P2, mode=SLICED_MODE)
        self.assertEqual(sliced['per_model_max_det'], 300)
        self.assertIsNone(sliced['final_max_det'])

    def test_all_prediction_events_share_a_serial_queue(self):
        events = [dep for dep in app.demo.config['dependencies'] if dep.get('api_name') in ['detect', 'compare']]
        self.assertEqual(len(events), 2)
        # Internal BlockFunction objects carry the scheduling group, not the public config.
        functions = [function for function in app.demo.fns.values() if function.fn in [app.detect, app.compare]]
        self.assertEqual({function.concurrency_id for function in functions}, {'helmet_inference'})
        self.assertTrue(all(function.concurrency_limit == 1 for function in functions))

    def test_uploads_and_clears_lock_until_request_finishes(self):
        locked = app.lock_controls()
        self.assertEqual(len(locked), len(app.RUN_CONTROLS))
        self.assertTrue(all(update['interactive'] is False for update in locked))
        restored = app.unlock_controls(P2, SLICED_MODE)
        self.assertTrue(restored[0]['interactive'])
        self.assertFalse(restored[4]['interactive'])
        self.assertFalse(restored[5]['interactive'])
        self.assertTrue(app.unlock_controls(P2, NORMAL_MODE)[4]['interactive'])
        failure_recovery = [event for event in app.demo.config['dependencies'] if event.get('trigger_only_on_failure')]
        self.assertEqual(len(failure_recovery), 2)


if __name__ == '__main__':
    unittest.main()
