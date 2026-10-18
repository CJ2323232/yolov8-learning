from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import numpy as np
from PIL import Image

from helmet_inference import HelmetPredictor, NORMAL_MODE, SLICED_MODE, count_slices, sahi_to_result


class InferenceTests(unittest.TestCase):
    def test_merged_boxes_keep_original_coordinates_classes_and_rgb(self):
        image = Image.new("RGB", (1000, 800), (220, 10, 30))
        box = SimpleNamespace(
            bbox=SimpleNamespace(to_xyxy=lambda: [700, 400, 740, 450]),
            score=SimpleNamespace(value=0.8), category=SimpleNamespace(id=1),
        )
        result = sahi_to_result([box], image, {0: "hat", 1: "person"})
        np.testing.assert_allclose(result.boxes.xyxy, [[700, 400, 740, 450]])
        self.assertEqual(result.boxes.cls.tolist(), [1])
        self.assertAlmostEqual(result.boxes.conf.item(), 0.8, places=6)
        self.assertEqual(result.orig_img[0, 0].tolist(), [30, 10, 220])

    def test_no_detections_still_returns_a_valid_original_size_result(self):
        result = sahi_to_result([], Image.new("RGB", (100, 80)), {0: "hat", 1: "person"})
        self.assertEqual(tuple(result.boxes.data.shape), (0, 6))
        self.assertEqual(result.plot().shape, (80, 100, 3))

    def test_normal_path_never_loads_sahi(self):
        image = Image.new("RGB", (40, 30), (200, 10, 20))
        result = sahi_to_result([], image, {0: "hat", 1: "person"})
        model = SimpleNamespace(names={0: "hat", 1: "person"}, predict=lambda **kw: [result])
        predictor = HelmetPredictor(model, "cpu")
        with patch.dict("sys.modules", {"sahi": None}):
            annotated, counts, _ = predictor.predict(image, 0.25, NORMAL_MODE)
            self.assertEqual(annotated.getpixel((0, 0)), (200, 10, 20))
            self.assertEqual(counts, {})
            self.assertIsNone(predictor._sahi_model)

    def test_slice_count_agrees_with_sahi_at_boundaries(self):
        from sahi.slicing import get_slice_bboxes
        for width, height in [(400, 300), (640, 640), (640, 300), (300, 640),
                              (641, 640), (640, 641), (1152, 1153), (1500, 900)]:
            actual = get_slice_bboxes(height, width, 640, 640, 0.2, 0.2)
            self.assertEqual(count_slices(width, height), len(actual))

    def test_single_tile_sliced_mode_preserves_normal_output_without_sahi(self):
        # These same-class boxes can survive YOLO NMS (IoU=0.49), but an
        # additional IOS-based merge would combine them (IOS=1.0).
        boxes = [
            SimpleNamespace(bbox=SimpleNamespace(to_xyxy=lambda: [10, 10, 30, 30]),
                            score=SimpleNamespace(value=0.9), category=SimpleNamespace(id=1)),
            SimpleNamespace(bbox=SimpleNamespace(to_xyxy=lambda: [11, 11, 25, 25]),
                            score=SimpleNamespace(value=0.8), category=SimpleNamespace(id=1)),
        ]
        for size in [(40, 30), (640, 640), (640, 300), (300, 640)]:
            for predictions in [boxes, []]:
                for conf in [0.25, 0.65]:
                    with self.subTest(size=size, count=len(predictions), conf=conf):
                        image = Image.new("RGB", size, (200, 10, 20))
                        names = {0: "hat", 1: "person"}
                        result = sahi_to_result(predictions, image, names)
                        model = SimpleNamespace(names=names, predict=Mock(return_value=[result]))
                        predictor = HelmetPredictor(model, "cpu")
                        with patch.dict("sys.modules", {"sahi": None}):
                            normal, normal_counts, _ = predictor.predict(image, conf, NORMAL_MODE)
                            sliced, counts, details = predictor.predict(image, conf, SLICED_MODE)
                        np.testing.assert_array_equal(np.asarray(normal), np.asarray(sliced))
                        self.assertEqual(counts, normal_counts)
                        self.assertEqual(counts, {"person": 2} if predictions else {})
                        self.assertEqual(details["tiles"], 1)
                        self.assertEqual(details["mode"], SLICED_MODE)
                        self.assertIsNone(predictor._sahi_model)
                        self.assertEqual(model.predict.call_count, 2)
                        for call in model.predict.call_args_list:
                            kwargs = dict(call.kwargs)
                            np.testing.assert_array_equal(np.asarray(kwargs.pop("source")), np.asarray(image))
                            self.assertEqual(kwargs, dict(conf=conf, imgsz=640, device="cpu",
                                                          iou=0.7, max_det=300, verbose=False))

    def test_single_tile_bypasses_existing_sahi_adapter(self):
        image = Image.new("RGB", (640, 640))
        names = {0: "hat", 1: "person"}
        result = sahi_to_result([], image, names)
        model = SimpleNamespace(names=names, predict=Mock(return_value=[result]))
        predictor = HelmetPredictor(model, "cpu")
        cached_adapter = SimpleNamespace(confidence_threshold=0.25)
        predictor._sahi_model = cached_adapter
        with patch("sahi.predict.get_sliced_prediction",
                   return_value=SimpleNamespace(object_prediction_list=[])) as sliced_prediction:
            _, counts, details = predictor.predict(image, 0.65, SLICED_MODE)
        sliced_prediction.assert_not_called()
        model.predict.assert_called_once()
        self.assertIs(predictor._sahi_model, cached_adapter)
        self.assertEqual(cached_adapter.confidence_threshold, 0.25)
        self.assertEqual(counts, {})
        self.assertEqual(details["tiles"], 1)

    def test_partial_same_class_boxes_merge_but_other_class_remains(self):
        from sahi.prediction import ObjectPrediction
        from sahi.postprocess.combine import GreedyNMMPostprocess
        items = [
            ObjectPrediction(bbox=[100, 100, 160, 160], category_id=0, score=0.9),
            ObjectPrediction(bbox=[100, 100, 130, 140], category_id=0, score=0.6),
            ObjectPrediction(bbox=[100, 100, 160, 160], category_id=1, score=0.8),
        ]
        merge = GreedyNMMPostprocess(match_threshold=0.5, match_metric="IOS", class_agnostic=False)
        result = merge(items)
        self.assertEqual(sorted(p.category.id for p in result), [0, 1])
        self.assertEqual(next(p for p in result if p.category.id == 0).bbox.to_xyxy(), [100, 100, 160, 160])

    def test_sliced_requests_reuse_weights_and_apply_current_confidence(self):
        model = SimpleNamespace(names={0: "hat", 1: "person"})
        predictor = HelmetPredictor(model, "cpu")
        seen = []
        def fake_prediction(**kw):
            adapter = kw["detection_model"]
            seen.append((adapter.model, adapter.confidence_threshold))
            self.assertEqual((kw["slice_height"], kw["slice_width"]), (640, 640))
            self.assertEqual((kw["overlap_height_ratio"], kw["overlap_width_ratio"]), (0.2, 0.2))
            self.assertTrue(kw["perform_standard_pred"])
            self.assertEqual(kw["postprocess_type"], "GREEDYNMM")
            self.assertEqual(kw["postprocess_match_metric"], "IOS")
            self.assertEqual(kw["postprocess_match_threshold"], 0.5)
            self.assertFalse(kw["postprocess_class_agnostic"])
            return SimpleNamespace(object_prediction_list=[])
        with patch("sahi.predict.get_sliced_prediction", side_effect=fake_prediction):
            for conf in [0.25, 0.65]:
                result, counts, details = predictor.predict(Image.new("RGB", (641, 640)), conf, SLICED_MODE)
                self.assertEqual(result.size, (641, 640))
                self.assertEqual(counts, {})
                self.assertEqual(details["tiles"], 2)
        self.assertEqual(seen, [(model, 0.25), (model, 0.65)])


if __name__ == "__main__":
    unittest.main()
