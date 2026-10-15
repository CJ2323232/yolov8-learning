"""Single-image inference using the same YOLO instance in both modes."""
import math
from threading import Lock
from time import perf_counter

import cv2
import numpy as np
from PIL import Image, ImageOps
from ultralytics.engine.results import Results

NORMAL_MODE = "普通检测"
SLICED_MODE = "SAHI 切片检测"
SLICE_SIZE = 640
OVERLAP = 0.2
MAX_SLICES = 256


def count_slices(width, height):
    step = SLICE_SIZE - int(SLICE_SIZE * OVERLAP)
    return (1 + math.ceil(max(0, width - SLICE_SIZE) / step)) * (
        1 + math.ceil(max(0, height - SLICE_SIZE) / step)
    )


def sahi_to_result(predictions, image, names):
    """SAHI returns merged full-image xyxy boxes; retain scores and classes."""
    boxes = np.asarray([
        [*p.bbox.to_xyxy(), p.score.value, p.category.id]
        for p in predictions
    ], dtype=np.float32).reshape(-1, 6)
    return Results(
        orig_img=np.asarray(image)[:, :, ::-1].copy(),
        path="uploaded_image", names=names, boxes=boxes,
    )


class HelmetPredictor:
    def __init__(self, model, device):
        self.model = model
        self.device = "cuda:" + device if str(device).isdigit() else str(device)
        self._sahi_model = None
        self._lock = Lock()

    def predict(self, image, conf, mode=NORMAL_MODE):
        if mode not in (NORMAL_MODE, SLICED_MODE):
            raise ValueError("请选择普通检测或 SAHI 切片检测。")
        if not math.isfinite(conf) or not 0 <= conf <= 1:
            raise ValueError("置信度必须在 0 到 1 之间。")
        # Both inference paths receive the same oriented RGB pixels.
        image = ImageOps.exif_transpose(image).convert("RGB")
        tiles = count_slices(*image.size) if mode == SLICED_MODE else 1
        if tiles > MAX_SLICES:
            raise ValueError("图片需要超过 256 个切片，请先缩小图片或选择普通检测。")

        # YOLO's predictor and SAHI's confidence/output fields are mutable.
        with self._lock:
            start = perf_counter()
            if mode == NORMAL_MODE:
                result = self.model.predict(
                    source=image, conf=conf, imgsz=640,
                    device=self.device, iou=0.7, max_det=300, verbose=False,
                )[0]
            else:
                try:
                    from sahi.models.ultralytics import UltralyticsDetectionModel
                    from sahi.predict import get_sliced_prediction
                except ImportError as exc:
                    raise ImportError(
                        "切片检测需要 SAHI，请在项目环境安装 requirements-gradio.txt。"
                    ) from exc
                if self._sahi_model is None:
                    self._sahi_model = UltralyticsDetectionModel(
                        model=self.model, device=self.device,
                        confidence_threshold=conf, image_size=640,
                    )
                self._sahi_model.confidence_threshold = conf
                prediction = get_sliced_prediction(
                    image=image, detection_model=self._sahi_model,
                    slice_height=SLICE_SIZE, slice_width=SLICE_SIZE,
                    overlap_height_ratio=OVERLAP, overlap_width_ratio=OVERLAP,
                    perform_standard_pred=True,
                    postprocess_type="GREEDYNMM", postprocess_match_metric="IOS",
                    postprocess_match_threshold=0.5,
                    postprocess_class_agnostic=False, verbose=0,
                )
                result = sahi_to_result(
                    prediction.object_prediction_list, image, self.model.names,
                )

            annotated = Image.fromarray(cv2.cvtColor(result.plot(), cv2.COLOR_BGR2RGB))
            counts = {}
            for cls in result.boxes.cls.tolist():
                name = self.model.names[int(cls)]
                counts[name] = counts.get(name, 0) + 1
            return annotated, counts, {
                "mode": mode, "tiles": tiles,
                "seconds": perf_counter() - start,
            }
