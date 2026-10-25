"""Lazy, identity-checked single-model and ensemble inference for the local UI."""
import json
import math
import os
from pathlib import Path
from threading import Lock
from time import perf_counter

import cv2
import numpy as np
from PIL import Image, ImageOps

from workflow import sha256
from helmet_inference import HelmetPredictor, NORMAL_MODE
from helmet_fusion import predictions, fuse_nms, calibrated_wbf

STANDARD = '标准版（单模型）'
P2 = 'P2（单模型）'
NMS = '双模型融合（NMS，召回优先）'
WBF = '双模型融合（WBF，均衡）'
LEGACY = '自选／历史演示模型'
KEYS = {STANDARD: 'standard', P2: 'p2', NMS: 'ensemble_nms', WBF: 'ensemble_wbf'}


class HelmetDetectionService:
    def __init__(self, root, device='cpu', legacy_path=None, loader=None):
        self.root = Path(root)
        self.config = json.loads((self.root/'models/fusion_config.json').read_text())
        self.device = 'cuda:'+device if str(device).isdigit() else str(device)
        self.legacy_path = Path(legacy_path) if legacy_path else None
        self.loader = loader
        self.models, self.predictors = {}, {}
        self._lock = Lock()

    def model_path(self, key):
        name = f'YOLO_{key.upper()}_MODEL_PATH'
        path = Path(os.environ.get(name) or self.config['models'][key]['path']).expanduser()
        return path if path.is_absolute() else self.root/path

    def available(self, key):
        path = self.model_path(key)
        return path.is_file() and sha256(path) == self.config['models'][key]['sha256']

    def threshold(self, preset):
        return self.config['thresholds'][KEYS[preset]] if preset in KEYS else .25

    def _load(self, key):
        if key in self.models:
            return self.models[key]
        path = self.legacy_path if key == 'legacy' else self.model_path(key)
        if path is None or not path.is_file():
            raise FileNotFoundError('缺少所选模型权重。标准版和P2需要对应的两份best.pt；请查看 docs/FUSION.md，历史下载权重不能代替这两份模型。')
        if key != 'legacy' and sha256(path) != self.config['models'][key]['sha256']:
            raise ValueError(f'{key} 权重与本次验证使用的模型不一致，停止推理；请检查模型路径。')
        if self.loader is None:
            from ultralytics import YOLO
            model = YOLO(str(path))
        else:
            model = self.loader(str(path))
        if model.task != 'detect' or model.names != {0: 'hat', 1: 'person'}:
            raise ValueError('所选模型必须为 0=hat、1=person 的头部检测模型。')
        self.models[key] = model
        return model

    def predict(self, image, preset, conf=None, mode=NORMAL_MODE, ordinary_imgsz=640):
        if preset not in [*KEYS, LEGACY]:
            raise ValueError('请选择有效的检测方案。')
        conf = self.threshold(preset) if conf is None else float(conf)
        if not math.isfinite(conf) or not 0 <= conf <= 1:
            raise ValueError('置信度必须在0到1之间。')
        fusion = preset in [NMS, WBF]
        if fusion and (mode != NORMAL_MODE or ordinary_imgsz != 640):
            raise ValueError('本次双模型融合固定使用640整图检测，不与SAHI或960分辨率叠加。')
        if fusion and conf < .05:
            raise ValueError('融合最终置信度请选择0.05到1之间。')
        with self._lock:
            start = perf_counter()
            if not fusion:
                key = KEYS.get(preset, 'legacy')
                if key not in self.predictors:
                    self.predictors[key] = HelmetPredictor(self._load(key), self.device)
                annotated, counts, details = self.predictors[key].predict(image, conf, mode, ordinary_imgsz)
                details['seconds'] = perf_counter()-start
            else:
                from ultralytics.engine.results import Results
                image = ImageOps.exif_transpose(image).convert('RGB')
                src = []
                for key in ['standard', 'p2']:
                    result = self._load(key).predict(
                        image, imgsz=640, conf=self.config['input_confidence'],
                        iou=self.config['per_model_nms_iou'], max_det=self.config['max_det'],
                        rect=True, half=False, batch=1, device=self.device,
                        augment=False, verbose=False, save=False)[0]
                    raw = result.boxes.data
                    rows = np.asarray(raw.cpu().numpy() if hasattr(raw, 'cpu') else raw,
                                      dtype=float).reshape(-1, 6)
                    if not np.isfinite(rows).all() or np.any(rows[:, 2:4] < rows[:, :2]):
                        raise ValueError('模型输出了无效框。')
                    rows = rows[np.all(rows[:, 2:4] > rows[:, :2], axis=1)]
                    src.append(predictions(rows))
                if preset == NMS:
                    combined = fuse_nms(src, iou=self.config['fusion_iou'], conf=conf, max_det=self.config['max_det'])
                else:
                    combined = calibrated_wbf(src, image.width, image.height,
                        input_conf=self.config['input_confidence'], output_conf=conf,
                        iou=self.config['fusion_iou'], max_det=self.config['max_det'])
                result = Results(orig_img=np.asarray(image)[:, :, ::-1].copy(), path='uploaded_image',
                                 names={0: 'hat', 1: 'person'}, boxes=combined.astype(np.float32))
                annotated = Image.fromarray(cv2.cvtColor(result.plot(line_width=2, font_size=12), cv2.COLOR_BGR2RGB))
                counts = {}
                for cls in combined[:, 5]:
                    name = result.names[int(cls)]
                    counts[name] = counts.get(name, 0)+1
                details = {'mode': NORMAL_MODE, 'tiles': 1, 'imgsz': 640, 'seconds': perf_counter()-start}
            details.update(preset=preset, confidence=conf,
                           recommended_confidence=self.threshold(preset),
                           calibrated_settings=(preset in KEYS and mode == NORMAL_MODE and ordinary_imgsz == 640
                                                and abs(conf-self.threshold(preset)) < 1e-9),
                           model_keys=['standard', 'p2'] if fusion else [KEYS.get(preset, 'legacy')])
            return annotated, counts, details
