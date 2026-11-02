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
from helmet_inference import HelmetPredictor, NORMAL_MODE, SLICED_MODE, detection_records, render_result
from helmet_fusion import predictions, fuse_nms, calibrated_wbf

STANDARD = '标准版（单模型）'
P2 = 'P2（单模型）'
STANDARD_P3_CBAM = '标准版 + P3-CBAM（单模型）'
P2_P3_CBAM = 'P2 + P3-CBAM（单模型）'
P2_P2_CBAM = 'P2 + P2-CBAM（单模型）'
NMS = '双模型融合（NMS，召回优先）'
WBF = '双模型融合（WBF，均衡）'
LEGACY = '自选／历史演示模型'
SINGLE_MODELS = {STANDARD: 'standard', STANDARD_P3_CBAM: 'standard_p3_cbam', P2: 'p2',
                 P2_P3_CBAM: 'p2_p3_cbam', P2_P2_CBAM: 'p2_p2_cbam'}
KEYS = {**SINGLE_MODELS, NMS: 'ensemble_nms', WBF: 'ensemble_wbf'}
MANIFEST_KEYS = {'01_standard': 'standard', '02_standard_P3_CBAM': 'standard_p3_cbam',
                 '03_P2': 'p2', '04_P2_P3_CBAM': 'p2_p3_cbam', '05_P2_P2_CBAM': 'p2_p2_cbam'}
MAX_INPUT_PIXELS = 40_000_000


class HelmetDetectionService:
    def __init__(self, root, device='cpu', legacy_path=None, loader=None):
        self.root = Path(root).resolve()
        self.config = json.loads((self.root/'models/fusion_config.json').read_text())
        self.model_specs = {key: dict(spec) for key, spec in self.config['models'].items()}
        manifest = self.root/'models/release_manifest.json'
        if manifest.is_file():
            for spec in json.loads(manifest.read_text(encoding='utf-8'))['models']:
                key = MANIFEST_KEYS.get(spec['id'])
                if key is not None:
                    relative = Path(spec['source'])
                    if relative.is_absolute() or not (self.root/relative).resolve().is_relative_to(self.root.resolve()):
                        raise ValueError('模型清单路径必须位于项目目录内。')
                    self.model_specs[key] = {'path': relative.as_posix(), 'sha256': spec['sha256']}
        self.device = 'cuda:'+device if str(device).isdigit() else str(device)
        self.legacy_path = Path(legacy_path) if legacy_path else None
        self.loader = loader
        self.models, self.predictors, self.loaded_identity = {}, {}, {}
        self._lock = Lock()

    def model_path(self, key):
        if key not in self.model_specs:
            raise FileNotFoundError(f'{key} 尚未登记权重；请查看 models/release_manifest.json。')
        name = f'YOLO_{key.upper()}_MODEL_PATH'
        path = Path(os.environ.get(name) or self.model_specs[key]['path']).expanduser()
        return path if path.is_absolute() else self.root/path

    def available(self, key):
        try:
            path = self.model_path(key)
            return path.is_file() and sha256(path) == self.model_specs[key]['sha256']
        except OSError:
            return False

    def available_presets(self):
        choices = [label for label, key in SINGLE_MODELS.items() if self.available(key)]
        if STANDARD in choices and P2 in choices:
            choices.extend([NMS, WBF])
        if self.legacy_path and self.legacy_path.is_file():
            choices.append(LEGACY)
        return choices

    def model_status(self):
        rows = []
        for label, key in SINGLE_MODELS.items():
            try:
                path = self.model_path(key)
                status = ('可用（哈希匹配）' if self.available(key) else
                          '哈希不匹配，已禁用' if path.is_file() else '缺少权重，已禁用')
                filename = path.relative_to(self.root).as_posix() if path.is_relative_to(self.root) else path.name
            except OSError:
                status, filename = '未登记或无法读取，已禁用', '—'
            rows.append([label, status, filename])
        return rows

    def threshold(self, preset):
        return self.config['thresholds'].get(KEYS.get(preset), .25)

    def _load(self, key):
        if key in self.models:
            return self.models[key]
        path = self.legacy_path if key == 'legacy' else self.model_path(key)
        if path is None or not path.is_file():
            raise FileNotFoundError('缺少所选模型权重；请查看模型状态与 models/release_manifest.json。历史演示权重不能代替五组实验模型。')
        if key != 'legacy' and sha256(path) != self.model_specs[key]['sha256']:
            raise ValueError(f'{key} 权重与本次验证使用的模型不一致，停止推理；请检查模型路径。')
        if self.loader is None:
            from helmet_cbam import register_cbam
            from ultralytics import YOLO
            register_cbam()
            model = YOLO(str(path))
        else:
            model = self.loader(str(path))
        if model.task != 'detect' or model.names != {0: 'hat', 1: 'person'}:
            raise ValueError('所选模型必须为 0=hat、1=person 的头部检测模型。')
        self.models[key] = model
        self.loaded_identity[key] = {'key': key, 'weight': path.relative_to(self.root).as_posix()
            if path.is_relative_to(self.root) else path.name, 'sha256': sha256(path)}
        return model

    def predict(self, image, preset, conf=None, mode=NORMAL_MODE, ordinary_imgsz=640, *,
                nms_iou=.7, show_labels=True, show_conf=True, show_box_ids=False):
        if not isinstance(image, Image.Image) or min(image.size) < 1:
            raise ValueError('请输入可读取的图片。')
        if image.width * image.height > MAX_INPUT_PIXELS:
            raise ValueError('图片超过4000万像素，请先缩小原图后再上传。')
        image.load()
        if preset not in [*KEYS, LEGACY]:
            raise ValueError('请选择有效的检测方案。')
        conf = self.threshold(preset) if conf is None else float(conf)
        if not math.isfinite(conf) or not 0 <= conf <= 1:
            raise ValueError('置信度必须在0到1之间。')
        fusion = preset in [NMS, WBF]
        nms_iou = float(nms_iou)
        if not math.isfinite(nms_iou) or not 0 <= nms_iou <= 1:
            raise ValueError('NMS IoU必须在0到1之间。')
        if (fusion or mode == SLICED_MODE) and abs(nms_iou - .7) > 1e-9:
            raise ValueError('SAHI及历史融合保留原协议，模型NMS IoU固定为0.7。')
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
                annotated, counts, details = self.predictors[key].predict(image, conf, mode, ordinary_imgsz,
                    nms_iou=nms_iou, show_labels=show_labels, show_conf=show_conf, show_box_ids=show_box_ids)
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
                annotated = render_result(result, show_labels=show_labels, show_conf=show_conf, show_box_ids=show_box_ids)
                counts = {}
                for cls in combined[:, 5]:
                    name = result.names[int(cls)]
                    counts[name] = counts.get(name, 0)+1
                details = {'mode': NORMAL_MODE, 'tiles': 1, 'imgsz': 640, 'seconds': perf_counter()-start,
                           'detections': detection_records(result), 'image_width': image.width, 'image_height': image.height}
            details.update(preset=preset, confidence=conf,
                           recommended_confidence=self.threshold(preset),
                           calibrated_settings=(KEYS.get(preset) in self.config['thresholds'] and mode == NORMAL_MODE and ordinary_imgsz == 640
                                                and abs(nms_iou-.7) < 1e-9
                                                and abs(conf-self.threshold(preset)) < 1e-9),
                           model_keys=['standard', 'p2'] if fusion else [KEYS.get(preset, 'legacy')])
            details.update(device=self.device, nms_iou=nms_iou, per_model_max_det=self.config['max_det'],
                           final_max_det=None if mode == SLICED_MODE and details['tiles'] > 1 else self.config['max_det'],
                           models=[self.loaded_identity[key] for key in details['model_keys']],
                           display={'labels': show_labels, 'confidence': show_conf, 'box_ids': show_box_ids})
            if fusion:
                details['fusion'] = {'iou': self.config['fusion_iou'], 'input_confidence': self.config['input_confidence']}
            if mode == SLICED_MODE:
                details['sahi'] = {'slice_size': 640, 'overlap': .2, 'merge_metric': 'IOS', 'merge_threshold': .5}
            return annotated, counts, details
