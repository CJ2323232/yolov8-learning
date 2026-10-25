"""Postprocess frozen xyxy,score,class predictions; never consult ground truth."""
import numpy as np

from tools.third_party.weighted_boxes_fusion import weighted_boxes_fusion


def predictions(rows):
    result = np.asarray(rows, dtype=float).reshape(-1, 6)
    if not np.isfinite(result).all():
        raise ValueError('Non-finite prediction')
    if len(result) and (np.any(result[:, 2:4] <= result[:, :2])
                        or np.any((result[:, 4] < 0) | (result[:, 4] > 1))
                        or not np.isin(result[:, 5], [0, 1]).all()):
        raise ValueError('Invalid box, confidence, or class')
    return result


def finish(rows, conf=.25, max_det=300):
    p = predictions(rows)
    p = p[p[:, 4] >= conf]
    return p[np.argsort(-p[:, 4], kind='stable')[:max_det]]


def fuse_nms(models, iou=.55, conf=.25, max_det=300):
    """Class-aware greedy NMS; suppress IoU > threshold, stable score tie order."""
    p = np.concatenate([predictions(m) for m in models], axis=0)
    p = p[p[:, 4] >= conf]
    order = np.argsort(-p[:, 4], kind='stable')
    keep = []
    while len(order) and len(keep) < max_det:
        i = order[0]
        keep.append(i)
        rest = order[1:]
        if not len(rest):
            break
        lo = np.maximum(p[i, :2], p[rest, :2])
        hi = np.minimum(p[i, 2:4], p[rest, 2:4])
        inter = np.maximum(hi - lo, 0).prod(axis=1)
        union = (p[i, 2:4] - p[i, :2]).prod() + (p[rest, 2:4] - p[rest, :2]).prod(axis=1) - inter
        overlaps = np.divide(inter, union, out=np.zeros_like(inter), where=union > 0)
        order = rest[~((p[rest, 5] == p[i, 5]) & (overlaps > iou))]
    return p[keep].reshape(-1, 6)


def fuse_wbf(models, width, height, iou=.55, conf=.25, max_det=300, input_conf=None):
    """Upstream WBF defaults: equal weights, average confidence, no overflow."""
    scale = np.array([width, height, width, height], dtype=float)
    if width <= 0 or height <= 0:
        raise ValueError('Invalid image dimensions')
    parts = [predictions(m) for m in models]
    for p in parts:
        if len(p) and (np.any(p[:, :4] < -1e-5) or np.any(p[:, :4] > scale + 1e-5)):
            raise ValueError('Prediction extends outside original image')
    boxes, scores, labels = weighted_boxes_fusion(
        [np.clip(p[:, :4] / scale, 0, 1).tolist() for p in parts],
        [p[:, 4].tolist() for p in parts], [p[:, 5].astype(int).tolist() for p in parts],
        weights=[1.0] * len(parts), iou_thr=iou, skip_box_thr=conf if input_conf is None else input_conf,
        conf_type='avg', allows_overflow=False)
    return finish(np.column_stack([boxes * scale, scores, labels]), conf, max_det)


def calibrated_wbf(models, width, height, input_conf=.01, output_conf=.17, iou=.55, max_det=300):
    """Separate candidate and final score floors, matching the frozen study."""
    return fuse_wbf(models, width, height, iou=iou, conf=output_conf,
                    max_det=max_det, input_conf=input_conf)
