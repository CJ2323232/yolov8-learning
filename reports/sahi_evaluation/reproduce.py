"""Reproduce saved SAHI configurations with a hash-verified historical dataset.

This publication entry point was added after the measured experiments. It does
not generate or download data, change labels, train, or overwrite an output run.
"""
import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import sys
from unittest.mock import patch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))
MODEL_SHA256 = '55be821e9eefdd0a69caf4abd3fdb2a22277059e21ec155222102b32cef2e650'
CONFIGS = ['ordinary', 'sahi_640', 'sahi_960', 'sahi_1280', 'sahi_640_fixed']


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def dataset_path(root, relative):
    relative = Path(relative)
    if relative.is_absolute() or '..' in relative.parts:
        raise ValueError('Manifest paths must be relative to --data-root')
    path = (root / relative).resolve()
    if not path.is_relative_to(root):
        raise ValueError('Manifest path escapes --data-root')
    return path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-root', type=Path, required=True, help='Historical dataset root containing images/val and labels/val')
    parser.add_argument('--manifest', type=Path, default=HERE / 'selected_manifest.csv')
    parser.add_argument('--model', type=Path, default=ROOT / 'models/helmet_yolov8n.pt')
    parser.add_argument('--output', type=Path, required=True, help='A new output directory')
    parser.add_argument('--configs', nargs='+', choices=CONFIGS, default=CONFIGS)
    parser.add_argument('--limit', type=int, help='Smoke check only; first N manifest images, not the full published benchmark')
    args = parser.parse_args()
    if len(set(args.configs)) != len(args.configs):
        parser.error('Duplicate configurations are not allowed')
    if args.limit is not None and not 1 <= args.limit <= 529:
        parser.error('--limit must be in [1, 529]')
    if digest(args.model) != MODEL_SHA256:
        parser.error('Model does not match the published historical checkpoint')
    all_rows = list(csv.DictReader(args.manifest.open()))
    if len(all_rows) != 529 or len({r['image'] for r in all_rows}) != 529:
        parser.error('The published manifest must contain 529 unique images')
    rows = all_rows[:args.limit] if args.limit else all_rows
    data_root = args.data_root.resolve()
    for row in rows:
        for kind in ['image', 'label']:
            path = dataset_path(data_root, row[kind + '_path'])
            if digest(path) != row[kind + '_sha256']:
                parser.error(f'{kind} hash mismatch: {row["image"]}')
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=False)
    os.environ.setdefault('YOLO_CONFIG_DIR', str(out / 'runtime_config'))
    os.environ.setdefault('MPLCONFIGDIR', str(out / 'matplotlib'))
    import cv2
    import numpy as np
    from PIL import Image, ImageOps
    import torch
    import ultralytics
    import sahi
    from ultralytics import YOLO
    import helmet_inference as fixed
    from repro import legacy_inference as legacy
    from repro.metrics import summarize, add_counts, with_rates
    from analyze_detection_errors import ground_truth

    versions = {'python': sys.version.split()[0], 'torch': torch.__version__,
                'ultralytics': ultralytics.__version__, 'sahi': sahi.__version__}
    expected_versions = json.loads((HERE / 'slice_run_identity.json').read_text())['versions']
    if versions != expected_versions:
        raise RuntimeError(f'Published-version reproduction requires {expected_versions}; found {versions}')
    torch.set_num_threads(4); cv2.setNumThreads(2)
    model = YOLO(str(args.model))
    predictors = {'legacy': legacy.HelmetPredictor(model, 'cpu'), 'fixed': fixed.HelmetPredictor(model, 'cpu')}
    captured = []
    original_plot, original_predict = fixed.Results.plot, model.predict
    def capture_plot(result, *a, **kw):
        data = result.boxes.data
        captured.append((data.cpu().numpy() if hasattr(data, 'cpu') else np.asarray(data)).copy())
        return original_plot(result, *a, **kw)
    def checked_predict(*a, **kw):
        result = original_predict(*a, **kw)
        settings = model.predictor.args
        assert (float(settings.conf), int(settings.imgsz), float(settings.iou), int(settings.max_det),
                bool(settings.rect), bool(settings.augment), str(settings.device)) == (.25, 640, .7, 300, True, False, 'cpu')
        return result
    model.predict = checked_predict
    # Set effective block defaults explicitly, including when only legacy SAHI is selected.
    model.overrides.update(iou=.7, max_det=300, rect=True, augment=False)
    def read(row):
        with Image.open(dataset_path(data_root, row['image_path'])) as image:
            assert image.getexif().get(274, 1) in (None, 0, 1), 'Active rotation requires transformed labels'
            rgb = ImageOps.exif_transpose(image).convert('RGB')
        assert rgb.size == (int(row['width']), int(row['height']))
        return rgb
    def predict(image, config):
        captured.clear()
        legacy.SLICE_SIZE = int(config.rsplit('_', 1)[1]) if config in CONFIGS[1:4] else 640
        fixed.SLICE_SIZE = 640
        predictor = predictors['fixed' if config in ['ordinary', 'sahi_640_fixed'] else 'legacy']
        mode = fixed.NORMAL_MODE if config == 'ordinary' else fixed.SLICED_MODE
        _, _, details = predictor.predict(image, .25, mode)
        assert len(captured) == 1
        return captured[0], details
    warm = sorted(rows, key=lambda r: int(r['width']) * int(r['height']))
    warm = [warm[0], warm[len(warm)//2], warm[-1]]
    protocol = {'scope': 'full 529-image reproduction' if len(rows) == 529 else 'subset smoke check only',
                'images': len(rows), 'configs': args.configs, 'versions': versions,
                'manifest_sha256': digest(args.manifest), 'model_sha256': digest(args.model),
                'runner_sha256': digest(__file__), 'inference_sha256': digest(ROOT / 'helmet_inference.py'),
                'legacy_inference_sha256': digest(HERE / 'repro/legacy_inference.py'),
                'metrics_sha256': digest(HERE / 'repro/metrics.py'),
                'matching_sha256': digest(ROOT / 'analyze_detection_errors.py'),
                'warmup_images': [r['image'] for r in warm],
                'fixed': json.loads((HERE / 'slice_protocol.json').read_text())['fixed'],
                'timing': 'includes plotting and capture; warm-up excluded; rotate config order by image index',
                'training_performed': False}
    (out / 'protocol.json').write_text(json.dumps(protocol, indent=2) + '\n')
    totals = {n: {} for n in args.configs}
    with patch.object(fixed.Results, 'plot', capture_plot), (out / 'predictions.jsonl').open('w') as handle:
        for row in warm:
            for config in args.configs:
                predict(read(row), config)
        for i, row in enumerate(rows):
            image = read(row)
            gt = ground_truth(dataset_path(data_root, row['label_path']), *image.size)
            record = {'image': row['image'], 'configs': {}}
            offset = i % len(args.configs)
            for config in args.configs[offset:] + args.configs[:offset]:
                boxes, details = predict(image, config)
                counts, pairs, missed, unmatched = summarize(gt, boxes, *image.size)
                add_counts(totals[config], counts)
                record['configs'][config] = {'counts': counts, 'pred_xyxy_score_class': boxes.tolist(),
                    'matches_prediction_gt_iou': pairs, 'missed_gt': missed, 'unmatched_predictions': unmatched,
                    'seconds': details['seconds'], 'tiles': details['tiles']}
            handle.write(json.dumps(record) + '\n'); handle.flush()
            if (i + 1) % 25 == 0 or i + 1 == len(rows):
                print(f'{i + 1}/{len(rows)} images', flush=True)
    for row in rows:
        for kind in ['image', 'label']:
            assert digest(dataset_path(data_root, row[kind + '_path'])) == row[kind + '_sha256']
    assert digest(args.model) == MODEL_SHA256
    result = {'scope': protocol['scope'], 'images': len(rows), 'configs': {n: with_rates(c) for n, c in totals.items()},
              'source_and_model_hashes': protocol, 'inputs_unchanged': True}
    (out / 'summary.json').write_text(json.dumps(result, indent=2) + '\n')
    print('Completed. This output is separate from the published measured records.')


if __name__ == '__main__':
    main()
