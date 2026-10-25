"""Frozen-threshold validation/test study. Inference only; never trains models."""
import argparse
import csv
import json
import platform
import sys
import time
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from PIL import Image

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
PREVIOUS = ROOT / 'error_analysis/fusion_validation_2026-10-24'
sys.path[:0] = [str(ROOT), str(PREVIOUS)]
from workflow import sha256, verify_dataset, write_json
from analyze_detection_errors import match, ground_truth
from fusion import fuse_nms, fuse_wbf, predictions, finish

DATA = ROOT / 'datasets/helmet_scene_isolated_v1'
MODELS = {v: ROOT / 'runs2' / f'helmet_{v}_isolated_640_windows/weights/best.pt'
          for v in ['standard', 'p2']}
METHODS = ['standard', 'p2', 'ensemble_nms', 'ensemble_wbf']


def read(name):
    return json.loads((HERE / name).read_text())


def save_once(name, value):
    path = HERE / name
    if path.exists():
        if read(name) != value:
            raise ValueError(f'Refusing to replace frozen artifact: {name}')
    else:
        write_json(path, value)


def now():
    return datetime.now(timezone.utc).isoformat()


def manifest(split):
    with (DATA / 'split_manifest.csv').open() as f:
        return sorted([r for r in csv.DictReader(f) if r['split'] == split], key=lambda r: r['image'])


def integrity():
    p = read('protocol.json')
    assert sha256(DATA / 'split_manifest.csv') == p['dataset']['manifest_sha256']
    for v, path in MODELS.items():
        assert sha256(path) == p['model_sha256'][v]
    for name, fingerprint in p['dependency_sha256'].items():
        assert sha256(ROOT / name) == fingerprint, name
    assert sha256(HERE / 'experiment.py') == p['experiment_script_sha256']
    return p


def prepare():
    import torch, ultralytics
    assert not (HERE / 'protocol.json').exists(), 'Protocol already prepared'
    identity = verify_dataset(DATA / 'data.yaml')
    previous = json.loads((PREVIOUS / 'protocol.json').read_text())
    assert identity == previous['source_protocol']['dataset']
    hashes = {v: sha256(path) for v, path in MODELS.items()}
    assert hashes == previous['source_protocol']['model_sha256']
    dependencies = ['analyze_detection_errors.py', 'workflow.py',
                    str((PREVIOUS / 'fusion.py').relative_to(ROOT)),
                    str((PREVIOUS / 'vendor/ensemble_boxes_wbf.py').relative_to(ROOT))]
    protocol = {
        'created_at_utc': now(), 'dataset': identity, 'model_sha256': hashes,
        'experiment_script_sha256': sha256(__file__),
        'dependency_sha256': {n: sha256(ROOT / n) for n in dependencies},
        'validation_images': 511, 'test_images': 508,
        'methods': METHODS, 'input_confidence': .01, 'fusion_output_floor': .005,
        'invalid_candidate_policy': 'Discard and count exactly zero-area boxes after image-boundary clipping; reject nonfinite, inverted, wrong-class or invalid-score rows. Applied identically to both models.',
        'protocol_revision': 'v2: preflight v1 stopped at validation standard image 245 on a zero-area low-score candidate. No threshold selection or test evaluation occurred. Partial v1 cache preserved, all inference regenerated under v2.',
        'imgsz': 640, 'per_model_nms_iou': .7, 'fusion_iou': .55,
        'max_det': 300, 'rect': True, 'half': False, 'batch': 1, 'device': 'cpu', 'threads': 4,
        'wbf': {'weights': [1, 1], 'conf_type': 'avg', 'allows_overflow': False,
                'input_skip_box_thr': .01, 'output_score_floor': .005},
        'matching': 'same-class score-ordered one-to-one IoU >= 0.5',
        'threshold_grid': [i / 100 for i in range(5, 96)],
        'validation_unmatched_budget': 759,
        'budget_source': 'previous P2 at confidence 0.25 on these 511 validation images',
        'threshold_selection': 'Each method: among FP <= 759, maximize TP, then minimize FP, then highest threshold. No class-specific tuning.',
        'method_selection': 'Choose strongest single and strongest fusion on validation: highest TP, then lowest FP; fixed method order resolves ties.',
        'test_rule': 'Freeze all four thresholds and primary single/fusion identities BEFORE test inference/evaluation; evaluate once; no test retuning.',
        'test_screening_gate': 'Chosen fusion recall > chosen single; precision >= single minus 0.01; F1 >= single; helmet recall >= single. Operational screen, not significance proof.',
        'uncertainty': 'Paired bootstrap 2000 resamples over manifest group_id, seed 20261024; retrieved-risk groups are not proven independent scenes.',
        'benchmark': {'images': 24, 'rounds': 2, 'warmup_calls_per_model': 3,
                      'selection': '24 evenly spaced validation filenames, independent of labels/results',
                      'scope': 'CPU resident-model single-image wall time including disk image decoding, prediction, fusion and final filtering; excludes model loading, rendering, saving, UI/network. Four threads. Warm filesystem cache.',
                      'device_memory': 'CPU process peak RSS only; no CUDA GPU/VRAM measurement on this Mac.'},
        'python': platform.python_version(), 'torch': torch.__version__, 'ultralytics': ultralytics.__version__,
        'limits': ['Prediction floor .01 and max_det 300 truncate candidates; no AP/mAP claimed.',
                   'Validation has been used for model selection and visual exploration.',
                   'Test images come from the existing corpus; historical models/tests used related data. This round keeps test out of threshold selection, not a newly collected external test.',
                   'No guarantee that undiscovered related scenes are isolated; no PartA classroom generalization claim.',
                   'Unmatched predictions are annotation-relative, not all human-confirmed false alarms.',
                   'Single trained checkpoint per structure; cannot establish architecture-only causal effects.'],
    }
    save_once('protocol.json', protocol)
    print('Protocol frozen. All 7507 image/label hashes and isolation constraints verified.', flush=True)


def load_image(row):
    assert sha256(DATA / row['image']) == row['image_sha256']
    with Image.open(DATA / row['image']) as source:
        assert source.getexif().get(274, 1) == 1
        return source.convert('RGB')


def prediction_kwargs(p):
    return dict(imgsz=p['imgsz'], conf=p['input_confidence'], iou=p['per_model_nms_iou'],
                max_det=p['max_det'], rect=p['rect'], half=p['half'], batch=1,
                device=p['device'], augment=False, verbose=False, save=False)


def sanitize_model_output(rows):
    a = np.asarray(rows, dtype=float).reshape(-1, 6)
    assert np.isfinite(a).all()
    assert np.all(a[:, 2:4] >= a[:, :2]), 'Inverted box'
    assert np.isin(a[:, 5], [0, 1]).all() and np.all((a[:, 4] >= 0) & (a[:, 4] <= 1))
    keep = np.all(a[:, 2:4] > a[:, :2], axis=1)
    return predictions(a[keep]), int((~keep).sum())


def infer(split):
    import torch
    from ultralytics import YOLO
    p = integrity()
    torch.set_num_threads(p['threads'])
    if split == 'test':
        selection = read('selection.json')
        assert selection['protocol_sha256'] == sha256(HERE / 'protocol.json')
        for v, fingerprint in selection['validation_predictions_sha256'].items():
            assert sha256(HERE / f'val_{v}.jsonl') == fingerprint
        save_once('test_started.json', {'selection_sha256': sha256(HERE / 'selection.json'),
                  'frozen_at_utc': selection['frozen_at_utc']})
    rows = manifest(split)
    assert len(rows) == p['validation_images' if split == 'val' else 'test_images']
    protocol_hash = sha256(HERE / 'protocol.json')
    for v, path in MODELS.items():
        cache = HERE / f'{split}_{v}.jsonl'
        saved = [json.loads(line) for line in cache.read_text().splitlines()] if cache.exists() else []
        assert len(saved) <= len(rows)
        assert all(c['image'] == rows[i]['image'] and c['protocol_sha256'] == protocol_hash for i, c in enumerate(saved))
        if len(saved) == len(rows):
            print(split, v, 'cache complete', flush=True)
            continue
        model = YOLO(str(path))
        assert model.names == {0: 'hat', 1: 'person'} and model.task == 'detect'
        started = time.perf_counter()
        with cache.open('a') as f:
            for i, row in enumerate(rows[len(saved):], len(saved)):
                im = load_image(row)
                result = model.predict(im, **prediction_kwargs(p))[0]
                pred, discarded = sanitize_model_output(result.boxes.data.cpu().numpy())
                c = {'image': row['image'], 'width': im.width, 'height': im.height, 'pred': pred.tolist(),
                     'discarded_zero_area': discarded,
                     'protocol_sha256': protocol_hash}
                f.write(json.dumps(c, separators=(',', ':')) + '\n')
                f.flush()
                if (i+1) % 50 == 0 or i+1 == len(rows):
                    print(split, v, i+1, '/', len(rows), 'seconds', round(time.perf_counter()-started, 1), flush=True)
        del model
    integrity()


def outputs(a, b, width, height, p):
    # WBF input floor and output floor are separate; retain single-model boxes for later calibration.
    from ensemble_boxes_wbf import weighted_boxes_fusion
    scale = np.array([width, height, width, height], dtype=float)
    src = [predictions(a), predictions(b)]
    for part in src:
        assert np.all(part[:, :4] >= -1e-5) and np.all(part[:, :4] <= scale + 1e-5)
    boxes, scores, labels = weighted_boxes_fusion(
        [np.clip(x[:, :4]/scale, 0, 1).tolist() for x in src],
        [x[:, 4].tolist() for x in src], [x[:, 5].astype(int).tolist() for x in src],
        weights=[1, 1], iou_thr=p['fusion_iou'], skip_box_thr=p['input_confidence'],
        conf_type='avg', allows_overflow=False)
    return {'standard': src[0], 'p2': src[1],
            'ensemble_nms': fuse_nms(src, iou=p['fusion_iou'], conf=p['input_confidence'], max_det=p['max_det']),
            'ensemble_wbf': finish(np.column_stack([boxes*scale, scores, labels]), p['fusion_output_floor'], p['max_det'])}


def cases(split):
    p = integrity()
    rows = manifest(split)
    cached = {v: [json.loads(x) for x in (HERE / f'{split}_{v}.jsonl').read_text().splitlines()] for v in MODELS}
    assert all(len(x) == len(rows) for x in cached.values())
    for i, row in enumerate(rows):
        a, b = [cached[v][i] for v in MODELS]
        assert a['image'] == b['image'] == row['image']
        assert a['protocol_sha256'] == b['protocol_sha256'] == sha256(HERE / 'protocol.json')
        assert (a['width'], a['height']) == (b['width'], b['height'])
        assert sha256(DATA / row['label']) == row['label_sha256']
        gt = ground_truth(DATA / row['label'], a['width'], a['height'])
        yield row, a['width'], a['height'], gt, outputs(a['pred'], b['pred'], a['width'], a['height'], p)


def metrics(tp, fp, gt):
    tp, fp, gt = int(tp), int(fp), int(gt)
    return {'tp': tp, 'fp': fp, 'fn': gt-tp, 'gt': gt,
            'precision': tp/(tp+fp) if tp+fp else 0,
            'recall': tp/gt if gt else 0, 'f1': 2*tp/(gt+tp+fp) if gt+tp+fp else 0}


def choose(points, budget):
    candidates = [x for x in points if x['fp'] <= budget]
    if not candidates:
        raise ValueError('No threshold satisfies the frozen budget')
    return max(candidates, key=lambda x: (x['tp'], -x['fp'], x['threshold']))


def select():
    p = integrity()
    assert not (HERE / 'test_started.json').exists(), 'Test already started; cannot retune'
    grid = np.array(p['threshold_grid'])
    counts = {v: np.zeros((len(grid), 2), dtype=int) for v in METHODS}
    total_gt = 0
    cap = Counter()
    for row, w, h, gt, variants in cases('val'):
        total_gt += len(gt)
        for v, pred in variants.items():
            pairs, _, unmatched = match(gt, pred, .5)
            # Removing lower-score predictions cannot change higher-score greedy assignments.
            tp_scores = np.sort([pred[i, 4] for i, _, _ in pairs])
            fp_scores = np.sort([pred[i, 4] for i in unmatched])
            counts[v][:, 0] += len(tp_scores)-np.searchsorted(tp_scores, grid, side='left')
            counts[v][:, 1] += len(fp_scores)-np.searchsorted(fp_scores, grid, side='left')
            cap[v] += int(len(pred) == p['max_det'])
    curves = {v: [dict(threshold=float(t), **metrics(*count, total_gt)) for t, count in zip(grid, counts[v])]
              for v in METHODS}
    selected = {v: choose(curves[v], p['validation_unmatched_budget']) for v in METHODS}
    rank = lambda v: (selected[v]['tp'], -selected[v]['fp'])
    single = max(METHODS[:2], key=rank)
    fusion = max(METHODS[2:], key=rank)
    selection = {'frozen_at_utc': now(), 'protocol_sha256': sha256(HERE / 'protocol.json'),
                 'validation_predictions_sha256': {v: sha256(HERE / f'val_{v}.jsonl') for v in MODELS},
                 'thresholds_and_validation_metrics': selected, 'primary_single': single,
                 'primary_fusion': fusion, 'validation_images': 511, 'gt': total_gt,
                 'candidate_cap_images_by_method': dict(cap),
                 'test_labels_or_metrics_used_for_selection': False}
    save_once('validation_curves.json', curves)
    save_once('selection.json', selection)
    print(json.dumps(selection, ensure_ascii=False, indent=2), flush=True)


def detail(gt, pred, w, h):
    pairs, missed, unmatched = match(gt, pred, .5)
    found = {g for _, g, _ in pairs}
    result = {'overall': metrics(len(pairs), len(unmatched), len(gt)), 'classes': {}, 'sizes': {}}
    for cid in [0, 1]:
        result['classes'][str(cid)] = metrics(sum(gt[g, 4] == cid for g in found),
            sum(pred[i, 5] == cid for i in unmatched), sum(gt[:, 4] == cid))
    short = np.minimum(gt[:, 2]-gt[:, 0], gt[:, 3]-gt[:, 1])*640/max(w, h)
    for name, lo, hi in [('lt8', 0, 8), ('8to16', 8, 16), ('16to32', 16, 32), ('ge32', 32, float('inf'))]:
        ids = set(np.where((short >= lo) & (short < hi))[0])
        result['sizes'][name] = {'gt': len(ids), 'tp': len(ids & found), 'fn': len(ids-found)}
    return result


def aggregate(details):
    total = lambda rows: metrics(sum(r['tp'] for r in rows), sum(r['fp'] for r in rows), sum(r['gt'] for r in rows))
    result = {'overall': total([r['overall'] for r in details]),
              'classes': {cid: total([r['classes'][cid] for r in details]) for cid in ['0', '1']},
              'sizes': {}}
    for group in ['lt8', '8to16', '16to32', 'ge32']:
        c = {k: sum(r['sizes'][group][k] for r in details) for k in ['gt', 'tp', 'fn']}
        c['recall'] = c['tp']/c['gt'] if c['gt'] else None
        result['sizes'][group] = c
    return result


def bootstrap(records, single, fusion):
    groups = defaultdict(lambda: np.zeros((2, 3), dtype=int))
    for r in records:
        for i, v in enumerate([single, fusion]):
            m = r['methods'][v]['overall']
            groups[r['group_id']][i] += [m['tp'], m['fp'], m['gt']]
    values = np.array(list(groups.values()))
    rng = np.random.default_rng(20261024)
    deltas = []
    for _ in range(2000):
        summed = values[rng.integers(0, len(values), len(values))].sum(axis=0)
        a, b = [metrics(*row) for row in summed]
        deltas.append([b[k]-a[k] for k in ['recall', 'precision', 'f1']])
    q = np.quantile(deltas, [.025, .975], axis=0)
    return {'risk_groups': len(groups), 'resamples': 2000,
            'delta_95pct_percentile_interval': {k: [float(q[0, i]), float(q[1, i])] for i, k in enumerate(['recall', 'precision', 'f1'])},
            'limit': 'Paired resampling of retrieved-risk groups, not proof of fully independent scenes or multiple-training-run stability.'}


def evaluate(split):
    p = integrity()
    selection = read('selection.json')
    if split == 'test':
        assert read('test_started.json')['selection_sha256'] == sha256(HERE / 'selection.json')
    records = []
    cap = Counter()
    for row, w, h, gt, variants in cases(split):
        record = {'image': row['image'], 'group_id': row['group_id'], 'methods': {}}
        for v, pred in variants.items():
            cap[v] += int(len(pred) == p['max_det'])
            threshold = selection['thresholds_and_validation_metrics'][v]['threshold']
            record['methods'][v] = detail(gt, finish(pred, threshold, p['max_det']), w, h)
        records.append(record)
    summary = {v: aggregate([r['methods'][v] for r in records]) for v in METHODS}
    if split == 'val':
        for v in METHODS:
            expected = selection['thresholds_and_validation_metrics'][v]
            assert all(summary[v]['overall'][k] == expected[k] for k in ['tp', 'fp', 'fn', 'gt'])
    a, b = selection['primary_single'], selection['primary_fusion']
    sa, sb = summary[a]['overall'], summary[b]['overall']
    gate = {'recall_increased': sb['recall'] > sa['recall'],
            'precision_drop_at_most_1pp': sb['precision'] >= sa['precision']-.01,
            'f1_not_lower': sb['f1'] >= sa['f1'],
            'helmet_recall_not_lower': summary[b]['classes']['0']['recall'] >= summary[a]['classes']['0']['recall']}
    result = {'split': split, 'images': len(records), 'metrics': summary,
              'primary_single': a, 'primary_fusion': b, 'screening_gate': gate,
              'passes_all_screening_checks': all(gate.values()),
              'candidate_cap_images_by_method': dict(cap), 'bootstrap': bootstrap(records, a, b),
              'selection_sha256': sha256(HERE / 'selection.json'),
              'prediction_sha256': {v: sha256(HERE / f'{split}_{v}.jsonl') for v in MODELS}}
    save_once(f'{split}_per_image.json', records)
    save_once(f'{split}_results.json', result)
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('stage', choices=['prepare', 'infer-val', 'select', 'evaluate-val', 'infer-test', 'evaluate-test', 'verify'])
    stage = ap.parse_args().stage
    if stage == 'prepare': prepare()
    elif stage.startswith('infer-'): infer(stage.split('-')[1])
    elif stage == 'select': select()
    elif stage.startswith('evaluate-'): evaluate(stage.split('-')[1])
    else:
        p = integrity()
        assert verify_dataset(DATA / 'data.yaml') == p['dataset']
        print('All dataset images/labels, weights and frozen dependencies unchanged.', flush=True)
