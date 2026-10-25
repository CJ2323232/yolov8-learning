"""Isolated-process CPU wall-time benchmark, already-trained weights only."""
import argparse
import json
import platform
import resource
import subprocess
import sys
import time

import numpy as np
from PIL import Image

from experiment import (HERE, DATA, MODELS, METHODS, read, integrity, manifest,
                        prediction_kwargs, predictions, finish, outputs, fuse_nms, sha256, save_once, sanitize_model_output)


def worker(variant):
    import torch
    from ultralytics import YOLO
    from ensemble_boxes_wbf import weighted_boxes_fusion
    p = integrity()
    selection = read('selection.json')
    torch.set_num_threads(p['threads'])
    needed = [variant] if variant in MODELS else list(MODELS)
    loaded = {v: YOLO(str(MODELS[v])) for v in needed}
    rows = manifest('val')
    indices = np.linspace(0, len(rows)-1, p['benchmark']['images'], dtype=int).tolist()
    threshold = selection['thresholds_and_validation_metrics'][variant]['threshold']
    kwargs = prediction_kwargs(p)
    for _ in range(p['benchmark']['warmup_calls_per_model']):
        with Image.open(DATA / rows[indices[0]]['image']) as src:
            im = src.convert('RGB')
        for model in loaded.values(): model.predict(im, **kwargs)
    effective_threads = torch.get_num_threads()

    def run(row):
        started = time.perf_counter_ns()
        with Image.open(DATA / row['image']) as src:
            im = src.convert('RGB')
        src = [sanitize_model_output(model.predict(im, **kwargs)[0].boxes.data.cpu().numpy())[0] for model in loaded.values()]
        if variant in MODELS:
            pred = finish(src[0], threshold, p['max_det'])
        elif variant == 'ensemble_nms':
            pred = fuse_nms(src, iou=p['fusion_iou'], conf=threshold, max_det=p['max_det'])
        else:
            scale = np.array([im.width, im.height, im.width, im.height], float)
            boxes, scores, labels = weighted_boxes_fusion(
                [np.clip(x[:, :4]/scale, 0, 1).tolist() for x in src],
                [x[:, 4].tolist() for x in src], [x[:, 5].astype(int).tolist() for x in src],
                weights=[1, 1], iou_thr=p['fusion_iou'], skip_box_thr=p['input_confidence'],
                conf_type='avg', allows_overflow=False)
            pred = finish(np.column_stack([boxes*scale, scores, labels]), threshold, p['max_det'])
        elapsed = (time.perf_counter_ns()-started)/1e6
        return elapsed, pred, src, im.width, im.height

    if variant not in MODELS:
        _, check_pred, check_src, w, h = run(rows[indices[0]])
        expected = finish(outputs(*check_src, w, h, p)[variant], threshold, p['max_det'])
        np.testing.assert_allclose(check_pred, expected, rtol=1e-6, atol=1e-5)
    records = []
    for repeat in range(p['benchmark']['rounds']):
        # Same order for each method, reversed on second round.
        for i in (indices if repeat % 2 == 0 else indices[::-1]):
            elapsed, pred, src, w, h = run(rows[i])
            assert torch.get_num_threads() == effective_threads
            records.append({'image': rows[i]['image'], 'repeat': repeat, 'ms': elapsed, 'detections': len(pred)})
    times = np.array([r['ms'] for r in records])
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    result = {'method': variant, 'sample_images': len(indices), 'timed_calls': len(records),
              'mean_ms': float(times.mean()), 'median_ms': float(np.median(times)),
              'p95_ms': float(np.quantile(times, .95)), 'throughput_images_per_second': float(1000/times.mean()),
              'process_peak_rss_mib': float(rss/(1024**2 if sys.platform == 'darwin' else 1024)),
              'rss_scope': 'whole isolated Python process including imports, warmup and consistency checks; CPU RAM, not VRAM',
              'records': records, 'requested_torch_threads': p['threads'], 'torch_threads': effective_threads,
              'thread_note': 'Ultralytics select_device resets torch threads during CPU setup. Record effective value after warmup; all timed calls assert the same value.',
              'machine': platform.platform(), 'device': 'cpu', 'cuda_vram_measured': False,
              'scope': p['benchmark']['scope'].replace('Four threads.', f'{effective_threads} effective threads.'), 'fusion_outputs_match_evaluation': True,
              'selection_sha256': sha256(HERE / 'selection.json'), 'script_sha256': sha256(__file__)}
    save_once(f'benchmark_{variant}.json', result)
    print(json.dumps({k: v for k, v in result.items() if k != 'records'}, ensure_ascii=False), flush=True)


def main():
    results = {}
    # Separate processes give each method its own model residency and RSS maximum.
    for v in ['p2', 'ensemble_nms', 'standard', 'ensemble_wbf']:
        print('Benchmark', v, flush=True)
        path = HERE / f'benchmark_{v}.json'
        if not path.exists():
            subprocess.run([sys.executable, '-B', __file__, '--worker', v], check=True, cwd=HERE)
        results[v] = read(path.name)
    save_once('benchmark_results.json', results)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--worker', choices=METHODS)
    args = ap.parse_args()
    if args.worker: worker(args.worker)
    else: main()
