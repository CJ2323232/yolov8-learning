"""Fixed-threshold error diagnosis on a frozen, already-used test subset.

Uses score-ordered, class-aware, one-to-one IoU matching across ALL ground truth
before stratifying by GT size. Custom size bins are measured after scaling the
long image side to 640. This is not COCO AP, independent testing, or training.
"""
import argparse
from collections import Counter, defaultdict
import csv
import html
import json
import math
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from workflow import read_labels, sha256, write_json

NAMES = {0: 'hat', 1: 'person'}
BINS = ('tiny', 'small', 'medium', 'large')
CONF = 0.25
MATCH_IOU = 0.5
POLICY = {
    'scope': 'post_hoc_error_diagnosis_on_used_original_test_subset',
    'mode': 'ordinary_full_image', 'confidence': CONF, 'input_size': 640,
    'matching_iou': MATCH_IOU, 'matching': 'score_ordered_class_aware_one_to_one',
    'nms_iou': 0.7, 'max_det': 300, 'device': 'cpu', 'batch': 1,
    'difficult_policy': 'include_as_ordinary',
    'size_definition': 'sqrt(GT box area) after scaling longest original image side to 640',
    'size_bins': {'tiny': '[0,16)', 'small': '[16,32)', 'medium': '[32,96)', 'large': '[96,infinity)'},
    'size_metrics': 'Recall by GT size; no per-size Precision because FP has no GT size',
    'not_comparable_to_prior_map_report': 'fixed confidence 0.25; micro P/R; custom matcher; no AP computed',
}


def overlap(a, b):
    a = np.asarray(a, dtype=float).reshape(-1, 4)
    b = np.asarray(b, dtype=float).reshape(-1, 4)
    lo = np.maximum(a[:, None, :2], b[None, :, :2])
    hi = np.minimum(a[:, None, 2:], b[None, :, 2:])
    area = np.maximum(hi - lo, 0).prod(axis=2)
    aa = np.maximum(a[:, 2:] - a[:, :2], 0).prod(axis=1)
    bb = np.maximum(b[:, 2:] - b[:, :2], 0).prod(axis=1)
    return np.divide(area, aa[:, None] + bb[None, :] - area,
                     out=np.zeros_like(area), where=(aa[:, None] + bb[None, :] - area) > 0)


def match(gt, pred, threshold=MATCH_IOU):
    """gt: xyxy,class; pred: xyxy,score,class. Match before size filtering."""
    gt = np.asarray(gt, dtype=float).reshape(-1, 5)
    pred = np.asarray(pred, dtype=float).reshape(-1, 6)
    iou = overlap(pred[:, :4], gt[:, :4])
    used, pairs = set(), []
    for p in np.argsort(-pred[:, 4], kind='stable'):
        available = [g for g in range(len(gt)) if g not in used and gt[g, 4] == pred[p, 5]]
        if available:
            g = max(available, key=lambda k: (iou[p, k], -k))
            if iou[p, g] >= threshold:
                used.add(g)
                pairs.append((int(p), int(g), float(iou[p, g])))
    found = {p for p, _, _ in pairs}
    return pairs, [g for g in range(len(gt)) if g not in used], [p for p in range(len(pred)) if p not in found]


def size_bin(box, width, height):
    side = math.sqrt(max(0, box[2]-box[0]) * max(0, box[3]-box[1])) * 640 / max(width, height)
    return BINS[0 if side < 16 else 1 if side < 32 else 2 if side < 96 else 3], side


def ground_truth(label, width, height):
    return np.asarray([
        [(x-w/2)*width, (y-h/2)*height, (x+w/2)*width, (y+h/2)*height, c]
        for c, x, y, w, h in read_labels(label)
    ], dtype=float).reshape(-1, 5)


def verify(root, rows):
    names = [r['image'] for r in rows]
    if len(names) != len(set(names)) or any(Path(n).name != n for n in names):
        raise ValueError('Manifest contains duplicate/unsafe image names')
    actual = {p.name for p in (root/'images/test').iterdir() if p.suffix.lower() in {'.jpg', '.jpeg', '.png'}}
    if actual != set(names):
        raise ValueError('Image directory does not match frozen manifest')
    for r in rows:
        image, label = root/'images/test'/r['image'], root/'labels/test'/Path(r['image']).with_suffix('.txt')
        if sha256(image) != r['image_sha256'] or sha256(label) != r['label_sha256']:
            raise ValueError('Image/label changed: ' + r['image'])


def write_csv(path, rows):
    if not rows:
        raise ValueError('Cannot write an unspecified empty table')
    with Path(path).open('w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)


def ratio(a, b):
    return a / b if b else None


def review_gallery(output, root, cases):
    """Local-only inspection; annotations remain unchanged."""
    chosen = []
    rankings = [
        ('tiny_misses', lambda x: (x['tiny_fn'], x['fn'])),
        ('small_misses', lambda x: (x['small_fn'], x['fn'])),
        ('false_positives', lambda x: (x['fp'], x['fn'])),
        ('class_candidates', lambda x: (x['wrong_class_candidates'], x['fp'])),
    ]
    for kind, key in rankings:
        added = 0
        for case in sorted(cases, key=lambda c: (*key(c), c['image']), reverse=True):
            if case['image'] in {x['image'] for x in chosen} or key(case)[0] == 0:
                continue
            chosen.append({**case, 'selection': kind}); added += 1
            if added == 5: break
    directory = output/'review'; directory.mkdir()
    cards, review_rows = [], []
    for k, case in enumerate(chosen, 1):
        with Image.open(root/'images/test'/case['image']) as im:
            original = im.convert('RGB')
        gt_image, pred_image = original.copy(), original.copy()
        dg, dp = ImageDraw.Draw(gt_image), ImageDraw.Draw(pred_image)
        missed = set(case['missed_gt']); bad = set(case['false_predictions'])
        for i, g in enumerate(case['gt']):
            color = '#e11d48' if i in missed else '#2563eb'
            dg.rectangle(g[:4], outline=color, width=3)
            dg.text((g[0], max(0,g[1]-12)), f'G{i} {NAMES[int(g[4])]}'+(' FN' if i in missed else ''), fill=color)
        for i, p in enumerate(case['pred']):
            color = '#e11d48' if i in bad else '#15803d'
            dp.rectangle(p[:4], outline=color, width=3)
            dp.text((p[0], max(0,p[1]-12)), f'P{i} {NAMES[int(p[5])]} {p[4]:.2f}'+(' FP' if i in bad else ''), fill=color)
        stem = f'{k:02d}'
        for tag, im in [('original',original), ('gt',gt_image), ('pred',pred_image)]:
            im.save(directory/f'{stem}_{tag}.jpg', quality=92)
        review_rows.append({'case':k,'image':case['image'],'selection':case['selection'],
                            'annotation_correct':'','error_type':'','notes':''})
        cards.append(f'''<article><h2>{k:02d} · {html.escape(case['image'])}</h2>
<p>筛选依据：{case['selection']}；标注 {len(case['gt'])}，检出 {case['tp']}，漏检 {case['fn']}，疑似误检 {case['fp']}。这些是程序判定，待核查。</p>
<div class="images">{''.join(f'<figure><figcaption>{label}</figcaption><a href="{stem}_{tag}.jpg" target="_blank"><img loading="lazy" src="{stem}_{tag}.jpg"></a></figure>' for tag,label in [('original','原图'),('gt','标注：蓝框；漏检标注：红框'),('pred','预测：检出绿框；疑似误检红框')])}</div></article>''')
    write_csv(output/'manual_review.csv', review_rows)
    page='''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>第一步：检测错误核查</title><style>
body{font:16px/1.6 system-ui,sans-serif;background:#f3f5f8;color:#182333;margin:0;padding:28px;max-width:1800px}h1{margin-top:0}article{background:white;border:1px solid #ddd;border-radius:12px;padding:18px;margin:24px 0}.images{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:12px}figure{margin:0}img{width:100%;height:auto}figcaption{font-weight:600;padding:8px 0}@media(max-width:900px){.images{grid-template-columns:1fr}}
</style><h1>检测错误核查 · 普通模式 / conf=0.25 / IoU=0.5</h1>
<p>本页只在本地查看，不上传原始数据图片。点击图片可放大。按错误数量筛选的案例不是随机代表样本，汇总统计使用全部 474 张图片。</p>
<p>先确认红色标注框是否正确、目标是否可辨认，再看模型输出。疑似误检也可能是漏标，模糊目标可以写“不确定”，不要为了让模型正确而修改答案。</p>
<p>请在上一级 manual_review.csv 填写 annotation_correct（是/否/不确定）、error_type（微小/遮挡/模糊/分类/定位/背景/漏标/其他）和 notes。不要改写原始标签。</p>'''+''.join(cards)+'</html>'
    (directory/'index.html').write_text(page, encoding='utf-8')
    return len(chosen)


def run(args):
    root, manifest, model_path, output = [Path(p).expanduser().resolve() for p in [args.data_root,args.manifest,args.model,args.output]]
    if output.exists(): raise FileExistsError('Refusing to overwrite frozen analysis: '+str(output))
    with manifest.open(newline='',encoding='utf-8') as f: rows=list(csv.DictReader(f))
    if len(rows)!=474: raise ValueError('This protocol expects the reviewed 474-image subset')
    verify(root, rows)
    model_hash, manifest_hash = sha256(model_path), sha256(manifest)
    if model_hash != args.model_sha256: raise ValueError('Checkpoint SHA256 mismatch')
    output.mkdir(parents=True)
    protocol={**POLICY,'model_sha256':model_hash,'manifest_sha256':manifest_hash,'images':len(rows),'analysis_script_sha256':sha256(__file__)}
    write_json(output/'protocol.json',protocol)
    import platform,torch,ultralytics
    from ultralytics import YOLO
    model=YOLO(str(model_path))
    if model.task!='detect' or model.names!=NAMES: raise ValueError('Wrong task/class mapping')
    objects, images, cases = [], [], []
    categories=defaultdict(Counter); sizes=defaultdict(Counter); per_class_size=defaultdict(Counter)
    totals=Counter(); fp_reasons=Counter(); fn_reasons=Counter()
    for start in range(0,len(rows),4):
        chunk=rows[start:start+4]; inputs=[]; gts=[]
        for row in chunk:
            with Image.open(root/'images/test'/row['image']) as im:
                if im.getexif().get(274,1) not in (0,1): raise ValueError('Unexpected EXIF orientation')
                rgb=im.convert('RGB');inputs.append(rgb)
                gts.append(ground_truth(root/'labels/test'/Path(row['image']).with_suffix('.txt'),*rgb.size))
        predictions=[model.predict(source=im,imgsz=640,conf=CONF,iou=0.7,max_det=300,
                                       device='cpu',batch=1,rect=True,augment=False,verbose=False)[0]
                     for im in inputs]
        if len(predictions)!=len(chunk): raise ValueError('Prediction count mismatch')
        for row,im,gt,result in zip(chunk,inputs,gts,predictions):
            pred=result.boxes.data.cpu().numpy().astype(float).reshape(-1,6)
            pairs, missed, false=match(gt,pred)
            matched_gt={g:p for p,g,_ in pairs}; matched_pred={p:g for p,g,_ in pairs}
            ious=overlap(pred[:,:4],gt[:,:4]); case_sizes=Counter(); wrong=0
            for g,box in enumerate(gt):
                cls=int(box[4]); bucket,side=size_bin(box,*im.size); ok=g in matched_gt
                categories[cls]['gt']+=1; categories[cls]['tp']+=int(ok);categories[cls]['fn']+=int(not ok)
                sizes[bucket]['gt']+=1;sizes[bucket]['tp']+=int(ok);sizes[bucket]['fn']+=int(not ok)
                per_class_size[(cls,bucket)]['gt']+=1;per_class_size[(cls,bucket)]['tp']+=int(ok);per_class_size[(cls,bucket)]['fn']+=int(not ok)
                reason='matched'
                if not ok:
                    case_sizes[bucket]+=1
                    other=[p for p in range(len(pred)) if pred[p,5]!=cls and ious[p,g]>=MATCH_IOU]
                    near=[p for p in range(len(pred)) if pred[p,5]==cls and 0.1<=ious[p,g]<MATCH_IOU]
                    reason='wrong_class_candidate' if other else 'localization_candidate' if near else 'no_matching_detection'
                    fn_reasons[reason]+=1
                objects.append({'image':row['image'],'gt_index':g,'class':NAMES[cls],'size_bin':bucket,'scaled_sqrt_area':side,'status':'TP' if ok else 'FN','prediction_index':matched_gt.get(g,''),'reason':reason})
            for p,box in enumerate(pred):
                cls=int(box[5]);categories[cls]['predictions']+=1
                if p not in matched_pred:
                    categories[cls]['fp']+=1
                    same=[g for g in range(len(gt)) if gt[g,4]==cls and ious[p,g]>=MATCH_IOU]
                    other=[g for g in range(len(gt)) if gt[g,4]!=cls and ious[p,g]>=MATCH_IOU]
                    near=[g for g in range(len(gt)) if gt[g,4]==cls and 0.1<=ious[p,g]<MATCH_IOU]
                    reason='duplicate_candidate' if same else 'wrong_class_candidate' if other else 'localization_candidate' if near else 'background_or_missing_label_candidate'
                    fp_reasons[reason]+=1;wrong+=int(reason=='wrong_class_candidate')
            tp,fn,fp=len(pairs),len(missed),len(false)
            totals.update(gt=len(gt),predictions=len(pred),tp=tp,fn=fn,fp=fp)
            images.append({'image':row['image'],'width':im.width,'height':im.height,'gt':len(gt),'predictions':len(pred),'tp':tp,'fn':fn,'fp':fp,'tiny_fn':case_sizes['tiny'],'small_fn':case_sizes['small'],'wrong_class_candidates':wrong,'at_max_det':len(pred)>=300})
            cases.append({**images[-1],'gt':gt.tolist(),'pred':pred.tolist(),'matches':pairs,'missed_gt':missed,'false_predictions':false})
        print(f'Analyzed {min(start+4,len(rows))}/{len(rows)}',flush=True)
    verify(root,rows)
    if sha256(model_path)!=model_hash or sha256(manifest)!=manifest_hash:raise ValueError('Identity changed during analysis')
    def values(counter):
        d=dict(counter);d['recall']=ratio(counter['tp'],counter['gt'])
        if 'predictions' in counter:d['precision']=ratio(counter['tp'],counter['predictions'])
        return d
    summary={'protocol':protocol,'overall':values(totals),'by_class':{NAMES[c]:values(categories[c]) for c in NAMES},
             'by_gt_size':{b:values(sizes[b]) for b in BINS},
             'by_class_and_gt_size':{NAMES[c]:{b:values(per_class_size[(c,b)]) for b in BINS} for c in NAMES},
             'fp_diagnostic_candidates':dict(fp_reasons),'fn_diagnostic_candidates':dict(fn_reasons),
             'images_at_max_det':sum(r['at_max_det'] for r in images),
             'environment':{'python':platform.python_version(),'torch':torch.__version__,'ultralytics':ultralytics.__version__},
             'model_and_data_verified_unchanged':True,
             'limitations':['Used post-hoc subset; scene independence unproven.','Label-relative errors require visual review; missing/wrong annotations can affect counts.','Custom scale bins and matching; not COCO AP or prior max-F1 P/R.','No annotation of occlusion/blur; those causes cannot be inferred from size alone.']}
    assert totals['tp']+totals['fn']==totals['gt'] and totals['tp']+totals['fp']==totals['predictions']
    assert sum(sizes[b]['gt'] for b in BINS)==totals['gt']
    write_csv(output/'ground_truth_outcomes.csv',objects);write_csv(output/'per_image.csv',images)
    write_json(output/'predictions_and_matches.json',cases)
    summary['review_cases']=review_gallery(output,root,cases)
    write_json(output/'summary.json',summary)
    print(json.dumps(summary,ensure_ascii=False),flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data-root',required=True);p.add_argument('--manifest',required=True)
    p.add_argument('--model',required=True);p.add_argument('--model-sha256',required=True)
    p.add_argument('--output',required=True)
    run(p.parse_args())


if __name__=='__main__':main()
