"""Five frozen trained models: COCO area AP, fixed-conf errors and CPU latency.

Never trains, changes labels, writes original caches, or selects a checkpoint.
pip install --target <output>/metrics_dependencies --no-deps pycocotools==2.0.11
"""
from pathlib import Path
import argparse
import collections
import contextlib
import csv
from datetime import datetime, timezone
import hashlib
import io
import json
import os
import platform
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'error_analysis/final_five_model_evaluation_2026-10-28'
DATA = ROOT / 'datasets/helmet_scene_isolated_v1'
OUT.mkdir(parents=True, exist_ok=True)
for folder in ['settings', 'matplotlib']:
    (OUT / folder).mkdir(exist_ok=True)
os.environ.setdefault('YOLO_CONFIG_DIR', str(OUT / 'settings'))
os.environ.setdefault('MPLCONFIGDIR', str(OUT / 'matplotlib'))
sys.path[:0] = [str(ROOT), str(OUT / 'metrics_dependencies')]
import numpy as np
from PIL import Image
from analyze_detection_errors import ground_truth, match
from workflow import sha256, verify_dataset

NAMES = {0: 'hat', 1: 'person'}
SHORT = [('lt8', 0, 8), ('8to16', 8, 16), ('16to32', 16, 32), ('ge32', 32, float('inf'))]
AREA = [('small', 0, 32**2), ('medium', 32**2, 96**2), ('large', 96**2, float('inf'))]

def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))

def write(path, obj):
    Path(path).write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False) + '\n')

def once(path, obj):
    if Path(path).exists():
        raise FileExistsError(path)
    write(path, obj)

def now():
    return datetime.now(timezone.utc).isoformat()

def rows(split):
    return sorted([r for r in csv.DictReader((DATA / 'split_manifest.csv').open()) if r['split'] == split], key=lambda x: x['image'])

def protocol():
    p = read(OUT / 'protocol.json')
    assert sha256(__file__) == p['evaluation_script_sha256']
    assert sha256(DATA / 'split_manifest.csv') == p['dataset']['manifest_sha256']
    for rel, expected in p['protected_sources'].items():
        assert sha256(ROOT / rel) == expected, rel
    for rel, expected in p['metric_library_files'].items():
        assert sha256(OUT / rel) == expected, rel
    return p

def prepare():
    import torch, ultralytics
    from pycocotools.cocoeval import COCOeval
    from importlib.metadata import distribution
    dist = distribution('pycocotools')
    assert dist.version == '2.0.11'
    audit = read(ROOT / 'reports/completed_experiments_2026-10-28/verification.json')
    dataset = verify_dataset(DATA / 'data.yaml')
    assert dataset['images'] == 7507
    records = audit['records']
    frozen = [Path(__file__).relative_to(ROOT), Path('helmet_cbam.py'), Path('analyze_detection_errors.py'), Path('workflow.py')]
    for r in records:
        for name in ['weights/best.pt', 'weights/last.pt', 'results.csv', 'effective_args.json', 'initialization.json', 'input_identity.json']:
            frozen.append(Path(r['source']) / name)
    val = rows('val')
    indices = np.linspace(0, len(val)-1, 32, dtype=int)
    library = {str(p.relative_to(OUT)):sha256(p) for p in (OUT/'metrics_dependencies/pycocotools').iterdir() if p.is_file() and p.suffix in ['.py','.so']}
    for r in records:
        assert sha256(ROOT/r['source']/'weights/best.pt') == r['best_sha256']
    p = {
        'created_at_utc': now(), 'scope': 'post_hoc_completion_on_existing_previously_explored_test_set',
        'prior_knowledge': 'Five model overall validation and GPU test scores were already known. Freeze this completion protocol before size/fixed-conf analysis and latency; not a preregistered blind evaluation.',
        'dataset': dataset, 'models': records, 'fixed_confidence': .25, 'matching_iou': .5,
        'prediction': {'imgsz':640,'conf':.001,'iou':.7,'max_det':300,'rect':True,'half':False,'batch':1,'device':'cpu','augment':False,'save':False,'verbose':False},
        'size_ap': {'implementation':'pycocotools2.0.11 official COCOeval bbox','area':'original-image GT box width*height, no input-resize scaling','ranges':'official inclusive [0,32²], [32²,96²], [96²,1e10]; exact boundary belongs to adjacent ranges','maxDets':[1,10,300], 'iou':'.50:.05:.95','recall_grid':'101 points','names':'COCO bbox AP@300, not standard COCO AP@100 or Ultralytics GPU AP','ignore':'official area ignore and unmatched-detection-area rules; never filter GT before matching'},
        'fixed_metrics': {'matching':'existing project stable score-ordered class-aware one-to-one IoU>=.5; all GT before GT-size grouping','overall':'micro TP/FP/FN/P/R/F1 at conf=.25','sizes':'disjoint original-area groups: <32², [32²,96²), >=96²; no size-specific FP/P because unmatched detections have no GT size','short_side':'GT short side * 640/max(original_width,original_height); bins <8,8–16,16–32,>=32; cumulative helmet<16 as tiny-head proxy, not measured distance','labels':'hat=helmeted head, person=bare head; difficult treated as ordinary; unmatched predictions can include missing labels'},
        'benchmark': {'split':'val','images':[val[int(i)]['image'] for i in indices], 'images_per_model':32,'rounds':5,'warmup_calls_per_model':5,'threads':8,'interop_threads':1,'opencv_threads':1,'device':'cpu','precision':'FP32','batch':1,'conf':.25,'order':'rotate five model order once per round; identical seeded shuffled image order for each model within round','seed':20261028,'core':'resident model: read/decode image, RGB conversion, preprocess, forward, NMS, copy boxes and class counting','rendered':'core plus prediction drawing and PNG encoding in memory (compression3)','excluded':'model import/loading, UI/browser/network, disk output and concurrent user work; filesystem caches are warm','stability':'report 5 round means, median, P95 and coefficient of variation; not multi-seed training stability','cold_start':'model constructor loading time separately, not whole application cold-start'},
        'uncertainty': {'resamples':2000,'seed':20261028,'unit':'retrieved-risk group_id, paired model results; groups may not be exhaustive true independent scenes','quantities':'microF1 and helmet recall differences; uncertainty conditional on single trained models, not repeated-training evidence'},
        'selection_policy': 'Existing P2 remains the reference deployment candidate. Describe Pareto tradeoffs across fixed-conf F1, helmet precision/recall, size metrics and CPU latency; do not claim a tiny test AP gain guarantees a better system. No new threshold search or checkpoint selection on test; any recommendation is retrospective, not a new blind proof.',
        'environment': {'python':platform.python_version(),'platform':platform.platform(),'machine':platform.machine(),'torch':torch.__version__,'ultralytics':ultralytics.__version__,'numpy':np.__version__,'pycocotools':dist.version},
        'metric_library_files':library,'protected_sources':{str(x):sha256(ROOT/x) for x in frozen},
        'evaluation_script_sha256':sha256(__file__), 'optimizer_steps':0,
        'references':['https://github.com/cocodataset/cocoapi/blob/master/PythonAPI/pycocotools/cocoeval.py'],
    }
    once(OUT/'protocol.json', p)
    print('PROTOCOL_FROZEN',len(records),'models, 508 test images; zero training',flush=True)

def load_model(record):
    import torch
    from ultralytics import YOLO
    from ultralytics.utils import LOGGER
    from helmet_cbam import register_cbam
    register_cbam(); LOGGER.setLevel(40)
    torch.set_num_threads(8)
    m = YOLO(str(ROOT / record['source'] / 'weights/best.pt'))
    assert m.names == NAMES and m.task == 'detect'
    return m

def image(row):
    assert sha256(DATA/row['image']) == row['image_sha256']
    assert sha256(DATA/row['label']) == row['label_sha256']
    with Image.open(DATA/row['image']) as im:
        assert im.getexif().get(274,1) == 1
        return im.convert('RGB')

def sanitize(pred):
    a=np.asarray(pred,dtype=float).reshape(-1,6)
    assert np.isfinite(a).all() and np.isin(a[:,5],[0,1]).all()
    assert ((a[:,4]>=0)&(a[:,4]<=1)).all()
    assert (a[:,2:4]>=a[:,:2]).all()
    valid=(a[:,2:4]>a[:,:2]).all(axis=1)
    return a[valid], int((~valid).sum())

def infer():
    import torch
    torch.set_num_interop_threads(1)
    p=protocol(); rs=rows('test'); assert len(rs)==508
    once(OUT/'started.json', {'time':now(),'protocol_sha256':sha256(OUT/'protocol.json')})
    metadata=[]; annotations=[]; aid=1
    for iid,row in enumerate(rs,1):
        im=image(row)
        gt=ground_truth(DATA/row['label'],im.width,im.height)
        assert (gt[:,2:4]>gt[:,:2]).all()
        metadata.append({'id':iid,'file_name':row['image'],'width':im.width,'height':im.height,'group_id':row['group_id'],'gt':gt.tolist()})
        for g in gt:
            x1,y1,x2,y2,c=g
            annotations.append({'id':aid,'image_id':iid,'category_id':int(c),'bbox':[x1,y1,x2-x1,y2-y1],'area':float((x2-x1)*(y2-y1)),'iscrowd':0})
            aid+=1
    once(OUT/'ground_truth_coco.json',{'info':{},'images':[{k:v for k,v in x.items() if k not in ['gt','group_id']} for x in metadata],'annotations':annotations,'categories':[{'id':c,'name':n} for c,n in NAMES.items()]})
    once(OUT/'image_metadata.json',metadata)
    checks=[]
    for r in p['models']:
        m=load_model(r); cache=OUT/f'{r["id"]}_predictions.jsonl'; count_cap=0
        with cache.open('x') as f:
            for i,row in enumerate(rs):
                im=image(row)
                pred,zero=sanitize(m.predict(im,**p['prediction'])[0].boxes.data.cpu().numpy())
                assert torch.get_num_threads()==8
                count_cap += int(len(pred)==300)
                f.write(json.dumps({'image':row['image'],'width':im.width,'height':im.height,'pred':pred.tolist(),'discarded_zero_area':zero,'protocol_sha256':sha256(OUT/'protocol.json')},separators=(',',':'))+'\n')
                if (i+1)%100==0 or i+1==len(rs):
                    f.flush(); print('INFER',r['label'],i+1,'/508',flush=True)
        cached=[json.loads(s) for s in cache.read_text().splitlines()]
        max_diff=0.
        # Confirm AP candidate filtering reproduces direct deployment-conf inference.
        for idx in np.linspace(0,len(rs)-1,12,dtype=int):
            actual,_=sanitize(m.predict(image(rs[int(idx)]),**{**p['prediction'],'conf':.25})[0].boxes.data.cpu().numpy())
            old=np.asarray(cached[int(idx)]['pred']).reshape(-1,6);old=old[old[:,4]>=.25]
            assert old.shape==actual.shape
            np.testing.assert_allclose(actual,old,rtol=0,atol=1e-5)
            if len(old):max_diff=max(max_diff,float(np.abs(old-actual).max()))
        checks.append({'id':r['id'],'prediction_sha256':sha256(cache),'candidate_cap_images':count_cap,'direct_conf25_check_images':12,'max_abs_prediction_difference':max_diff})
        del m
    once(OUT/'prediction_verification.json',checks)

def rates(tp,fp,gt):
    return {'gt':int(gt),'tp':int(tp),'fp':int(fp),'fn':int(gt-tp),
            'precision':float(tp/(tp+fp)) if tp+fp else 0.,'recall':float(tp/gt) if gt else None,
            'f1':float(2*tp/(gt+tp+fp)) if gt+tp+fp else 0.}

def detail(gt,pred,w,h):
    gt=np.asarray(gt).reshape(-1,5);pred=np.asarray(pred).reshape(-1,6)
    pairs,misses,unmatched=match(gt,pred,.5); found={g for _,g,_ in pairs}
    out={'overall':rates(len(pairs),len(unmatched),len(gt)),'classes':{},'area':{},'short':{},'helmet_short':{},'matched_gt':sorted(found)}
    for c in [0,1]:
        ids=set(np.flatnonzero(gt[:,4]==c))
        out['classes'][str(c)]=rates(len(ids&found),sum(pred[j,5]==c for j in unmatched),len(ids))
    area=(gt[:,2]-gt[:,0])*(gt[:,3]-gt[:,1]);short=np.minimum(gt[:,2]-gt[:,0],gt[:,3]-gt[:,1])*640/max(w,h)
    def count(ids):
        ids=set(ids);return {'gt':len(ids),'tp':len(ids&found),'fn':len(ids-found),'recall':len(ids&found)/len(ids) if ids else None}
    for name,lo,hi in AREA:
        ids=np.flatnonzero((area>=lo)&(area<hi))
        out['area'][name]={'all':count(ids),**{str(c):count([j for j in ids if gt[j,4]==c]) for c in [0,1]}}
    for name,lo,hi in SHORT:
        ids=np.flatnonzero((short>=lo)&(short<hi));out['short'][name]=count(ids)
        out['helmet_short'][name]=count([j for j in ids if gt[j,4]==0])
    out['helmet_short']['lt16']=count(np.flatnonzero((short<16)&(gt[:,4]==0)))
    return out

def merge_details(details):
    def total(values):return rates(sum(x['tp'] for x in values),sum(x['fp'] for x in values),sum(x['gt'] for x in values))
    def recalls(values):
        gt=sum(x['gt'] for x in values);tp=sum(x['tp'] for x in values)
        return {'gt':gt,'tp':tp,'fn':gt-tp,'recall':tp/gt if gt else None}
    return {'overall':total([d['overall'] for d in details]),'classes':{str(c):total([d['classes'][str(c)] for d in details]) for c in [0,1]},
            'area':{name:{c:recalls([d['area'][name][c] for d in details]) for c in ['all','0','1']} for name,_,_ in AREA},
            'short':{name:recalls([d['short'][name] for d in details]) for name,_,_ in SHORT},
            'helmet_short':{name:recalls([d['helmet_short'][name] for d in details]) for name in [x[0] for x in SHORT]+['lt16']}}

def coco_ap(ground, predictions):
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval
    with contextlib.redirect_stdout(io.StringIO()):
        gt=COCO();gt.dataset=ground;gt.createIndex()
        dt=gt.loadRes(predictions) if predictions else COCO()
        if not predictions:
            dt.dataset={**ground,'annotations':[]};dt.createIndex()
        ce=COCOeval(gt,dt,'bbox');ce.params.maxDets=[1,10,300];ce.evaluate();ce.accumulate()
    def avg(a):
        a=a[a>-1];return float(a.mean()) if a.size else None
    result={}
    for ai,label in enumerate(ce.params.areaRngLbl):
        pr=ce.eval['precision'][:,:,:,ai,-1];rc=ce.eval['recall'][:,:,ai,-1]
        result[label]={'AP50_95':avg(pr),'AP50':avg(pr[0]),'AR50_95':avg(rc),'classes':{}}
        for ki,cid in enumerate(ce.params.catIds):
            result[label]['classes'][str(cid)]={'AP50_95':avg(pr[:,:,ki]),'AP50':avg(pr[0,:,ki]),'AR50_95':avg(rc[:,ki])}
    return result

def score():
    p=protocol();meta=read(OUT/'image_metadata.json');ground=read(OUT/'ground_truth_coco.json')
    per_image={};result={};checks=read(OUT/'prediction_verification.json')
    for record,check in zip(p['models'],checks):
        mid=record['id'];cache=OUT/f'{mid}_predictions.jsonl'
        assert sha256(cache)==check['prediction_sha256']
        cached=[json.loads(s) for s in cache.read_text().splitlines()];assert len(cached)==508
        details=[];detections=[]
        for info,row in zip(meta,cached):
            assert row['image']==info['file_name']
            a=np.asarray(row['pred']).reshape(-1,6)
            for x1,y1,x2,y2,conf,cls in a:
                detections.append({'image_id':info['id'],'category_id':int(cls),'bbox':[float(x1),float(y1),float(x2-x1),float(y2-y1)],'score':float(conf)})
            d=detail(info['gt'],a[a[:,4]>=.25],info['width'],info['height']);details.append(d)
        per_image[mid]=details
        result[mid]={'label':record['label'],'fixed':merge_details(details),'coco_bbox_ap300':coco_ap(ground,detections),'prediction_sha256':sha256(cache)}
        print('SCORE',record['label'],'fixed',result[mid]['fixed']['overall'],'areaAP', {k:v['AP50_95'] for k,v in result[mid]['coco_bbox_ap300'].items()},flush=True)
    once(OUT/'per_image_details.json',{'images':meta,'models':per_image})
    once(OUT/'metrics.json',result)
    # Risk-group paired bootstrap; conditional on these five fixed trained checkpoints.
    groups=collections.defaultdict(lambda:np.zeros((5,2,3),dtype=int))
    for j,info in enumerate(meta):
        for mi,r in enumerate(p['models']):
            d=per_image[r['id']][j]
            for ci,x in enumerate([d['overall'],d['classes']['0']]):groups[info['group_id']][mi,ci]+=[x['tp'],x['fp'],x['gt']]
    array=np.stack(list(groups.values()));rng=np.random.default_rng(20261028);pairs=[(2,0),(3,2),(1,0),(4,2)];samples=collections.defaultdict(list)
    for _ in range(2000):
        counts=array[rng.integers(0,len(array),len(array))].sum(axis=0)
        for child,parent in pairs:
            a,b=counts[child],counts[parent]
            f1=lambda x:2*x[0]/(x[2]+x[0]+x[1]) if (x[2]+x[0]+x[1]) else 0
            recall=lambda x:x[0]/x[2] if x[2] else 0
            samples[(child,parent)].append([f1(a[0])-f1(b[0]),recall(a[1])-recall(b[1])])
    bootstrap={}
    for child,parent in pairs:
        intervals=np.percentile(samples[(child,parent)],[2.5,97.5],axis=0)
        mid,pid=p['models'][child]['id'],p['models'][parent]['id']
        gained=sum(len(set(a['matched_gt'])-set(b['matched_gt'])) for a,b in zip(per_image[mid],per_image[pid]))
        lost=sum(len(set(b['matched_gt'])-set(a['matched_gt'])) for a,b in zip(per_image[mid],per_image[pid]))
        bootstrap[f'{mid}_minus_{pid}']={'micro_f1_delta95':intervals[:,0].tolist(),'helmet_recall_delta95':intervals[:,1].tolist(),'gained_gt':gained,'lost_gt':lost}
    once(OUT/'paired_uncertainty.json',{'risk_groups':len(array),'resamples':2000,'comparisons':bootstrap})

def benchmark():
    import torch,cv2
    torch.set_num_interop_threads(1);cv2.setNumThreads(1)
    p=protocol();selected={r['image']:r for r in rows('val')};bs=p['benchmark'];images=[selected[x] for x in bs['images']]
    models={};load_times={};kwargs={**p['prediction'],'conf':.25}
    for r in p['models']:
        start=time.perf_counter_ns();m=load_model(r);load_times[r['id']]=(time.perf_counter_ns()-start)/1e6
        for _ in range(5):m.predict(image(images[0]),**kwargs)
        assert torch.get_num_threads()==8;models[r['id']]=m
    raw=[];rng=np.random.default_rng(20261028);load_before=os.getloadavg()
    for repeat in range(5):
        order=np.roll(list(models),-repeat);indices=rng.permutation(len(images))
        for slot,mid in enumerate(order):
            model=models[mid]
            for idx in indices:
                row=images[int(idx)]
                # Hashes validated outside timer. Image I/O is inside timer.
                assert sha256(DATA/row['image'])==row['image_sha256']
                start=time.perf_counter_ns()
                with Image.open(DATA/row['image']) as source:im=source.convert('RGB')
                result=model.predict(im,**kwargs)[0]
                pred,zero=sanitize(result.boxes.data.cpu().numpy())
                counts={str(c):int((pred[:,5]==c).sum()) for c in [0,1]}
                core=(time.perf_counter_ns()-start)/1e6
                annotated=result.plot(line_width=1,font_size=10,pil=False,labels=True,conf=True)
                ok,encoded=cv2.imencode('.png',annotated,[cv2.IMWRITE_PNG_COMPRESSION,3]);assert ok
                rendered=(time.perf_counter_ns()-start)/1e6
                assert torch.get_num_threads()==8 and cv2.getNumThreads()==1
                raw.append({'model':str(mid),'round':repeat+1,'slot':slot+1,'image':row['image'],'core_ms':core,'rendered_ms':rendered,'detections':len(pred),'class_counts':counts,'png_bytes':int(encoded.nbytes)})
            print('BENCHMARK',repeat+1,mid,flush=True)
    result={}
    for mid in models:
        rr=[r for r in raw if r['model']==mid];record={'calls':len(rr),'model_constructor_ms':load_times[mid]}
        for key in ['core_ms','rendered_ms']:
            a=np.asarray([r[key] for r in rr]);means=[float(np.mean([r[key] for r in rr if r['round']==k])) for k in range(1,6)]
            record[key]={'mean':float(a.mean()),'median':float(np.median(a)),'p95':float(np.percentile(a,95)),'cv':float(a.std()/a.mean()),'round_means':means,'round_mean_cv':float(np.std(means)/np.mean(means))}
        result[mid]=record
    once(OUT/'benchmark.json',{'results':result,'raw':raw,'threads':torch.get_num_threads(),'opencv_threads':cv2.getNumThreads(),'cpu_load_before':load_before,'cpu_load_after':os.getloadavg(),'finished_at_utc':now()})
    print('BENCHMARK_COMPLETE',json.dumps(result,ensure_ascii=False),flush=True)

def verify():
    p=protocol();dataset=verify_dataset(DATA/'data.yaml');assert dataset==p['dataset']
    metrics=read(OUT/'metrics.json');bench=read(OUT/'benchmark.json')
    for r in p['models']:
        m=metrics[r['id']]['fixed'];assert m['overall']['gt']==4875
        assert sum(x['gt'] for x in m['classes'].values())==4875
        assert sum(x['all']['gt'] for x in m['area'].values())==4875
        assert sum(x['gt'] for x in m['short'].values())==4875
        assert bench['results'][r['id']]['calls']==160
    original=['protocol.json','metrics.json','paired_uncertainty.json','benchmark.json','ground_truth_coco.json','image_metadata.json','prediction_verification.json','per_image_details.json']
    once(OUT/'verification.json',{'status':'passed','time':now(),'all7507_image_label_hashes_verified':True,'training_code_and_weights_unchanged':True,'optimizer_steps':0,'protocol_sha256':sha256(OUT/'protocol.json'),'output_sha256':{n:sha256(OUT/n) for n in original}})
    print('FINAL_VERIFICATION_PASSED',flush=True)

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('stage',choices=['prepare','infer','score','benchmark','verify']);args=parser.parse_args()
    globals()[args.stage]()
