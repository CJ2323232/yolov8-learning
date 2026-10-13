"""Screen the ORIGINAL test split against original train/val, then evaluate an unchanged model.

Post-hoc sensitivity analysis, not a newly collected external test set or proof of scene independence.
No train() calls. Candidate thresholds are fixed before evaluation and never use detector predictions.
"""
import argparse
from collections import Counter, defaultdict
import csv
import hashlib
import json
from pathlib import Path
import shutil

import cv2
import numpy as np
from PIL import Image, ImageOps
import yaml

from workflow import IMAGE_EXTENSIONS, RELEASE_SHA256, ROOT, read_labels, sha256, write_json

POLICY = {'phash_bits': 64, 'dhash_bits_each_axis': 64,
          'near_rule': '(p<=8 and (horizontal<=10 or vertical<=10)) or (horizontal<=4 and vertical<=8)',
          'nearest_hash_candidates_each': 10, 'nearest_embedding_candidates': 20,
          'embedding_layers': [4, 6, 8], 'embedding_imgsz': 224,
          'orb_features': 600, 'orb_ratio_test': 0.75, 'ransac_pixels': 4,
          'conservative_scene_min_inliers': 8, 'scene_min_inlier_fraction': .30,
          'scene_min_hull_fraction_each_image': .03}


def write_csv(path, rows, fields=None):
    with Path(path).open('w', encoding='utf-8', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fields or list(rows[0]))
        w.writeheader(); w.writerows(rows)


def read_csv(path):
    with Path(path).open(encoding='utf-8', newline='') as f:
        return list(csv.DictReader(f))


def bits64(bits):
    return np.uint64(sum(int(v) << i for i, v in enumerate(np.asarray(bits).ravel())))


def image_features(path, orb):
    with Image.open(path) as original:
        rgb = ImageOps.exif_transpose(original).convert('RGB')
        pixels = np.asarray(rgb)
        pixel_sha = hashlib.sha256(str(rgb.size).encode()+pixels.tobytes()).hexdigest()
        gray = cv2.cvtColor(pixels, cv2.COLOR_RGB2GRAY)
    coefficients = cv2.dct(cv2.resize(gray, (32,32)).astype(np.float32))[:8,:8]
    p = bits64(coefficients > np.median(coefficients.ravel()[1:]))
    h = cv2.resize(gray, (9,8)); horizontal = bits64(h[:,1:] > h[:,:-1])
    v = cv2.resize(gray, (8,9)); vertical = bits64(v[1:,:] > v[:-1,:])
    scale = min(1.,640/max(gray.shape))
    small = cv2.resize(gray,None,fx=scale,fy=scale) if scale<1 else gray
    keypoints, descriptors = orb.detectAndCompute(small,None)
    points = np.asarray([k.pt for k in keypoints], dtype=np.float32).reshape(-1,2)
    return pixel_sha, p, horizontal, vertical, points, descriptors, small.shape


def geometry(a, b):
    """Conservative local-feature relation; a candidate, not an automatic semantic verdict."""
    points_a, desc_a, shape_a = a
    points_b, desc_b, shape_b = b
    if desc_a is None or desc_b is None or min(len(desc_a),len(desc_b)) < 8:
        return None
    pairs = cv2.BFMatcher(cv2.NORM_HAMMING).knnMatch(desc_a,desc_b,k=2)
    good = sorted((m for pair in pairs if len(pair)==2 for m,n in [pair]
                   if m.distance < POLICY['orb_ratio_test']*n.distance), key=lambda m:m.distance)
    seen, unique = set(), []
    for match in good:
        if match.trainIdx not in seen:
            seen.add(match.trainIdx);unique.append(match)
    if len(unique)<8:
        return None
    x = np.asarray([points_a[m.queryIdx] for m in unique])
    y = np.asarray([points_b[m.trainIdx] for m in unique])
    cv2.setRNGSeed(0)
    transform, mask = cv2.findHomography(x,y,cv2.RANSAC,POLICY['ransac_pixels'])
    if transform is None or mask is None:
        return None
    mask=mask.ravel().astype(bool);inliers=int(mask.sum())
    if inliers<8:
        return None
    coverage_a=cv2.contourArea(cv2.convexHull(x[mask]))/(shape_a[0]*shape_a[1])
    coverage_b=cv2.contourArea(cv2.convexHull(y[mask]))/(shape_b[0]*shape_b[1])
    fraction=inliers/len(unique)
    if fraction < .30 or min(coverage_a,coverage_b)<.03:
        return None
    return {'inliers':inliers,'inlier_fraction':round(fraction,4),
            'coverage_test':round(coverage_a,4),'coverage_reference':round(coverage_b,4)}


def contact_sheets(output, rows, records, legacy):
    """Local review aids only; never publish source dataset photos."""
    from PIL import ImageDraw
    directory=output/'contact_sheets';directory.mkdir()
    for start in range(0,len(rows),12):
        sheet=Image.new('RGB',(1440,1440),'white');draw=ImageDraw.Draw(sheet)
        for k,row in enumerate(rows[start:start+12]):
            test=records[int(row['test_index'])];ref=records[int(row['reference_index'])]
            x,y=(k%2)*720,(k//2)*240
            for offset,record in [(0,test),(350,ref)]:
                with Image.open(legacy/'helmet_dataset'/record['image']) as im:
                    thumb=ImageOps.contain(im.convert('RGB'),(340,190))
                sheet.paste(thumb,(x+offset,y+35))
            draw.text((x+4,y+4),f"{start+k}: {test['image']} | {ref['image']}",fill='black')
            draw.text((x+4,y+225),f"{row['reason']} p={row.get('phash_distance','')} inliers={row.get('inliers','')}",fill='black')
        sheet.save(directory/f'pairs_{start//12:03d}.jpg',quality=90)


def prepare(legacy, output, retrieval_model):
    legacy,output=Path(legacy).resolve(),Path(output).resolve()
    if output.exists():
        raise FileExistsError(f'不会覆盖已冻结的筛查：{output}')
    output.mkdir(parents=True)
    records, hashes, local_features=[],[],[]
    orb=cv2.ORB_create(nfeatures=POLICY['orb_features'],edgeThreshold=15,fastThreshold=10)
    for split in ('train','val','test'):
        directory=legacy/'helmet_dataset/images'/split
        for image in sorted(p for p in directory.iterdir() if p.suffix.lower() in IMAGE_EXTENSIONS):
            label=legacy/'helmet_dataset/labels'/split/(image.stem+'.txt')
            read_labels(label)
            pixel,p,h,v,points,desc,shape=image_features(image,orb)
            records.append({'split':split,'image':f'images/{split}/{image.name}',
                            'label':f'labels/{split}/{label.name}','image_sha256':sha256(image),
                            'label_sha256':sha256(label),'pixel_sha256':pixel})
            hashes.append((p,h,v));local_features.append((points,desc,shape))
            if len(records)%1000==0:print('IMAGE_FEATURES',len(records),flush=True)
    hashes=np.asarray(hashes,dtype=np.uint64)
    if sum(r['split']=='test' for r in records)!=759:
        raise ValueError('原测试集数量已变化；不能沿用已审查历史协议')
    from ultralytics import YOLO
    model=YOLO(str(retrieval_model))
    embeddings=[]
    paths=[str(legacy/'helmet_dataset'/r['image']) for r in records]
    # Ultralytics eagerly decodes list inputs; explicitly chunk instead of loading the whole corpus.
    for start in range(0,len(paths),32):
        for vector in model.embed(source=paths[start:start+32],stream=True,batch=32,imgsz=224,device='cpu',
                                  embed=POLICY['embedding_layers'],verbose=False):
            embeddings.append(vector.detach().cpu().numpy().ravel())
        if start%1024==0:print('RETRIEVAL_EMBEDDINGS',len(embeddings),flush=True)
    if len(embeddings)!=len(records):
        raise ValueError('检索向量与图片数量不同')
    embeddings=np.asarray(embeddings,dtype=np.float32)
    embeddings/=np.maximum(np.linalg.norm(embeddings,axis=1,keepdims=True),1e-12)
    tests=[i for i,r in enumerate(records) if r['split']=='test']
    by_bytes,by_pixels=defaultdict(list),defaultdict(list)
    for i,r in enumerate(records):
        by_bytes[r['image_sha256']].append(i);by_pixels[r['pixel_sha256']].append(i)
    evidence,excluded={},set()
    geometry_cache={}
    for number,i in enumerate(tests):
        exact=[j for j in by_bytes[records[i]['image_sha256']] if j!=i]
        pixels=[j for j in by_pixels[records[i]['pixel_sha256']] if j!=i]
        distances=np.bitwise_count(hashes^hashes[i]).astype(np.int16)
        distances[i]=65
        near=np.where(((distances[:,0]<=8)&((distances[:,1]<=10)|(distances[:,2]<=10))) |
                      ((distances[:,1]<=4)&(distances[:,2]<=8)))[0].tolist()
        chosen=exact or pixels or near
        if chosen:
            # Prefer a train/val relation as evidence; otherwise exclude within-test redundancy.
            j=min(chosen,key=lambda j:(records[j]['split']=='test',int(distances[j].sum())))
            reason='exact_bytes' if exact else ('identical_decoded_pixels' if pixels else 'perceptual_candidate_conservative_exclusion')
            evidence[i]={'test_index':i,'reference_index':j,'reason':reason,
                         'phash_distance':int(distances[j,0]),'horizontal_distance':int(distances[j,1]),
                         'vertical_distance':int(distances[j,2])}
            excluded.add(i)
            if records[j]['split']=='test':
                excluded.add(j)
                evidence.setdefault(j,{'test_index':j,'reference_index':i,'reason':'within_test_redundancy'})
            continue
        candidates=set()
        for axis in range(3):
            candidates.update(np.argsort(distances[:,axis])[:10].tolist())
        similarities=embeddings@embeddings[i];similarities[i]=-2
        candidates.update(np.argsort(similarities)[-20:].tolist())
        candidates.discard(i)
        best=None
        for j in sorted(candidates):
            key=tuple(sorted((i,j)))
            if key not in geometry_cache:
                geometry_cache[key]=geometry(local_features[i],local_features[j])
            match=geometry_cache[key]
            if match and (best is None or match['inliers']>best['inliers']):
                best={**match,'test_index':i,'reference_index':j,
                      'reason':'geometric_scene_candidate_conservative_exclusion',
                      'phash_distance':int(distances[j,0])}
        if best:
            evidence[i]=best;excluded.add(i)
            j=best['reference_index']
            if records[j]['split']=='test':
                excluded.add(j)
                evidence.setdefault(j,{'test_index':j,'reference_index':i,'reason':'within_test_scene_redundancy'})
        if (number+1)%50==0:print('TEST_SCREENING',number+1,'excluded_so_far',len(excluded),flush=True)
    # Freeze decisions BEFORE running the detector on the retained subset.
    fields=['test_image','decision','reason','reference_split','reference_image','phash_distance',
            'horizontal_distance','vertical_distance','inliers','inlier_fraction','coverage_test','coverage_reference']
    decisions=[]
    for i in tests:
        row=evidence.get(i,{})
        reference=records[row['reference_index']] if row else {}
        decisions.append({'test_image':Path(records[i]['image']).name,'decision':'exclude' if i in excluded else 'retain',
                          'reason':row.get('reason','no_relation_found_by_screening'),
                          'reference_split':reference.get('split',''),'reference_image':reference.get('image',''),
                          **{k:row.get(k,'') for k in fields[5:]}})
    write_csv(output/'screening_decisions.csv',decisions,fields)
    write_csv(output/'original_reference_manifest.csv',[r for r in records if r['split']!='test'])
    write_csv(output/'original_test_manifest.csv',[r for r in records if r['split']=='test'])
    ds=output/'dataset'
    (ds/'images/test').mkdir(parents=True);(ds/'labels/test').mkdir(parents=True)
    retained,objects=[],Counter()
    for i in tests:
        if i in excluded:continue
        r=records[i];src=legacy/'helmet_dataset'/r['image'];label=legacy/'helmet_dataset'/r['label']
        dest=ds/'images/test'/src.name;dl=ds/'labels/test'/label.name
        shutil.copy2(src,dest);shutil.copy2(label,dl)
        if dest.suffix.lower() in ('.jpg','.jpeg'):
            with dest.open('rb') as f:f.seek(-2,2);needs_eoi=f.read()!=b'\xff\xd9'
            if needs_eoi:
                with dest.open('ab') as f:f.write(b'\xff\xd9')
        retained.append({'image':dest.name,'source_image_sha256':r['image_sha256'],
                         'image_sha256':sha256(dest),'label_sha256':r['label_sha256']})
        objects.update(int(b[0]) for b in read_labels(dl))
    if not retained:
        raise ValueError('没有剩余测试图片，不进行评价')
    write_csv(output/'retained_test_manifest.csv',retained)
    (ds/'evaluation_only.yaml').write_text(yaml.safe_dump({'path':str(ds),
          'train':str(legacy/'helmet_dataset/images/train'),'val':str(legacy/'helmet_dataset/images/val'),
          'test':'images/test','names':{0:'hat',1:'person'},'evaluation_only':True}),encoding='utf-8')
    visual=[evidence[i] for i in sorted(evidence)]
    contact_sheets(output,visual,records,legacy)
    report={'purpose':'Post-hoc original-test sensitivity analysis; no retraining',
            'policy':POLICY,'original_test_images':len(tests),'retained_test_images':len(retained),
            'excluded_test_images':len(excluded),'exclusion_reasons':dict(Counter(d['reason'] for d in decisions if d['decision']=='exclude')),
            'retained_objects':dict(objects),'retrieval_model_sha256':sha256(retrieval_model),
            'decision_manifest_sha256':sha256(output/'screening_decisions.csv'),
            'retained_manifest_sha256':sha256(output/'retained_test_manifest.csv'),
            'reference_manifest_sha256':sha256(output/'original_reference_manifest.csv'),
            'original_test_manifest_sha256':sha256(output/'original_test_manifest.csv'),
            'evaluation_yaml_sha256':sha256(ds/'evaluation_only.yaml'),
            'difficult_policy':'include_as_ordinary',
            'scene_independence':'NOT_PROVEN: automatic retrieval is finite; original XML has no usable scene/video IDs',
            'training_provenance':'Current original train/val files are reference; immutable training-time data manifest was not recorded',
            'comparison_limit':'Different test composition; differences in mAP are not model improvements. Original test had already been evaluated.'}
    write_json(output/'screening_report.json',report)
    print('SCREENING_FROZEN',json.dumps(report,ensure_ascii=False),flush=True)
    return report


def evaluate(legacy,output,checkpoint):
    legacy,output,checkpoint=Path(legacy).resolve(),Path(output).resolve(),Path(checkpoint).resolve()
    model_sha=sha256(checkpoint)
    if model_sha!=RELEASE_SHA256:
        raise ValueError('模型身份与历史 best.pt 不符；不能声称沿用同一模型')
    report=json.loads((output/'screening_report.json').read_text())
    checks={'decision_manifest_sha256':'screening_decisions.csv','retained_manifest_sha256':'retained_test_manifest.csv',
            'reference_manifest_sha256':'original_reference_manifest.csv','original_test_manifest_sha256':'original_test_manifest.csv',
            'evaluation_yaml_sha256':'dataset/evaluation_only.yaml'}
    for key,file in checks.items():
        if sha256(output/file)!=report[key]:raise ValueError('筛查决策或清单已变化')
    for file in ['original_reference_manifest.csv','original_test_manifest.csv']:
        for r in read_csv(output/file):
            if sha256(legacy/'helmet_dataset'/r['image'])!=r['image_sha256'] or sha256(legacy/'helmet_dataset'/r['label'])!=r['label_sha256']:
                raise ValueError('原始划分已改变，停止评价')
    retained=read_csv(output/'retained_test_manifest.csv')
    original={Path(r['image']).name:r for r in read_csv(output/'original_test_manifest.csv')}
    selected={r['test_image'] for r in read_csv(output/'screening_decisions.csv') if r['decision']=='retain'}
    if {r['image'] for r in retained}!=selected or not selected.issubset(original):
        raise ValueError('评价集合不是被冻结的原测试子集')
    config=yaml.safe_load((output/'dataset/evaluation_only.yaml').read_text())
    if Path(config['path']).resolve()!=output/'dataset' or config['test']!='images/test' or config['names']!={0:'hat',1:'person'}:
        raise ValueError('评价配置与冻结数据身份不一致')
    for r in retained:
        if r['source_image_sha256']!=original[r['image']]['image_sha256'] or r['label_sha256']!=original[r['image']]['label_sha256']:
            raise ValueError('子集图片或标签不属于原测试划分')
        if sha256(output/'dataset/images/test'/r['image'])!=r['image_sha256'] or sha256(output/'dataset/labels/test'/Path(r['image']).with_suffix('.txt'))!=r['label_sha256']:
            raise ValueError('评价副本已变化')
    import platform,torch,ultralytics
    from ultralytics import YOLO
    model=YOLO(str(checkpoint))
    metrics=model.val(data=str(output/'dataset/evaluation_only.yaml'),split='test',imgsz=640,batch=4,
                      device='cpu',workers=0,conf=.001,iou=.7,max_det=300,rect=True,
                      augment=False,plots=True,verbose=False,seed=0,deterministic=True,
                      project=str(output/'runs'),name='unchanged_model_screened_test')
    for r in retained:
        if sha256(output/'dataset/images/test'/r['image'])!=r['image_sha256']:
            raise ValueError('评价过程改写了图片，拒绝发布成绩')
    if sha256(checkpoint)!=model_sha:raise ValueError('模型在评价期间变化')
    result={'evaluation_scope':'post_hoc_screened_original_test_not_proven_scene_independent',
            'model_sha256':model_sha,'images':len(retained),'training_performed':False,
            'retained_manifest_sha256':report['retained_manifest_sha256'],
            'screening_decisions_sha256':report['decision_manifest_sha256'],
            'device':'cpu','imgsz':640,'batch':4,'conf':.001,'nms_iou':.7,'max_det':300,
            'python':platform.python_version(),'torch':torch.__version__,'ultralytics':ultralytics.__version__,
            'precision':float(metrics.box.mp),'recall':float(metrics.box.mr),
            'map50':float(metrics.box.map50),'map50_95':float(metrics.box.map),
            'precision_recall_definition':'class means at library max-mean-F1 confidence, not fixed conf=.25',
            'per_class':[{'id':int(c),'name':metrics.names[int(c)],'precision':float(metrics.box.p[i]),
                          'recall':float(metrics.box.r[i]),'map50':float(metrics.box.ap50[i]),
                          'map50_95':float(metrics.box.ap[i])} for i,c in enumerate(metrics.box.ap_class_index)],
            'limitations':report['scene_independence']+'; '+report['training_provenance']+'; '+report['comparison_limit']}
    write_json(output/'screened_test_metrics.json',result)
    print('UNCHANGED_MODEL_EVALUATION',json.dumps(result,ensure_ascii=False),flush=True)
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('mode',choices=['prepare','evaluate'])
    p.add_argument('--legacy-root',type=Path,default=ROOT)
    p.add_argument('--output',type=Path,default=ROOT/'reviewed_legacy_test')
    p.add_argument('--retrieval-model',type=Path)
    p.add_argument('--checkpoint',type=Path)
    a=p.parse_args()
    if a.mode=='prepare':
        prepare(a.legacy_root,a.output,a.retrieval_model or a.legacy_root/'yolov8n.pt')
    else:
        evaluate(a.legacy_root,a.output,a.checkpoint or a.legacy_root/'runs/helmet_50/weights/best.pt')


if __name__=='__main__':
    main()
