"""Metric functions extracted unchanged from the measured slice experiment."""
import numpy as np
from analyze_detection_errors import match, size_bin, BINS

def summarize(gt,boxes,width,height):
    pairs,missed,bad=match(gt,boxes)
    hit={g for _,g,_ in pairs}
    summary={'gt':len(gt),'predictions':len(boxes),'tp':len(pairs),'fn':len(missed),'fp':len(bad)}
    summary['by_class']={}
    for cls,name in [(0,'hat'),(1,'person')]:
        ng=sum(int(g[4])==cls for g in gt);npred=sum(int(b[5])==cls for b in boxes)
        tp=sum(int(gt[g,4])==cls for _,g,_ in pairs)
        summary['by_class'][name]={'gt':ng,'predictions':npred,'tp':tp,'fn':ng-tp,'fp':npred-tp}
    summary['by_size']={};summary['by_class_size']={}
    for band in BINS:
        ids={g for g in range(len(gt)) if size_bin(gt[g],width,height)[0]==band}
        summary['by_size'][band]={'gt':len(ids),'tp':len(ids&hit),'fn':len(ids-hit)}
        summary['by_class_size'][band]={}
        for cls,name in [(0,'hat'),(1,'person')]:
            cid={g for g in ids if int(gt[g,4])==cls}
            summary['by_class_size'][band][name]={'gt':len(cid),'tp':len(cid&hit),'fn':len(cid-hit)}
    assert summary['tp']+summary['fn']==len(gt)
    assert summary['tp']+summary['fp']==len(boxes)
    assert sum(x['gt'] for x in summary['by_size'].values())==len(gt)
    assert sum(x['tp'] for x in summary['by_size'].values())==summary['tp']
    return summary,pairs,missed,bad

def add_counts(total,current):
    for k,v in current.items():
        if isinstance(v,dict):add_counts(total.setdefault(k,{}),v)
        else:total[k]=total.get(k,0)+v

def with_rates(c):
    c={k:with_rates(v) if isinstance(v,dict) else v for k,v in c.items()}
    if 'predictions' in c:
        c['precision']=c['tp']/c['predictions'] if c['predictions'] else None
        c['f1']=2*c['tp']/(c['gt']+c['predictions']) if c['gt']+c['predictions'] else None
    if 'gt' in c:c['recall']=c['tp']/c['gt'] if c['gt'] else None
    return c
