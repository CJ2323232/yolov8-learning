"""Evaluate a completed CBAM/P2 experiment on the SAME audited validation set.
No test selection, threshold tuning or checkpoint replacement.
"""
import argparse
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))


def main():
    from train_p2_cbam import protocol, prepare_data, check_runtime, read_json, code_identity
    from helmet_cbam import register_cbam
    from workflow import sha256
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run',type=Path,default=ROOT/'runs/helmet_p2_cbam_isolated_640_windows')
    args=parser.parse_args()
    directory=args.run.expanduser().resolve()
    if not directory.is_relative_to(ROOT):
        raise ValueError('模型必须来自本项目已登记的实验目录')
    p=protocol()
    check_runtime('0',p)
    data,identity=prepare_data(p)
    meta=read_json(directory/'experiment_identity.json')
    completed=read_json(directory/'completed.json')
    if meta['dataset']!=identity or meta['protocol']!=p or meta['code_identity']!=code_identity():
        raise ValueError('评价数据、协议或代码与训练记录不匹配')
    best=directory/'weights/best.pt'
    if sha256(best)!=completed['best_sha256']:
        raise ValueError('best.pt发生变化')
    register_cbam()
    from run import main as evaluate
    return evaluate(['val','--model',str(best),'--data',str(data),'--device','0',
                     '--imgsz','640','--batch','2','--seed','0','--split','val',
                     '--project',str(ROOT/'runs/cbam_validation'),'--name',directory.name])


if __name__=='__main__':
    main()
