"""Post-training val-only AP; never test or change app default."""
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from workflow import sha256
from helmet_cbam import register_cbam
from tools.p3_cbam_standard.train import protocol,check_runtime,prepare_data
from ultralytics import YOLO

def main():
 p=protocol();check_runtime('0',p);data,_=prepare_data(p)
 run=ROOT/'runs/helmet_standard_p3_cbam_isolated_640_windows'
 completed=json.loads((run/'completed.json').read_text());best=run/'weights/best.pt'
 assert completed['variant']=='standard_p3_cbam' and sha256(best)==completed['best_sha256']
 target=run/'validation_recheck'
 if target.exists():raise FileExistsError(target)
 register_cbam();model=YOLO(str(best));results=model.val(data=str(data),split='val',imgsz=640,batch=2,workers=0,device='0',conf=.001,iou=.7,max_det=300,plots=True,project=str(run),name='validation_recheck',exist_ok=False)
 (target/'metrics.json').write_text(json.dumps({'split':'val','mAP50':float(results.box.map50),'mAP50_95':float(results.box.map),'model_sha256':sha256(best)},indent=2)+'\n')
if __name__=='__main__':main()
