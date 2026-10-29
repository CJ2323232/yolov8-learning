"""Paired P3-only CBAM: matched64-channel attention and parent initialization."""
import json
from pathlib import Path
import torch
ROOT=Path(__file__).resolve().parents[2]
HERE=Path(__file__).resolve().parent
from helmet_cbam import P2CBAM,register_cbam
from helmet_models import VARIANTS,build_initialized_model,tensor_digest
from workflow import sha256
SPEC=json.loads((HERE/'spec.json').read_text())
VARIANT=SPEC['variant'];CBAM_INDEX=SPEC['cbam_index'];DETECT_INDEX=SPEC['detect_index'];BASE_DETECT=SPEC['base_detect'];FEATURE_SOURCE=SPEC['feature_source']
CONFIG=HERE/('helmet_yolov8n_'+('standard_p3_cbam' if VARIANT=='standard' else 'p2_p3_cbam')+'.yaml')


def validate_config():
 import yaml
 expected={'standard':{'variant':'standard','feature_source':15,'cbam_index':22,'base_detect':22,'detect_index':23,'detect_inputs':[22,18,21],'strides':[8,16,32],'channels':64,'reduction':16,'kernel':7,'cbam_seed':0},
           'p2':{'variant':'p2','feature_source':21,'cbam_index':28,'base_detect':28,'detect_index':29,'detect_inputs':[18,28,24,27],'strides':[4,8,16,32],'channels':64,'reduction':16,'kernel':7,'cbam_seed':0}}
 if SPEC!=expected.get(VARIANT):raise ValueError('Unreviewed P3-CBAM topology')
 metadata=json.loads((HERE/'sources.json').read_text())
 for path,h in [(CONFIG,metadata['config_sha256']),(VARIANTS[VARIANT],metadata['parent_config_sha256']),(ROOT/'helmet_cbam.py',metadata['cbam_module_sha256']),(HERE/'spec.json',metadata['spec_sha256'])]:
  if sha256(path)!=h:raise ValueError('Unreviewed source change: '+str(path))
 cfg=yaml.safe_load(CONFIG.read_text());parent=yaml.safe_load(VARIANTS[VARIANT].read_text())
 if (cfg['nc']!=2 or cfg['scale']!='n' or cfg['scales']!={'n':parent['scales']['n']} or cfg['backbone']!=parent['backbone'] or cfg['head'][:-2]!=parent['head'][:-1] or cfg['head'][-2:]!=[[FEATURE_SOURCE,1,'P2CBAM',[64,16,7]],[SPEC['detect_inputs'],1,'Detect',['nc']]]):raise ValueError('Only frozen P3 detection-only routing supported')
 return metadata


def build_model(pretrained=None,seed=0):
 from ultralytics import YOLO
 metadata=validate_config();register_cbam();pretrained=ROOT/'yolov8n.pt' if pretrained is None else Path(pretrained)
 with torch.random.fork_rng(devices=[]):
  baseline,base_audit=build_initialized_model(VARIANTS[VARIANT],pretrained,seed)
  torch.random.default_generator.manual_seed(seed)
  target=YOLO(str(CONFIG),task='detect',verbose=False)
  src,dst=baseline.model.state_dict(),target.model.state_dict();mapped={}
  for key,value in src.items():
   parts=key.split('.');old=int(parts[1]);new=DETECT_INDEX if old==BASE_DETECT else old
   a,b=baseline.model.model[old],target.model.model[new]
   if type(a) is not type(b) or (old<BASE_DETECT and a.f!=b.f):raise ValueError(f'Topology mismatch {old}->{new}')
   parts[1]=str(new);nk='.'.join(parts)
   if nk not in dst or dst[nk].shape!=value.shape:raise ValueError('State mismatch '+key)
   mapped[nk]=value
  expected={k for k in dst if not k.startswith(f'model.{CBAM_INDEX}.')}
  if set(mapped)!=expected:raise ValueError('Incomplete parent state coverage')
  missing=target.model.load_state_dict(mapped,strict=False)
  if set(missing.missing_keys)!=set(dst)-expected or missing.unexpected_keys:raise ValueError('Unexpected missing state')
  # Explicit shared gate initialization removes architecture-dependent RNG consumption.
  torch.random.default_generator.manual_seed(SPEC['cbam_seed'])
  gate=P2CBAM(64,16,7)
  fresh=tensor_digest(gate.state_dict().items())
  target.model.model[CBAM_INDEX].load_state_dict(gate.state_dict(),strict=True)
  target.model.names={0:'hat',1:'person'};target.load(target.model)
  state=target.model.state_dict()
  if not all(torch.equal(state[k],v) for k,v in mapped.items()):raise ValueError('Parent initialization changed')
  if fresh!=tensor_digest(target.model.model[CBAM_INDEX].state_dict().items()):raise ValueError('Gate initialization changed')
 historical=metadata['historical_parent_initialization']
 for key in ['pretrained_sha256','shared_parameters_sha256']:
  if base_audit[key]!=historical[key]:raise ValueError('Official/shared initialization differs from historical parent')
 remaining_equal=base_audit['remaining_state_sha256']==historical['remaining_state_sha256']
 same_torch_build=torch.__version__==historical['runtime']['torch']
 if seed==0 and same_torch_build and not remaining_equal:raise ValueError('Parent fresh initialization changed within the historical torch build')
 audit={'historical_shared_initialization_matches':True,'historical_remaining_state_equal':remaining_equal,'historical_torch_build':historical['runtime']['torch'],'current_torch_build':torch.__version__,'historical_remaining_check_note':'Different torch builds/platforms may produce different random initial state fingerprints. Windows preflight verifies again; no cross-platform bitwise equality claim.', 'status':'initialized_not_trained','parent_variant':VARIANT,'seed':seed,'config_sha256':sha256(CONFIG),'module_sha256':sha256(__file__),'cbam_module_sha256':sha256(ROOT/'helmet_cbam.py'),'baseline_initialization':base_audit,'mapping':f'unchanged0..{BASE_DETECT-1}; Detect{BASE_DETECT}->{DETECT_INDEX}; CBAM{CBAM_INDEX} from{FEATURE_SOURCE}',
        'mapped_state_tensors':len(mapped),'all_non_cbam_state_equal_to_parent_initial_state':True,'mapped_state_sha256':tensor_digest(mapped.items()),'cbam_state_sha256':fresh,'cbam_initialization_seed':SPEC['cbam_seed'],'parent_parameters':sum(p.numel() for p in baseline.model.parameters()),'target_parameters':sum(p.numel() for p in target.model.parameters()),'cbam_parameters':sum(p.numel() for p in gate.parameters()),'strides':target.model.stride.tolist(),'p3_feature_channels':64,'p3_feature_stride':8,'training_performed':False}
 return target,audit
