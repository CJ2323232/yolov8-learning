"""Independent standard-CBAM regression tests; no formal training."""
import copy
import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch,Mock
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
import torch
from torch import nn
from helmet_cbam import P2CBAM,register_cbam
from helmet_models import VARIANTS,build_initialized_model,tensor_digest
from tools.p3_cbam_p2.model import build_model,validate_config,CBAM_INDEX,DETECT_INDEX,BASE_DETECT,FEATURE_SOURCE,SPEC
from tools.p3_cbam_p2 import train
from tools.p3_cbam_p2.check import inspect_model

class StructureTests(unittest.TestCase):
 @classmethod
 def setUpClass(cls):
  cls.old=torch.get_num_threads();torch.set_num_threads(2)
  cls.wrapper,cls.audit=build_model(seed=0)
  cls.base,cls.base_audit=build_initialized_model(VARIANTS['p2'],ROOT/'yolov8n.pt',0)
  (ROOT/'project_work/p3_cbam_p2_training').mkdir(parents=True,exist_ok=True)
 @classmethod
 def tearDownClass(cls):torch.set_num_threads(cls.old)
 def test_non_cbam_states_and_topology_identical_to_standard(self):
  a,b=self.wrapper.model,self.base.model
  for i in range(BASE_DETECT):
   self.assertEqual(a.model[i].f,b.model[i].f)
   self.assertEqual(tensor_digest(a.model[i].state_dict().items()),tensor_digest(b.model[i].state_dict().items()))
  self.assertEqual(tensor_digest(a.model[DETECT_INDEX].state_dict().items()),tensor_digest(b.model[BASE_DETECT].state_dict().items()))
  self.assertEqual(a.model[CBAM_INDEX].f,FEATURE_SOURCE);self.assertEqual(a.model[DETECT_INDEX].f,SPEC['detect_inputs']);self.assertIs(type(a.model[CBAM_INDEX]),P2CBAM)
  self.assertEqual(self.audit['cbam_parameters'],678)
 def test_identity_gate_exactly_recovers_standard(self):
  target=copy.deepcopy(self.wrapper.model).eval();replacement=nn.Identity()
  for name in ['f','i','type','np']:setattr(replacement,name,getattr(target.model[CBAM_INDEX],name))
  target.model[CBAM_INDEX]=replacement
  with torch.inference_mode():
   x=torch.zeros(1,3,128,160);a=target(x)[0];b=self.base.model.eval()(x)[0]
  torch.testing.assert_close(a,b,rtol=0,atol=0)
 def test640_rectangular_and_real_detection_gradients(self):
  result=inspect_model(self.wrapper,'cpu',True)
  self.assertEqual(result['forward'][0]['prediction'],[1,6,34000])
  self.assertEqual(result['optimizer_steps'],0)
 def test_trainer_rebuild_keeps_all_initialized_states(self):
  from ultralytics.models.yolo.detect.train import DetectionTrainer
  t=object.__new__(DetectionTrainer);t.args=SimpleNamespace(cls_remap=False);t.data={'nc':2,'channels':3,'names':{0:'hat',1:'person'}}
  rebuilt=t.get_model(cfg=self.wrapper.model.yaml,weights=self.wrapper.model,verbose=False)
  self.assertEqual(tensor_digest(rebuilt.state_dict().items()),tensor_digest(self.wrapper.model.state_dict().items()))
 def test_rng_preserved_by_builder(self):
  before=torch.get_rng_state().clone();build_model(seed=17);self.assertTrue(torch.equal(before,torch.get_rng_state()))
 def test_pretrained_digest_matches_historical_standard(self):
  old=json.loads((next(p for p in [ROOT/'runs2/helmet_p2_isolated_640_windows/initialization.json',ROOT/'runs/helmet_p2_isolated_640_windows/initialization.json'] if p.exists())).read_text())
  for key in ['pretrained_sha256','shared_parameters_sha256']:
   self.assertEqual(self.base_audit[key],old[key])
  if torch.__version__==old['runtime']['torch']:self.assertEqual(self.base_audit['remaining_state_sha256'],old['remaining_state_sha256'])
  self.assertEqual(self.audit['historical_remaining_state_equal'],self.base_audit['remaining_state_sha256']==old['remaining_state_sha256'])
 def test_save_reload_keeps_existing_module_and_half_precision_state(self):
  from ultralytics import YOLO
  with tempfile.TemporaryDirectory(dir=ROOT/'project_work/p3_cbam_p2_training') as tmp:
   path=Path(tmp)/'synthetic_structure_only.pt';self.wrapper.save(path);register_cbam();restored=YOLO(str(path),task='detect').model
   self.assertIsInstance(restored.model[CBAM_INDEX],P2CBAM)
   for k,v in self.wrapper.model.state_dict().items():torch.testing.assert_close(restored.state_dict()[k],v.half().float() if v.is_floating_point() else v,rtol=0,atol=0)

class ProtocolTests(unittest.TestCase):
 def setUp(self):
  (ROOT/'project_work/p3_cbam_p2_training').mkdir(parents=True,exist_ok=True)
 def test67_parameters_match_all_three_historical_runs(self):
  p=train.protocol();self.assertEqual(len(p['train']),67)
  for folder in ['runs2/helmet_standard_isolated_640_windows','runs2/helmet_p2_isolated_640_windows','CBAM outputs/helmet_p2_cbam_isolated_640_windows']:
   candidates=[ROOT/folder/'effective_args.json',ROOT/'runs'/Path(folder).name/'effective_args.json']
   record=next((q for q in candidates if q.exists()),None)
   if record is None:continue
   args=json.loads(record.read_text());train.check_parameters(args,p['train'])
 def test_changed_and_missing_parameters_rejected(self):
  expected=train.protocol()['train']
  for key in expected:
   args=copy.deepcopy(expected);args.pop(key)
   with self.assertRaises(ValueError):train.check_parameters(args,expected)
  args={**expected,'batch':1}
  with self.assertRaises(ValueError):train.check_parameters(args,expected)
 def test_cpu_cannot_start_training(self):
  with self.assertRaises(SystemExit),patch.object(train,'check_runtime') as runtime:train.main(['--train','--device','cpu'])
  runtime.assert_not_called()
 def test_hyperparameter_cli_overrides_rejected(self):
  for options in [['--epochs','1'],['--batch','1'],['--lr0','.1'],['--seed','3']]:
   with self.assertRaises(SystemExit):train.main(options)
 def test_default_check_never_calls_train(self):
  p=train.protocol();fake=SimpleNamespace(model=SimpleNamespace(),train=Mock())
  with tempfile.TemporaryDirectory(dir=ROOT/'project_work/p3_cbam_p2_training') as tmp:
   with patch.object(train,'ROOT',Path(tmp)),patch.object(train,'code_identity',return_value={}),patch.object(train,'check_runtime',return_value={}),patch.object(train,'prepare_data',return_value=(Path(tmp)/'data.yaml',{})),patch.object(train,'check_structure',return_value={}),patch('tools.p3_cbam_p2.model.build_model',return_value=(fake,{})):
    result=train.main(['--device','cpu'])
  fake.train.assert_not_called();self.assertEqual(result['status'],'preflight_passed_no_training')
 def test_completed_or_foreign_resume_rejected(self):
  p=train.protocol();prior={'dataset':{},'code_identity':{},'variant':'p2_p3_cbam','protocol':p}
  with self.assertRaises(ValueError):train.validate_resume({'train_args':p['train'],'epoch':-1},prior,p,{}, {},'p2_p3_cbam')
  with self.assertRaises(ValueError):train.validate_resume({'train_args':p['train'],'epoch':1},prior,p,{}, {},'p2_cbam')

class MigrationTests(unittest.TestCase):
 def setUp(self):
  (ROOT/'project_work/p3_cbam_p2_training').mkdir(parents=True,exist_ok=True)
 def fixture(self,tmp):
  from tools.p3_cbam_p2.install_update import sha
  root=Path(tmp)/'old';bundle=Path(tmp)/'update';root.mkdir();bundle.mkdir()
  (root/'train_helmet.py').write_text('old training code')
  src=bundle/'payload/tools/p3_cbam_p2/new.py';src.parent.mkdir(parents=True);src.write_text('new experiment')
  manifest={'required_existing_files':{'train_helmet.py':sha(root/'train_helmet.py')},'files':{'tools/p3_cbam_p2/new.py':sha(src)}}
  (bundle/'update_manifest.json').write_text(json.dumps(manifest))
  return root,bundle
 def test_additive_install_and_idempotence(self):
  from tools.p3_cbam_p2.install_update import install,sha
  with tempfile.TemporaryDirectory(dir=ROOT/'project_work/p3_cbam_p2_training') as tmp:
   root,bundle=self.fixture(tmp);old=sha(root/'train_helmet.py');before=sorted(str(p) for p in root.rglob('*'))
   self.assertEqual(install(bundle,root,True)['status'],'check_only');self.assertEqual(before,sorted(str(p) for p in root.rglob('*')))
   self.assertEqual(install(bundle,root)['status'],'installed');self.assertEqual(old,sha(root/'train_helmet.py'))
   self.assertEqual(install(bundle,root)['status'],'already_installed')
 def test_corrupt_payload_or_old_dependency_stops_before_writes(self):
  from tools.p3_cbam_p2.install_update import install
  for corrupt_old in [False,True]:
   with tempfile.TemporaryDirectory(dir=ROOT/'project_work/p3_cbam_p2_training') as tmp:
    root,bundle=self.fixture(tmp)
    (root/'train_helmet.py' if corrupt_old else bundle/'payload/tools/p3_cbam_p2/new.py').write_text('changed')
    with self.assertRaises(ValueError):install(bundle,root)
    self.assertFalse((root/'tools').exists())
 def test_old_paths_and_path_traversal_rejected(self):
  from tools.p3_cbam_p2.install_update import safe
  for rel in ['train_helmet.py','datasets/a.txt','runs/weights/best.pt','../outside','tools/p3_cbam_p2/../../old.py','tools/p3_cbam_p2/best.pt']:
   with self.assertRaises(ValueError):safe(ROOT,rel)
 def test_code_update_refused_after_run_exists(self):
  from tools.p3_cbam_p2.install_update import install
  with tempfile.TemporaryDirectory(dir=ROOT/'project_work/p3_cbam_p2_training') as tmp:
   root,bundle=self.fixture(tmp);(root/'runs/helmet_p2_p3_cbam_isolated_640_windows').mkdir(parents=True)
   with self.assertRaises(ValueError):install(bundle,root)
   self.assertFalse((root/'tools').exists())
 def test_launcher_finds_existing_ancestor_environment(self):
  from tools.p3_cbam_p2.launch import choose_python
  import os
  with tempfile.TemporaryDirectory(dir=ROOT/'project_work/p3_cbam_p2_training') as tmp:
   parent=Path(tmp);root=parent/'nested/project';root.mkdir(parents=True)
   python=parent/'.venv-yolo/Scripts/python.exe';python.parent.mkdir(parents=True);python.write_bytes(b'')
   with patch.dict(os.environ,{},clear=True):self.assertEqual(choose_python(root),str(python))

if __name__=='__main__':unittest.main()
