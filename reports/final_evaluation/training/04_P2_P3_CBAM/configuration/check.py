"""Synthetic structure/gradient checks; no optimizer, data training or formal output."""
import copy
from pathlib import Path
from types import SimpleNamespace
import torch
from torch import nn
from ultralytics.utils import DEFAULT_CFG_DICT


def inspect_model(wrapper,device='cpu',backward=True):
 from helmet_cbam import P2CBAM
 from helmet_models import tensor_digest
 from tools.p3_cbam_p2.train import protocol
 from tools.p3_cbam_p2.model import SPEC,CBAM_INDEX,DETECT_INDEX,FEATURE_SOURCE
 original=wrapper.model
 if len(original.model)!=DETECT_INDEX+1 or original.stride.tolist()!=SPEC['strides']:raise ValueError('Unexpected parent detection scales')
 if original.model[CBAM_INDEX].f!=FEATURE_SOURCE or original.model[DETECT_INDEX].f!=SPEC['detect_inputs']:raise ValueError('Wrong CBAM branch routing')
 blocks=[m for m in original.modules() if isinstance(m,P2CBAM)]
 if len(blocks)!=1 or blocks[0].channels!=64:raise ValueError('Expected existing CBAM with64 channels')
 before=tensor_digest(original.state_dict().items())
 target='cuda:0' if device=='0' else 'cpu'
 network=copy.deepcopy(original).to(target).float().eval();forward=[]
 for height,width in [(640,640),(320,640)]:
  shapes=[];cbam_shapes=[]
  h1=network.model[DETECT_INDEX].register_forward_pre_hook(lambda mod,args:shapes.extend([list(x.shape) for x in args[0]]))
  h2=network.model[CBAM_INDEX].register_forward_hook(lambda mod,args,out:cbam_shapes.append({'in':list(args[0].shape),'out':list(out.shape)}))
  try:
   with torch.inference_mode():pred=network(torch.zeros(1,3,height,width,device=target))[0]
   expected=[1,6,sum((height//s)*(width//s) for s in SPEC['strides'])]
   if list(pred.shape)!=expected or not torch.isfinite(pred).all():raise ValueError('Forward shape/nonfinite failure')
   assert shapes==[[1,32,height//4,width//4],[1,64,height//8,width//8],[1,128,height//16,width//16],[1,256,height//32,width//32]]
   assert cbam_shapes==[{'in':shapes[1],'out':shapes[1]}]
   forward.append({'input':[1,3,height,width],'prediction':list(pred.shape),'detect_inputs':shapes,'cbam':cbam_shapes,'finite':True})
  finally:h1.remove();h2.remove()
 del network
 gradients={}
 if backward:
  network=copy.deepcopy(original).to(target).float().train();network.args=SimpleNamespace(**{**DEFAULT_CFG_DICT,**protocol()['train']})
  with torch.random.fork_rng(devices=[0] if device=='0' else []):
   torch.manual_seed(20261027)
   batch={'img':torch.rand(2,3,128,128,device=target),'batch_idx':torch.tensor([0.,1.],device=target),'cls':torch.tensor([[0.],[1.]],device=target),'bboxes':torch.tensor([[.5,.5,.12,.12],[.4,.6,.2,.2]],device=target)}
   parameters_before=tensor_digest(network.named_parameters());loss,_=network.loss(batch)
   if not torch.isfinite(loss).all():raise ValueError('Detection loss not finite')
   loss.sum().backward()
   for name,module in [('channel',network.model[CBAM_INDEX].channel_mlp),('spatial',network.model[CBAM_INDEX].spatial),('backbone',network.model[0])]:
    grads=[p.grad for p in module.parameters() if p.requires_grad]
    if not grads or not all(g is not None and torch.isfinite(g).all() for g in grads):raise ValueError('Missing/nonfinite '+name+' gradient')
    norm=sum(g.abs().sum().item() for g in grads)
    if norm<=0:raise ValueError('Zero '+name+' gradient')
    gradients[name]={'finite':True,'absolute_sum':norm}
   assert tensor_digest(network.named_parameters())==parameters_before
  del network
 assert tensor_digest(original.state_dict().items())==before
 return {'forward':forward,'backward_input':[2,3,128,128] if backward else None,'gradients':gradients,'optimizer_steps':0,'original_state_unchanged':True,'limitations':'Synthetic forward/backward on a discarded copy. Not50epoch convergence, performance, CUDA training memory or Windows execution proof.'}
