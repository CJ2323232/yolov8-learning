"""Pinned YOLOv8 variants and audited shared-layer initialization.

Torch and Ultralytics are imported only when actually building a model.
This module does not train, save checkpoints or change the active demo model.
"""
import hashlib
import json
from pathlib import Path

from workflow import ROOT, sha256

CONFIG_ROOT = ROOT/'models/configs'
VARIANTS = {'standard': CONFIG_ROOT/'helmet_yolov8n.yaml',
            'p2': CONFIG_ROOT/'helmet_yolov8n_p2.yaml'}
SHARED_LAYERS = 16


def initialization_spec(config, pretrained):
    """Validate frozen local inputs without importing torch or loading weights."""
    config, pretrained = Path(config).expanduser().resolve(), Path(pretrained).expanduser().resolve()
    variant = next((name for name, path in VARIANTS.items() if path.resolve() == config), None)
    if variant is None:
        raise ValueError('共有层初始化只支持项目中已固定的 standard/p2 配置')
    sources = json.loads((CONFIG_ROOT/'sources.json').read_text())
    if sha256(config) != sources['variants'][variant]['config_sha256']:
        raise ValueError('模型配置已变化，请为新实验重新审核配置和初始化范围')
    if sha256(pretrained) != sources['pretrained_sha256']:
        raise ValueError('预训练文件不是本次对照固定的官方 yolov8n.pt；不能混入历史安全帽模型')
    return {'variant': variant, 'architecture': str(config), 'architecture_sha256': sha256(config),
            'pretrained': str(pretrained), 'pretrained_sha256': sha256(pretrained),
            'policy': 'shared_backbone_and_neck_layers_0_through_15',
            'shared_layer_indices': list(range(SHARED_LAYERS)),
            'remaining_layers': 'fresh/default initialization, including fixed DFL constants',
            'coco_class_name_remapping': False}


def _in_shared(key):
    parts = key.split('.')
    return len(parts) > 2 and parts[0] == 'model' and parts[1].isdigit() and int(parts[1]) < SHARED_LAYERS


def tensor_digest(items):
    digest = hashlib.sha256()
    for key, value in sorted(items):
        value = value.detach().cpu().contiguous()
        digest.update(key.encode())
        digest.update(str((tuple(value.shape), value.dtype)).encode())
        digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def shared_parameter_digest(network):
    network = getattr(network, 'module', network)
    return tensor_digest((key, value) for key, value in network.named_parameters() if _in_shared(key))


def build_initialized_model(config, pretrained, seed=0):
    """Load only verified common layers, preserving all fresh target branches.

    Layers after 15 change semantic position in the P2 graph. Shape-compatible
    layer indices alone are insufficient evidence that a transfer is correct.
    Both variants deliberately use this SAME initialization coverage.
    """
    spec = initialization_spec(config, pretrained)
    import torch
    from ultralytics import YOLO

    # Architecture-dependent RNG consumption must not change data-order RNG.
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(seed)
        target = YOLO(str(config), task='detect')
        source = YOLO(str(pretrained), task='detect').model.cpu().float()
        if target.model.model[-1].nc != 2 or source.model[-1].nc != 80:
            raise ValueError('期望目标为两类、官方起点为80类')
        for index in range(SHARED_LAYERS):
            dst, src = target.model.model[index], source.model[index]
            if type(dst) is not type(src) or dst.f != src.f:
                raise ValueError(f'共有层拓扑不匹配：{index}')
        target_state, source_state = target.model.state_dict(), source.state_dict()
        selected = {key: value for key, value in source_state.items() if _in_shared(key)}
        expected = {key for key in target_state if _in_shared(key)}
        if set(selected) != expected or any(selected[k].shape != target_state[k].shape for k in selected):
            raise ValueError('共有层参数缺失或尺寸不一致，停止加载')
        fresh_before = tensor_digest((k, v) for k, v in target_state.items() if not _in_shared(k))
        target.model.load_state_dict(selected, strict=False)
        target.model.names = {0: 'hat', 1: 'person'}
        # Public object-load API marks this already-initialized model for reuse
        # by YOLO.train(). Without this, YAML training can rebuild random weights.
        target.load(target.model)
        state = target.model.state_dict()
        if any(not torch.equal(state[k], selected[k]) for k in selected):
            raise ValueError('共有层加载后数值不一致')
        if tensor_digest((k, v) for k, v in state.items() if not _in_shared(k)) != fresh_before:
            raise ValueError('共有范围之外的参数被意外覆盖')
    audit = {**spec, 'seed': seed, 'loaded_state_tensors': len(selected),
             'loaded_tensor_shapes': {k: list(v.shape) for k, v in selected.items()},
             'loaded_shared_state_sha256': tensor_digest(selected.items()),
             'shared_parameters_sha256': shared_parameter_digest(target.model),
             'remaining_state_unchanged': True,
             'remaining_state_sha256': fresh_before,
             'parameters': sum(p.numel() for p in target.model.parameters()),
             'detection_strides': [int(s) for s in target.model.stride.tolist()],
             'runtime': {'torch': torch.__version__},
             'training_performed': False}
    return target, audit
