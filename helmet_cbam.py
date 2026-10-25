"""P2-only CBAM experiment. No training or registration on import.

Implements the CBAM paper's avg+max shared channel MLP, followed by spatial
attention. Spatial Conv has no BatchNorm (unlike the authors' reference code).
This is not Ultralytics' simplified average-only ChannelAttention.
"""
from pathlib import Path
import torch
from torch import nn

ROOT = Path(__file__).resolve().parent
CONFIG = ROOT / 'models/configs/helmet_yolov8n_p2_cbam.yaml'


class P2CBAM(nn.Module):
    """Shape-preserving attention, explicit actual channels (nano P2: 32)."""
    def __init__(self, channels=32, reduction=16, kernel_size=7):
        super().__init__()
        if channels < reduction or reduction < 1 or kernel_size not in (3, 7):
            raise ValueError('Invalid CBAM channels/reduction/kernel')
        self.channels = channels
        self.channel_mlp = nn.Sequential(
            nn.Conv2d(channels, channels // reduction, 1, bias=True),
            nn.ReLU(),
            nn.Conv2d(channels // reduction, channels, 1, bias=True),
        )
        self.spatial = nn.Conv2d(2, 1, kernel_size, padding=kernel_size // 2, bias=False)

    def forward(self, x):
        if x.ndim != 4 or x.shape[1] != self.channels:
            raise ValueError(f'CBAM expects BCHW with {self.channels} channels')
        # Same MLP parameters are used for BOTH pooled descriptors.
        channel_gate = torch.sigmoid(
            self.channel_mlp(x.mean((2, 3), keepdim=True))
            + self.channel_mlp(x.amax((2, 3), keepdim=True)))
        channel_refined = x * channel_gate
        spatial_descriptor = torch.cat((channel_refined.amax(1, keepdim=True),
                                        channel_refined.mean(1, keepdim=True)), dim=1)
        return channel_refined * torch.sigmoid(self.spatial(spatial_descriptor))


def register_cbam():
    """Process-local registration; never edit installed Ultralytics files."""
    import ultralytics.nn.tasks as tasks
    previous = getattr(tasks, 'P2CBAM', None)
    if previous is not None and previous is not P2CBAM:
        raise RuntimeError('P2CBAM name is already occupied by another implementation')
    tasks.P2CBAM = P2CBAM
    # Allow exactly this project class when Ultralytics uses restricted loading.
    torch.serialization.add_safe_globals([P2CBAM])


def validate_config():
    import json
    import yaml
    from helmet_models import VARIANTS
    from workflow import sha256
    metadata = json.loads((CONFIG.parent / 'p2_cbam_sources.json').read_text())
    for path, expected in ((CONFIG, metadata['config_sha256']),
                           (VARIANTS['p2'], metadata['parent_config_sha256'])):
        if sha256(path) != expected:
            raise ValueError(f'Unreviewed architecture change: {path.name}')
    cfg = yaml.safe_load(CONFIG.read_text())
    parent = yaml.safe_load(VARIANTS['p2'].read_text())
    if (cfg['scale'] != 'n' or set(cfg['scales']) != {'n'}
            or cfg['scales']['n'] != parent['scales']['n'] or cfg['nc'] != 2
            or cfg['backbone'] != parent['backbone']
            or cfg['head'][:-2] != parent['head'][:-1]
            or cfg['head'][-2:] != [[18, 1, 'P2CBAM', [32, 16, 7]],
                                     [[28, 21, 24, 27], 1, 'Detect', ['nc']]]):
        raise ValueError('Only the audited nano P2 detection-branch insertion is supported')
    return metadata


def build_cbam_model(pretrained=None, seed=0):
    """Match the existing P2 initial state EXACTLY outside CBAM.

    Existing baseline policy: official COCO layers 0..15; other P2 layers fresh.
    Copy those initial tensors by semantic position, mapping Detect 28 -> 29.
    No fine-tuned best.pt is used. This is preparation, not a trained detector.
    """
    from ultralytics import YOLO
    from helmet_models import VARIANTS, build_initialized_model, tensor_digest
    from workflow import sha256
    metadata = validate_config()
    register_cbam()
    pretrained = ROOT / 'yolov8n.pt' if pretrained is None else Path(pretrained)
    with torch.random.fork_rng(devices=[]):
        baseline, base_audit = build_initialized_model(VARIANTS['p2'], pretrained, seed)
        torch.random.default_generator.manual_seed(seed)
        target = YOLO(str(CONFIG), task='detect', verbose=False)
        src, dst = baseline.model.state_dict(), target.model.state_dict()
        mapped = {}
        for key, value in src.items():
            parts = key.split('.')
            index = int(parts[1])
            new_index = 29 if index == 28 else index
            source_layer, target_layer = baseline.model.model[index], target.model.model[new_index]
            if type(source_layer) is not type(target_layer):
                raise ValueError(f'Layer type mismatch: {index} -> {new_index}')
            if index < 28 and source_layer.f != target_layer.f:
                raise ValueError(f'Layer connection mismatch at {index}')
            parts[1] = str(new_index)
            new_key = '.'.join(parts)
            if new_key not in dst or dst[new_key].shape != value.shape:
                raise ValueError(f'State mismatch: {key} -> {new_key}')
            mapped[new_key] = value
        expected = {k for k in dst if not k.startswith('model.28.')}
        if set(mapped) != expected:
            raise ValueError('Non-CBAM state coverage is incomplete')
        fresh = tensor_digest((k, v) for k, v in dst.items() if k.startswith('model.28.'))
        missing = target.model.load_state_dict(mapped, strict=False)
        if set(missing.missing_keys) != set(dst) - expected or missing.unexpected_keys:
            raise ValueError('Unexpected missing/extra tensors')
        target.model.names = {0: 'hat', 1: 'person'}
        target.load(target.model)  # preserve explicit initialization during trainer rebuild
        state = target.model.state_dict()
        if not all(torch.equal(state[k], v) for k, v in mapped.items()):
            raise ValueError('Transferred state changed unexpectedly')
        if fresh != tensor_digest((k, v) for k, v in state.items() if k.startswith('model.28.')):
            raise ValueError('CBAM initial state was overwritten')
    audit = {
        'status': 'structure_only_not_trained', 'seed': seed,
        'config_sha256': sha256(CONFIG), 'module_sha256': sha256(Path(__file__)),
        'parent_config_sha256': metadata['parent_config_sha256'],
        'baseline_initialization': base_audit,
        'mapping': 'layers 0..27 unchanged; Detect 28 -> 29; new CBAM at 28 from layer 18',
        'mapped_state_tensors': len(mapped), 'all_non_cbam_state_equal_to_p2_initial_state': True,
        'mapped_state_sha256': tensor_digest(mapped.items()), 'cbam_state_sha256': fresh,
        'p2_parameters': sum(p.numel() for p in baseline.model.parameters()),
        'p2_cbam_parameters': sum(p.numel() for p in target.model.parameters()),
        'cbam_parameters': sum(p.numel() for p in target.model.model[28].parameters()),
        'strides': target.model.stride.tolist(), 'training_performed': False,
    }
    return target, audit
