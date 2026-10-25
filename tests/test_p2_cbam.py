"""Structural checks only: synthetic inputs, no optimizer or real dataset."""
import copy
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
import torch
from torch import nn
from torch.nn import functional as F
from helmet_cbam import ROOT, P2CBAM, build_cbam_model, register_cbam
from helmet_models import VARIANTS, build_initialized_model, tensor_digest


class CBAMMechanismTests(unittest.TestCase):
    def test_shared_avg_max_mlp_then_spatial_formula(self):
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(123)
            block = P2CBAM()
            x = torch.randn(2, 32, 11, 17)
        a, b = block.channel_mlp[0], block.channel_mlp[2]
        def mlp(z):
            return F.linear(F.relu(F.linear(z, a.weight[:, :, 0, 0], a.bias)),
                            b.weight[:, :, 0, 0], b.bias)
        g = torch.sigmoid(mlp(x.mean((2, 3))) + mlp(x.amax((2, 3))))[:, :, None, None]
        y = x * g
        spatial = torch.cat((y.amax(1, keepdim=True), y.mean(1, keepdim=True)), 1)
        expected = y * torch.sigmoid(F.conv2d(spatial, block.spatial.weight, padding=3))
        torch.testing.assert_close(block(x), expected)
        self.assertEqual(block(x).shape, x.shape)

    def test_wrong_channels_rejected(self):
        with self.assertRaisesRegex(ValueError, '32 channels'):
            P2CBAM()(torch.zeros(1, 64, 8, 8))


class CBAMIntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        (ROOT/'project_work/p2_cbam_structure_2026-10-24').mkdir(parents=True, exist_ok=True)
        cls.old_threads = torch.get_num_threads()
        torch.set_num_threads(2)
        cls.wrapper, cls.audit = build_cbam_model(seed=0)
        cls.baseline, _ = build_initialized_model(VARIANTS['p2'], ROOT/'yolov8n.pt', seed=0)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.old_threads)

    def test_only_p2_detection_route_changed_and_weights_equal(self):
        target, base = self.wrapper.model, self.baseline.model
        self.assertEqual(target.model[28].f, 18)
        self.assertEqual(target.model[29].f, [28, 21, 24, 27])
        self.assertEqual(target.model[19].f, -1)
        for index in range(28):
            self.assertEqual(target.model[index].f, base.model[index].f)
            self.assertEqual(tensor_digest(target.model[index].state_dict().items()),
                             tensor_digest(base.model[index].state_dict().items()))
        self.assertEqual(tensor_digest(target.model[29].state_dict().items()),
                         tensor_digest(base.model[28].state_dict().items()))
        self.assertEqual(self.audit['cbam_parameters'], 260)
        self.assertEqual(self.audit['strides'], [4, 8, 16, 32])

    def test_identity_ablation_recovers_exact_p2_output(self):
        # Disable attention mathematically; graph reindexing must not change anything else.
        target = copy.deepcopy(self.wrapper.model).eval()
        replacement = nn.Identity()
        for name in ('f', 'i', 'type', 'np'):
            setattr(replacement, name, getattr(target.model[28], name))
        target.model[28] = replacement
        with torch.inference_mode():
            x = torch.zeros(1, 3, 128, 160)
            expected = self.baseline.model.eval()(x)[0]
            actual = target(x)[0]
        torch.testing.assert_close(actual, expected, atol=0, rtol=0)

    def test_640_forward_and_rectangular_forward(self):
        network = self.wrapper.model.eval()
        for height, width in ((640, 640), (320, 640)):
            shapes = []
            hook = network.model[-1].register_forward_pre_hook(
                lambda m, args: shapes.extend([list(t.shape) for t in args[0]]))
            try:
                with torch.inference_mode():
                    output = network(torch.zeros(1, 3, height, width))[0]
                self.assertTrue(torch.isfinite(output).all())
                self.assertEqual([s[2:] for s in shapes],
                                 [[height//s, width//s] for s in (4, 8, 16, 32)])
                self.assertEqual(output.shape[1], 6)  # xywh + 2 class scores
            finally:
                hook.remove()

    def test_real_detection_loss_reaches_both_attention_stages(self):
        from ultralytics.utils import DEFAULT_CFG_DICT
        network = copy.deepcopy(self.wrapper.model).train()
        network.args = SimpleNamespace(**DEFAULT_CFG_DICT)
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(211)
            batch = {'img': torch.rand(2, 3, 128, 128),
                     'batch_idx': torch.tensor([0., 1.]), 'cls': torch.tensor([[0.], [1.]]),
                     'bboxes': torch.tensor([[.5, .5, .12, .12], [.4, .6, .2, .2]])}
        params_before = tensor_digest(network.named_parameters())
        loss, _ = network.loss(batch)
        self.assertTrue(torch.isfinite(loss).all())
        loss.sum().backward()
        for name, stage in (('channel', network.model[28].channel_mlp),
                            ('spatial', network.model[28].spatial)):
            grads = [p.grad for p in stage.parameters()]
            self.assertTrue(all(g is not None and torch.isfinite(g).all() for g in grads), name)
            self.assertGreater(sum(g.abs().sum().item() for g in grads), 0, name)
        self.assertEqual(params_before, tensor_digest(network.named_parameters()))
        # No optimizer.step(); BN running statistics only exist on this discarded copy.

    def test_trainer_rebuild_preserves_all_initialized_weights(self):
        from ultralytics.models.yolo.detect.train import DetectionTrainer
        trainer = object.__new__(DetectionTrainer)
        trainer.args = SimpleNamespace(cls_remap=False)
        trainer.data = {'nc': 2, 'channels': 3, 'names': {0: 'hat', 1: 'person'}}
        rebuilt = trainer.get_model(cfg=self.wrapper.model.yaml, weights=self.wrapper.model, verbose=False)
        self.assertEqual(tensor_digest(rebuilt.state_dict().items()),
                         tensor_digest(self.wrapper.model.state_dict().items()))

    def test_checkpoint_reload_with_explicit_custom_module_registration(self):
        from ultralytics import YOLO
        register_cbam()
        with tempfile.TemporaryDirectory(dir=ROOT/'project_work/p2_cbam_structure_2026-10-24') as tmp:
            path = Path(tmp)/'structure_only.pt'
            self.wrapper.save(path)  # Ultralytics stores half precision; compare accordingly.
            with torch.serialization.safe_globals([P2CBAM]):
                restored = YOLO(str(path), task='detect').model
            self.assertIsInstance(restored.model[28], P2CBAM)
            for key, value in self.wrapper.model.state_dict().items():
                torch.testing.assert_close(restored.state_dict()[key], value.half().float()
                                           if value.is_floating_point() else value, atol=0, rtol=0)
        self.assertFalse(path.exists())

    def test_build_preserves_cpu_rng(self):
        before = torch.get_rng_state().clone()
        build_cbam_model(seed=42)
        self.assertTrue(torch.equal(before, torch.get_rng_state()))


if __name__ == '__main__':
    unittest.main()
