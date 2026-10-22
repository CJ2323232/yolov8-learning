import contextlib
import io
import json
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch

from helmet_models import (VARIANTS, build_initialized_model, initialization_spec,
                           shared_parameter_digest, tensor_digest)
from workflow import ROOT, sha256


class ModelInputTests(unittest.TestCase):
    def test_wrong_checkpoint_is_rejected_before_model_construction(self):
        with self.assertRaisesRegex(ValueError, '官方 yolov8n.pt'):
            initialization_spec(VARIANTS['p2'], ROOT/'models/helmet_yolov8n.pt')

    def test_project_yaml_cannot_silently_train_from_random_initialization(self):
        import run
        with contextlib.redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit):
                run.main(['train', '--model', str(VARIANTS['p2']), '--dry-run'])

    def test_model_variant_and_checkpoint_are_forwarded_to_training(self):
        import train_helmet
        with patch.object(train_helmet, 'run', side_effect=lambda args: args):
            args = train_helmet.main(['--variant', 'p2', '--dry-run'])
        self.assertEqual(args[args.index('--model')+1], str(VARIANTS['p2']))
        self.assertEqual(args[args.index('--pretrained')+1], str(ROOT/'yolov8n.pt'))
        self.assertIn('p2', args[args.index('--name')+1])

    def test_resume_cannot_apply_new_initialization(self):
        import run
        with contextlib.redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit):
                run.main(['train', '--model', str(ROOT/'yolov8n.pt'), '--pretrained', str(ROOT/'yolov8n.pt'),
                          '--resume', '--dry-run'])


class InitializedModelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not (ROOT/'yolov8n.pt').is_file():
            raise unittest.SkipTest('Download the pinned official yolov8n.pt before initialization tests')
        cls.old_threads = torch.get_num_threads()
        torch.set_num_threads(2)
        cls.models, cls.audits = {}, {}
        for variant in VARIANTS:
            cls.models[variant], cls.audits[variant] = build_initialized_model(
                VARIANTS[variant], ROOT/'yolov8n.pt', seed=0)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.old_threads)

    def test_both_variants_have_identical_common_pretrained_parameters(self):
        first, second = self.audits['standard'], self.audits['p2']
        self.assertEqual(first['shared_parameters_sha256'], second['shared_parameters_sha256'])
        self.assertEqual(first['loaded_shared_state_sha256'], second['loaded_shared_state_sha256'])
        self.assertEqual(first['loaded_state_tensors'], 210)
        self.assertTrue(first['remaining_state_unchanged'])
        self.assertTrue(second['remaining_state_unchanged'])
        self.assertTrue(all(int(k.split('.')[1]) < 16 for k in second['loaded_tensor_shapes']))

    def test_object_load_prevents_yaml_training_from_losing_initialization(self):
        from ultralytics.models.yolo.detect.train import DetectionTrainer
        for variant, wrapper in self.models.items():
            with self.subTest(variant=variant):
                self.assertIsNotNone(wrapper.ckpt)
                trainer = object.__new__(DetectionTrainer)
                trainer.args = SimpleNamespace(cls_remap=False)
                trainer.data = {'nc': 2, 'channels': 3, 'names': {0: 'hat', 1: 'person'}}
                rebuilt = trainer.get_model(cfg=wrapper.model.yaml, weights=wrapper.model, verbose=False)
                self.assertEqual(shared_parameter_digest(rebuilt), self.audits[variant]['shared_parameters_sha256'])
                self.assertEqual(tensor_digest(rebuilt.state_dict().items()),
                                 tensor_digest(wrapper.model.state_dict().items()))

    def test_initialization_preserves_callers_cpu_rng(self):
        torch.manual_seed(137)
        before = torch.get_rng_state().clone()
        build_initialized_model(VARIANTS['p2'], ROOT/'yolov8n.pt', seed=9)
        self.assertTrue(torch.equal(before, torch.get_rng_state()))

    def test_p2_exposes_four_scales_and_finite_synthetic_prediction(self):
        for variant, expected in [('standard', [8, 16, 32]), ('p2', [4, 8, 16, 32])]:
            network = self.models[variant].model.eval()
            self.assertEqual(network.model[-1].nc, 2)
            self.assertEqual([int(x) for x in network.stride], expected)
            shapes = []
            hook = network.model[-1].register_forward_pre_hook(
                lambda module, args: shapes.extend([tuple(t.shape[-2:]) for t in args[0]]))
            try:
                with torch.inference_mode():
                    output = network(torch.zeros(1, 3, 128, 128))
                prediction = output[0] if isinstance(output, tuple) else output
                if isinstance(prediction, torch.Tensor):
                    self.assertTrue(torch.isfinite(prediction).all())
            finally:
                hook.remove()
            self.assertEqual(shapes, [(128//s, 128//s) for s in expected])


if __name__ == '__main__':
    unittest.main()
