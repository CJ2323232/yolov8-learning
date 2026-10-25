"""Training contract tests: never start a real training epoch."""
import copy
import contextlib
import io
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import train_p2_cbam as entry
from workflow import ROOT

WORK = ROOT/'project_work/p2_cbam_training_2026-10-25'
WORK.mkdir(parents=True, exist_ok=True)


class FrozenProtocolTests(unittest.TestCase):
    def test_all_locked_parameters_equal_historical_p2(self):
        import yaml
        p = entry.protocol()
        path = ROOT/p['historical_reference']['args_file']
        if not path.exists():
            self.skipTest('Original history is not part of the small Windows update')
        historical = yaml.safe_load(path.read_text(encoding='utf-8'))
        self.assertEqual(entry.sha256(path), p['historical_reference']['args_sha256'])
        entry.check_parameters(historical, p['train'])
        entry.check_parameters(historical, p['operational'])

    def test_all_changes_and_missing_keys_are_rejected(self):
        expected = entry.protocol()['train']
        for key in expected:
            altered = dict(expected)
            altered.pop(key)
            with self.subTest(key=key), self.assertRaisesRegex(ValueError, '参数不一致'):
                entry.check_parameters(altered, expected)

    def test_training_flags_cannot_override_locked_protocol(self):
        for flags in (['--epochs','1'], ['--batch','1'], ['--lr0','.01'],
                      ['--seed','1'], ['--data','other.yaml'], ['--pretrained','old.pt']):
            with self.subTest(flags=flags), contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                entry.main(flags)

    def test_cpu_cannot_accidentally_start_training(self):
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            entry.main(['--device','cpu'])

    def test_legacy_entry_routes_cbam_and_rejects_960(self):
        import train_helmet
        with patch.object(entry, 'main', return_value='checked') as call:
            self.assertEqual(train_helmet.main(['--variant','p2_cbam','--check']), 'checked')
            call.assert_called_once_with(['--check'])
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            train_helmet.main(['--variant','p2_cbam','--profile','highres960'])

    def test_oom_batch_reduction_is_rejected_before_pipeline_rebuild(self):
        from ultralytics.models.yolo.detect import DetectionTrainer
        cls = entry.trainer_class(ROOT/'runs/test', ROOT/'data.yaml')
        trainer = object.__new__(cls)
        trainer.batch_size = 1
        trainer.args = SimpleNamespace(batch=1)
        with patch.object(DetectionTrainer, '_build_train_pipeline') as build:
            with self.assertRaisesRegex(ValueError, 'batch=2'):
                trainer._build_train_pipeline()
            build.assert_not_called()

    def test_resume_trainer_rebases_paths_only(self):
        from ultralytics.models.yolo.detect import DetectionTrainer
        cls = entry.trainer_class(ROOT/'runs/new_location', ROOT/'project_work/local.yaml')
        trainer = object.__new__(cls)
        trainer.args = SimpleNamespace(batch=2, lr0=.001667, seed=0)
        with patch.object(DetectionTrainer, 'check_resume'):
            trainer.check_resume({})
        self.assertEqual(trainer.args.batch, 2)
        self.assertEqual(trainer.args.lr0, .001667)
        self.assertEqual(trainer.args.data, str(ROOT/'project_work/local.yaml'))


class ResumeTests(unittest.TestCase):
    def setUp(self):
        self.p = entry.protocol()
        self.identity = {'manifest_sha256': self.p['dataset_manifest_sha256']}
        self.files = {'test.py':'hash'}
        self.prior = {'dataset': self.identity, 'code_identity':self.files,
                      'variant':'p2_cbam', 'protocol':self.p}
        self.ckpt = {'epoch':13, 'train_args':dict(self.p['train']),
                     'optimizer':{'state':{1:{'step':10}}}, 'ema':object(),
                     'scaler':{'scale':65536.}, 'version':'8.4.164'}

    def verify(self, ckpt=None, prior=None):
        return entry.validate_resume(ckpt or self.ckpt, prior or self.prior, self.p,
                                     self.identity, self.files, 'p2_cbam')

    def test_resume_restores_fourteen_completed_epochs(self):
        self.assertEqual(self.verify(), 14)

    def test_finished_or_stripped_checkpoints_never_restart(self):
        for epoch in (-1,49,50):
            self.ckpt['epoch']=epoch
            with self.subTest(epoch=epoch), self.assertRaises(ValueError):
                self.verify()

    def test_missing_optimizer_ema_or_scaler_rejected(self):
        for key in ('optimizer','ema','scaler'):
            ckpt=dict(self.ckpt)
            ckpt[key]=None
            with self.subTest(key=key), self.assertRaises(ValueError):
                self.verify(ckpt)

    def test_changed_dataset_code_or_variant_rejected(self):
        for key in ('dataset','code_identity','variant','protocol'):
            prior=dict(self.prior)
            prior[key]='changed'
            with self.subTest(key=key), self.assertRaises(ValueError):
                self.verify(prior=prior)

    def test_changed_hyperparameter_checkpoint_rejected(self):
        self.ckpt['train_args']['batch']=1
        with self.assertRaises(ValueError):
            self.verify()

    def test_csv_requires_exact_epoch_sequence(self):
        with tempfile.TemporaryDirectory(dir=WORK) as tmp:
            p=Path(tmp)/'results.csv'
            p.write_text('epoch,time\n1,0\n2,0\n')
            entry.check_history(p,2)
            with self.assertRaises(ValueError):
                entry.check_history(p,3)
            p.write_text('epoch,time\n1,0\n3,0\n')
            with self.assertRaises(ValueError):
                entry.check_history(p,2)


class DispatchTests(unittest.TestCase):
    def test_check_never_calls_train_and_existing_run_is_protected(self):
        import helmet_cbam
        from unittest.mock import MagicMock
        model=MagicMock()
        frozen=entry.protocol()
        with tempfile.TemporaryDirectory(dir=WORK) as tmp:
            root=Path(tmp)
            patches=[patch.object(entry,'protocol',return_value=frozen),patch.object(entry,'ROOT',root),patch.object(entry,'check_runtime',return_value={}),
                     patch.object(entry,'prepare_data',return_value=(root/'data.yaml',{})),
                     patch.object(entry,'code_identity',return_value={}),
                     patch.object(entry,'check_structure',return_value={}),
                     patch.object(helmet_cbam,'build_cbam_model',return_value=(model,{}))]
            with contextlib.ExitStack() as stack:
                for p in patches: stack.enter_context(p)
                with contextlib.redirect_stdout(io.StringIO()):
                    entry.main(['--check','--device','cpu'])
                model.train.assert_not_called()
                (root/'runs'/entry.DEFAULT_NAME).mkdir(parents=True)
                with self.assertRaises(FileExistsError):
                    entry.main(['--check','--device','cpu'])


    def test_formal_dispatch_checks_effective_parameters_before_any_update(self):
        import helmet_cbam
        from unittest.mock import MagicMock
        frozen=entry.protocol()
        class EndBeforeEpoch(Exception): pass
        for changed in (False, True):
            with self.subTest(changed=changed), tempfile.TemporaryDirectory(dir=WORK) as tmp:
                root=Path(tmp)
                model=MagicMock()
                callbacks={}
                model.add_callback.side_effect=lambda key, fn: callbacks.update({key:fn})
                def train(**kwargs):
                    entry.check_parameters(kwargs,frozen['train'])
                    self.assertFalse(kwargs['resume'])
                    self.assertEqual(kwargs['device'],'0')
                    actual={k:v for k,v in kwargs.items() if k not in ('trainer','resume')}
                    if changed: actual['batch']=1
                    trainer=SimpleNamespace(args=SimpleNamespace(**actual),
                        save_dir=kwargs['save_dir'], start_epoch=0, amp=True,
                        batch_size=2, accumulate=32, model=model.model)
                    callbacks['on_train_start'](trainer)
                    raise EndBeforeEpoch()
                model.train.side_effect=train
                with contextlib.ExitStack() as stack:
                    for patcher in [patch.object(entry,'ROOT',root),
                        patch.object(entry,'protocol',return_value=frozen),
                        patch.object(entry,'check_runtime',return_value={}),
                        patch.object(entry,'prepare_data',return_value=(root/'data.yaml',{})),
                        patch.object(entry,'code_identity',return_value={}),
                        patch.object(entry,'check_structure',return_value={}),
                        patch.object(helmet_cbam,'build_cbam_model',return_value=(model,{}))]:
                        stack.enter_context(patcher)
                    with contextlib.redirect_stdout(io.StringIO()), self.assertRaises(ValueError if changed else EndBeforeEpoch):
                        entry.main([])
                output=root/'runs'/entry.DEFAULT_NAME
                self.assertTrue((output/'experiment_identity.json').exists())
                self.assertEqual((output/'launch_events.json').exists(), not changed)
                self.assertFalse((output/'completed.json').exists())


if __name__ == '__main__':
    unittest.main()
