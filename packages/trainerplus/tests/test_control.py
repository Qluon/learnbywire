"""Contract tests with a test double; these do NOT qualify real GPU execution."""
from dataclasses import replace
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import torch
from trainerplus import TrainerPlus, TrainingConfig
from trainerplus.integration import load_optimizerplus
from helpers import data, model


class ControlTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.calls = []
        self.sessions = []
        owner = self

        class Batches:
            def __init__(self, **kwargs):
                owner.batch_options = kwargs

        class Session:
            def __init__(self, optimizer, **kwargs):
                owner.calls.append((optimizer, kwargs))
                owner.sessions.append(self)
                self.total = kwargs['total_steps']
                self.committed_steps = 2  # Simulated multi-update startup.
                if kwargs.get('resume_from'):
                    self.committed_steps = json.loads((Path(kwargs['resume_from'])/'state.json').read_text())['step']
                self.closed = False
                self.runtime_summary = dict(full_stack=True, test_double=True)

            def __iter__(self):
                while self.committed_steps < self.total:
                    yield {'mock_batch': True}

            def train_step(self, batch):
                self.committed_steps = min(self.committed_steps+2, self.total)
                return torch.tensor(1.0)

            def save_checkpoint(self, path, *, workload_id):
                assert not self.closed
                path.mkdir()
                (path/'state.json').write_text(json.dumps(dict(step=self.committed_steps, workload=workload_id)))

            def close(self):
                self.closed = True

        self.control = Session
        self.module = SimpleNamespace(__version__='1.2.2', OptimizerPlus=Session, TokenBatches=Batches)
        self.cfg = TrainingConfig(output_dir=self.temp.name, max_steps=6,
            gradient_accumulation_steps=2, mixed_precision='bf16', save_model=False,
            lr_schedule='cosine', warmup_ratio=0.25)

    def trainer(self, *, control_options=None, **changes):
        net = model()
        return TrainerPlus(model=net, optimizer=torch.optim.AdamW(net.parameters(), lr=0.002),
            train_dataset=data(12), args=replace(self.cfg, **changes), control=self.control,
            control_options=control_options)

    def patched(self):
        from contextlib import ExitStack
        stack = ExitStack()
        stack.enter_context(patch.object(TrainerPlus, '_validate_control_device'))
        stack.enter_context(patch('trainerplus.trainer.load_optimizerplus', return_value=self.module))
        return stack

    def test_public_session_owns_updates_and_startup_counts(self):
        trainer = self.trainer()
        def forbidden_step(*args, **kwargs):
            raise AssertionError('outer step')
        with self.patched(), patch.object(trainer.optimizer, 'step', new=forbidden_step):
            result = trainer.train()
        self.assertIs(self.calls[0][0], trainer.optimizer)
        options = self.calls[0][1]
        self.assertEqual(options['lr_schedule'], 'cosine')
        self.assertEqual(options['warmup_ratio'], 0.25)
        self.assertEqual(options['loss_reduction'], 'token_mean')
        self.assertEqual((self.batch_options['examples_per_update'], self.batch_options['microbatch_size']), (2, 1))
        self.assertEqual(result.metrics['updates_this_run'], 6)
        self.assertEqual(result.metrics['examples'], 12)
        self.assertTrue(self.sessions[0].closed)

    def test_stop_saves_actual_transaction_boundary_and_resume_same_budget(self):
        with self.patched():
            partial = self.trainer().train(stop_after_steps=3)
            self.assertEqual(partial.committed_steps, 4)
            resumed = self.trainer().train(resume_from_checkpoint=partial.checkpoint)
        self.assertEqual(resumed.committed_steps, 6)
        self.assertEqual(resumed.metrics['updates_this_run'], 2)
        self.assertEqual(self.calls[-1][1]['total_steps'], 6)
        self.assertEqual(self.calls[-1][1]['resume_from'], partial.checkpoint)
        self.assertEqual(len(self.calls[-1][1]['workload_id']), 64)
        self.assertTrue(all(session.closed for session in self.sessions))

    def test_session_is_closed_after_training_failure(self):
        with self.patched(), patch.object(self.control, 'train_step', side_effect=RuntimeError('test failure')):
            with self.assertRaisesRegex(RuntimeError, 'test failure'):
                self.trainer().train()
        self.assertTrue(self.sessions[0].closed)

    def test_constant_maps_to_no_external_schedule(self):
        with self.patched():
            self.trainer(lr_schedule='constant', warmup_ratio=0).train()
        self.assertIsNone(self.calls[0][1]['lr_schedule'])

    def test_loss_reduction_reaches_session_and_workload_manifest(self):
        for reduction in ('token_mean', 'microbatch_mean'):
            with self.subTest(reduction=reduction), self.patched():
                supplied = {'loss_reduction': reduction, 'runtime': 'fast'}
                trainer = self.trainer(control_options=supplied,
                                       lr_schedule='constant', warmup_ratio=0)
                trainer.train()
                options = self.calls[-1][1]
                self.assertEqual(options['loss_reduction'], reduction)
                self.assertEqual(options['runtime'], 'fast')
                self.assertIsNone(options['lr_schedule'])
                self.assertEqual(options['warmup_ratio'], 0)
                self.assertEqual(trainer._workload['control_options']['loss_reduction'], reduction)
                manifest = json.loads((trainer.run_dir/'manifest.json').read_text())
                self.assertEqual(manifest['contract']['control_options']['loss_reduction'], reduction)
                self.assertEqual(supplied, {'loss_reduction': reduction, 'runtime': 'fast'})

    def test_invalid_loss_reduction_fails_before_session_or_run(self):
        for reduction in ('sum', 'example_mean', None, True, [], {}):
            with self.subTest(reduction=reduction):
                with self.assertRaisesRegex(ValueError, 'loss_reduction'):
                    self.trainer(control_options={'loss_reduction': reduction})
        self.assertFalse(self.calls)
        self.assertFalse(list(Path(self.temp.name).iterdir()))

    def test_default_is_explicit_in_contract_without_mutating_user_options(self):
        supplied = {'runtime': 'fast'}
        trainer = self.trainer(control_options=supplied)
        self.assertEqual(trainer.control_options['loss_reduction'], 'token_mean')
        self.assertEqual(supplied, {'runtime': 'fast'})

    def test_microbatch_checkpoint_resumes_only_with_matching_reduction(self):
        with self.patched():
            partial = self.trainer(control_options={'loss_reduction': 'microbatch_mean'}).train(stop_after_steps=3)
            calls_before = len(self.calls)
            with self.assertRaisesRegex(ValueError, 'Checkpoint workload mismatch'):
                self.trainer().train(resume_from_checkpoint=partial.checkpoint)
            self.assertEqual(len(self.calls), calls_before)
            resumed = self.trainer(control_options={'loss_reduction': 'microbatch_mean'}).train(
                resume_from_checkpoint=partial.checkpoint)
            self.assertEqual(resumed.committed_steps, 6)
            self.assertEqual(self.calls[-1][1]['loss_reduction'], 'microbatch_mean')

    def test_implicit_and_explicit_default_checkpoint_contracts_match(self):
        with self.patched():
            partial = self.trainer().train(stop_after_steps=3)
            resumed = self.trainer(control_options={'loss_reduction': 'token_mean'}).train(
                resume_from_checkpoint=partial.checkpoint)
        self.assertEqual(resumed.committed_steps, 6)

    def test_control_reduction_cannot_silently_change_native_training(self):
        net = model()
        with self.assertRaisesRegex(ValueError, 'control_options require an explicit control'):
            TrainerPlus(model=net, optimizer=torch.optim.AdamW(net.parameters()),
                train_dataset=data(12), control_options={'loss_reduction': 'microbatch_mean'})

    def test_cpu_control_and_nonfixed_inputs_are_rejected(self):
        with self.assertRaisesRegex(ValueError, 'CUDA'):
            self.trainer().train()
        net = model()
        with self.assertRaisesRegex(ValueError, 'fixed padding'):
            TrainerPlus(model=net, optimizer=torch.optim.AdamW(net.parameters()), control=self.control,
                train_dataset=data(12, dynamic=True), args=self.cfg)

    def test_version_gate_and_nested_import_errors(self):
        with patch('trainerplus.integration.importlib.import_module', return_value=self.module):
            self.assertIs(load_optimizerplus(), self.module)
        with patch('trainerplus.integration.importlib.import_module', return_value=SimpleNamespace(__version__='1.1')):
            with self.assertRaisesRegex(RuntimeError, '1.2.2'):
                load_optimizerplus()
        error = ModuleNotFoundError('missing internal dependency', name='dependency')
        with patch('trainerplus.integration.importlib.import_module', side_effect=error):
            with self.assertRaises(ModuleNotFoundError) as context:
                load_optimizerplus()
        self.assertIs(context.exception, error)


if __name__ == '__main__':
    unittest.main()
