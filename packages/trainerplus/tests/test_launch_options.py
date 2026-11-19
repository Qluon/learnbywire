"""Validate launch settings without importing ML libraries or initializing CUDA."""
from contextlib import redirect_stderr, redirect_stdout
import builtins
import importlib.util
import io
import os
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch


class BeforeTorchImport(Exception):
    pass


class LaunchOptionsTests(unittest.TestCase):
    def setUp(self):
        path = Path(__file__).resolve().parents[1]/'examples/train.py'
        spec = importlib.util.spec_from_file_location('trainerplus_example', path)
        self.example = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(self.example)
        self.base = ['train.py', '--model', '/unused/model', '--data', '/unused/data', '--task', 'cpt']

    def call_until_torch(self, args):
        original = builtins.__import__

        def intercept(name, *pos, **kwargs):
            if name == 'torch':
                raise BeforeTorchImport()
            return original(name, *pos, **kwargs)

        with patch.object(sys, 'argv', self.base+args), patch('builtins.__import__', side_effect=intercept):
            self.example.main()

    def test_explicit_allocator_is_set_before_torch_import(self):
        with patch.dict(os.environ, {'PYTORCH_CUDA_ALLOC_CONF': 'expandable_segments:False'}, clear=True):
            with self.assertRaises(BeforeTorchImport):
                self.call_until_torch(['--device', 'cuda', '--cuda-alloc-conf', 'expandable_segments:True'])
            self.assertEqual(os.environ['PYTORCH_ALLOC_CONF'], 'expandable_segments:True')
            self.assertNotIn('PYTORCH_CUDA_ALLOC_CONF', os.environ)

    def test_omitted_allocator_preserves_environment(self):
        settings = {'PYTORCH_CUDA_ALLOC_CONF': 'expandable_segments:False', 'UNRELATED': 'preserved'}
        with patch.dict(os.environ, settings, clear=True):
            with self.assertRaises(BeforeTorchImport):
                self.call_until_torch([])
            self.assertEqual(dict(os.environ), settings)

    def test_allocator_requires_cuda_and_nonempty_configuration(self):
        for args in (['--cuda-alloc-conf', 'expandable_segments:True'],
                     ['--device', 'cuda', '--cuda-alloc-conf', '  ']):
            with self.subTest(args=args), redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit) as error:
                    self.call_until_torch(args)
                self.assertEqual(error.exception.code, 2)

    def test_allocator_cannot_be_applied_after_cuda_initialization(self):
        fake = SimpleNamespace(cuda=SimpleNamespace(is_initialized=lambda: True))
        with patch.dict(sys.modules, {'torch': fake}), redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit) as error:
                self.call_until_torch(['--device', 'cuda', '--cuda-alloc-conf', 'expandable_segments:True'])
            self.assertEqual(error.exception.code, 2)

    def test_graph_grouping_requires_positive_size_and_explicit_control(self):
        cases = [
            ['--microbatches-per-replay', '2'],
            ['--device', 'cuda', '--control', 'optimizerplus', '--microbatches-per-replay', '0'],
            ['--device', 'cuda', '--control', 'optimizerplus', '--microbatches-per-replay', '-1'],
        ]
        for args in cases:
            with self.subTest(args=args), redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit) as error:
                    self.call_until_torch(args)
                self.assertEqual(error.exception.code, 2)

    def test_public_control_options_are_accepted_without_launching_training(self):
        with self.assertRaises(BeforeTorchImport):
            self.call_until_torch(['--device', 'cuda', '--control', 'optimizerplus',
                '--control-loss-reduction', 'microbatch_mean', '--microbatches-per-replay', '2'])

    def test_parallel_shape_rejected_before_model_import(self):
        invalid = [
            ['--microbatch-execution', 'parallel'],
            ['--device','cuda','--control','optimizerplus','--microbatch-execution','parallel'],
            ['--device','cuda','--control','optimizerplus','--microbatch-execution','parallel',
             '--microbatch','2','--accumulation','2','--lora-rank','16','--lora-targets','q_proj',
             '--control-loss-reduction','microbatch_mean'],
        ]
        for args in invalid:
            with self.subTest(args=args), redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit) as error:self.call_until_torch(args)
                self.assertEqual(error.exception.code,2)

    def test_qualified_parallel_settings_accepted_before_model_import(self):
        with self.assertRaises(BeforeTorchImport):
            self.call_until_torch(['--device','cuda','--control','optimizerplus',
                '--microbatch-execution','parallel','--microbatch','1','--accumulation','2',
                '--lora-rank','16','--lora-targets','q_proj','--control-loss-reduction','microbatch_mean'])

    def test_help_documents_both_launch_options(self):
        output = io.StringIO()
        with patch.object(sys, 'argv', ['train.py', '--help']), redirect_stdout(output):
            with self.assertRaises(SystemExit) as error:
                self.example.main()
        self.assertEqual(error.exception.code, 0)
        self.assertIn('--microbatch-execution', output.getvalue())
        self.assertIn('--cuda-alloc-conf', output.getvalue())
        self.assertIn('--microbatches-per-replay', output.getvalue())


if __name__ == '__main__':
    unittest.main()
