from dataclasses import replace
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import torch
from trainerplus import TrainerPlus, TrainingConfig
from helpers import data, model

torch.set_num_threads(1)


class NativeTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.cfg = TrainingConfig(output_dir=self.temp.name, max_steps=4,
            gradient_accumulation_steps=2, lr_schedule='cosine', warmup_ratio=0.25,
            logging_steps=2, save_model=False)

    def trainer(self, cfg=None, dataset=None, evaluation=None, seed=91):
        net = model(seed)
        return TrainerPlus(model=net, optimizer=torch.optim.AdamW(net.parameters(), lr=0.002),
                           train_dataset=dataset or data(), eval_dataset=evaluation,
                           args=cfg or self.cfg)

    def test_real_hf_trainer_completes_and_native_never_imports_control(self):
        import builtins
        original = builtins.__import__
        def guarded(name, *args, **kwargs):
            if name.startswith('optimizerplus'):
                raise AssertionError('native path imported proprietary code')
            return original(name, *args, **kwargs)
        trainer = self.trainer(evaluation=data(2))
        original_optimizer = trainer.optimizer
        before = {name: p.clone() for name, p in trainer.model.named_parameters()}
        with patch('builtins.__import__', side_effect=guarded):
            result = trainer.train()
        from transformers import Trainer
        self.assertIs(type(trainer._hf_trainer), Trainer)
        self.assertIs(trainer.optimizer, original_optimizer)
        self.assertEqual((result.status, result.committed_steps), ('complete', 4))
        self.assertEqual(result.metrics['input_tokens'], 32)
        self.assertEqual(result.metrics['supervised_tokens'], 24)
        self.assertEqual(result.metrics['evaluation']['supervised_tokens'], 6)
        self.assertTrue(any(not torch.equal(before[n], p) for n, p in trainer.model.named_parameters()))
        self.assertTrue((result.run_dir/'source/trainerplus/trainer.py').is_file())
        manifest = json.loads((result.run_dir/'manifest.json').read_text())
        self.assertEqual(manifest['contract']['control'], None)
        with self.assertRaisesRegex(RuntimeError, 'single-use'):
            trainer.train()

    def test_resume_matches_uninterrupted_weights_and_schedule_with_dropout(self):
        continuous = self.trainer()
        continuous.train()
        interrupted = self.trainer()
        partial = interrupted.train(stop_after_steps=2)
        self.assertEqual((partial.status, partial.committed_steps), ('stopped', 2))
        self.assertTrue(partial.checkpoint.is_dir())
        resumed = self.trainer(seed=17)
        final = resumed.train(resume_from_checkpoint=partial.checkpoint)
        self.assertNotEqual(partial.run_dir, final.run_dir)
        self.assertEqual(final.metrics['updates_this_run'], 2)
        self.assertEqual(final.metrics['input_tokens'], 16)
        for name, value in continuous.model.state_dict().items():
            torch.testing.assert_close(value, resumed.model.state_dict()[name], rtol=0, atol=0, msg=name)
        self.assertEqual(continuous._hf_trainer.lr_scheduler.state_dict(), resumed._hf_trainer.lr_scheduler.state_dict())

    def test_changed_workload_or_tampered_checkpoint_is_rejected(self):
        partial = self.trainer().train(stop_after_steps=2)
        changed = self.trainer(cfg=replace(self.cfg, seed=99))
        with self.assertRaisesRegex(ValueError, 'workload mismatch'):
            changed.train(resume_from_checkpoint=partial.checkpoint)
        state = partial.checkpoint/'trainer_state.json'
        state.write_text(state.read_text()+' ')
        with self.assertRaisesRegex(ValueError, 'integrity mismatch'):
            self.trainer().train(resume_from_checkpoint=partial.checkpoint)

    def test_periodic_evaluation_preserves_rng_and_training(self):
        plain = self.trainer()
        plain.train()
        evaluated = self.trainer(cfg=replace(self.cfg, eval_steps=1), evaluation=data(2))
        result = evaluated.train()
        for name, value in plain.model.state_dict().items():
            torch.testing.assert_close(value, evaluated.model.state_dict()[name], rtol=0, atol=0)
        events = [json.loads(line) for line in (result.run_dir/'metrics.jsonl').read_text().splitlines()]
        self.assertEqual(sum(event['event'] == 'evaluation' for event in events), 4)
        self.assertTrue(evaluated.model.training)

    def test_dynamic_padding_trains_and_export_reloads(self):
        trainer = self.trainer(cfg=replace(self.cfg, save_model=True), dataset=data(dynamic=True))
        result = trainer.train()
        from transformers import GPT2LMHeadModel
        restored = GPT2LMHeadModel.from_pretrained(result.model_path, local_files_only=True)
        for name, value in trainer.model.state_dict().items():
            torch.testing.assert_close(value, restored.state_dict()[name])

    def test_two_tiny_decoder_families_use_public_trainer(self):
        from transformers import LlamaConfig, LlamaForCausalLM, Qwen2Config, Qwen2ForCausalLM
        for config_type, model_type in [(LlamaConfig, LlamaForCausalLM), (Qwen2Config, Qwen2ForCausalLM)]:
            with self.subTest(model=model_type.__name__):
                config = config_type(vocab_size=16, hidden_size=16, intermediate_size=32,
                    num_hidden_layers=1, num_attention_heads=2, num_key_value_heads=1,
                    max_position_embeddings=32, use_cache=False, pad_token_id=0)
                net = model_type(config)
                trainer = TrainerPlus(model=net, optimizer=torch.optim.AdamW(net.parameters(), lr=0.001),
                    train_dataset=data(), args=replace(self.cfg, max_steps=2))
                self.assertEqual(trainer.train().committed_steps, 2)

    def test_partial_groups_require_explicit_drop_and_never_silently_repeat(self):
        with self.assertRaisesRegex(ValueError, 'Incomplete'):
            self.trainer(dataset=data(9)).train()
        with self.assertRaisesRegex(ValueError, 'exceeds'):
            self.trainer(cfg=replace(self.cfg, max_steps=5)).train()
        result = self.trainer(cfg=replace(self.cfg, drop_last=True), dataset=data(9)).train()
        self.assertEqual(result.metrics['examples'], 8)

    def test_accumulation_matches_full_batch_with_unequal_target_counts(self):
        from trainerplus import TokenDataset
        rows = [data()[i] for i in range(4)]
        rows[0]['labels'][1:3] = -100
        rows[2]['labels'][1] = -100
        dataset = TokenDataset.from_rows(rows)
        accumulated = self.trainer(cfg=replace(self.cfg, max_steps=1, lr_schedule='constant', warmup_ratio=0),
                                   dataset=dataset)
        full = self.trainer(cfg=replace(self.cfg, max_steps=1, microbatch_size=2,
                           gradient_accumulation_steps=1, lr_schedule='constant', warmup_ratio=0), dataset=dataset)
        gradients = []
        for trainer in (accumulated, full):
            for module in trainer.model.modules():
                if isinstance(module, torch.nn.Dropout):
                    module.p = 0.0
            captured = {}
            def capture(optimizer, args, kwargs):
                captured.update({name:p.grad.clone() for name,p in trainer.model.named_parameters()})
            hook = trainer.optimizer.register_step_pre_hook(capture)
            trainer.train()
            hook.remove()
            gradients.append(captured)
        for name, value in gradients[0].items():
            torch.testing.assert_close(value, gradients[1][name], atol=1e-7, rtol=1e-5)

    def test_peft_lora_keeps_base_frozen_and_exports_adapter(self):
        from peft import LoraConfig, get_peft_model
        from transformers import LlamaConfig, LlamaForCausalLM
        config = LlamaConfig(vocab_size=16, hidden_size=16, intermediate_size=32,
            num_hidden_layers=1, num_attention_heads=2, num_key_value_heads=1,
            max_position_embeddings=32, use_cache=False, pad_token_id=0)
        net = get_peft_model(LlamaForCausalLM(config), LoraConfig(r=2, lora_alpha=4,
                            target_modules=['q_proj', 'v_proj'], task_type='CAUSAL_LM'))
        before = {name:p.clone() for name, p in net.named_parameters() if not p.requires_grad}
        trainer = TrainerPlus(model=net,
            optimizer=torch.optim.AdamW([p for p in net.parameters() if p.requires_grad], lr=0.001),
            train_dataset=data(), args=replace(self.cfg, max_steps=2, save_model=True))
        result = trainer.train()
        self.assertTrue((result.model_path/'adapter_model.safetensors').is_file())
        for name, parameter in net.named_parameters():
            if name in before:
                self.assertTrue(torch.equal(before[name], parameter))


if __name__ == '__main__':
    unittest.main()
