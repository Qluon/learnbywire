import json
from pathlib import Path
import tempfile
import unittest
import torch
from trainerplus import DataConfig, TokenDataset, TrainingConfig, prepare_dataset
from trainerplus.data import TokenCollator
from helpers import tokenizer


class DataTests(unittest.TestCase):
    def setUp(self):
        self.tokenizer = tokenizer()

    def prepare(self, source, task='cpt', **options):
        return prepare_dataset(source, task=task, tokenizer=self.tokenizer,
                               config=DataConfig(max_length=8, **options))

    def test_cpt_fixed_loss_excludes_padding_and_first_token(self):
        row = self.prepare(['a b c'])[0]
        self.assertEqual(row['input_ids'].tolist(), [3, 4, 5, 2, 0, 0, 0, 0])
        self.assertEqual(row['labels'].tolist(), [-100, 4, 5, 2, -100, -100, -100, -100])

    def test_completion_mask_uses_single_tokenization_and_offsets(self):
        row = self.prepare([dict(prompt='question ', completion='answer')], 'sft')[0]
        self.assertEqual(row['labels'].tolist()[:3], [-100, 7, 2])
        # 'ab' straddles the prompt/completion boundary: never train the prompt.
        row = self.prepare([dict(prompt='a', completion='b answer')], 'sft')[0]
        self.assertEqual(row['input_ids'].tolist()[:3], [11, 7, 2])
        self.assertEqual(row['labels'].tolist()[:3], [-100, 7, 2])

    def test_assistant_mask_from_template_excludes_user_and_role_tokens(self):
        record = dict(messages=[dict(role='user', content='question'), dict(role='assistant', content='answer')])
        row = self.prepare([record], 'sft')[0]
        self.assertEqual(row['labels'].tolist()[:5], [-100, -100, -100, 7, 2])
        self.assertEqual(int(row['input_ids'].eq(2).sum()), 1)

    def test_missing_generation_blocks_requires_explicit_all_loss(self):
        self.tokenizer.chat_template = "{% for message in messages %}{{ message['content'] + ' ' }}{% endfor %}"
        record = dict(messages=[dict(role='user', content='question'), dict(role='assistant', content='answer')])
        with self.assertRaisesRegex(ValueError, 'generation blocks'):
            self.prepare([record], 'sft')
        self.assertEqual(self.prepare([record], 'sft', sft_loss='all')[0]['labels'][1].item(), 7)

    def test_truncation_is_explicit_and_no_targets_is_error(self):
        with self.assertRaisesRegex(ValueError, 'exceeds'):
            self.prepare(['a '*9])
        self.assertEqual(self.prepare(['a '*9], truncation='truncate').metadata['truncated_records'], 1)
        with self.assertRaisesRegex(ValueError, 'no supervised'):
            self.prepare([dict(prompt='a '*9, completion='answer')], 'sft', truncation='truncate')

    def test_dynamic_collator_padding_and_counts(self):
        dataset = self.prepare(['a b', 'a b c d'], padding='dynamic')
        self.assertFalse(dataset.fixed_length)
        batch = TokenCollator(0)([dataset[0], dataset[1]])
        self.assertEqual(tuple(batch['input_ids'].shape), (2, 5))
        self.assertTrue(batch['labels'][~batch['attention_mask']].eq(-100).all())
        self.assertEqual(dataset.counts(), dict(examples=2, input_tokens=8, supervised_tokens=6))
        with self.assertRaisesRegex(ValueError, 'fixed-length'):
            dataset.tensors()

    def test_packing_preserves_document_eos_and_reports_single_token_tail(self):
        dataset = self.prepare(['a b c', 'a b c d'], packing=True)
        self.assertEqual(len(dataset), 1)
        self.assertEqual(dataset.metadata['discarded_packing_tail_tokens'], 1)
        self.assertEqual(dataset[0]['input_ids'].tolist(), [3, 4, 5, 2, 3, 4, 5, 6])
        with self.assertRaisesRegex(ValueError, 'CPT only'):
            self.prepare([dict(prompt='a ', completion='b')], 'sft', packing=True)

    def test_local_json_jsonl_csv_are_equivalent(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root/'a.json').write_text(json.dumps([dict(text='a b')]))
            (root/'a.jsonl').write_text(json.dumps(dict(text='a b'))+'\n')
            (root/'a.csv').write_text('text\na b\n')
            self.assertEqual(len({self.prepare(path).fingerprint() for path in root.iterdir()}), 1)

    def test_dataset_copies_source_and_rejects_invalid_masks(self):
        ids = torch.tensor([[3, 4, 2]])
        dataset = TokenDataset(ids, torch.ones_like(ids), ids)
        digest = dataset.fingerprint()
        ids.zero_()
        dataset[0]['input_ids'].zero_()
        self.assertEqual(dataset.fingerprint(), digest)
        with self.assertRaisesRegex(ValueError, 'binary'):
            TokenDataset(ids, torch.full_like(ids, 2), ids)

    def test_invalid_settings_and_streaming_are_rejected(self):
        with self.assertRaises(ValueError):
            TrainingConfig(microbatch_size=True)
        with self.assertRaises(ValueError):
            TrainingConfig(lr_schedule='constant', warmup_ratio=0.1)
        with self.assertRaisesRegex(TypeError, 'finite'):
            self.prepare(iter(['a b']))


if __name__ == '__main__':
    unittest.main()
