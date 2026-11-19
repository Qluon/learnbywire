"""Exact loss reduction and accuracy with bounded evaluation workspace."""
import unittest
import torch
import torch.nn.functional as F
from trainerplus.evaluation import _dense_loss_and_correct, evaluate


class EvaluationWorkspaceTests(unittest.TestCase):
    def compare(self, device, shape, dtype, chunk_rows):
        torch.manual_seed(4007)
        logits = (torch.randn(shape, device=device) * 4).to(dtype)
        labels = torch.randint(shape[-1], shape[:2], device=device)[:, 1:]
        # Include ties and ragged chunk boundaries.
        logits[0, 0].zero_()
        with torch.inference_mode():
            shifted = logits[:, :-1].float()
            expected = F.cross_entropy(shifted.reshape(-1, shape[-1]), labels.reshape(-1), reduction='sum')
            expected_correct = (shifted.argmax(-1) == labels).sum()
            actual, correct = _dense_loss_and_correct(logits, labels, chunk_rows)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        torch.testing.assert_close(correct, expected_correct, rtol=0, atol=0)

    def test_cpu_shapes_dtypes_and_boundaries(self):
        torch.set_num_threads(1)
        for dtype in (torch.bfloat16, torch.float32):
            for shape, chunk in [((2, 18, 67), 1), ((2, 130, 257), 64), ((2, 34, 151936), 16)]:
                with self.subTest(dtype=dtype, shape=shape, chunk=chunk):
                    self.compare('cpu', shape, dtype, chunk)

    @unittest.skipUnless(torch.cuda.is_available(), 'CUDA evaluation regression')
    def test_cuda_shapes_dtypes_and_boundaries(self):
        for dtype in (torch.bfloat16, torch.float32):
            for shape, chunk in [((2, 18, 67), 1), ((2, 130, 257), 64), ((2, 512, 151936), 64)]:
                with self.subTest(dtype=dtype, shape=shape, chunk=chunk):
                    self.compare('cuda', shape, dtype, chunk)

    def test_invalid_chunk_size_rejected_before_model_execution(self):
        records = {'labels': torch.zeros(2, 4, dtype=torch.long)}
        for value in (0, -1, True, 1.5):
            with self.subTest(value=value), self.assertRaisesRegex(ValueError, 'loss_chunk_rows'):
                evaluate(None, records, (2,), loss_chunk_rows=value)


if __name__ == '__main__':
    unittest.main()
