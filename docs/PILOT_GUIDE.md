# Run a useful customer evaluation

Start with LearnByWire 1.2.2, OptimizerPlus 1.2.2 and TrainerPlus 1.1.1. OptimizerPlus controls a supplied public optimizer such as AdamW; TrainerPlus handles data preparation and training orchestration. The examples are starting points for your own compatible model, data and settings.

## First successful run

Complete [installation, licence activation and the runnable example](../GETTING_STARTED.md). Use the synthetic example to verify loading, tokenization, training, validation and adapter export. Its speed and loss are not evidence of task quality or production speedup.

Record `status: complete`, the planned number of committed updates and the exported adapter location. Keep the base model ID/revision: a LoRA adapter is not a standalone base model. Use [the reload example](USER_GUIDE.md#saving-and-reusing-the-result) to verify that the export can be used.

## Choose the real comparison before running

Agree one workload and the outcome that matters: for example, validation loss or exact-match accuracy at a fixed token budget, together with elapsed time. Choose the acceptance threshold before inspecting results. No particular speedup or accuracy gain is guaranteed.

| Keep matched | Record explicitly |
|---|---|
| Model revision and adapter initialization | LoRA rank, alpha, target modules and dropout |
| Training data, order, tokenizer and masking | Prepared rows, input tokens and supervised tokens |
| Public base optimizer and starting configuration | Learning rate, betas, weight decay, clipping and schedule |
| Batch size and training budget | Microbatch, accumulation, updates and epochs |
| Evaluation | Dataset, metric, generation settings and evaluation timing |
| Hardware and timing | GPU, visible devices, power settings, cache state and measured time boundary |

Run the same public first-run script with `--control native` and `--control optimizerplus`, using different output directories. The native path uses public Hugging Face Trainer and the same public AdamW constructor. An optional `--native-compile` arm is a separate baseline. Record actual loss normalization and any execution differences; matched inputs alone do not prove numerical identity.

## Read the metrics correctly

| Metric | What to use |
|---|---|
| Task quality | Held-out validation and the task's own evaluator; not synthetic demonstration loss |
| Training seconds | `result.json` → `metrics.training_seconds`; includes engine/session setup and any periodic evaluation/saves |
| Trainer workflow seconds | `metrics.workflow_seconds`; adds final evaluation/export and trainer bookkeeping, but excludes earlier model loading/tokenization/order preparation |
| Complete process seconds | An external timer around the entire Python invocation; state whether model download is included |
| Tokens per second | Use actual input or supervised tokens divided by the chosen time boundary; label the numerator and denominator |
| GPU-hours | Elapsed seconds × number of GPUs used / 3600, for the same boundary in both arms |
| Peak allocated/reserved memory | Collect per-process CUDA allocator peaks with the same reset points; these are distinct from device-wide NVML memory |
| Energy | Integrate measured GPU power over the stated interval; do not estimate watt-hours from GPU-hours alone |
| Retention | Evaluate a separately defined held-out reference corpus before and after training |

The generic first-run script does not collect all optional telemetry above. Mark unmeasured values as unavailable. Do not fill them from another run. Its `input_tokens_per_second` uses training seconds, not complete process seconds. Compare complete workflow throughput only after recording matching complete workflow boundaries.

Use this reporting order consistently:

| Metric | Native | OptimizerPlus | Change versus Native |
|---|---:|---:|---:|
| Held-out task metric | | | |
| Complete process seconds | | | |
| Valid input tokens / process second | | | |
| GPU-hours | | | |
| Peak allocated / reserved memory | | | |
| Measured GPU energy | | | |

For accuracy report percentage-point change as well as relative change when useful. For elapsed time report `(Native − OptimizerPlus) / Native × 100`; for throughput report `(OptimizerPlus / Native − 1) × 100`. Speedup is `Native seconds / OptimizerPlus seconds` at the same workload. Report failures and omitted measurements alongside successes.

## Follow-up

Send a reviewed [support report](SUPPORT_REPORT.md) after the installation check. Agree a follow-up after the first real workload and a final review before the licence expires. Share run summaries and evaluation outcomes through your agreed private channel. Raw customer data and weights stay in your environment unless separately authorized.

A first seed is useful for integration. Confirm a promising result with additional matched seeds and, where practical, alternate execution order to reduce temperature/cache effects. Conclude whether the agreed outcome was met, partially met, not met or could not be assessed. Distinguish integration success from a measured performance benefit.
