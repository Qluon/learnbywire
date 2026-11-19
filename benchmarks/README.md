# Performance evidence: LearnByWire 1.2.2

These measurements use OptimizerPlus 1.2.2 and TrainerPlus 1.1.1. They compare complete training systems on one GPU and one seed per model. They do not isolate an optimizer-only or trainer-only effect.

All arms use public fused AdamW, the same initial adapter within each model, identical token tensors/order, one epoch, 7,812 updates, 7,999,488 valid input tokens, sequence length 512, microbatch 1 and accumulation 2. Learning rate is constant 2e-4 with no external warmup. LoRA uses rank 16, alpha 32 and dropout 0. Base weights are BF16 and adapters FP32. Validation and retention use the same fixed panels and 33 evaluation checkpoints.

Full-workflow timing includes model loading, setup/compilation, training, evaluation and adapter export; shared download/tokenization is excluded. Data uses pinned FineWeb-Edu and WikiText sources. Each model uses its own tokenizer; raw-text coverage therefore differs at a fixed token budget. Compare perplexity within each model.

The tested GPU was an RTX 5090 Laptop GPU (24 GiB, SM120), with Python 3.12, PyTorch 2.11.0+cu128, Transformers 5.8.0, PEFT 0.19.1 and Accelerate 1.13.0.

## TinyLlama 1.1B

| Metric | TrainerPlus native eager | TrainerPlus native compiled | TrainerPlus + OptimizerPlus |
|---|---:|---:|---:|
| Full-workflow tokens/s | 2,202.7 | 4,917.4 | 9,981.3 |
| Workflow seconds | 3,631.68 | 1,626.78 | 801.45 |
| GPU-hours | 1.0088 | 0.4519 | 0.2226 |
| Original validation PPL | 9.675793 | 9.675793 | 9.675793 |
| Final validation PPL | 9.798278 | 9.795411 | 9.420217 |
| Original retention PPL | 13.789109 | 13.789109 | 13.789109 |
| Final retention PPL | 14.166541 | 14.212439 | 13.737172 |
| GPU energy (Wh) | 84.81 | 51.56 | 31.43 |
| Peak allocated VRAM (GiB) | 3.509 | 3.162 | 3.413 |
| Peak reserved VRAM (GiB) | 3.631 | 3.336 | 4.281 |

## SmolLM2 1.7B

| Metric | TrainerPlus native eager | TrainerPlus native compiled | TrainerPlus + OptimizerPlus |
|---|---:|---:|---:|
| Full-workflow tokens/s | 2,167.3 | 4,568.3 | 7,140.6 |
| Workflow seconds | 3,690.94 | 1,751.08 | 1,120.28 |
| GPU-hours | 1.0253 | 0.4864 | 0.3112 |
| Original validation PPL | 10.018842 | 10.018842 | 10.018842 |
| Final validation PPL | 10.092818 | 10.090511 | 10.015581 |
| Original retention PPL | 16.525959 | 16.525959 | 16.525959 |
| Final retention PPL | 17.253361 | 17.208597 | 17.163446 |
| GPU energy (Wh) | 100.82 | 64.01 | 46.23 |
| Peak allocated VRAM (GiB) | 5.245 | 4.685 | 5.093 |
| Peak reserved VRAM (GiB) | 5.328 | 4.795 | 6.375 |

## Direct HF Trainer check: TinyLlama

The direct baseline instantiates public `transformers.Trainer` without importing TrainerPlus, OptimizerPlus or LearnByWire. It uses public `default_data_collator`, fused PyTorch AdamW and `torch_compile=True`. Training settings, initial adapter, data order and evaluation panels match the TrainerPlus-hosted native run. Evaluation is a shared standalone benchmark utility; this is not a third-party replication.

| Metric | Direct HF compiled | TrainerPlus native compiled | TrainerPlus + OptimizerPlus |
|---|---:|---:|---:|
| Full-workflow tokens/s | 4,779.2 | 4,917.4 | 9,981.3 |
| Workflow seconds | 1,673.81 | 1,626.78 | 801.45 |
| GPU-hours | 0.4649 | 0.4519 | 0.2226 |
| Validation PPL | 9.795411 | 9.795411 | 9.420217 |
| Retention PPL | 14.212439 | 14.212439 | 13.737172 |
| GPU energy (Wh) | 53.59 | 51.56 | 31.43 |
| Peak allocated VRAM (GiB) | 3.162 | 3.162 | 3.413 |
| Peak reserved VRAM (GiB) | 3.336 | 3.336 | 4.281 |

All 33 evaluation results and exported adapter tensors match exactly between direct HF and TrainerPlus-native. The 2.9% observed TrainerPlus-native throughput difference is not evidence of a repeatable TrainerPlus speed advantage. The complete OptimizerPlus system has 2.09x throughput and 52.1% lower workflow time than this direct HF observation, with 41.4% lower GPU energy and 28.3% higher reserved VRAM.

## Interpreting these measurements

The extension study has one seed per model/arm and uses the same GPU at different times. Both models are Llama-style decoders. These measurements do not predict speed or quality on every workload, and do not restrict which compatible workloads users may try. The measured OptimizerPlus wheel uses the QA verification key and a valid signed QA entitlement. Customer-key and portable installation checks are reported separately. Exact-match weights in the HF check apply to its tested configuration.

SmolLM2's OptimizerPlus retention PPL is 17.163 versus 16.526 before training: retention worsened from the original model despite beating the trained baselines. GPU energy is sampled whole-device energy, excluding CPU and wall-power overhead. Greater reserved VRAM is a real trade-off.

Machine-readable values and provenance hashes: [results.json](results.json).
