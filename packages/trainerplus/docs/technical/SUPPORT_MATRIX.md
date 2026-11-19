# Configuration, compatibility and measured evidence

The published experiments are examples and performance evidence, not a list of permitted uses. Users can supply their own compatible models, datasets, optimizers and training settings; Qluon does not need to benchmark each configuration before users try it. Runtime compatibility depends on the implemented interfaces and execution requirements.

## Configurable interfaces

| Area | TrainerPlus native | OptimizerPlus control |
|---|---|---|
| Model | Caller-supplied causal language model | Caller-supplied compatible model; acceleration is selected by structure, not a model-name whitelist |
| Optimizer | Caller-supplied public optimizer | Control over caller-supplied AdamW, Lion or Muon through the current optimizer adapters |
| Data and task | User data; CPT and SFT interfaces | User data; CPT and SFT interfaces |
| Training settings | Configurable length, epochs, update budget, batch sizes, LR and schedule | Configurable settings through TrainerPlus and the OP session API |
| Execution | Public HF Trainer; optional public torch.compile | OP-owned session with Opt+ and Run+ |
| Padding | Fixed or dynamic | Fixed shapes in the current session integration |

The benchmark presets are reproducible examples. They do not restrict users to the benchmark's model names, datasets, seed, token length or update count. The explicit parallel OP mode is a specialized execution path; serial mode is the general default for other batching settings.

## Technical requirements of this implementation

The combined OP wheel targets CPython 3.12 on Linux x86-64. OP requires CUDA and its pinned PyTorch stack. Its current TrainerPlus integration uses a single device/process, BF16 execution, fixed token shapes and a causal-LM loss. Full OP execution uses its own session; it is not a drop-in standard HF Trainer optimizer. Quantized and distributed OP integrations are not implemented in this adapter. These requirements come from code, not from which models were benchmarked.

The explicit parallel mode requires a compatible structural LoRA model, dense attention, microbatch 1, accumulation 2, microbatch-mean reduction and standard loss. Serial mode permits other batching configurations subject to runtime checks. Native GPU kernels require a matching compiled specialization. The catalog includes SM80, SM86, SM89, SM90, SM100 and SM120 targets; a compiled target is not a measured performance result. A manylinux tag describes the OP wheel's CPU ABI, not the requirements of the entire CUDA dependency stack.

## Experiments recorded for this release

| Evidence | Configuration measured |
|---|---|
| Full epochs | TinyLlama 1.1B and SmolLM2 1.7B; fused AdamW; seed 4007 |
| Independent native baseline | Direct public HF Trainer on TinyLlama; no TP/OP imports |
| GPU packaging/equivalence check | Qwen2.5-1.5B; AdamW; 60 updates |
| Device and precision | One RTX 5090 Laptop GPU, SM120; BF16 frozen base and FP32 LoRA |
| Software | PyTorch 2.11.0+cu128, Transformers 5.8.0, PEFT 0.19.1, Accelerate 1.13.0, Triton 3.6.0 |

These observations describe the measurements, not a usage boundary. Speed, memory and quality for a user's workload must be measured on that workload. New-version durable recovery and GPU sanitizer instrumentation have not been revalidated in these experiments; those are test-coverage notes, not model restrictions.

The low-memory evaluator in `trainerplus.evaluation` is used by the benchmark. It is not automatically used by every `TrainerPlus.evaluate()` call. General API defaults are not benchmark settings.
