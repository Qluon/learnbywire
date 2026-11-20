# Public-release installation and first-run validation

On 20 November 2026, Qluon installed **LearnByWire 1.2.2, OptimizerPlus 1.2.2 and TrainerPlus 1.1.1** from PyPI into a fresh CPython 3.12.3 virtual environment, then completed the documented first-run example and reloaded its saved LoRA adapter. OptimizerPlus supplied control over public fused PyTorch AdamW.

The environment did not inherit system site packages or previously installed training dependencies. Downloads used a new empty package cache. PyTorch came from its official CUDA 12.8 wheel index; the three LearnByWire distributions came from PyPI, with their exact release hashes checked. Model weights were reused read-only from the local Hugging Face cache; model download was not tested in this check.

## What passed

| Check | Observed result |
|---|---|
| Fresh package installation | All three requested versions and dependencies installed |
| Dependency consistency | `pip check` passed |
| Release identity | All three wheel SHA-256 values matched the published release manifest |
| OptimizerPlus local checks | Version, installed integrity, production licence acceptance and doctor passed |
| Customer example | 50 planned and 50 committed updates; status `complete` |
| Data use | 100 synthetic training records; 2,386 valid input tokens and 2,286 supervised tokens |
| Validation | 8 synthetic held-out records; finite loss 0.287816 and perplexity 1.333512 |
| Adapter export | 392 adapter tensors, 18,464,768 parameters; all finite |
| Separate-process reload | Public PEFT reloaded the base plus exported adapter; finite logits and 12 generated tokens |
| Run isolation | A unique experiment directory, frozen input/script hashes and fresh compiler caches |

These checks demonstrate the installation-to-inference workflow. The tiny synthetic corpus, short token length and startup-heavy timing do not establish a quality improvement or speed advantage. This was a fresh software environment on Qluon's existing test machine, not an independent customer's hardware validation. Meaningful comparisons use [the workload evaluation guide](PILOT_GUIDE.md).

## Exact test configuration

| Setting | Value |
|---|---|
| Source example revision | `8a423661c3260a0494615272b9b4e9fc10a299e4` (public `v1.2.2` tag) |
| Model | `Qwen/Qwen2.5-1.5B` |
| Model revision | `8faed761d45a263340a0528343f099c05c9a4323` |
| GPU | NVIDIA GeForce RTX 5090 Laptop GPU, 24 GB; driver 573.13 |
| Python / PyTorch | 3.12.3 / 2.11.0+cu128 |
| Transformers / PEFT | 5.8.0 / 0.19.1 |
| Accelerate / Triton / safetensors | 1.13.0 / 3.6.0 / 0.7.0 |
| Task / adaptation | CPT / LoRA rank 16, alpha 32, all-linear targets, zero dropout |
| Precision | BF16 base model and FP32 trainable adapters |
| Data and budget | Supplied `make_demo_data.py`; one epoch, 100 records, 50 updates |
| Padding / sequence length | Fixed / 128 |
| Microbatch / accumulation | 1 / 2 |
| AdamW | Fused; LR 0.0002, betas (0.9, 0.95), epsilon 1e-8, weight decay 0.01 |
| Schedule / seed | Constant, zero external warmup / 42 |
| Execution configuration | Automatic model acceleration, serial microbatches, token-mean loss; public example defaults |

The copied public script was run with the documented arguments and a local model path plus `--local-files-only`. No training-runtime code or published wheel was modified for this check. A private Qluon production validation entitlement authorized the local test; customers request their own [licence](LICENSING.md).

## Timing scope

| Measurement | Seconds | Scope |
|---|---:|---|
| Trainer training timer | 69.37 | Session setup, startup and training |
| Trainer workflow timer | 74.22 | Trainer work including final validation and export |
| Whole training invocation | 90.92 | Python process start to exit, including model loading and data preparation; excludes prior installation and cached model download |
| Separate reload check | 15.70 | Imports, adapter validation, base/adapter reload and bounded inference |

The workflow and process rows overlap; they must not be added together. No native comparison, full-epoch benchmark, retention measurement, energy measurement or peak-memory measurement was added by this installation check.

## Artifact identity

Experiment ID: `lbw122_customer_first_run_20261120T125624Z`.
Public example: [first_run.py at the tested revision](https://github.com/Qluon/learnbywire/blob/8a423661c3260a0494615272b9b4e9fc10a299e4/packages/trainerplus/examples/first_run.py).
Reproduction instructions: [quickstart](../GETTING_STARTED.md).

| Artifact | SHA-256 |
|---|---|
| LearnByWire wheel | `69231552e0e43fd467e72071cc471d6f7aab32e456794d1edbb4d4479693c40a` |
| OptimizerPlus wheel | `26838980c73e540a6cdaef8af60465af1226f08b6d8bdad54b3fc392cfd4ac58` |
| TrainerPlus wheel | `efbc1f6409669c3599c337ff2bc77965ac50e5343124abb9cdc5aa43cc989536` |
| Frozen execution manifest | `837dcaf1ea2e46b23261bbddba9d0a41d78b6435526a897de9990ff47a8b6dc0` |
| Completed result JSON | `b19f51829a6a7ce77966cc37622f2437b014522201a68e69ab523b1410aefea0` |
| Exported adapter | `2c305923a3125e03d1ee6860f85531375f9d5b5b89f0ebc878941400bb14162e` |

Qluon retains the full local logs, installation reports, frozen configuration and adapter. The table identifies them; it does not imply that private run paths, credentials or full artifacts are publicly hosted.
