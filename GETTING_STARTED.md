# Your first training run

LearnByWire combines TrainerPlus, which prepares data and manages training, with OptimizerPlus, a control layer over your existing optimizer. You provide the model, data and training settings. The published benchmarks are examples, not a list of permitted models or datasets.

This guide uses LearnByWire 1.2.2, OptimizerPlus 1.2.2 and TrainerPlus 1.1.1. TrainerPlus 1.1.1 is available on PyPI; publication of the other two packages remains pending. For native training, [install TrainerPlus](INSTALL.md#trainerplus-without-optimizerplus), skip OptimizerPlus activation and use `--control native` in the training command below. OptimizerPlus training requires the matching wheel bundle and entitlement supplied by Qluon. The example files below are in this source repository, not installed by the wheels.

## 1. Install

For OptimizerPlus use Linux x86-64, CPython 3.12, a compatible NVIDIA GPU and driver. Create a new environment and replace the wheel-bundle path with your supplied directory:

```bash
python3.12 -m venv .venv
. .venv/bin/activate
python -m pip install 'torch==2.11.0+cu128' --index-url https://download.pytorch.org/whl/cu128
python -m pip install --find-links /path/to/approved-wheel-bundle 'learnbywire==1.2.2'
python -m pip check
```

Expected last output: `No broken requirements found.` The combined installation supplies both components. For TrainerPlus alone, use the [standalone installation](INSTALL.md#trainerplus-without-optimizerplus).

## 2. Activate OptimizerPlus

```bash
optimizerplus machine-id
```

Give this fingerprint privately to the Qluon contact supplying your package and obtain an entitlement for OptimizerPlus 1.2.2. Keep the returned licence outside your source repository:

```bash
export QLUON_OPTIMIZER_PLUS_LICENSE_FILE=/private/path/customer.license
export CUDA_VISIBLE_DEVICES=0
optimizerplus version
optimizerplus verify-install
optimizerplus license-status
optimizerplus doctor --check
```

The version should be `1.2.2`; the other commands should exit successfully. `doctor` checks environment prerequisites without loading a model or training. It does not predict performance on your workload. Native TrainerPlus training does not need an OptimizerPlus licence.

## 3. Prepare data

For continued pretraining (CPT), save one JSON object per line in `train.jsonl`:

```json
{"text": "Your first training document goes here."}
{"text": "Your second training document goes here."}
```

Use a separate file for held-out validation. You can also provide JSON arrays, CSV, a list of records or a finite Hugging Face dataset through Python. TrainerPlus handles tokenization; you do not have to create token tensors yourself.

For a small installation check, generate the supplied synthetic data from the repository root:

```bash
python packages/trainerplus/examples/make_demo_data.py --output demo-data
```

This creates 100 training and 8 validation records. It tests the training workflow; it is not a useful training corpus or a quality benchmark. Replace it with your own data for meaningful training.

## 4. Train

```bash
python packages/trainerplus/examples/first_run.py \
  --model Qwen/Qwen2.5-1.5B \
  --data demo-data/train.jsonl \
  --validation demo-data/validation.jsonl \
  --control optimizerplus \
  --steps 50 --length 128 \
  --microbatch 1 --accumulation 2 \
  --output runs/first-op
```

The model ID is an example. Use your own compatible Hugging Face ID or a local model/tokenizer directory. The first use of a model ID may download weights; for an offline model add `--local-files-only`. Add `--revision COMMIT_ID` when you need a fixed model revision. Some model repositories require their own access permission.

This command uses LoRA, public fused AdamW, constant LR `2e-4`, BF16 base weights and FP32 trainable adapters. OptimizerPlus manages its controlled training session. The example uses serial execution and token-weighted loss, which suit padded user data; it does not force the specialized dense parallel benchmark settings.

The terminal first reports prepared rows, effective batch size and planned updates. Automatic OptimizerPlus control currently needs at least 50 updates; compilation and initial controller work make startup slower than steady training. This minimum is an implementation requirement, not a restriction on which workloads you may test.

For a full epoch on your data, replace the data paths and omit `--steps`. Use `--epochs 3` for three epochs. Effective batch size is microbatch × accumulation; with 100 rows and batch 2, one epoch is 50 updates. The example reports an error if the data or epoch budget cannot supply the requested updates. It never silently repeats data to meet the budget.

## 5. Find the output

At completion the script prints a JSON summary with `status`, `updates`, `artifacts`, `saved_model` and `metrics`. A successful run reports `status: "complete"` and the planned update count.

Each launch has a new directory under your output root. The launch directory contains your script and arguments; its timestamped training directory contains:

| File or directory | What it tells you |
|---|---|
| `manifest.json` | Model/data identity, settings, data order and package versions |
| `metrics.jsonl` | Training events, progress and runtime summary |
| `status.json` | Running, complete, stopped or failed |
| `result.json` | Completed updates, timings, token counts and optional validation |
| `model/` | Exported model or PEFT adapter plus tokenizer |
| `source/trainerplus/` | Snapshot of the public trainer code used |

For LoRA, `model/` contains adapter weights, not a complete standalone copy of the base model. Keep the base model ID/revision for reloading. See [saving and reusing the result](docs/USER_GUIDE.md#saving-and-reusing-the-result).

## 6. Compare with native training

Run the same command with `--control native` and a different output directory. Add `--native-compile` to use public HF Trainer's `torch.compile` path. Keep data, seed, model, length, optimizer and training budget the same. Both paths use public AdamW; OptimizerPlus adds its control layer on the controlled path.

If you only want to check TrainerPlus on CPU, use `--control native --device cpu --steps 2` with a suitably small model. This tests usage, not GPU performance.

Next: [complete Python usage, data and settings](docs/USER_GUIDE.md), [troubleshooting](docs/TROUBLESHOOTING.md), and [configuration requirements](packages/trainerplus/docs/technical/SUPPORT_MATRIX.md).
