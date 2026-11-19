# Using LearnByWire with your model and data

Start with [installation and a first run](../GETTING_STARTED.md). The examples below use the real APIs in LearnByWire 1.2.2, OptimizerPlus 1.2.2 and TrainerPlus 1.1.1. Models and datasets are supplied by you; the benchmark settings are not mandatory defaults.

## Which component does what?

| Component | Your choice | Responsibility |
|---|---|---|
| Model and tokenizer | Compatible Hugging Face model ID or local folder | Predictions and conversion of text into tokens |
| PEFT LoRA | Adapter rank, scale and target modules | Selects the parameters to train |
| Public optimizer | For example, `torch.optim.AdamW` | Owns trainable parameters and optimizer settings |
| OptimizerPlus | `control=OptimizerPlus` | Adds optimizer control and training execution over that optimizer |
| TrainerPlus | Model, optimizer, datasets and configuration | Prepares data, runs training/evaluation and saves artifacts |
| LearnByWire | `from learnbywire import ...` | Convenient imports for both packages |

`control=None` uses public HF Trainer. Full OptimizerPlus execution uses its own session. TrainerPlus calls the correct path; do not add your own `backward()` or `optimizer.step()` around `trainer.train()`.

## Complete Python example

Create `demo-data` with the generator in the first-run guide, or change the two paths below to your own data. This script defines every training object; it does not assume an already-created model, optimizer or token dataset.

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed
from peft import LoraConfig, get_peft_model
from learnbywire import TrainerPlus, OptimizerPlus, TrainingConfig, DataConfig

set_seed(42)
model_id = "Qwen/Qwen2.5-1.5B"  # Replace with your compatible model ID or folder.
tokenizer = AutoTokenizer.from_pretrained(model_id)
if tokenizer.pad_token_id is None:
    tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    model_id, dtype=torch.bfloat16, attn_implementation="sdpa",
)
model.config.use_cache = False
model = get_peft_model(model, LoraConfig(
    task_type="CAUSAL_LM", r=16, lora_alpha=32,
    lora_dropout=0.0, bias="none", target_modules="all-linear",
))
model = model.to("cuda")
for parameter in model.parameters():
    if parameter.requires_grad:
        parameter.data = parameter.data.float().contiguous()

adamw = torch.optim.AdamW(
    [p for p in model.parameters() if p.requires_grad],
    lr=2e-4, betas=(0.9, 0.95), eps=1e-8,
    weight_decay=0.01, fused=True, foreach=False,
)
trainer = TrainerPlus(
    model=model,
    optimizer=adamw,
    tokenizer=tokenizer,
    train_dataset="demo-data/train.jsonl",
    eval_dataset="demo-data/validation.jsonl",
    task="cpt",
    control=OptimizerPlus,
    args=TrainingConfig(
        output_dir="runs/python-example", num_epochs=1,
        microbatch_size=1, gradient_accumulation_steps=2,
        mixed_precision="bf16", seed=42,
        lr_schedule="constant", warmup_ratio=0.0,
        max_grad_norm=None,
    ),
    data_config=DataConfig(max_length=128, padding="fixed"),
    control_options={
        "audition": "checkpoint_continue",
        "acceleration": "auto",
        "loss_reduction": "token_mean",
        "microbatch_execution": "serial",
    },
)
result = trainer.train()
print(result.status, result.committed_steps, result.run_dir)
print(result.metrics)
```

Set `CUDA_VISIBLE_DEVICES` and the licence-file environment variable before running, as shown in the installation guide. If you use another model, choose appropriate LoRA targets; PEFT's `all-linear` is a convenient starting point, not a guarantee that every model structure fits every acceleration path. Tokenization uses that model's tokenizer. Pin the model revision for reproducible comparisons.

The [runnable command-line example](../packages/trainerplus/examples/first_run.py) additionally records the outer script, arguments and input-file hashes. Importing it does not start training.

## Native and OptimizerPlus choices

| Setting | Native | OptimizerPlus |
|---|---|---|
| `control` | `None` | Imported `OptimizerPlus` class |
| `control_options` | Omit or `None` | OptimizerPlus options dictionary |
| `native_compile` | `False` for eager; `True` for public HF compilation | `False`; OptimizerPlus owns compilation |
| Optimizer | Public AdamW instance | Same public AdamW configuration |
| Execution owner | Public HF Trainer | OptimizerPlus session through TrainerPlus |

For a fresh comparison create a fresh model, optimizer and TrainerPlus instance with the same seed and settings. Reusing an already-trained model changes the comparison. `TrainerPlus` instances are single-use, and each `.train()` creates a separate run directory. The user guide does not promise that every configuration will improve speed or quality; measure your workload.

You may also use direct imports with separately installed packages:

```python
from trainerplus import TrainerPlus, TrainingConfig, DataConfig
from optimizerplus import OptimizerPlus  # Needed only when you select OP.
```

## Data formats

Raw records are tokenized by TrainerPlus. `train_dataset` and `eval_dataset` accept a local JSONL/JSON/CSV path, a list of records, a finite Hugging Face dataset split or an already prepared `TokenDataset`. Pass a specific dataset split, not an entire split dictionary. Streaming datasets are not implemented by the current adapter.

### CPT: text documents

```json
{"text": "Your company documentation or other training text."}
```

Choose `task="cpt"`. Change the field name with `DataConfig(text_field="content")`. `packing=True` concatenates CPT documents with EOS boundaries and splits them into token blocks; it changes the training context and prepared row count. It is a deliberate data choice, not a required speed setting. SFT packing is not implemented here.

### SFT: prompts and completions

```json
{"prompt": "Question: What is 6 + 7?\nAnswer: ", "completion": "13."}
```

Choose `task="sft"`. `sft_loss="auto"` uses completion-only loss for this format. A fast tokenizer with offset mappings is required to mask the prompt correctly. Keep your desired delimiter, space or newline in the prompt: the preparation code concatenates `prompt + completion` directly. Use `prompt_field` and `completion_field` for other column names. Choose `sft_loss="all"` explicitly if you intend to train on both prompt and answer tokens.

### SFT: conversations

```json
{"messages": [{"role": "user", "content": "What is 6 + 7?"}, {"role": "assistant", "content": "13."}]}
```

Choose `task="sft"` and a tokenizer with a suitable chat template. Assistant-only loss needs generation blocks in that template to identify assistant tokens. If those blocks are absent, provide a suitable template or deliberately select `sft_loss="all"`; the trainer does not silently train on all messages. Chat templates own their end-of-turn tokens.

### Already tokenized data

```python
from learnbywire import TokenDataset

dataset = TokenDataset(
    input_ids=input_ids,         # CPU integer tensor: [rows, sequence_length]
    attention_mask=mask,         # Same shape: 1 for real tokens, 0 for padding
    labels=labels,               # Same shape: token IDs; -100 ignores a position
)
```

Here the three tensors come from your own preprocessing. Raw-text users should use the file or record APIs above instead. Padding labels must be `-100`; the first causal label is masked automatically. Every row must contain a supervised target. OptimizerPlus training through TrainerPlus requires fixed-length rows. Native training can use variable-length rows with `TokenDataset.from_rows(...)` or `DataConfig(padding="dynamic")`.

## Choosing settings

| Setting | Meaning and practical effect |
|---|---|
| Model ID/revision | Selects the base model; the revision makes its identity repeatable |
| `max_length` | Tokens per prepared row; longer sequences need more computation and memory |
| `microbatch_size` | Examples processed together in one forward/backward pass |
| `gradient_accumulation_steps` | Microbatches combined before an optimizer update |
| Effective batch | Microbatch × accumulation in this single-process trainer |
| `num_epochs` | Passes through the prepared data |
| `max_steps` | Optional update budget within those epochs; does not invent more data |
| `drop_last` | Explicitly discard an incomplete final effective batch; otherwise it raises |
| AdamW `lr` | Base learning rate you choose on the public optimizer |
| `lr_schedule` | `constant` or `cosine`; default is constant |
| `warmup_ratio` | External warmup fraction; the current API implements it with cosine |
| `eval_steps` | Optional periodic validation at completed update boundaries; requires validation data |
| `save_steps` | Optional resumable checkpoints at completed update boundaries |
| `save_model` | Export the completed model/adapter; default `True` |
| `truncation` | Default `error`; choose `truncate` deliberately to cut long records |
| `seed` | Controls initialization/order randomness; keep it fixed for a paired comparison |

Automatic OptimizerPlus startup currently requires at least 50 planned updates. This is a runtime guard in the current version. For `N` prepared rows, microbatch `m`, accumulation `a`, and `e` epochs, the available updates are `floor(N / (m*a)) * e`, with explicit `drop_last=True` required if there is a remainder. Packing changes `N`, so count prepared rows rather than source documents.

An external cosine schedule and OptimizerPlus's control are separate choices. The example deliberately uses constant base LR and no external warmup; it does not claim that this setting is best for every workload. When native and OptimizerPlus are compared, record any difference in schedules or loss weighting.

`token_mean` weights supervised tokens equally across an update. `microbatch_mean` averages microbatch losses equally; the two differ if microbatches have different numbers of supervised tokens. The quickstart uses token mean for ordinary padded data. The reported dense parallel benchmark explicitly uses microbatch mean. Do not silently switch reductions while claiming an otherwise identical quality comparison.

## Saving and reusing the result

The returned `TrainResult` has `status`, `committed_steps`, `run_dir`, `metrics`, `model_path` and `checkpoint`. Completed runs export to `model_path` when `save_model=True`.

For a LoRA export, reload the original base model plus adapter:

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

adapter_dir = "runs/first-training/launch_ID/RUN_ID/model"  # Use the printed saved_model path.
base_id = "Qwen/Qwen2.5-1.5B"  # Same base ID/revision used in training.
base = AutoModelForCausalLM.from_pretrained(base_id, dtype=torch.bfloat16)
model = PeftModel.from_pretrained(base, adapter_dir).to("cuda").eval()
tokenizer = AutoTokenizer.from_pretrained(adapter_dir)
inputs = tokenizer("A short example prompt:", return_tensors="pt").to("cuda")
with torch.inference_mode():
    output = model.generate(**inputs, max_new_tokens=32)
print(tokenizer.decode(output[0], skip_special_tokens=True))
```

A final adapter export is for reuse/inference, not an optimizer-state checkpoint. To resume training, enable `save_steps`, keep the original model/data/settings, create a new trainer and pass the recorded checkpoint path to `trainer.train(resume_from_checkpoint=...)`. The current recovery contract verifies workload identity; changing data, release or settings is a new run. Native recovery is covered by CPU tests; OptimizerPlus 1.2.2's recovery path has not been revalidated on GPU in the current release study.

## Reading metrics correctly

`result.metrics` reports completed examples, valid input tokens, supervised tokens, training seconds, input/supervised tokens per training second, and workflow seconds. With validation it also reports loss, perplexity and next-token accuracy. Next-token accuracy is not a downstream question-answer benchmark score. Lower perplexity means the model assigns greater probability to the held-out tokens; compare on the same dataset/tokenizer and masking.

Training seconds include session setup, controller/compile startup, periodic evaluations and saves. TrainerPlus workflow seconds additionally include its manifest work, final evaluation and export; model loading and prior tokenization happen before this timer. The paper's benchmark workflow timer starts earlier, so do not label TrainerPlus's built-in rate as the same measurement. For TrainerPlus's timed workflow rate, divide `input_tokens` by `workflow_seconds` and describe the timing boundary.

Energy, NVML sampling, peak VRAM, retention datasets and downstream accuracy are extra measurements in the benchmark harness, not automatically collected by a basic `trainer.train()` call. See the [measured protocol](../benchmarks/README.md) before comparing numbers. The low-memory evaluator is a separate helper; the standard TrainerPlus evaluation method does not automatically use it.

## Trying another model or sharing a finding

Change the model/tokenizer and model-appropriate LoRA targets, then use your own data and settings. OptimizerPlus selects acceleration by model structure; it does not check against the benchmark model-name list. Keep serial execution for general settings. The specialized parallel path has additional shape and reduction requirements described in the [configuration guide](../packages/trainerplus/docs/technical/SUPPORT_MATRIX.md).

Record versions, model revision, settings, runtime selection and the observed result. For a problem report include a small non-sensitive reproducer and the relevant error. Never include a licence, signing key, private training data or customer model weights in a public issue. See [troubleshooting](TROUBLESHOOTING.md).
