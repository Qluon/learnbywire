# Troubleshooting

Start with `python -m pip check`. For OptimizerPlus also run `optimizerplus doctor --check`, `optimizerplus verify-install` and `optimizerplus license-status`. These are local checks; they do not train a model or send a licence to a service.

| Symptom | Likely cause | Next action |
|---|---|---|
| No matching OP wheel | Python/platform mismatch or local wheels not supplied | Use CPython 3.12 Linux x86-64 and the approved wheel bundle; see [installation](../INSTALL.md) |
| Import conflict or wrong version | Mixed package ownership or stale environment | Create a fresh venv and install the pinned three-package combination |
| No CUDA device | Driver, CUDA-enabled PyTorch or device visibility | Check `nvidia-smi`, the CUDA PyTorch installation and `CUDA_VISIBLE_DEVICES` |
| More than one visible GPU | Current TP orchestrator expects one device | Select the intended device before launch, for example `CUDA_VISIBLE_DEVICES=0` |
| Licence missing, wrong version, expired or machine mismatch | Entitlement does not match this host/release | Ask the supplier for a matching entitlement; set its path, not its contents, in the environment |
| Record exceeds max length | Truncation defaults to an error to avoid silent data loss | Increase `--length`, prepare shorter records or deliberately select `--truncate` |
| Incomplete update group | Prepared rows not divisible by microbatch × accumulation | Adjust the batch sizes/data or explicitly use `--drop-last` |
| Fewer than 50 updates | Current automatic OP controller minimum | Supply enough data/epochs for 50 updates; do not assume the script repeats data automatically |
| Budget exceeds available examples | `--steps` is larger than the configured epochs supply | Reduce steps, add data or deliberately increase epochs |
| Missing padding token | Tokenizer has no pad token | Use a suitable pad token; the first-run example uses EOS when available and masks padding labels |
| No supervised targets | Truncation or SFT masking removed the answer | Inspect tokenized data, increase length or fix the template/format |
| Chat-template generation mask missing | Assistant-only training needs generation blocks | Supply a compatible template, use prompt/completion data, or deliberately choose all-token loss |
| Parallel replay eligibility error | Dense paired benchmark settings do not fit your data/model | Use serial execution and the appropriate loss reduction |
| Model acceleration eligibility error | Model/LoRA structure does not match that accelerator | Inspect the reported reason; try `acceleration="none"` for the model accelerator while retaining runtime checks |
| Compiler/data-dependent control-flow error | Model forward cannot be captured by this runtime | Use the reported trace to identify the incompatible branch; `acceleration="none"` does not disable graph compilation |
| Out of GPU memory | Model, token length, batching or evaluation exceeds capacity | Reduce microbatch or length; increase accumulation only if preserving effective batch is intended; use a smaller model |
| Constant LR with warmup error | This API implements external warmup with cosine | Use cosine plus warmup, or constant with zero external warmup |
| Cannot reuse a trainer | TrainerPlus instances are single-use | Create a new instance for a new run or resume |
| Cannot resume an exported adapter | Final adapter export lacks optimizer/runtime state | Use a saved training checkpoint with the matching workload and release |

## Startup and speed

Model loading, compilation and controller startup cost time. Inspect `status.json` and `metrics.jsonl` in the printed artifact directory. A low rate during a short startup window is not a steady-state measurement. Use complete-workflow timing when that is the quantity you care about, and include startup consistently for both systems.

For a native baseline choose `control=None`; compilation is a separate native setting. For OP choose `control=OptimizerPlus`. In the OP `control_runtime` event, inspect the reported runtime/acceleration summary. This event occurs at startup; a later active capture can differ from an initial summary. If a run fails, preserve the error and status rather than assuming that capture succeeded. A completed OP run and runtime telemetry together provide better evidence than package installation alone.

Reserved GPU memory is PyTorch's allocator pool; allocated memory is memory currently occupied by tensors. Reserved memory can exceed allocated memory because the allocator retains reusable blocks and graph capture has persistent requirements. Do not insert `empty_cache()` into every update as a blanket fix; measure its effect and preserve capture requirements.

## A useful support report

Include package/Python/PyTorch versions, GPU model, model ID/revision, task, batch sizes, token length, precision, schedule, control options, complete error text and a small non-sensitive data sample. State whether the issue also occurs with native training. Attach only reviewed, sanitized manifests and logs: they can contain paths, model identities and user configuration. Exclude licence files, signing keys and proprietary data. There is no requirement to wait for Qluon to benchmark your model before you try it.
