# TrainerPlus 1.1.1

TrainerPlus is LearnByWire's Apache-2.0 training orchestrator. It prepares CPT/SFT data, preserves experiment artifacts and offers an explicit choice between standard public HF Trainer and the proprietary OptimizerPlus control session.

Start with the repository's [first-run guide](../../GETTING_STARTED.md), [Python and data guide](../../docs/USER_GUIDE.md), or [runnable example](examples/first_run.py). The example accepts a model ID or local folder plus your own data files; no prebuilt token tensors are required.

```python
from trainerplus import TrainerPlus, TrainingConfig, DataConfig, TokenDataset
```

With the combined LearnByWire 1.2.2 installation:

```python
from learnbywire import TrainerPlus, OptimizerPlus, TrainingConfig
```

Supply the model, optimizer and dataset. Set `control=None` for native HF Trainer, or `control=OptimizerPlus` for OptimizerPlus 1.2.2. The caller's AdamW remains the public optimizer. `native_compile=True` selects public HF compilation only on the native path; OptimizerPlus manages its own runtime.

The recorded benchmark uses constant LR 2e-4, LoRA, microbatch 1, accumulation 2 and explicit parallel OptimizerPlus settings. The presets describe those experiments, not universal defaults. Models and data are supplied by the user; the benchmark does not define a model whitelist. See [examples](examples/train.py), [qualification](QUALIFICATION.md), [performance](docs/technical/PERFORMANCE.md) and [configuration and compatibility](docs/technical/SUPPORT_MATRIX.md).

The benchmark evaluator in `trainerplus.evaluation` bounds FP32 temporary workspace. It is a separate helper and is not automatically used by `TrainerPlus.evaluate()`.

This source does not contain OptimizerPlus internals. The native path works without OptimizerPlus. The current orchestrator runs in one process; this is an implementation requirement, independent of the benchmark coverage. TrainerPlus is licensed under Apache-2.0. No PyPI/GitHub publication is implied.
