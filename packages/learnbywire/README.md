# LearnByWire 1.2.2

For installation, activation and a complete first training run, read the repository's [getting-started guide](../../GETTING_STARTED.md). The [user guide](../../docs/USER_GUIDE.md) covers Python usage, your data, settings and saved outputs.

Shared imports for TrainerPlus 1.1.1 and OptimizerPlus 1.2.2:

```python
from learnbywire import TrainerPlus, OptimizerPlus, TrainingConfig
```

The planned `pip install learnbywire` installs both components and the declared Hugging Face/PEFT dependencies. The combined package currently targets CPython 3.12 on the supported OptimizerPlus platform (Linux x86-64 with NVIDIA CUDA for controlled training). Installation does not automatically enable OptimizerPlus: pass `control=OptimizerPlus` when constructing TrainerPlus. The base optimizer remains a public optimizer such as `torch.optim.AdamW`.

`learnbywire.__version__` identifies the shared package; `learnbywire.__trainer_version__` and `learnbywire.__optimizer_version__` identify its components. Importing the top-level package does not import torch or either component until an exported object is requested.

Only this wheel owns the `learnbywire/` import directory. The children own `trainerplus/` and `optimizerplus/` respectively. Do not co-install legacy `qluon-optimizerplus` packages that also own shared imports; migrate in a clean environment.

Release candidate only: not yet published. The OptimizerPlus package requires its declared platform and a valid licence. The public interface is licensed under Apache-2.0.
