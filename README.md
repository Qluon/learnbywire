# LearnByWire 1.2.2

LearnByWire provides a shared interface to TrainerPlus 1.1.1 and OptimizerPlus 1.2.2. OptimizerPlus is a control layer on a caller-supplied public optimizer such as PyTorch AdamW.

```python
from learnbywire import TrainerPlus, OptimizerPlus, TrainingConfig
```

`control=None` selects standard public HF Trainer. `control=OptimizerPlus` selects OptimizerPlus's training session. Installing the packages does not activate OptimizerPlus automatically. Full OptimizerPlus execution uses its own session rather than a drop-in standard HF Trainer optimizer.

## Start here

For a single download, get the [customer kit ZIP](https://github.com/Qluon/learnbywire/releases/download/v1.2.2/learnbywire-1.2.2-customer-kit.zip). It contains the public guides, runnable examples, licence-request and evaluation forms. Start with its README; the [archive checksum](https://github.com/Qluon/learnbywire/releases/download/v1.2.2/learnbywire-1.2.2-customer-kit.zip.sha256) is supplied separately.

1. [Install, activate and complete your first run](GETTING_STARTED.md).
2. [Use the complete Python example and your own data](docs/USER_GUIDE.md).
3. [Resolve common setup and training errors](docs/TROUBLESHOOTING.md).
4. [Request an OptimizerPlus licence or renew access](docs/LICENSING.md).
5. [Evaluate your own workload and follow up with Qluon](docs/PILOT_GUIDE.md).

[Fresh PyPI installation and customer-example validation](docs/ONBOARDING_VALIDATION.md) records the release's installation-to-adapter check. For help, create a [local support report](docs/SUPPORT_REPORT.md) and review it before sharing.

Already using the 1.2.0 packages? Follow the [migration guide](MIGRATION.md) to keep the new packages' import ownership separate.

The [first-run script](packages/trainerplus/examples/first_run.py) accepts your model ID or local folder and your training/validation files. It prepares LoRA and public AdamW, runs your chosen control and prints where to find the saved adapter and metrics. Use `--help` to see its settings.

| Package | Python namespace | Source policy |
|---|---|---|
| learnbywire 1.2.2 | learnbywire | Apache-2.0 interface |
| learnbywire-trainerplus 1.1.1 | trainerplus | Apache-2.0 orchestration |
| learnbywire-optimizerplus 1.2.2 | optimizerplus | Separately licensed proprietary binary |

This repository contains the LearnByWire interface and TrainerPlus source under Apache-2.0. OptimizerPlus remains separately proprietary and is distributed as a licensed binary; its implementation, build tools, signing keys and customer licences are excluded.

[LearnByWire 1.2.2](https://pypi.org/project/learnbywire/1.2.2/), [OptimizerPlus 1.2.2](https://pypi.org/project/learnbywire-optimizerplus/1.2.2/) and [TrainerPlus 1.1.1](https://pypi.org/project/learnbywire-trainerplus/1.1.1/) are available on PyPI. Install the combined package with `python -m pip install learnbywire==1.2.2` after preparing the environment in the [installation guide](INSTALL.md). OptimizerPlus training requires a signed entitlement supplied by Qluon; native TrainerPlus training does not.

The published experiments are examples and performance evidence, not a list of permitted uses. Users can supply their own compatible models, datasets, optimizers and training settings; Qluon does not need to benchmark each configuration before users try it. Runtime compatibility depends on the implemented interfaces and execution requirements.

See [installation](INSTALL.md), [TrainerPlus](packages/trainerplus/README.md), [configuration and compatibility](packages/trainerplus/docs/technical/SUPPORT_MATRIX.md) and [measured results](benchmarks/README.md). The results include a direct public HF baseline independent of TrainerPlus imports.

For maintainers: [publish the audited wheels through PyPI Trusted Publishing](docs/PUBLISHING.md).

## Explore the release

Read the [OptimizerPlus results article, demonstration guide and downloadable graphics](docs/launch/README.md). The comparisons include workflow time, validation and retention perplexity, GPU energy and peak memory, with the exact benchmark conditions.
