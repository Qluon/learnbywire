# Installation and activation

LearnByWire 1.2.2, OptimizerPlus 1.2.2 and TrainerPlus 1.1.1 are available on PyPI. The combined installation installs both components. OptimizerPlus training requires a signed entitlement supplied by Qluon.

## Combined installation

Use CPython 3.12 on Linux x86-64 and a fresh virtual environment. Install the required PyTorch 2.11.0+cu128 stack and use a compatible NVIDIA GPU with a matching compiled kernel target. A compatible NVIDIA driver is required; the wheel's manylinux tag describes CPU/glibc compatibility, not universal GPU support.

```bash
python3.12 -m venv .venv
. .venv/bin/activate
python -m pip install 'torch==2.11.0+cu128' --index-url https://download.pytorch.org/whl/cu128
python -m pip install 'learnbywire==1.2.2'
python -m pip check
```

The combined package pins TrainerPlus 1.1.1 and OptimizerPlus 1.2.2. Its OptimizerPlus dependency pins Transformers 5.8.0 and PEFT 0.19.1. Do not mix another OptimizerPlus distribution that also owns the `learnbywire` namespace into this environment.

The existing PyPI 1.2.0 packages used overlapping import directories. If migrating from either `learnbywire==1.2.0` or `learnbywire-optimizerplus==1.2.0`, follow the [migration guide](MIGRATION.md) and keep the previous environment separate.

## TrainerPlus without OptimizerPlus

Install the public package from PyPI in a fresh environment. Clone this repository for the runnable examples. The commands below use the CUDA stack from the measured NVIDIA runs:

```bash
git clone https://github.com/Qluon/learnbywire.git
cd learnbywire
python3.12 -m venv .venv
. .venv/bin/activate
python -m pip install 'torch==2.11.0+cu128' --index-url https://download.pytorch.org/whl/cu128
python -m pip install 'learnbywire-trainerplus[lora]==1.1.1'
python -m pip check
```

If Qluon has supplied a wheel bundle, you can instead install TrainerPlus from that bundle:

```bash
python -m pip install --find-links /path/to/approved-wheel-bundle 'learnbywire-trainerplus[lora]==1.1.1'
```

```python
from trainerplus import TrainerPlus, TrainingConfig, DataConfig, TokenDataset
```

Native training does not require an OptimizerPlus licence. The public `examples/train.py` supports this standalone TrainerPlus installation. The shared `learnbywire` convenience package deliberately installs both components.

## OptimizerPlus activation

A signed entitlement authorizing OptimizerPlus 1.2.2 is required for controlled training. Obtain the host fingerprint with `optimizerplus machine-id` and provide it privately to Qluon. Supply the returned licence path with `QLUON_OPTIMIZER_PLUS_LICENSE_FILE`; never commit the file.

```bash
export QLUON_OPTIMIZER_PLUS_LICENSE_FILE=/private/path/customer.license
optimizerplus verify-install
optimizerplus license-status
```

A QA entitlement is not a customer licence. A wheel without a matching entitlement does not activate OptimizerPlus. Native compilation and machine binding are not guarantees against reverse engineering or privileged local tampering.

## Your own workload

The published experiments are examples and performance evidence, not a list of permitted uses. Users can supply their own compatible models, datasets, optimizers and training settings; Qluon does not need to benchmark each configuration before users try it. Runtime compatibility depends on the implemented interfaces and execution requirements.

See the [configuration and compatibility guide](packages/trainerplus/docs/technical/SUPPORT_MATRIX.md) for API choices and actual technical requirements. Benchmark throughput includes a wider workflow than `trainer.train()` timing alone.
