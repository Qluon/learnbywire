# Moving from the 1.2.0 packages

LearnByWire 1.2.2 separates the public interface, TrainerPlus and OptimizerPlus into three distributions. The existing PyPI releases `learnbywire==1.2.0` and `learnbywire-optimizerplus==1.2.0` both contain the `learnbywire` and `optimizerplus` import directories and the same CLI entry points. They are two older packaging layouts, not the new parent/child combination.

## Use a fresh environment

Keep the previous environment intact for existing work. Create a separate Python 3.12 environment for the new packages instead of installing them over the shared files in an older environment:

```bash
python3.12 -m venv .venv-lbw122
. .venv-lbw122/bin/activate
python -m pip install 'torch==2.11.0+cu128' --index-url https://download.pytorch.org/whl/cu128
python -m pip install --find-links /path/to/approved-wheel-bundle 'learnbywire==1.2.2'
python -m pip check
```

The `--find-links` path is needed while these versions are supplied as a wheel bundle. After the versions are published on PyPI, install the same pinned version directly from PyPI.

| Distribution | Version | Import directory it owns | Licence |
|---|---|---|---|
| `learnbywire` | 1.2.2 | `learnbywire` | Apache-2.0 interface |
| `learnbywire-trainerplus` | 1.1.1 | `trainerplus` | Apache-2.0 |
| `learnbywire-optimizerplus` | 1.2.2 | `optimizerplus` | Proprietary |

Do not combine the new facade with an old package that owns the same import files. Uninstalling one shared owner can remove files another distribution needs. The new wheels have separate import ownership; a fresh environment avoids carrying over old files or CLI entry points.

## Check the installation

```bash
python -c 'import importlib.metadata as m; print({n: m.version(n) for n in ("learnbywire", "learnbywire-trainerplus", "learnbywire-optimizerplus")})'
optimizerplus verify-install
```

Expected versions are `1.2.2`, `1.1.1` and `1.2.2`, respectively. Activate a licence that authorizes OptimizerPlus 1.2.2 and matches the wheel's verification key and your machine. An older entitlement is not automatically valid for this release.

The shared imports remain familiar:

```python
from learnbywire import TrainerPlus, OptimizerPlus, TrainingConfig, DataConfig
```

Use the [complete first-run guide](GETTING_STARTED.md) to adapt your training code. TrainerPlus prepares data and runs the chosen native or OptimizerPlus session. Existing training checkpoints should stay with the environment and workload that produced them; do not assume cross-version optimizer/runtime recovery. For LoRA weights, retain the original base model ID/revision and follow the [adapter loading example](docs/USER_GUIDE.md#saving-and-reusing-the-result).

The Apache-2.0 decision applies to the new public interface and TrainerPlus sources. It does not relicense the old proprietary distributions or the separately supplied OptimizerPlus component. Benchmark configurations remain examples, not a whitelist of customer workloads.
