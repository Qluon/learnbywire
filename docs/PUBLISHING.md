# Publishing the audited wheel distributions

The public repository contains TrainerPlus and the LearnByWire interface. OptimizerPlus implementation and signing keys stay outside it. Its binary wheel is built and qualified privately; no public workflow compiles OptimizerPlus source.

`release/pypi-manifest.json` records each version, GitHub release tag, wheel filename, SHA-256 and publication readiness. The publishing workflow downloads that exact wheel, checks its archive and metadata, checks PyPI version conflicts, and publishes it through Trusted Publishing. It does not rebuild a wheel during upload. The LearnByWire interface also requires both child distributions to exist on PyPI with the approved hashes.

## One-time PyPI setup

Sign into PyPI as the owner of the projects. Register the same GitHub publisher fields for each project:

| Field | Value |
|---|---|
| Owner | `Qluon` |
| Repository | `learnbywire` |
| Workflow filename | `publish-pypi.yml` |
| Environment | `pypi` |

All three projects now exist on PyPI. Register the publisher separately on each project's **Manage → Publishing** page: `learnbywire`, `learnbywire-optimizerplus` and `learnbywire-trainerplus`. Registration on one project does not authorize uploads to the others.

The PyPI publisher owner is the GitHub repository owner, **Qluon**, even when a collaborator dispatches the workflow. No PyPI token belongs in this repository or in a chat.

Official instructions: [existing projects](https://docs.pypi.org/trusted-publishers/adding-a-publisher/) and [new projects](https://docs.pypi.org/trusted-publishers/creating-a-project-through-oidc/).

## Release and validate

1. Finish qualification and set the package's `approved_for_pypi` flag only after its recorded wheel hash has passed the release checks. OptimizerPlus requires successful production entitlement activation. Update the manifest if rebuilding changes the final wheel.
2. Attach only that wheel and its checksum to the package's GitHub release. Never upload an OptimizerPlus source distribution, licence, signing key, QA wheel or private build folder.
3. In **Actions → Publish audited wheels to PyPI → Run workflow**, select `main` and the package. Leave **Publish the verified wheel to PyPI** unchecked for validation only.
4. After validation and Trusted Publisher registration, run the workflow with publication checked. Existing PyPI versions are not silently skipped or replaced.
5. Verify the published file's SHA-256 and a fresh installation. The local release report records the remaining activation and customer example checks.

TrainerPlus can be published independently. Publish the OptimizerPlus dependency before the combined LearnByWire interface. These release checks control which binaries are published; they do not restrict customers to benchmark models or datasets.
