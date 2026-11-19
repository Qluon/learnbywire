# Release evidence and preparation status

LearnByWire 1.2.2, OptimizerPlus 1.2.2 and TrainerPlus 1.1.1 passed 99 CPU checks (one CUDA-only check skipped in that CPU suite), a separate 60-update GPU equivalence test, and six full epochs on two models. The additional direct-HF TinyLlama epoch completed with all 33 evaluations and final exported adapter tensors exactly matching TrainerPlus-native.

The measured OptimizerPlus wheel contains 90 native modules and no readable OptimizerPlus Python implementation or bytecode. Compilation is not a guarantee against reverse engineering. Native source remains private. Training/runtime code is unchanged during this publication-preparation revision.

The published experiments are examples and performance evidence, not a list of permitted uses. Users can supply their own compatible models, datasets, optimizers and training settings; Qluon does not need to benchmark each configuration before users try it. Runtime compatibility depends on the implemented interfaces and execution requirements.

See [performance](docs/technical/PERFORMANCE.md) and [configuration and compatibility](docs/technical/SUPPORT_MATRIX.md). The tests used QA entitlement; a fresh pip installation and 60-update manylinux-wheel check also passed. Production-key imports and rejection of QA credentials pass; positive customer-entitlement activation still needs the matching production signing key. The owner confirmed Apache-2.0 for TrainerPlus and the LearnByWire interface on 19 November 2026; OptimizerPlus remains separately proprietary. New-version checkpoint recovery and GPU sanitizer instrumentation have not been qualified. No public release is implied by these results.

The customer-facing first-run command also completed a 50-update OptimizerPlus GPU smoke test with synthetic padded data, serial execution and token-mean loss, including validation and adapter export. Native CPT and SFT examples completed offline CPU smoke checks with validation and adapter reload. These checks verify usage; their synthetic/tiny results are not performance claims.
