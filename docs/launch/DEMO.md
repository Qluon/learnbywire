# OptimizerPlus demonstration

Versions: LearnByWire 1.2.2, OptimizerPlus 1.2.2 and TrainerPlus 1.1.1.

The guided website walkthrough shows verified commands and recorded outcomes. It does not simulate a running terminal. For a recorded video, use this approximately 90-second storyboard, and identify any time cuts explicitly.

| Segment | Show | Say |
|---|---|---|
| 0–15 seconds | Product page and the native-left, OptimizerPlus-right TinyLlama comparison | “OptimizerPlus adds a control layer over your existing optimizer. Here is what we measured for a complete training workflow.” |
| 15–30 seconds | The install and activation instructions | “Install the released packages, obtain a signed entitlement from Qluon, and check the environment. TrainerPlus handles the training workflow.” |
| 30–50 seconds | The supplied first-run command and the already-recorded smoke-test summary | “This small synthetic run checks installation, training, export and reload. It completed 50 updates; it is not the speed benchmark.” |
| 50–75 seconds | The full-epoch figure, followed by reserved memory and benchmark scope | “The separate matched TinyLlama run recorded 2.09 times the workflow throughput, lower validation perplexity and less GPU energy. Reserved memory increased. This is one seed on one GPU.” |
| 75–90 seconds | Pilot guide and access link | “Bring your workload and native baseline. Match the settings, measure the full workflow and evaluate the result.” |

Use the exact commands from [GETTING_STARTED.md](../../GETTING_STARTED.md). They cover environment creation, CUDA-compatible PyTorch, package installation, activation, synthetic data, training and finding the output. Keep fingerprints, licence contents, signing material and private filesystem paths out of screen recordings.

For real training, use separate held-out data and the intended token budget. Changing only `--control` to `native` gives the native example; add `--native-compile` for compiled native execution. A small first-run example is not a reproduction of the specialized dense benchmark recipe.

The complete fresh-install validation is documented in [ONBOARDING_VALIDATION.md](../ONBOARDING_VALIDATION.md). Benchmark conditions and original model perplexities are in [the evidence report](../../benchmarks/README.md).
