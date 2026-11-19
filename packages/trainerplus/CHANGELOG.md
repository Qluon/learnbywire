# TrainerPlus 1.1.1

The package retains the validated bounded FP32 evaluation helper, original per-batch reduction, full-workspace option and padded fallback. Its public control integration requires OptimizerPlus 1.2.2.

Current-version installed wheels completed six full epochs across TinyLlama 1.1B and SmolLM2 1.7B. The additional direct HF Trainer TinyLlama run matched TP-native exported adapters and all 33 evaluation panels exactly. Full-workflow performance and memory trade-offs are documented in the performance report; these one-seed results do not establish universal speedup.

Publication preparation refreshes presets, scope and evidence documentation, and makes the command-line example usable with standalone TrainerPlus imports. Trainer and evaluator runtime source are unchanged. Publication and customer-key qualification remain separate gates.
