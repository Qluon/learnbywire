# OptimizerPlus launch materials

These materials describe LearnByWire 1.2.2, OptimizerPlus 1.2.2 and TrainerPlus 1.1.1. They contain public measurements and customer instructions, not proprietary implementation code.

- [Technical article](ARTICLE.md)
- [Demonstration storyboard and commands](DEMO.md)
- [TinyLlama graphic](assets/benchmark-overview.png), [vector SVG](assets/benchmark-overview.svg), [PDF](assets/benchmark-overview.pdf)
- [SmolLM2 graphic](assets/smollm2-comparison.png), [vector SVG](assets/smollm2-comparison.svg), [PDF](assets/smollm2-comparison.pdf)
- [Traceable comparison data](evidence.json)
- [Public benchmark source](../../benchmarks/README.md)
- [Customer first-run guide](../../GETTING_STARTED.md)
- [Customer kit](https://github.com/Qluon/learnbywire/releases/download/v1.2.2/learnbywire-1.2.2-customer-kit.zip)

To regenerate the figures, install Matplotlib in an isolated Python environment and run `python docs/launch/make_figures.py` from the repository root. All chart values derive from `evidence.json`, which records the exact upstream data hash and source commit.

The headline 2.09× comparison is specifically TinyLlama against direct public Hugging Face Trainer compiled AdamW. SmolLM2 uses a TrainerPlus-hosted native compiled baseline. These are complete-system observations with one seed per arm, not universal performance promises or independent replication. Both memory metrics must be reported with the gains.
