# Prepare a support or follow-up report

For licence requests use [the private access process](LICENSING.md). For other issues, contact your Qluon representative or use [Qluon's private request form](https://qluon.ai/contact.php?topic=pilot). Public reproducible software issues can be filed in [GitHub Issues](https://github.com/Qluon/learnbywire/issues).

The repository includes a read-only report helper. It records selected package versions, Python/platform information and visible GPU specifications. With `--run-dir`, it includes only selected numeric results and the completion status. It does not collect licence files, machine fingerprints, environment variables, datasets, weights, raw logs or manifests, and it sends nothing over the network.

```bash
python packages/trainerplus/examples/collect_support.py \
  --output support-report.json
```

After training, use the timestamped artifact directory printed by the first-run script, not the outer output directory:

```bash
python packages/trainerplus/examples/collect_support.py \
  --run-dir /path/to/printed/artifact-directory \
  --output support-after-training.json
```

The helper refuses to overwrite an existing output file. Review the JSON before sharing it; even model-free system and numeric information may be sensitive for your organization. Add the following manually through the agreed channel:

```text
Expected outcome:
What happened instead:
Model ID/revision or non-sensitive architecture description:
Task and data format (no customer examples unless approved):
Control: native / native compiled / OptimizerPlus
Optimizer, LR/schedule, precision, sequence length, microbatch/accumulation:
Planned and completed updates:
Does the native command reproduce the issue?:
Minimal reproduction command with private paths redacted:
Reviewed error excerpt:
```

Do not attach the entire run directory by default. It can contain customer configuration, paths, data identities, adapters or checkpoints. Never include licence contents, signing keys, tokens or raw customer data in a public issue. Review [troubleshooting](TROUBLESHOOTING.md) for common fixes before rerunning an expensive workload.
