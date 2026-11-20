# Request and activate an OptimizerPlus licence

LearnByWire 1.2.2 and TrainerPlus 1.1.1 are Apache-2.0 packages. OptimizerPlus 1.2.2 is a separately licensed proprietary control layer over your existing optimizer. Installing a wheel does not grant or activate an OptimizerPlus entitlement. Native TrainerPlus training does not need one.

## Request access

1. Complete the [installation steps](../GETTING_STARTED.md#1-install) on the Linux environment where training will run.
2. Run `optimizerplus machine-id` **inside that environment**. Keep the output for your private request; do not publish it in a GitHub issue.
3. Use [Qluon's private pilot request form](https://qluon.ai/contact.php?topic=pilot), or contact your existing Qluon representative. Request **OptimizerPlus 1.2.2 evaluation access** and provide the following:

```text
Contact name and work email:
Company/team:
Requested product: OptimizerPlus 1.2.2
Machine fingerprint (from optimizerplus machine-id):
GPU model and number of GPUs visible to this process:
Python and PyTorch versions:
Model ID or architecture and approximate size:
Task: CPT / SFT / other
Adaptation: LoRA / other
Optimizer:
Intended evaluation dates:
What would make the evaluation useful to you:
```

Use a general model description if its identity is confidential. Do not send training data, model weights, passwords or tokens. Qluon confirms access terms, duration and scope before issuing a signed licence file. Submission does not automatically activate access or promise a response time.

## Activate the returned file

Save the file outside the source repository, then use its actual path:

```bash
chmod 600 /private/path/customer.license
export QLUON_OPTIMIZER_PLUS_LICENSE_FILE=/private/path/customer.license
export CUDA_VISIBLE_DEVICES=0
optimizerplus version
optimizerplus verify-install
optimizerplus license-status
optimizerplus doctor --check
```

Expected outcomes: version `1.2.2`; installation verification succeeds; licence status is accepted; the environment check reports `ready: true`. These checks do not start training. Run the [first training example](../GETTING_STARTED.md#4-train) next to check the actual workload.

Customers receive a signed licence file. They do not need Qluon's private signing key. Set the environment variable in each new shell or approved job configuration; do not paste the licence contents into commands, source files or issue reports.

## Renewal and machine changes

Contact Qluon before expiry if more evaluation time is needed. For a new host, container or WSL installation, run `optimizerplus machine-id` in the new training environment and request a matching licence. Do not assume that a licence for another environment will work. Keep the existing environment and evidence until migration is checked.

For an error, report its category—missing licence, expired, wrong release, wrong machine or signature rejection—and the package version privately. If the error says permissions are too broad, apply `chmod 600` to the licence file. Follow the [support-report guide](SUPPORT_REPORT.md); do not attach the licence itself to a public report.
