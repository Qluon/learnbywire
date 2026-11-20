"""Write a small local report; never collect licences, raw logs or customer data."""
import argparse
import importlib.metadata
import json
import math
from pathlib import Path
import platform


def numeric_fields(source, names):
    if not isinstance(source, dict):
        return {}
    return {name: source[name] for name in names
            if isinstance(source.get(name), (int, float))
            and not isinstance(source[name], bool) and math.isfinite(source[name])}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--run-dir', type=Path, help='Printed trainer artifact directory containing result.json')
    args = parser.parse_args(argv)
    if args.output.exists():
        parser.error('Output already exists; choose a new filename')
    report = {'schema_version': 1, 'python': platform.python_version(),
              'platform': {'system': platform.system(), 'architecture': platform.machine()},
              'packages': {}, 'network_transmission': False}
    for name in ('learnbywire', 'learnbywire-optimizerplus', 'learnbywire-trainerplus',
                 'torch', 'transformers', 'peft', 'accelerate', 'triton', 'safetensors'):
        try:
            report['packages'][name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            report['packages'][name] = None
    try:
        import torch
        report['cuda_runtime'] = torch.version.cuda
        report['cuda_available'] = torch.cuda.is_available()
        report['visible_gpus'] = [
            {'name': torch.cuda.get_device_name(i),
             'memory_bytes': torch.cuda.get_device_properties(i).total_memory,
             'compute_capability': list(torch.cuda.get_device_capability(i))}
            for i in range(torch.cuda.device_count())] if report['cuda_available'] else []
    except (ImportError, RuntimeError, AssertionError) as error:
        report['gpu_check_error_type'] = type(error).__name__
    if args.run_dir is not None:
        path = args.run_dir/'result.json'
        try:
            if path.stat().st_size > 2_000_000:
                parser.error('Result is unexpectedly large; inspect it manually')
            result = json.loads(path.read_text())
        except (OSError, ValueError):
            parser.error('Cannot read result.json; use the completed trainer artifact directory')
        if not isinstance(result, dict):
            parser.error('result.json must contain an object')
        status = result.get('status')
        selected = {'status': status if status in ('complete','stopped','failed','running') else 'unknown'}
        selected.update(numeric_fields(result, ('committed_steps',)))
        metrics = result.get('metrics', {})
        if not isinstance(metrics, dict):
            metrics = {}
        selected['metrics'] = numeric_fields(metrics, ('training_seconds','workflow_seconds','examples','input_tokens',
            'supervised_tokens','input_tokens_per_second','supervised_tokens_per_second','updates_this_run'))
        evaluation = metrics.get('evaluation', {})
        if isinstance(evaluation, dict):
            selected['evaluation'] = numeric_fields(evaluation, ('loss','perplexity','eval_loss','eval_perplexity',
                                                                'next_token_accuracy','supervised_tokens','tokens'))
        report['run'] = selected
    with args.output.open('x') as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
        stream.write('\n')
    print(f'Written {args.output}. Review before sharing; nothing was transmitted.')
    return report


if __name__ == '__main__':
    main()
