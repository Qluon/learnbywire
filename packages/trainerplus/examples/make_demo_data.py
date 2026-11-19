"""Create small, synthetic plumbing-test data; not a benchmark or useful training corpus."""
import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', default='demo-data')
    args = parser.parse_args()
    root = Path(args.output)
    root.mkdir(parents=True, exist_ok=False)
    for name, numbers in [('train', range(100)), ('validation', range(100, 108))]:
        with (root/(name+'.jsonl')).open('x') as stream:
            for n in numbers:
                stream.write(json.dumps({'text': f'A box contains {n} red beads and 3 blue beads. It contains {n+3} beads in total.'})+'\n')
    print(f'Created 100 training and 8 validation examples in {root}. Replace these with your own data for useful training.')


if __name__ == '__main__':
    main()
