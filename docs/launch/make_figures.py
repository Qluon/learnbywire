"""Regenerate public launch figures from the adjacent, traceable measurement JSON."""
from pathlib import Path
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

HERE = Path(__file__).resolve().parent
DATA = json.loads((HERE / 'evidence.json').read_text())
COLORS = ['#718297', '#205eab']

def main():
    output = HERE / 'assets'
    output.mkdir(exist_ok=True)
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 11, 'axes.spines.top': False,
        'axes.spines.right': False, 'axes.spines.left': False, 'axes.edgecolor': '#ccd6e2',
        'text.color': '#102c55', 'axes.labelcolor': '#102c55', 'xtick.color': '#51627a',
        'ytick.color': '#51627a', 'savefig.facecolor': 'white'})
    specs = [('workflow_tok_s', 'Workflow tokens/s', 1, 0),
             ('workflow_seconds', 'Complete workflow (minutes)', 60, 2),
             ('val_ppl', 'Validation perplexity', 1, 4),
             ('retention_ppl', 'Retention perplexity', 1, 4),
             ('gpu_wh', 'GPU energy (Wh)', 1, 2),
             ('reserved_gib', 'Peak reserved VRAM (GiB)', 1, 3)]
    for model, comparison in DATA['comparisons'].items():
        a, b = comparison['native'], comparison['optimizerplus']
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        fig.subplots_adjust(top=.73, bottom=.18, hspace=.68, wspace=.30)
        title = 'TinyLlama 1.1B' if model == 'tiny' else 'SmolLM2 1.7B'
        baseline = 'Direct public Hugging Face Trainer · compiled · fused AdamW' if model == 'tiny' else 'TrainerPlus native compiled · fused AdamW'
        fig.text(.07, .95, 'QLUON  /  OPTIMIZERPLUS 1.2.2', size=12, weight='bold', color='#c9284b')
        fig.text(.07, .90, f'{title}: the complete training comparison', size=25, weight='bold')
        fig.text(.07, .86, f'Native baseline: {baseline}', size=12)
        fig.text(.07, .825, 'Native is always left. TrainerPlus + OptimizerPlus is always right. Bars begin at zero.', size=11, color='#51627a')
        for ax, (key, label, divisor, digits) in zip(axes.flat, specs):
            vals = [a[key] / divisor, b[key] / divisor]
            change = (vals[1]/vals[0]-1)*100
            ax.bar([0, 1], vals, color=COLORS, width=.58, zorder=3)
            ax.set_xticks([0, 1], ['Native', 'OptimizerPlus'])
            ax.set_ylim(0, max(vals)*1.27)
            ax.yaxis.set_major_locator(MaxNLocator(4))
            ax.grid(axis='y', color='#e5ebf2', zorder=0)
            ax.tick_params(axis='both', length=0)
            ax.set_title(label + '\n' + f'{change:+.1f}% versus native', loc='left', pad=18, size=12)
            for i, value in enumerate(vals):
                ax.text(i, value+max(vals)*.04, f'{value:,.{digits}f}', ha='center', weight='bold', size=12)
        scope = ('One seed (4007), one RTX 5090 Laptop GPU (24 GiB). 7,999,488 input tokens, one epoch, 7,812 updates.\n'
                 'Complete systems comparison: loading, setup/compilation, training, 33 evaluations and export. Downloads/tokenization excluded.\n'
                 'GPU energy excludes CPU and wall-power overhead. Higher reserved memory is a trade-off; perplexity is not answer accuracy.\n')
        if model == 'smol':
            scope += 'Retention worsened from the original model (16.5260 to 17.1634), despite slightly beating trained native.\n'
        else:
            scope += 'Original validation / retention perplexity: 9.6758 / 13.7891. QA-key measured wheel; production checks separate.\n'
        scope += 'Source: github.com/Qluon/learnbywire/tree/main/benchmarks · TrainerPlus 1.1.1 · Qluon measurements, not external replication.'
        fig.text(.07, .045, scope, size=9, linespacing=1.6, color='#51627a')
        name = 'benchmark-overview' if model == 'tiny' else 'smollm2-comparison'
        for extension in ['png', 'svg', 'pdf']:
            fig.savefig(output / f'{name}.{extension}', dpi=160, metadata={'Creator': 'Qluon Inc.'})
        plt.close(fig)
    print('Generated six figure files from evidence.json')

if __name__ == '__main__':
    main()
