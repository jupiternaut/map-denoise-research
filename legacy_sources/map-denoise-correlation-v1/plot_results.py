"""Plots of saved measurements; never generate or edit experimental outcomes."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--analysis', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    rows = json.loads((args.analysis / 'paired.json').read_text())['records']
    families = ['single', 'dual_balanced', 'dual_imbalanced', 'raycast_gap4', 'raycast_gap8', 'curved_dual']
    labels = ['Single\nplane', 'Dual\nbalanced', 'Dual\nimbalanced', 'Visible plates\n4 mm', 'Visible plates\n8 mm', 'Curved\ndual']
    plt.rcParams.update({'font.size': 10, 'axes.spines.top': False, 'axes.spines.right': False})
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.6), sharey=True, layout='constrained')
    for ax, method, title in zip(axes, ['xyz_mixture', 'fast'], ['XYZ-only scalar control', 'Frame-aware fast filter']):
        for side, color, offset in [('correlated', '#3273a8', -.19), ('shuffled', '#db7c34', .19)]:
            means, lows, highs = [], [], []
            for family in families:
                values = [r[f'{side}_normal_mae_mm'] for r in rows if r['family'] == family and
                          r['amplitude_mm'] == 4 and r['sigma_protocol'] == 'base_sigma' and r['method'] == method]
                mean = np.mean(values)
                means.append(mean); lows.append(mean-min(values)); highs.append(max(values)-mean)
            ax.bar(np.arange(6)+offset, means, .36, color=color, label=side,
                   yerr=np.array([lows, highs]), capsize=3, error_kw={'elinewidth': 1})
        ax.set_xticks(np.arange(6), labels)
        ax.set_title(title)
        ax.grid(axis='y', alpha=.2)
        ax.set_axisbelow(True)
        ax.legend(frameon=False)
    axes[0].set_ylabel('Mean absolute z error (mm)')
    fig.suptitle('Same error multiset; different assignment to acquisition frames\nFrame-bias RMS = 4 mm, supplied sigma = 1 mm; bars: 3 seeds, whiskers: seed range', fontsize=12)
    fig.savefig(args.out / 'paired-correlation.png', dpi=190)
    fig.savefig(args.out / 'paired-correlation.pdf')
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9.5, 4.5), layout='constrained')
    for mode, color, offset in [('base_sigma', '#3273a8', -.19), ('total_scale', '#ac4860', .19)]:
        vals = [np.mean([r['correlated_normal_mae_mm'] for r in rows if r['family'] == family and
                         r['amplitude_mm'] == 4 and r['sigma_protocol'] == mode and r['method'] == 'fast']) for family in families]
        ax.bar(np.arange(6)+offset, vals, .36, color=color,
               label='supplied sigma = 1 mm' if mode == 'base_sigma' else 'supplied sigma = sqrt(17) mm')
    ax.set_xticks(np.arange(6), labels)
    ax.set_ylabel('Mean absolute z error (mm)')
    ax.set_title('Shared-bias data: treating total error as residual noise erases real layers\nFrozen fast filter; same observations, 3-seed means')
    ax.legend(frameon=False)
    ax.grid(axis='y', alpha=.2); ax.set_axisbelow(True)
    fig.savefig(args.out / 'noise-scale-sensitivity.png', dpi=190)
    fig.savefig(args.out / 'noise-scale-sensitivity.pdf')
    plt.close(fig)
    print(args.out)


if __name__ == '__main__':
    main()
