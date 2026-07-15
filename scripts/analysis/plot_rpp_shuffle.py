"""
plot_rpp_shuffle.py — Bar chart of the rpp dual-shuffle ablation at convergence.

Plots NDS for the four conditions of Table~\\ref{tab:rpp-shuffle} (best-NDS
checkpoint, epoch 16, full validation split): true, radar-shuffle, paint-zero,
paint-shuffle. Annotates the camera contribution (True - Paint-zero) and the
radar collapse under radar-shuffle.

Usage:
    python scripts/analysis/plot_rpp_shuffle.py --out results/rpp/dual_shuffle.png
"""
from __future__ import annotations

import argparse
import os

import matplotlib.pyplot as plt

# NDS at the best-NDS checkpoint (epoch 16, full validation split),
# from Table~\ref{tab:rpp-shuffle}.
CONDITIONS = ['True', 'Radar-shuffle', 'Paint-zero', 'Paint-shuffle']
NDS = [0.3344, 0.0000, 0.1073, 0.0304]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', default='results/rpp/dual_shuffle.png')
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.out), exist_ok=True)

    fig, ax = plt.subplots(figsize=(8, 5))
    x = range(len(CONDITIONS))
    ax.bar(x, NDS, color='steelblue', alpha=0.85, edgecolor='none', width=0.6)

    for i, v in enumerate(NDS):
        ax.text(i, v + 0.01, f'{v:.4f}', ha='center', va='bottom', fontsize=9)

    # Camera contribution: True - Paint-zero.
    delta_paint = NDS[0] - NDS[2]
    ax.annotate(
        '', xy=(0, NDS[0] + 0.045), xytext=(2, NDS[2] + 0.045),
        arrowprops=dict(arrowstyle='<->', color='dimgray', lw=1.2),
    )
    ax.text(1, NDS[0] + 0.06, r'$\Delta$NDS(paint) $= +$' + f'{delta_paint:.4f}',
            ha='center', va='bottom', fontsize=9, color='dimgray')

    # Radar collapse under radar-shuffle.
    ax.annotate(
        'radar-shuffle\ncollapses NDS to 0',
        xy=(1, 0.0), xytext=(1, 0.16),
        ha='center', va='bottom', fontsize=9, color='firebrick',
        arrowprops=dict(arrowstyle='->', color='firebrick', lw=1.2),
    )

    ax.set_xticks(list(x))
    ax.set_xticklabels(CONDITIONS)
    ax.set_ylabel('NDS')
    ax.set_ylim(0, max(NDS) + 0.12)
    ax.set_title('rpp dual-shuffle ablation (epoch 16, full validation split)')
    ax.grid(True, axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig(args.out, dpi=150)
    print(f'Saved to {args.out}')


if __name__ == '__main__':
    main()
