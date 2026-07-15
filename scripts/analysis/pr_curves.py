"""
pr_curves.py — mean precision-recall curves for the temporal fusion chapter.

Runs the official nuScenes detection evaluation on each results.json, extracts
the per-class PR arrays at the 2 m distance threshold, averages over all 10
detection classes, and saves a figure suitable for inclusion in the thesis.

Results are cached alongside each results.json (pr_cache.json) so re-running
is fast after the first call.

Usage:
    python scripts/analysis/pr_curves.py \\
        --dataroot data/nuscenes --version v1.0-trainval \\
        --out results/pr_curves.png
"""
from __future__ import annotations

import argparse
import json
import sys
import tempfile
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_ROOT))

from nuscenes.nuscenes import NuScenes
from nuscenes.eval.detection.config import config_factory
from nuscenes.eval.detection.evaluate import NuScenesEval
from nuscenes.eval.detection.data_classes import DetectionMetricDataList

VARIANTS = [
    ('CenterFusion',        'results/center_fusion/full_learned/results.json'),
    ('T1 (smooth+rescore)', 'results/track_fusion/full/results.json'),
    ('T2 (+gap interp)',    'results/track_fusion/full/t2/results.json'),
]

DIST_TH = 2.0
RECALL_GRID = np.linspace(0.0, 1.0, 101)
COLORS = ['#4c72b0', '#dd8452', '#55a868']
LINESTYLES = ['-', '-', '-']


def load_or_eval(nusc, result_path: str, split: str, cache_path: Path) -> DetectionMetricDataList:
    if cache_path.exists():
        print(f'  cache hit: {cache_path.name}')
        with open(cache_path) as f:
            return DetectionMetricDataList.deserialize(json.load(f))

    print(f'  evaluating {result_path} ...')
    cfg = config_factory('detection_cvpr_2019')
    with tempfile.TemporaryDirectory() as tmp:
        ev = NuScenesEval(nusc, config=cfg, result_path=result_path,
                          eval_set=split, output_dir=tmp, verbose=False)
        _, mdl = ev.evaluate()

    with open(cache_path, 'w') as f:
        json.dump(mdl.serialize(), f)
    return mdl


def mean_pr(mdl: DetectionMetricDataList) -> np.ndarray:
    """Average precision over all classes, interpolated to RECALL_GRID at DIST_TH."""
    precisions = []
    for md, _ in mdl.get_dist_data(DIST_TH):
        interp = np.interp(RECALL_GRID, md.recall, md.precision, left=0.0, right=0.0)
        precisions.append(interp)
    return np.mean(precisions, axis=0)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataroot', default='data/nuscenes')
    parser.add_argument('--version',  default='v1.0-trainval')
    parser.add_argument('--split',    default='val')
    parser.add_argument('--out',      default='results/pr_curves.png')
    args = parser.parse_args()

    nusc = NuScenes(version=args.version, dataroot=args.dataroot, verbose=False)

    fig, ax = plt.subplots(figsize=(6, 4.5))

    for (label, rel_path), color, ls in zip(VARIANTS, COLORS, LINESTYLES):
        result_path = str(_ROOT / rel_path)
        cache_path  = (_ROOT / rel_path).with_name('pr_cache.json')
        print(f'\n[{label}]')
        mdl  = load_or_eval(nusc, result_path, args.split, cache_path)
        mean = mean_pr(mdl)
        ap   = float(np.trapz(mean, RECALL_GRID))
        ax.plot(RECALL_GRID, mean, label=f'{label}  (mAP={ap:.3f})',
                color=color, linestyle=ls, linewidth=1.8)

    ax.set_xlabel('Recall', fontsize=11)
    ax.set_ylabel('Precision', fontsize=11)
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.legend(fontsize=9, loc='upper right')
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    out = _ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150, bbox_inches='tight')
    print(f'\nSaved → {args.out}')


if __name__ == '__main__':
    main()
