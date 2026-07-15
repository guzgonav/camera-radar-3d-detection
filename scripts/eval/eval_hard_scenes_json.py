"""
eval_hard_scenes_json.py — Evaluate an existing nuScenes-submission results.json
on a hard-scene subset, without going through an mmdet3d Runner.

Companion to eval_hard_scenes.py: that script handles mmdet3d-Runner-based
methods (v3b, rpp) via SubsetNuScenesMetric (datasets/nuscenes_radar_dataset.py).
This script handles the JSON/post-processing-based methods (late fusion,
CenterFusion, T1, T2) that already terminate in a full-val results.json — no
new inference needed, just filtering + a patched NuScenesEval.

Mechanism (same trick as SubsetNuScenesMetric, applied directly to
nuscenes.eval.detection.evaluate.NuScenesEval instead of through mmdet3d):
the official NuScenesEval asserts pred_tokens == gt_tokens for the full split,
which fails on any subset. We filter the results.json down to the hard-scene
sample tokens (from the same nuscenes_infos_hard<N>_val.pkl used by
eval_hard_scenes.py, so "hard scenes" means the same scenes across every
method), then monkeypatch load_gt to only return GT for tokens present in the
filtered predictions.

Usage:
    python scripts/eval/eval_hard_scenes_json.py \\
        --results results/late_fusion/full/results.json \\
        --hard-pkl data/nuscenes/nuscenes_infos_hard20_val.pkl \\
        --out-dir results/late_fusion/hard20 \\
        --label "Late Fusion"

    # or pick by --n-scenes instead of an explicit --hard-pkl
    python scripts/eval/eval_hard_scenes_json.py \\
        --results results/track_fusion/full/t2/results.json \\
        --n-scenes 50 \\
        --out-dir results/track_fusion/full/t2/hard50 \\
        --label "T2"
"""
from __future__ import annotations

import argparse
import json
import os
import pickle
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_ROOT))

DETECTION_CLASSES = [
    'car', 'truck', 'bus', 'trailer', 'construction_vehicle',
    'pedestrian', 'motorcycle', 'bicycle', 'traffic_cone', 'barrier',
]


# ---------------------------------------------------------------------------
# Hard-scene token set + results filtering
# ---------------------------------------------------------------------------

def load_hard_tokens(hard_pkl: str) -> set[str]:
    with open(hard_pkl, 'rb') as f:
        data = pickle.load(f)
    return {info['token'] for info in data['data_list']}


def filter_results(results_path: str, tokens: set[str], out_path: str) -> int:
    """Filter a nuScenes-submission results.json down to `tokens`. Returns the
    number of hard-scene tokens actually found in the predictions."""
    with open(results_path) as f:
        data = json.load(f)

    full_results = data['results']
    missing = tokens - full_results.keys()
    if missing:
        print(f'WARNING: {len(missing)}/{len(tokens)} hard-scene tokens not '
              f'found in {results_path} (evaluating on the intersection)')

    data['results'] = {tok: dets for tok, dets in full_results.items()
                        if tok in tokens}

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, 'w') as f:
        json.dump(data, f)

    return len(data['results'])


# ---------------------------------------------------------------------------
# Evaluation (SubsetNuScenesMetric's load_gt trick, applied directly)
# ---------------------------------------------------------------------------

def evaluate_subset(nusc, filtered_results_path: str, split: str,
                     out_dir: str, label: str, verbose: bool = True) -> dict:
    import nuscenes.eval.detection.evaluate as _nus_eval_mod
    from nuscenes.eval.common.data_classes import EvalBoxes
    from nuscenes.eval.detection.config import config_factory

    with open(filtered_results_path) as f:
        pred_tokens = set(json.load(f)['results'].keys())

    _orig_load_gt = _nus_eval_mod.load_gt

    def _subset_load_gt(nusc_, eval_set, box_cls, verbose=False):
        all_gt = _orig_load_gt(nusc_, eval_set, box_cls, verbose=False)
        subset = EvalBoxes()
        for tok in sorted(pred_tokens):
            subset.add_boxes(tok, all_gt.boxes.get(tok, []))
        return subset

    _nus_eval_mod.load_gt = _subset_load_gt
    try:
        cfg = config_factory('detection_cvpr_2019')
        evaluator = _nus_eval_mod.NuScenesEval(
            nusc, config=cfg, result_path=filtered_results_path,
            eval_set=split, output_dir=out_dir, verbose=verbose)
        metrics, _ = evaluator.evaluate()
    finally:
        _nus_eval_mod.load_gt = _orig_load_gt

    summary = {
        'mAP': metrics.mean_ap,
        'NDS': metrics.nd_score,
        'mATE': metrics.tp_errors.get('trans_err', float('nan')),
        'mASE': metrics.tp_errors.get('scale_err', float('nan')),
        'mAOE': metrics.tp_errors.get('orient_err', float('nan')),
        'mAVE': metrics.tp_errors.get('vel_err', float('nan')),
        'mAAE': metrics.tp_errors.get('attr_err', float('nan')),
    }
    for cls in DETECTION_CLASSES:
        summary[f'{cls}_AP'] = metrics.mean_dist_aps.get(cls, float('nan'))

    metrics_path = os.path.join(out_dir, 'metrics.json')
    with open(metrics_path, 'w') as f:
        json.dump(summary, f, indent=2)

    if verbose:
        print(f'\n{"="*60}')
        print(f'  {label} — hard-scene subset ({len(pred_tokens)} samples)')
        print(f'{"="*60}')
        print(f'  mAP : {summary["mAP"]:.4f}')
        print(f'  NDS : {summary["NDS"]:.4f}')
        print(f'  mATE: {summary["mATE"]:.4f}  mASE: {summary["mASE"]:.4f}  '
              f'mAOE: {summary["mAOE"]:.4f}')
        print(f'  mAVE: {summary["mAVE"]:.4f}  mAAE: {summary["mAAE"]:.4f}')
        print(f'{"="*60}')
        print(f'Metrics saved → {metrics_path}')

    return summary


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description='Evaluate an existing results.json on a hard-scene subset '
                     '(no mmdet3d Runner required).')
    p.add_argument('--results', required=True,
                    help='Path to full-val nuScenes-submission results.json')
    p.add_argument('--hard-pkl', default=None,
                    help='Filtered hard-scene pkl (default: '
                         'data/nuscenes/nuscenes_infos_hard<N>_val.pkl, '
                         'must already exist — built by eval_hard_scenes.py)')
    p.add_argument('--n-scenes', type=int, default=20,
                    help='Used to derive --hard-pkl when not given explicitly '
                         '(default: 20)')
    p.add_argument('--version', default='v1.0-trainval')
    p.add_argument('--dataroot', default='data/nuscenes')
    p.add_argument('--split', default='val')
    p.add_argument('--out-dir', required=True,
                    help='Output directory for the filtered results.json and metrics.json')
    p.add_argument('--label', default=None,
                    help='Method name for print headers (default: derived from --results)')
    return p.parse_args()


def main():
    args = parse_args()
    os.chdir(_ROOT)

    hard_pkl = args.hard_pkl or f'data/nuscenes/nuscenes_infos_hard{args.n_scenes}_val.pkl'
    if not Path(hard_pkl).exists():
        raise FileNotFoundError(
            f'{hard_pkl} does not exist yet — build it once via eval_hard_scenes.py '
            f'(e.g. against v3b or rpp), then re-run this script.')

    label = args.label or Path(args.results).parent.name

    tokens = load_hard_tokens(hard_pkl)
    print(f'Loaded {len(tokens)} hard-scene tokens from {hard_pkl}')

    filtered_path = os.path.join(args.out_dir, 'results_filtered.json')
    n_filtered = filter_results(args.results, tokens, filtered_path)
    print(f'Filtered {args.results} → {n_filtered} hard-scene samples '
          f'→ {filtered_path}')

    from nuscenes.nuscenes import NuScenes
    print(f'Loading nuScenes {args.version} from {args.dataroot}...')
    nusc = NuScenes(version=args.version, dataroot=args.dataroot, verbose=False)

    evaluate_subset(nusc, filtered_path, args.split, args.out_dir, label,
                     verbose=True)


if __name__ == '__main__':
    main()
