"""
tp_histogram.py — True Positive count by distance for one or two detectors.

For each GT annotation, finds the nearest predicted detection of the same
category within a match threshold (greedy NN, same logic as nuScenes eval).
Plots a histogram of TP count by distance to ego.

Pass one --results path for a single detector, or two paths separated by a
comma for side-by-side / overlaid comparison (e.g. radar vs camera).

Usage
-----
    # Radar only (full val)
    python scripts/analysis/tp_histogram.py \
        --results results/radar_baseline/full/results.json \
        --labels "Radar baseline" \
        --dataroot data/nuscenes --version v1.0-trainval \
        --out results/tp_histogram_radar.png

    # Both overlaid
    python scripts/analysis/tp_histogram.py \
        --results "results/radar_baseline/full/results.json,results/fcos3d_baseline/full/detections/pred_instances_3d/results_nusc.json" \
        --labels "Radar,FCOS3D (camera)" \
        --dataroot data/nuscenes --version v1.0-trainval \
        --out results/tp_histogram_comparison.png
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from nuscenes.nuscenes import NuScenes
from nuscenes.utils.splits import create_splits_scenes

DETECTION_CLASSES = {
    'vehicle.car':                          'car',
    'vehicle.truck':                        'truck',
    'vehicle.bus.rigid':                    'bus',
    'vehicle.bus.bendy':                    'bus',
    'vehicle.motorcycle':                   'motorcycle',
    'vehicle.bicycle':                      'bicycle',
    'human.pedestrian.adult':               'pedestrian',
    'human.pedestrian.child':               'pedestrian',
    'human.pedestrian.construction_worker': 'pedestrian',
    'human.pedestrian.police_officer':      'pedestrian',
    'vehicle.trailer':                      'trailer',
    'vehicle.construction':                 'construction_vehicle',
    'movable_object.barrier':               'barrier',
    'movable_object.trafficcone':           'traffic_cone',
}


def ego_xy(nusc, sample_token: str) -> np.ndarray:
    sample = nusc.get('sample', sample_token)
    lidar_sd = nusc.get('sample_data', sample['data']['LIDAR_TOP'])
    ep = nusc.get('ego_pose', lidar_sd['ego_pose_token'])
    return np.array(ep['translation'][:2])


def xy_dist(a, b) -> float:
    return float(np.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2))


def collect_tp_distances(
    nusc,
    split_tokens: list[str],
    predictions: dict,
    threshold: float,
) -> np.ndarray:
    """
    Returns an array of GT distances to ego for every TP match.
    Greedy nearest-neighbour matching per category, same as nuScenes eval.
    """
    tp_dists = []

    for sample_token in split_tokens:
        preds = predictions.get(sample_token, [])
        if not preds:
            continue

        preds_by_cat: dict[str, list] = defaultdict(list)
        for p in preds:
            preds_by_cat[p['detection_name']].append(p)

        ego = ego_xy(nusc, sample_token)
        sample = nusc.get('sample', sample_token)
        gts = [nusc.get('sample_annotation', tok) for tok in sample['anns']]

        matched_pred_indices: set[int] = set()

        for gt in gts:
            cat = DETECTION_CLASSES.get(gt['category_name'])
            if cat is None:
                continue
            candidates = preds_by_cat.get(cat, [])
            if not candidates:
                continue

            gt_xy = np.array(gt['translation'][:2])
            best_d, best_i = float('inf'), -1
            for i, p in enumerate(candidates):
                if i in matched_pred_indices:
                    continue
                d = xy_dist(p['translation'][:2], gt_xy)
                if d < best_d:
                    best_d, best_i = d, i

            if best_d <= threshold:
                matched_pred_indices.add(best_i)
                gt_dist = xy_dist(gt_xy, ego)
                tp_dists.append(gt_dist)

    return np.array(tp_dists)


def get_split_tokens(nusc, split: str) -> list[str]:
    scenes_in_split = create_splits_scenes()[split]
    tokens = []
    for scene in nusc.scene:
        if scene['name'] in scenes_in_split:
            tok = scene['first_sample_token']
            while tok:
                tokens.append(tok)
                tok = nusc.get('sample', tok)['next']
    return tokens


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--results', required=True,
                        help='Path(s) to results JSON. Comma-separated for multiple.')
    parser.add_argument('--labels', default='',
                        help='Comma-separated display labels for each results file.')
    parser.add_argument('--dataroot', default='data/nuscenes')
    parser.add_argument('--version', default='v1.0-trainval')
    parser.add_argument('--split', default='val')
    parser.add_argument('--match-threshold', type=float, default=2.0,
                        help='Max XY distance (m) to count as a TP match')
    parser.add_argument('--max-distance', type=float, default=80.0)
    parser.add_argument('--bins', type=int, default=40)
    parser.add_argument('--out', default='results/tp_histogram.png')
    args = parser.parse_args()

    result_paths = [p.strip() for p in args.results.split(',')]
    labels_raw = [l.strip() for l in args.labels.split(',')] if args.labels else []
    labels = []
    for i, p in enumerate(result_paths):
        labels.append(labels_raw[i] if i < len(labels_raw) else Path(p).stem)

    print(f'Loading nuScenes {args.version} from {args.dataroot}...')
    nusc = NuScenes(version=args.version, dataroot=args.dataroot, verbose=False)
    tokens = get_split_tokens(nusc, args.split)
    print(f'Split "{args.split}": {len(tokens)} samples')

    colors = ['steelblue', 'tomato', 'seagreen', 'darkorange']
    fig, ax = plt.subplots(figsize=(9, 5))

    for i, (path, label) in enumerate(zip(result_paths, labels)):
        print(f'\n[{label}] Loading {path}...')
        with open(path) as f:
            data = json.load(f)
        preds = data['results']

        dists = collect_tp_distances(nusc, tokens, preds, args.match_threshold)
        print(f'  TPs: {len(dists):,}  |  median range: {np.median(dists):.1f} m  '
              f'|  90th pct: {np.percentile(dists, 90):.1f} m')

        ax.hist(
            dists,
            bins=args.bins,
            range=(0, args.max_distance),
            alpha=0.65,
            color=colors[i % len(colors)],
            edgecolor='none',
            label=f'{label}  (n={len(dists):,})',
        )

    ax.set_xlabel('GT distance to ego (m)')
    ax.set_ylabel('Ground-truth objects covered')
    title = 'Ground-truth coverage by distance to ego'
    if len(result_paths) > 1:
        title += f'  [match threshold <= {args.match_threshold} m]'
    ax.set_title(title)
    ax.legend()
    ax.grid(True, axis='y', alpha=0.3)

    plt.tight_layout()
    import os
    os.makedirs(Path(args.out).parent, exist_ok=True)
    plt.savefig(args.out, dpi=150)
    print(f'\nSaved -> {args.out}')


if __name__ == '__main__':
    main()
