"""
error_by_distance.py — Position error of late-fusion detections vs. distance to ego.

For each GT annotation in the val set, finds the nearest predicted detection
of the same category (within a match threshold). Computes XY position error
and GT distance to ego, then plots:
  - Scatter: error vs. distance (all matches)
  - Bar chart: mean + std error per distance bin

Usage:
    python scripts/analysis/error_by_distance.py \
        --results results/late_fusion/full/results.json \
        --dataroot data/nuscenes \
        --version v1.0-trainval \
        --match-threshold 2.0 \
        --max-distance 80.0 \
        --bin-size 10.0
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict

import numpy as np
import matplotlib.pyplot as plt
from nuscenes.nuscenes import NuScenes

DETECTION_CLASSES = {
    'vehicle.car':          'car',
    'vehicle.truck':        'truck',
    'vehicle.bus.rigid':    'bus',
    'vehicle.bus.bendy':    'bus',
    'vehicle.motorcycle':   'motorcycle',
    'vehicle.bicycle':      'bicycle',
    'human.pedestrian.adult':           'pedestrian',
    'human.pedestrian.child':           'pedestrian',
    'human.pedestrian.construction_worker': 'pedestrian',
    'human.pedestrian.police_officer':  'pedestrian',
    'vehicle.trailer':      'trailer',
    'vehicle.construction': 'construction_vehicle',
    'movable_object.barrier':   'barrier',
    'movable_object.trafficcone': 'traffic_cone',
}


def xy_dist(a, b):
    return float(np.sqrt((a[0] - b[0])**2 + (a[1] - b[1])**2))


def match_detections(preds, gts, ego_xy, threshold):
    """
    Greedy nearest-neighbour matching (same category).
    Returns list of (gt_dist_to_ego, xy_error).
    """
    # Group preds by category
    preds_by_cat = defaultdict(list)
    for p in preds:
        preds_by_cat[p['detection_name']].append(p)

    matched = []
    for gt in gts:
        cat = DETECTION_CLASSES.get(gt['category_name'])
        if cat is None:
            continue
        candidates = preds_by_cat.get(cat, [])
        if not candidates:
            continue

        gt_xy = gt['translation'][:2]
        gt_dist = xy_dist(gt_xy, ego_xy)

        best_dist = float('inf')
        best_pred = None
        for p in candidates:
            d = xy_dist(p['translation'][:2], gt_xy)
            if d < best_dist:
                best_dist = d
                best_pred = p

        if best_dist <= threshold:
            matched.append((gt_dist, best_dist))

    return matched


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--results', default='results/late_fusion/full/results.json')
    parser.add_argument('--dataroot', default='data/nuscenes')
    parser.add_argument('--version', default='v1.0-trainval')
    parser.add_argument('--match-threshold', type=float, default=2.0,
                        help='Max XY distance (m) to count as a TP match')
    parser.add_argument('--max-distance', type=float, default=80.0)
    parser.add_argument('--bin-size', type=float, default=10.0)
    parser.add_argument('--out', default='results/late_fusion/full/error_by_distance.png')
    parser.add_argument('--title', default='Late Fusion',
                        help='Method name for the plot title')
    args = parser.parse_args()

    print(f'Loading {args.version} from {args.dataroot}...')
    nusc = NuScenes(version=args.version, dataroot=args.dataroot, verbose=False)

    print(f'Loading predictions from {args.results}...')
    with open(args.results) as f:
        pred_data = json.load(f)
    predictions = pred_data['results']  # {sample_token: [det, ...]}

    all_matches = []  # (gt_dist_to_ego, xy_error)

    print(f'Matching {len(predictions)} samples...')
    for sample_token, preds in predictions.items():
        sample = nusc.get('sample', sample_token)
        lidar_sd = nusc.get('sample_data', sample['data']['LIDAR_TOP'])
        ego_pose = nusc.get('ego_pose', lidar_sd['ego_pose_token'])
        ego_xy = ego_pose['translation'][:2]

        gts = [nusc.get('sample_annotation', tok) for tok in sample['anns']]
        matches = match_detections(preds, gts, ego_xy, args.match_threshold)
        all_matches.extend(matches)

    if not all_matches:
        print('No matches found. Try increasing --match-threshold.')
        return

    dists = np.array([m[0] for m in all_matches])
    errors = np.array([m[1] for m in all_matches])
    print(f'Total TP matches: {len(all_matches)}')
    print(f'Mean error: {errors.mean():.3f} m | Median: {np.median(errors):.3f} m')

    # Bin statistics
    bins = np.arange(0, args.max_distance + args.bin_size, args.bin_size)
    bin_labels = [f'{int(b)}-{int(b+args.bin_size)}' for b in bins[:-1]]
    bin_means, bin_stds, bin_counts = [], [], []
    for lo, hi in zip(bins[:-1], bins[1:]):
        mask = (dists >= lo) & (dists < hi)
        if mask.sum() > 0:
            bin_means.append(errors[mask].mean())
            bin_stds.append(errors[mask].std())
            bin_counts.append(mask.sum())
        else:
            bin_means.append(0.)
            bin_stds.append(0.)
            bin_counts.append(0)

    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(f'{args.title} — Position Error vs. Distance to Ego', fontsize=13)

    # Scatter
    ax = axes[0]
    ax.scatter(dists, errors, s=4, alpha=0.3, color='steelblue')
    ax.set_xlabel('GT distance to ego (m)')
    ax.set_ylabel('XY position error (m)')
    ax.set_xlim(0, args.max_distance)
    ax.set_ylim(0, args.match_threshold + 0.1)
    ax.set_title('Scatter (all TP matches)')
    ax.grid(True, alpha=0.3)

    # Bar chart (mean ± std per bin)
    ax = axes[1]
    x = np.arange(len(bin_labels))
    bars = ax.bar(x, bin_means, yerr=bin_stds, capsize=4,
                  color='steelblue', alpha=0.8, error_kw={'elinewidth': 1.2})
    for i, (count, mean) in enumerate(zip(bin_counts, bin_means)):
        if count > 0:
            ax.text(i, mean + bin_stds[i] + 0.03, f'n={count}',
                    ha='center', va='bottom', fontsize=7)
    ax.set_xticks(x)
    ax.set_xticklabels(bin_labels, rotation=30, ha='right')
    ax.set_xlabel('Distance to ego (m)')
    ax.set_ylabel('Mean XY error ± std (m)')
    ax.set_title(f'Mean error per {int(args.bin_size)} m bin')
    ax.grid(True, axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig(args.out, dpi=150)
    print(f'Saved → {args.out}')
    plt.show()


if __name__ == '__main__':
    main()
