"""
radar_domain.py — Visualise the effective working domain of the radar.

Loads a random sample of cached radar point clouds and plots:
  1. Histogram of radar return density by distance to ego.
  2. 2D BEV heatmap of radar return density (x forward, y left).

Usage:
    python scripts/analysis/radar_domain.py \
        --radar-dir data/nuscenes/radar_bev \
        --n-samples 2000 \
        --max-distance 80 \
        --out results/radar_domain.png
"""

from __future__ import annotations

import argparse
import os
import random

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

# Output field indices from radar_preprocess.py
OUT_X, OUT_Y, OUT_Z, OUT_VX, OUT_VY, OUT_RCS = 0, 1, 2, 3, 4, 5


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--radar-dir', default='data/nuscenes/radar_bev')
    parser.add_argument('--n-samples', type=int, default=2000,
                        help='Number of .npy files to load (random subset)')
    parser.add_argument('--max-distance', type=float, default=80.0)
    parser.add_argument('--seed', type=int, default=0,
                        help='RNG seed for the random file subset (reproducibility)')
    parser.add_argument('--out', default='results/radar_domain.png')
    args = parser.parse_args()

    random.seed(args.seed)
    files = [f for f in os.listdir(args.radar_dir) if f.endswith('.npy')]
    if len(files) > args.n_samples:
        files = random.sample(files, args.n_samples)
    print(f'Loading {len(files)} radar clouds...')

    all_x, all_y, all_dist = [], [], []

    for fname in files:
        pts = np.load(os.path.join(args.radar_dir, fname))  # (6, M)
        if pts.shape[1] == 0:
            continue
        x = pts[OUT_X]
        y = pts[OUT_Y]
        dist = np.sqrt(x**2 + y**2)
        mask = dist <= args.max_distance
        all_x.append(x[mask])
        all_y.append(y[mask])
        all_dist.append(dist[mask])

    all_x    = np.concatenate(all_x)
    all_y    = np.concatenate(all_y)
    all_dist = np.concatenate(all_dist)

    print(f'Total radar returns: {len(all_dist):,}')
    print(f'Range: {all_dist.min():.1f} – {all_dist.max():.1f} m')
    print(f'50th pct: {np.percentile(all_dist, 50):.1f} m  |  '
          f'90th pct: {np.percentile(all_dist, 90):.1f} m  |  '
          f'95th pct: {np.percentile(all_dist, 95):.1f} m')

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(all_dist, bins=80, range=(0, args.max_distance),
            color='steelblue', alpha=0.85, edgecolor='none')
    ax.set_xlabel('Distance to ego (m)')
    ax.set_ylabel('Number of radar returns')
    ax.set_title('Radar return density by distance (nuScenes trainval)')
    ax.grid(True, axis='y', alpha=0.3)

    plt.tight_layout()
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    plt.savefig(args.out, dpi=150)
    print(f'Saved → {args.out}')
    plt.show()


if __name__ == '__main__':
    main()
