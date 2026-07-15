"""
radar_coverage_oracle.py — Radar coverage ceiling for a radar-primary detector.

For every valid GT box on a split (within its class eval range), count how many
accumulated radar returns fall inside its BEV footprint (+margin). The per-class
fraction of boxes with >=1 return is the RECALL CEILING of any detector whose
proposals/pillars come from radar — the load-bearing number behind the
radar-primary early-fusion design (see the `rpp` detector).

Also reports: coverage by range band, coverage of moving objects (|v|>1 m/s),
and |median in-box radar range − GT range| (radar as a depth oracle; inflated
for moving objects by multi-sweep trails — the cache has no per-point dt).

Two point sources:
  --source cache   read data/<...>/radar_bev/<token>.npy (the strict 6-sweep,
                   min_rcs=0, dyn-prop-filtered cache from precompute_radar.py)
  --source devkit  rebuild clouds from raw sweeps with arbitrary filters
                   (--nsweeps / --min-rcs / --keep-all-dyn-props) to measure
                   how much ceiling the preprocessing filters cost.

  strict cache (6 sweeps, rcs>=0):        car 63.4%  ped 33.9%  barrier 40.5%
  devkit relaxed (7 sweeps, no filters):  car 73.4%  ped 48.0%  barrier 65.9%
                                          (300-sample subset; ~1480 pts/sample)

Usage:
  .venv/bin/python scripts/radar/radar_coverage_oracle.py --source cache
  .venv/bin/python scripts/radar/radar_coverage_oracle.py --source devkit \
      --nsweeps 7 --min-rcs -1e9 --keep-all-dyn-props --subset 300
"""
from __future__ import annotations

import argparse
import json
import pickle
import random
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_ROOT))

CLS = ['car', 'truck', 'trailer', 'bus', 'construction_vehicle',
       'bicycle', 'motorcycle', 'pedestrian', 'traffic_cone', 'barrier']
# Official nuScenes detection eval ranges per class (m).
EVAL_RANGE = {'car': 50, 'truck': 50, 'trailer': 50, 'bus': 50,
              'construction_vehicle': 50, 'bicycle': 40, 'motorcycle': 40,
              'pedestrian': 40, 'traffic_cone': 30, 'barrier': 30}
BANDS = [(0, 20), (20, 35), (35, 55)]


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dataroot', default='data/nuscenes')
    p.add_argument('--version', default='v1.0-trainval')
    p.add_argument('--ann-file', default='data/nuscenes/nuscenes_infos_val.pkl')
    p.add_argument('--source', choices=['cache', 'devkit'], default='cache')
    p.add_argument('--cache-dir', default='data/nuscenes/radar_bev')
    p.add_argument('--nsweeps', type=int, default=6, help='devkit source only')
    p.add_argument('--min-rcs', type=float, default=0.0, help='devkit source only')
    p.add_argument('--keep-all-dyn-props', action='store_true',
                   help='devkit source only: keep dyn_prop 0..7 (default drops 3,4)')
    p.add_argument('--margin', type=float, default=0.5,
                   help='BEV footprint expansion in m (strict association)')
    p.add_argument('--loose-margin', type=float, default=1.5,
                   help='second, loose footprint expansion in m')
    p.add_argument('--subset', type=int, default=0,
                   help='evaluate on a random subset of N samples (0 = all)')
    p.add_argument('--out', default='',
                   help='optional JSON output path')
    return p.parse_args()


def points_in_footprint(px, py, c_ego, yaw, l, w, margin):
    dx, dy = px - c_ego[0], py - c_ego[1]
    co, si = np.cos(-yaw), np.sin(-yaw)
    u, v = co * dx - si * dy, si * dx + co * dy
    return (np.abs(u) <= l / 2 + margin) & (np.abs(v) <= w / 2 + margin)


def main():
    args = parse_args()
    with open(args.ann_file, 'rb') as f:
        infos = pickle.load(f)['data_list']
    if args.subset:
        random.seed(0)
        infos = random.sample(infos, args.subset)

    get_cloud = None
    if args.source == 'devkit':
        from nuscenes.nuscenes import NuScenes
        from scripts.radar.radar_preprocess import get_radar_pointcloud
        print(f'Loading nuScenes {args.version} devkit...', file=sys.stderr)
        nusc = NuScenes(version=args.version, dataroot=args.dataroot, verbose=False)
        dyn = tuple(range(8)) if args.keep_all_dyn_props else (0, 1, 2, 5, 6, 7)

        def get_cloud(token):
            return get_radar_pointcloud(nusc, token, nsweeps=args.nsweeps,
                                        min_rcs=args.min_rcs, valid_dyn_props=dyn)
    else:
        def get_cloud(token):
            return np.load(Path(args.cache_dir) / f'{token}.npy')

    # (cls, band) -> [n_gt, n_ge1, n_ge3, n_ge1_loose]
    stats = defaultdict(lambda: np.zeros(4))
    rng_err = defaultdict(list)
    vel_cov = defaultdict(lambda: np.zeros(2))  # moving GT: [n, n_with_pt]
    n_pts = []

    for k, d in enumerate(infos):
        pts = get_cloud(d['token'])  # (6, M) ego frame
        n_pts.append(pts.shape[1])
        l2e = np.array(d['lidar_points']['lidar2ego'])
        R, t = l2e[:3, :3], l2e[:3, 3]
        yaw_r = np.arctan2(R[1, 0], R[0, 0])
        px, py = (pts[0], pts[1]) if pts.shape[1] else (np.zeros(0), np.zeros(0))

        for inst in d['instances']:
            lab = inst['bbox_label_3d']
            if lab < 0 or lab >= 10 or not inst.get('bbox_3d_isvalid', True):
                continue
            cls = CLS[lab]
            b = np.array(inst['bbox_3d'])  # [x,y,z,l,w,h,yaw] lidar frame
            c_ego = R @ b[:3] + t
            r_gt = np.hypot(c_ego[0], c_ego[1])
            if r_gt > EVAL_RANGE[cls]:
                continue
            band = next((i for i, (a, bb) in enumerate(BANDS) if a <= r_gt < bb), None)
            if band is None:
                continue
            yaw = b[6] + yaw_r
            inb = points_in_footprint(px, py, c_ego, yaw, b[3], b[4], args.margin)
            inl = points_in_footprint(px, py, c_ego, yaw, b[3], b[4], args.loose_margin)
            n = int(inb.sum())
            s = stats[(cls, band)]
            s[0] += 1
            s[1] += n >= 1
            s[2] += n >= 3
            s[3] += inl.sum() >= 1
            if n >= 1:
                rng_err[cls].append(abs(np.median(np.hypot(px[inb], py[inb])) - r_gt))
            vel = np.array(inst['velocity'][:2])
            if np.isfinite(vel).all() and np.hypot(*vel) > 1.0:
                vc = vel_cov[cls]
                vc[0] += 1
                vc[1] += n >= 1
        if (k + 1) % 1500 == 0:
            print(f'  {k + 1}/{len(infos)}', file=sys.stderr)

    src = (f'{args.source} nsweeps={args.nsweeps} min_rcs={args.min_rcs} '
           f'all_dyn={args.keep_all_dyn_props}' if args.source == 'devkit'
           else f'cache {args.cache_dir}')
    print(f'\nsource: {src} | samples: {len(infos)} | '
          f'pts/sample p50={np.median(n_pts):.0f} p95={np.percentile(n_pts, 95):.0f}')
    print(f"{'class':<22}{'n_GT':>7} | >=1pt: 0-20m 20-35m 35-55m   all | "
          f">=3pt | loose | moving>=1pt | rng_err")
    out = {'source': src, 'margin': args.margin, 'per_class': {}}
    for cls in CLS:
        row = [stats[(cls, i)] for i in range(3)]
        tot = np.sum(row, axis=0)
        if tot[0] == 0:
            continue
        pb = [(r[1] / r[0] * 100 if r[0] else float('nan')) for r in row]
        e = float(np.median(rng_err[cls])) if rng_err[cls] else float('nan')
        vc = vel_cov[cls]
        mv = vc[1] / vc[0] * 100 if vc[0] else float('nan')
        print(f'{cls:<22}{int(tot[0]):>7} |       {pb[0]:5.1f}  {pb[1]:5.1f}  '
              f'{pb[2]:5.1f} {tot[1] / tot[0] * 100:5.1f} | {tot[2] / tot[0] * 100:5.1f} | '
              f'{tot[3] / tot[0] * 100:5.1f} |       {mv:5.1f} | {e:.2f} m')
        out['per_class'][cls] = dict(
            n=int(tot[0]), ge1=tot[1] / tot[0], ge3=tot[2] / tot[0],
            ge1_loose=tot[3] / tot[0], rng_err_med=e,
            ge1_by_band=[r[1] / r[0] if r[0] else None for r in row],
            moving_ge1=(vc[1] / vc[0]) if vc[0] else None)

    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        with open(args.out, 'w') as f:
            json.dump(out, f, indent=1)
        print(f'\nwritten: {args.out}')


if __name__ == '__main__':
    main()
