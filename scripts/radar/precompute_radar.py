"""
precompute_radar.py — Cache per-sample radar point clouds as .npy

Runs the existing radar preprocessing pipeline (filter + multi-sweep
accumulation + 5-sensor merge → ego frame) once over every sample in a
nuScenes split and writes a small `.npy` per sample. The BEV fusion
training loop then just `np.load`s the file instead of re-doing this work
every iteration.

Output layout
-------------
    <out-dir>/<sample_token>.npy        # shape (6, M), float32
    <out-dir>/_meta.json                # parameters used to generate the cache

The 6 rows are: [x, y, z, vx_comp, vy_comp, rcs] in the ego frame at the
sample timestamp — exactly what `get_radar_pointcloud()` returns.

Usage
-----
    # Mini (cheap smoke test)
    python scripts/radar/precompute_radar.py \\
        --version v1.0-mini --dataroot data/nuscenes-mini \\
        --out data/nuscenes-mini/radar_bev

    # Full (run once; ~35 MB total)
    python scripts/radar/precompute_radar.py \\
        --version v1.0-trainval --dataroot data/nuscenes \\
        --out data/nuscenes/radar_bev

If the file already exists and `--overwrite` is not set, it is skipped —
re-runs are cheap.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_ROOT))

from scripts.radar.radar_preprocess import (
    DEFAULT_MIN_RCS,
    DEFAULT_NSWEEPS,
    DEFAULT_VALID_DYN_PROPS,
    V2_MIN_RCS,
    V2_NSWEEPS,
    V2_VALID_DYN_PROPS,
    get_radar_pointcloud,
    iterate_samples,
)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--version', default='v1.0-mini',
                   help='nuScenes version (v1.0-mini or v1.0-trainval)')
    p.add_argument('--dataroot', default='data/nuscenes-mini',
                   help='Path to nuScenes data root')
    p.add_argument('--out', required=True,
                   help='Output directory for the per-sample .npy files')
    p.add_argument('--layout', choices=['v1', 'v2'], default='v1',
                   help="v2 = (8, M) with dt + dyn_prop rows, relaxed "
                        'filters (nsweeps=7, no rcs/dyn_prop cut) — the '
                        'week-17 radar-primary cache')
    p.add_argument('--nsweeps', type=int, default=None,
                   help='default: 6 for v1, 7 for v2')
    p.add_argument('--min-rcs', type=float, default=None,
                   help='default: 0 for v1, -inf for v2')
    p.add_argument('--shard', default=None, metavar='I/N',
                   help='process only samples with index %% N == I '
                        '(run N processes in parallel, e.g. 0/4 .. 3/4)')
    p.add_argument('--overwrite', action='store_true',
                   help='Re-generate files even if they exist')
    args = p.parse_args()
    if args.nsweeps is None:
        args.nsweeps = V2_NSWEEPS if args.layout == 'v2' else DEFAULT_NSWEEPS
    if args.min_rcs is None:
        args.min_rcs = V2_MIN_RCS if args.layout == 'v2' else DEFAULT_MIN_RCS
    args.valid_dyn_props = (V2_VALID_DYN_PROPS if args.layout == 'v2'
                            else DEFAULT_VALID_DYN_PROPS)
    if args.shard is not None:
        i, n = args.shard.split('/')
        args.shard = (int(i), int(n))
        assert 0 <= args.shard[0] < args.shard[1]
    return args


def main():
    os.chdir(_ROOT)
    args = parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    from nuscenes.nuscenes import NuScenes
    print(f'Loading nuScenes {args.version} from {args.dataroot}...')
    nusc = NuScenes(version=args.version, dataroot=args.dataroot, verbose=False)

    sample_tokens = list(iterate_samples(nusc))
    if args.shard is not None:
        shard_i, shard_n = args.shard
        sample_tokens = sample_tokens[shard_i::shard_n]
        print(f'Shard {shard_i}/{shard_n}: {len(sample_tokens)} samples')
    print(f'Found {len(sample_tokens)} samples — caching radar to {out_dir}')

    n_existing = 0
    n_written = 0
    n_empty = 0
    t0 = time.time()
    for i, token in enumerate(sample_tokens):
        out_path = out_dir / f'{token}.npy'
        if out_path.exists() and not args.overwrite:
            n_existing += 1
            continue

        pts = get_radar_pointcloud(
            nusc, token,
            nsweeps=args.nsweeps,
            min_rcs=args.min_rcs,
            valid_dyn_props=args.valid_dyn_props,
            layout=args.layout,
        )
        np.save(out_path, pts.astype(np.float32))
        n_written += 1
        if pts.shape[1] == 0:
            n_empty += 1

        if (i + 1) % 500 == 0:
            elapsed = time.time() - t0
            rate = (i + 1) / max(elapsed, 1e-6)
            eta = (len(sample_tokens) - (i + 1)) / max(rate, 1e-6)
            print(f'  [{i+1:5d}/{len(sample_tokens)}] '
                  f'{rate:5.1f} samples/s, ETA {eta/60:.1f} min')

    meta = {
        'version': args.version,
        'dataroot': args.dataroot,
        'layout': args.layout,
        'nsweeps': args.nsweeps,
        'min_rcs': args.min_rcs,
        'valid_dyn_props': list(args.valid_dyn_props),
        'n_samples': len(sample_tokens),
    }
    with open(out_dir / '_meta.json', 'w') as f:
        json.dump(meta, f, indent=2)

    print(f'Done. wrote={n_written}, existing={n_existing}, empty_after_filter={n_empty}')
    print(f'Meta written to {out_dir / "_meta.json"}')


if __name__ == '__main__':
    main()
