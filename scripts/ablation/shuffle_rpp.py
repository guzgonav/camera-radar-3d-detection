"""
shuffle_rpp.py — dual load-bearing ablations for the rpp detector (G3/G4).

Evaluates one trained checkpoint on full val under four conditions:

  true           unmodified inputs (reference)
  radar_shuffle  each sample gets ANOTHER sample's radar cloud (points and
                 paint permuted together via paired symlink dirs, so the
                 pair stays point-aligned) — the radar-content test. The
                 detector is radar-primary, so this must collapse.
  paint_zero     painted rows zeroed (model.paint_zero=True) — the true
                 radar-only ablation; NDS(true) − NDS(paint_zero) is the
                 camera contribution.
  paint_shuffle  paint columns permuted across points within each sample
                 (LoadRadarPoints.paint_shuffle) — camera content detached
                 from radar geometry with both marginals preserved.

Success criteria: radar_shuffle collapses; ΔNDS(paint) = true − paint_zero
≥ +0.01 at G3 (≥ +0.02 at G4); paint_shuffle degrades vs true.

Usage:
    .venv/bin/python scripts/ablation/shuffle_rpp.py \
        --config configs/rpp_full_gate3.py \
        --checkpoint work_dirs/rpp_full_gate3/epoch_4.pth \
        --modes true radar_shuffle paint_zero paint_shuffle
"""
from __future__ import annotations

import argparse
import json
import os
import pickle
import sys
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_ROOT))

MODES = ('true', 'radar_shuffle', 'paint_zero', 'paint_shuffle')


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config', default='configs/rpp_full_gate3.py')
    p.add_argument('--checkpoint', required=True)
    p.add_argument('--modes', nargs='+', default=list(MODES),
                   choices=list(MODES))
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--out', default=None,
                   help='output root (default: <ckpt dir>/ablations)')
    return p.parse_args()


def build_shuffled_dirs(data_root: Path, ann_file: str, pts_dir: str,
                        paint_dir: str | None, seed: int) -> tuple[str, ...]:
    """Create paired symlink dirs where sample i points at sample perm(i)'s
    radar cloud AND (if given) paint — a consistent, point-aligned pair.
    Derangement-enforced. Also used by scripts/ablation/shuffle_bev_fusion.py with
    ``paint_dir=None``."""
    with open(data_root / ann_file, 'rb') as f:
        tokens = [d['token'] for d in pickle.load(f)['data_list']]
    rng = np.random.RandomState(seed)
    perm = rng.permutation(len(tokens))
    fixed = np.where(perm == np.arange(len(tokens)))[0]
    if len(fixed):  # break fixed points by rolling them one position
        perm[fixed] = np.roll(perm[fixed], 1)

    src_dirs = [pts_dir] + ([paint_dir] if paint_dir else [])
    out_dirs = [data_root / f'{d}_shuffled_seed{seed}' for d in src_dirs]
    if all(d.exists() for d in out_dirs):
        return tuple(d.name for d in out_dirs)
    for src_name, out in zip(src_dirs, out_dirs):
        out.mkdir(exist_ok=True)
        src = (data_root / src_name).resolve()
        for i, tok in enumerate(tokens):
            dst = out / f'{tok}.npy'
            if dst.is_symlink() or dst.exists():
                dst.unlink()
            dst.symlink_to(src / f'{tokens[perm[i]]}.npy')
    print(f'built shuffled symlink dirs ({len(tokens)} samples, '
          f'seed {seed}): {[d.name for d in out_dirs]}')
    return tuple(d.name for d in out_dirs)


def run_mode(base_cfg_path: str, checkpoint: str, mode: str, seed: int,
             out_root: Path) -> dict:
    from mmengine.config import Config
    from mmengine.runner import Runner

    cfg = Config.fromfile(base_cfg_path)
    cfg.load_from = checkpoint
    cfg.work_dir = str(out_root / mode)

    ds = cfg.test_dataloader.dataset
    if mode == 'radar_shuffle':
        pts_dir, paint_dir = build_shuffled_dirs(
            Path(ds.data_root), ds.ann_file,
            ds.radar_bev_dir, ds.radar_paint_dir, seed)
        ds.radar_bev_dir = pts_dir
        ds.radar_paint_dir = paint_dir
    elif mode == 'paint_zero':
        cfg.model.paint_zero = True
    elif mode == 'paint_shuffle':
        for t in ds.pipeline:
            if t['type'] == 'LoadRadarPoints':
                t['paint_shuffle'] = True

    runner = Runner.from_cfg(cfg)
    metrics = runner.test()
    # strip the metric-prefix noise for the summary table
    clean = {k.split('/')[-1]: v for k, v in metrics.items()}
    return clean


def main():
    args = parse_args()
    os.chdir(_ROOT)
    out_root = Path(args.out) if args.out else (
        Path(args.checkpoint).parent / 'ablations')
    out_root.mkdir(parents=True, exist_ok=True)

    results = {}
    for mode in args.modes:
        print(f'\n=== mode: {mode} ===', flush=True)
        results[mode] = run_mode(
            args.config, args.checkpoint, mode, args.seed, out_root)

    keys = ('NDS', 'mAP', 'mATE', 'mASE', 'mAOE', 'mAVE', 'mAAE')
    print(f'\n{"mode":<15}' + ''.join(f'{k:>9}' for k in keys))
    for mode, m in results.items():
        print(f'{mode:<15}' + ''.join(
            f'{m.get(k, float("nan")):>9.4f}' for k in keys))
    if 'true' in results:
        for ref in ('radar_shuffle', 'paint_zero', 'paint_shuffle'):
            if ref in results:
                d = results['true'].get('NDS', 0) - results[ref].get('NDS', 0)
                print(f'ΔNDS true − {ref:<14} = {d:+.4f}')

    with open(out_root / 'summary.json', 'w') as f:
        json.dump(results, f, indent=1)
    print(f'\nwritten: {out_root}/summary.json')


if __name__ == '__main__':
    main()
