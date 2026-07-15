"""
shuffle_bev_fusion.py — radar load-bearing test for BEV-fusion checkpoints.

Evaluates one checkpoint on full val under:
  true           unmodified radar
  radar_shuffle  each sample rasterises ANOTHER sample's radar cloud
                 (symlink-permutation dir, derangement-enforced)

shuffle-ΔNDS = NDS(true) − NDS(radar_shuffle). Works for any config whose
val dataset is a NuScenesRadarDataset.

Usage:
    .venv/bin/python scripts/ablation/shuffle_bev_fusion.py \
        --config configs/bev_fusion_full_v3b_framefix.py \
        --checkpoint work_dirs/bev_fusion_full_v3b_framefix/epoch_24.pth
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_ROOT))

from scripts.ablation.shuffle_rpp import build_shuffled_dirs  # noqa: E402

MODES = ('true', 'radar_shuffle')


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config', required=True)
    p.add_argument('--checkpoint', required=True)
    p.add_argument('--modes', nargs='+', default=list(MODES),
                   choices=list(MODES))
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--out', default=None,
                   help='output root (default: <ckpt dir>/ablations)')
    return p.parse_args()


def run_mode(cfg_path: str, checkpoint: str, mode: str, seed: int,
             out_root: Path) -> dict:
    from mmengine.config import Config
    from mmengine.runner import Runner

    cfg = Config.fromfile(cfg_path)
    cfg.load_from = checkpoint
    cfg.work_dir = str(out_root / mode)

    ds = cfg.test_dataloader.dataset
    if mode == 'radar_shuffle':
        (shuf_dir,) = build_shuffled_dirs(
            Path(ds.data_root), ds.ann_file, ds.radar_bev_dir,
            None, seed)
        ds.radar_bev_dir = shuf_dir

    runner = Runner.from_cfg(cfg)
    metrics = runner.test()
    return {k.split('/')[-1]: v for k, v in metrics.items()}


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
    if 'true' in results and 'radar_shuffle' in results:
        d = results['true'].get('NDS', 0) - results['radar_shuffle'].get('NDS', 0)
        print(f'\nshuffle-ΔNDS = {d:+.4f}   '
              f'(no-fix references: v3e −0.0003, v3f −0.0004, v3c 0.000)')

    with open(out_root / 'summary.json', 'w') as f:
        json.dump(results, f, indent=1)
    print(f'written: {out_root}/summary.json')


if __name__ == '__main__':
    main()
