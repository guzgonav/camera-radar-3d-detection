"""
shuffle_center_fusion.py — radar-shuffle load-bearing test for CenterFusion.

The mirror of the BEV-fusion chapter's shuffle ablation. We permute each val
detection's radar evidence to a random other val detection — breaking the true
geometric association — then re-run inference with the same MLP trained on
real train radar. If the gain comes from real radar content, NDS collapses
back toward camera-only and the gate self-closes (g → 0).

    python scripts/ablation/shuffle_center_fusion.py --config configs/center_fusion_full.py
"""
from __future__ import annotations

import argparse
import copy
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch

_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_ROOT))

from scripts.train.train_center_fusion import (
    build_or_load_records, load_config,
    assemble_arrays, train_model, infer_records,
)
from scripts.fusion.center_fusion import evaluate, RANGE_NORM

META = {'use_camera': True, 'use_lidar': False, 'use_radar': True,
        'use_map': False, 'use_external': False}

# Radar scalar slots in the feature vector (see center_fusion.py / radar_refinement.py).
RAD_IDX = list(range(7, 14))


def shuffle_radar(records: list[dict], seed: int) -> list[dict]:
    """Deep-copy records and permute radar evidence across all associated dets."""
    rng = np.random.default_rng(seed)
    locs, radar_range, radar_vel, feat_radar = [], [], [], []
    for ri, rec in enumerate(records):
        for dj, d in enumerate(rec['dets']):
            if d.get('has_radar'):
                locs.append((ri, dj))
                radar_range.append(d['radar_range'])
                radar_vel.append(d['radar_vel_ego'])
                feat_radar.append(d['feat'][RAD_IDX].copy())
    perm = rng.permutation(len(locs))

    shuf = copy.deepcopy(records)
    for k, (ri, dj) in enumerate(locs):
        src = perm[k]
        d = shuf[ri]['dets'][dj]
        d['radar_range'] = float(radar_range[src])
        d['radar_vel_ego'] = radar_vel[src].copy()
        f = d['feat'].copy()
        f[RAD_IDX] = feat_radar[src]
        # Recompute range residual against this detection's real camera range so
        # the mismatched radar yields a realistically large residual.
        f[8] = (d['radar_range'] - d['cam_range']) / RANGE_NORM
        d['feat'] = f
    return shuf


def main():
    p = argparse.ArgumentParser(description='CenterFusion radar-shuffle load-bearing test')
    p.add_argument('--config', required=True)
    p.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    args = p.parse_args()

    os.chdir(_ROOT)
    config = load_config(args.config)

    from nuscenes.nuscenes import NuScenes
    nusc = NuScenes(version=config['version'], dataroot=config['dataroot'], verbose=False)

    # Val records (inference target, will be shuffled).
    val_records = build_or_load_records(nusc, config, with_gt=True)

    # Train MLP on real train radar (same as the main pipeline).
    train_cam_json = config.get('train_cam_json')
    if train_cam_json:
        train_cfg = dict(config)
        train_cfg.update({
            'cam_json': train_cam_json,
            'split': config['train_split'],
            'cache': config.get('train_cache'),
        })
        train_records = build_or_load_records(nusc, train_cfg, with_gt=True)
    else:
        print('[WARN] No train_cam_json — training on val records (shuffle test still valid).')
        train_records = val_records

    arrays = assemble_arrays(train_records)
    full_mask = np.ones(len(arrays['weight']), dtype=bool)
    torch.manual_seed(config['seed'])
    model = train_model(arrays, full_mask, config, args.device)

    # Infer on shuffled val records.
    shuf_records = shuffle_radar(val_records, config['seed'])
    results, diag = infer_records(shuf_records, model, args.device)

    out = os.path.join(config['out'], 'shuffle')
    os.makedirs(out, exist_ok=True)
    with open(os.path.join(out, 'results.json'), 'w') as f:
        json.dump({'meta': META, 'results': results}, f)
    s = evaluate(nusc, os.path.join(out, 'results.json'), config['split'], out, verbose=False)

    # Pull camera_only / learned_gate reference rows from comparison.json.
    cmp_path = os.path.join(config['out'], 'comparison.json')
    ref = json.load(open(cmp_path)) if os.path.exists(cmp_path) else {}

    print(f'\n{"="*60}\n  RADAR-SHUFFLE TEST ({config["split"]})\n{"="*60}')
    print(f'{"system":<24}{"mAP":>8}{"NDS":>8}{"mATE":>8}{"mAVE":>8}')
    for label, key in [('camera_only', 'camera_only'),
                       ('learned (true radar)', 'learned_gate')]:
        if key in ref:
            r = ref[key]
            print(f'{label:<24}{r["mAP"]:>8.4f}{r["NDS"]:>8.4f}'
                  f'{r["mATE"]:>8.4f}{r["mAVE"]:>8.4f}')
    print(f'{"learned (shuffled)":<24}{s["mAP"]:>8.4f}{s["NDS"]:>8.4f}'
          f'{s["mATE"]:>8.4f}{s["mAVE"]:>8.4f}')
    delta_nds = s['NDS'] - ref.get('learned_gate', {}).get('NDS', float('nan'))
    print(f'\nΔNDS_shuffle = {delta_nds:+.4f}  (→0 or negative ⇒ gate uses real radar)')
    print(f'gate under shuffle: mean g_depth={diag[:,3].mean():.3f} '
          f'g_vel={diag[:,4].mean():.3f}  (→0 ⇒ gate correctly ignores noise)')
    print('=' * 60)


if __name__ == '__main__':
    main()
