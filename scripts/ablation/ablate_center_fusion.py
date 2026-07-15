"""
ablate_center_fusion.py — hyper-parameter sensitivity of the CenterFusion gate.

Reuses the cached radar-association records (no re-association), so a full
lambda_vel × residual sweep costs only MLP training (tiny) + one NuScenesEval
per setting.

Two protocols are supported via --protocol:
  * trainval (default): train each setting on the train split and evaluate on
    val, exactly as the headline model does — so the ablation numbers are
    directly comparable to the main results table.
  * cv: scene-fold cross-validation within the val split (out-of-fold estimate;
    leakage-free across settings but NOT comparable to the train→val headline).

    python scripts/ablation/ablate_center_fusion.py --config configs/center_fusion_full.py
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_ROOT))

from scripts.train.train_center_fusion import (
    build_or_load_records, load_config,
    assemble_arrays, train_model, infer_records,
)
from scripts.fusion.center_fusion import evaluate

# (lambda_vel, use_residual) settings to sweep.
SETTINGS = [
    ('lam1_res0', dict(lambda_vel=1.0, use_residual=False)),
    ('lam3_res0', dict(lambda_vel=3.0, use_residual=False)),
    ('lam6_res0', dict(lambda_vel=6.0, use_residual=False)),
    ('lam1_res1', dict(lambda_vel=1.0, use_residual=True)),
    ('lam3_res1', dict(lambda_vel=3.0, use_residual=True)),
]

META = {'use_camera': True, 'use_lidar': False, 'use_radar': True,
        'use_map': False, 'use_external': False}


def cross_validate(records: list[dict], config: dict, device: str
                   ) -> tuple[dict, np.ndarray]:
    """k-fold CV over val scenes → out-of-fold results for the whole split."""
    arrays = assemble_arrays(records)
    print(f'Training rows (radar-associated & GT-matched): {len(arrays["weight"])}')

    scenes = sorted({rec['scene_name'] for rec in records})
    rng = np.random.default_rng(config['seed'])
    rng.shuffle(scenes)
    k = min(config['k_folds'], len(scenes))
    folds = {s: i % k for i, s in enumerate(scenes)}
    print(f'Scene-fold CV: {len(scenes)} scenes, k={k}')

    arr_scene_fold = np.array([folds[s] for s in arrays['scene']])

    results, diag_all = {}, []
    for fi in range(k):
        train_mask = arr_scene_fold != fi
        model = train_model(arrays, train_mask, config, device)
        held = [r for r in records if folds[r['scene_name']] == fi]
        part, diag = infer_records(held, model, device)
        results.update(part)
        diag_all.append(diag)
        gd = diag[:, 3].mean() if len(diag) else float('nan')
        gv = diag[:, 4].mean() if len(diag) else float('nan')
        print(f'  fold {fi}: train_rows={int(train_mask.sum())} '
              f'held_scenes={len(held)} mean g_depth={gd:.3f} g_vel={gv:.3f}')

    return results, np.concatenate(diag_all) if diag_all else np.zeros((0, 5), np.float32)


def train_val_eval(train_records: list[dict], val_records: list[dict],
                   config: dict, device: str) -> tuple[dict, np.ndarray]:
    """Train on the train split, evaluate on val — the same protocol as the
    headline model (train_center_fusion.py). Unlike scene-fold CV within val,
    these numbers are directly comparable to the main results table."""
    arrays = assemble_arrays(train_records)
    print(f'Training rows (radar-associated & GT-matched): {len(arrays["weight"])}')
    full_mask = np.ones(len(arrays['weight']), dtype=bool)
    torch.manual_seed(config['seed'])
    model = train_model(arrays, full_mask, config, device)
    return infer_records(val_records, model, device)


def main():
    p = argparse.ArgumentParser(description='CenterFusion gate hyper-parameter ablation')
    p.add_argument('--config', required=True)
    p.add_argument('--protocol', choices=('trainval', 'cv'), default='trainval',
                   help="'trainval' (train on the train split, evaluate on val — "
                        "directly comparable to the main results table) or 'cv' "
                        "(scene-fold cross-validation within the val split).")
    p.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    args = p.parse_args()

    os.chdir(_ROOT)
    config = load_config(args.config)

    from nuscenes.nuscenes import NuScenes
    nusc = NuScenes(version=config['version'], dataroot=config['dataroot'], verbose=False)
    val_records = build_or_load_records(nusc, config)

    if args.protocol == 'cv':
        if 'k_folds' not in config:
            raise KeyError('cv ablation requires k_folds in config')
        train_records = None
        sub = 'ablation'
    else:
        train_cam_json = config.get('train_cam_json')
        if not train_cam_json:
            raise KeyError('trainval ablation requires train_cam_json in config')
        train_cfg = dict(config)
        train_cfg.update({
            'cam_json': train_cam_json,
            'split': config['train_split'],
            'cache': config.get('train_cache'),
        })
        print(f'\nBuilding train records ({config["train_split"]})...')
        train_records = build_or_load_records(nusc, train_cfg)
        sub = 'ablation_trainval'

    rows = []
    for name, ov in SETTINGS:
        cfg = dict(config)
        cfg.update(ov)
        torch.manual_seed(cfg['seed'])
        t0 = time.time()
        if args.protocol == 'cv':
            results, diag = cross_validate(val_records, cfg, args.device)
        else:
            results, diag = train_val_eval(train_records, val_records, cfg, args.device)
        out = os.path.join(config['out'], sub, name)
        os.makedirs(out, exist_ok=True)
        with open(os.path.join(out, 'results.json'), 'w') as f:
            json.dump({'meta': META, 'results': results}, f)
        s = evaluate(nusc, os.path.join(out, 'results.json'),
                     cfg['split'], out, verbose=False)
        rows.append((name, s, float(diag[:, 3].mean()), float(diag[:, 4].mean())))
        print(f'### {name}: NDS={s["NDS"]:.4f} mAP={s["mAP"]:.4f} '
              f'mATE={s["mATE"]:.4f} mAVE={s["mAVE"]:.4f} '
              f'| g_depth={diag[:,3].mean():.3f} g_vel={diag[:,4].mean():.3f} '
              f'({time.time()-t0:.0f}s)')

    label = 'scene-fold CV' if args.protocol == 'cv' else 'train→val'
    print(f'\n{"="*64}\n  ABLATION SUMMARY ({config["split"]}, {label})\n{"="*64}')
    print(f'{"setting":<12}{"mAP":>8}{"NDS":>8}{"mATE":>8}{"mAVE":>8}{"g_dep":>7}{"g_vel":>7}')
    for name, s, gd, gv in rows:
        print(f'{name:<12}{s["mAP"]:>8.4f}{s["NDS"]:>8.4f}'
              f'{s["mATE"]:>8.4f}{s["mAVE"]:>8.4f}{gd:>7.3f}{gv:>7.3f}')


if __name__ == '__main__':
    main()
