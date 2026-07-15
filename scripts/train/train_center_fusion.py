"""
train_center_fusion.py — train the CenterFusion uncertainty gate, evaluate on val.

Stage 2 of the CenterFusion contribution (Stage 1 = rule-based fixed gate in
``scripts/fusion/center_fusion.py``). A small MLP (``models/radar_refinement.py``)
learns per-detection depth / velocity gates that decide how much radar to
trust. FCOS3D stays frozen — only the MLP is trainable.

The MLP trains on the train split (GT-supervised) and evaluates on val, so
the reported NDS is leakage-free and directly comparable to the camera-only
baseline (falls back to training on val — a biased eval — if
``train_cam_json`` is absent from the config; useful for smoke-testing on
mini). Each radar-associated detection is matched to a same-class GT box
(strict azimuth, lenient range) to get a target range/velocity, and the
gate is trained implicitly by minimising the refined range/velocity error
against GT, weighted by detection score.

Usage
-----
    python scripts/train/train_center_fusion.py --config configs/center_fusion_mini.py
    python scripts/train/train_center_fusion.py --config configs/center_fusion_full.py
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import pickle
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from pyquaternion import Quaternion

_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_ROOT))

from scripts.fusion.center_fusion import (
    FrustumConfig,
    FEATURE_DIM,
    extract_sample_records,
    split_sample_tokens,
    apply_refinement,
    infer_records_fixed,
    evaluate,
    load_camera_detections,
)


def _load_module(path: str, name: str):
    """Import a .py file directly (bypasses package __init__ side effects)."""
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# Load the model by file path so we never trigger models/__init__.py (which
# pulls in the mmdet3d bev_fusion registry — irrelevant here).
_rr = _load_module(str(_ROOT / 'models' / 'radar_refinement.py'), 'radar_refinement')
RadarRefinementMLP = _rr.RadarRefinementMLP
assert _rr.FEATURE_DIM == FEATURE_DIM, (_rr.FEATURE_DIM, FEATURE_DIM)


# ---------------------------------------------------------------------------
# Records cache (the expensive radar association — built once, reused)
# ---------------------------------------------------------------------------
def build_or_load_records(nusc, config: dict, rebuild: bool = False,
                          with_gt: bool = True) -> list[dict]:
    cache = config.get('cache')
    if cache and os.path.exists(cache) and not rebuild:
        print(f'Loading cached records → {cache}')
        with open(cache, 'rb') as f:
            return pickle.load(f)

    cfg = FrustumConfig(**config['frustum'])
    cam_results = load_camera_detections(config['cam_json'])
    tokens = split_sample_tokens(nusc, config['split'])
    print(f'Building records for {len(tokens)} samples ({config["split"]})...')

    t0 = time.time()
    records = []
    for i, tok in enumerate(tokens):
        rec = extract_sample_records(
            nusc, tok, cam_results.get(tok, []), cfg, with_gt=with_gt)
        records.append(rec)
        if (i + 1) % 200 == 0:
            dt = time.time() - t0
            print(f'  {i+1}/{len(tokens)}  ({dt:.0f}s, {(i+1)/dt:.1f} samp/s)')

    if cache:
        os.makedirs(os.path.dirname(cache), exist_ok=True)
        with open(cache, 'wb') as f:
            pickle.dump(records, f)
        print(f'Cached records → {cache}')
    return records


# ---------------------------------------------------------------------------
# Training-array assembly
# ---------------------------------------------------------------------------
def assemble_arrays(records: list[dict]) -> dict:
    """Flatten radar-associated, GT-matched detections into training arrays."""
    feat, cam_range, radar_range = [], [], []
    cam_vel, radar_vel, tgt_range, tgt_vel, weight, scene = \
        [], [], [], [], [], []
    for rec in records:
        for d in rec['dets']:
            if not d.get('has_radar') or not d.get('has_target'):
                continue
            feat.append(d['feat'])
            cam_range.append(d['cam_range'])
            radar_range.append(d['radar_range'])
            cam_vel.append(d['cam_vel_ego'])
            radar_vel.append(d['radar_vel_ego'])
            tgt_range.append(d['target_range'])
            tgt_vel.append(d['target_vel_ego'])
            weight.append(float(np.clip(d['score'], 0.05, 1.0)))
            scene.append(rec['scene_name'])
    return {
        'feat': np.asarray(feat, np.float32),
        'cam_range': np.asarray(cam_range, np.float32),
        'radar_range': np.asarray(radar_range, np.float32),
        'cam_vel': np.asarray(cam_vel, np.float32),
        'radar_vel': np.asarray(radar_vel, np.float32),
        'target_range': np.asarray(tgt_range, np.float32),
        'target_vel': np.asarray(tgt_vel, np.float32),
        'weight': np.asarray(weight, np.float32),
        'scene': np.asarray(scene, dtype=object),
    }


def train_model(arrays: dict, train_mask: np.ndarray, config: dict,
                device: str) -> RadarRefinementMLP:
    """Full-batch train the gate MLP on the selected rows."""
    def t(key):
        return torch.as_tensor(arrays[key][train_mask], device=device).float()

    feat = t('feat')
    cam_range, radar_range = t('cam_range'), t('radar_range')
    cam_vel, radar_vel = t('cam_vel'), t('radar_vel')
    tgt_range, tgt_vel = t('target_range'), t('target_vel')
    w = t('weight')
    wsum = w.sum().clamp_min(1.0)

    model = RadarRefinementMLP(
        input_dim=FEATURE_DIM, hidden_dim=config['hidden_dim'],
        use_residual=config['use_residual'], gate_bias=config['gate_bias'],
    ).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=config['lr'],
                           weight_decay=config['weight_decay'])
    lam = config['lambda_vel']

    model.train()
    for _ in range(config['epochs']):
        opt.zero_grad()
        out = model(feat, cam_range, radar_range, cam_vel, radar_vel)
        rl = F.huber_loss(out.refined_range, tgt_range, delta=2.0, reduction='none')
        vl = F.huber_loss(out.refined_vel, tgt_vel, delta=1.0,
                          reduction='none').mean(dim=1)
        loss = (w * (rl + lam * vl)).sum() / wsum
        loss.backward()
        opt.step()
    return model


# ---------------------------------------------------------------------------
# Inference: refine a set of records with a trained model
# ---------------------------------------------------------------------------
@torch.no_grad()
def infer_records(records: list[dict], model: RadarRefinementMLP,
                  device: str) -> tuple[dict, np.ndarray]:
    """Refine every radar-associated detection; keep the rest as camera.

    Returns (results dict keyed by sample_token, diagnostics array). The
    diagnostics columns are [cam_range, depth_var, score, g_depth, g_vel].
    """
    model.eval()

    feats, cam_r, rad_r, cam_v, rad_v, locs = [], [], [], [], [], []
    for ri, rec in enumerate(records):
        for dj, d in enumerate(rec['dets']):
            if d.get('has_radar'):
                feats.append(d['feat'])
                cam_r.append(d['cam_range'])
                rad_r.append(d['radar_range'])
                cam_v.append(d['cam_vel_ego'])
                rad_v.append(d['radar_vel_ego'])
                locs.append((ri, dj))

    refined_range = refined_vel = g_depth = g_vel = None
    if feats:
        feat = torch.as_tensor(np.asarray(feats, np.float32), device=device)
        out = model(
            feat,
            torch.as_tensor(np.asarray(cam_r, np.float32), device=device),
            torch.as_tensor(np.asarray(rad_r, np.float32), device=device),
            torch.as_tensor(np.asarray(cam_v, np.float32), device=device),
            torch.as_tensor(np.asarray(rad_v, np.float32), device=device),
        )
        refined_range = out.refined_range.cpu().numpy()
        refined_vel = out.refined_vel.cpu().numpy()
        g_depth = out.gate_depth.cpu().numpy()
        g_vel = out.gate_vel.cpu().numpy()

    loc_to_k = {loc: k for k, loc in enumerate(locs)}

    results, diag = {}, []
    for ri, rec in enumerate(records):
        ego_t = rec['ego_t']
        ego_r = Quaternion(rec['ego_q'])
        out_dets = []
        for dj, d in enumerate(rec['dets']):
            if d.get('has_radar'):
                k = loc_to_k[(ri, dj)]
                out_dets.append(apply_refinement(
                    d['det'], d['cam_center_ego'],
                    float(refined_range[k]), refined_vel[k], ego_t, ego_r))
                diag.append((d['cam_range'], float(d['feat'][2]),
                             float(d['feat'][0]), float(g_depth[k]),
                             float(g_vel[k])))
            else:
                out_dets.append(dict(d['det']))
        results[rec['sample_token']] = out_dets

    return results, np.asarray(diag, np.float32) if diag else np.zeros((0, 5), np.float32)


def report_gate_diagnostics(diag: np.ndarray, label: str = '') -> None:
    """Summarise gate behaviour — supports the 'trust radar where camera is
    weak' thesis claim."""
    if len(diag) == 0:
        print('No radar-associated detections — no gate diagnostics.')
        return
    cam_range, depth_var, score, g_depth, g_vel = (diag[:, i] for i in range(5))

    def corr(a, b):
        if a.std() < 1e-6 or b.std() < 1e-6:
            return float('nan')
        return float(np.corrcoef(a, b)[0, 1])

    tag = f' [{label}]' if label else ''
    print(f'\n{"="*60}\n  Gate diagnostics{tag} ({len(diag)} refined detections)\n{"="*60}')
    print(f'  mean g_depth: {g_depth.mean():.3f}   mean g_vel: {g_vel.mean():.3f}')
    print(f'  corr(g_depth, range)     = {corr(g_depth, cam_range):+.3f}  '
          '(expect > 0: far → trust radar)')
    print(f'  corr(g_depth, depth_var) = {corr(g_depth, depth_var):+.3f}  '
          '(expect > 0: uncertain → trust radar)')
    print(f'  corr(g_depth, score)     = {corr(g_depth, score):+.3f}  '
          '(expect < 0: confident → keep camera)')
    for lo, hi in [(0, 15), (15, 30), (30, 100)]:
        m = (cam_range >= lo) & (cam_range < hi)
        if m.any():
            print(f'  range [{lo:>2}-{hi:<3}m]  n={int(m.sum()):>6}  '
                  f'g_depth={g_depth[m].mean():.3f}  g_vel={g_vel[m].mean():.3f}')
    print('=' * 60)


# ---------------------------------------------------------------------------
# Config loading
# ---------------------------------------------------------------------------
DEFAULTS = dict(
    frustum=dict(nsweeps=6, radial_frac=0.25, min_radial=2.5,
                 max_radial=15.0, lateral_margin=1.0, min_angle_deg=3.0),
    epochs=300, lr=1e-3, weight_decay=1e-4, hidden_dim=64,
    lambda_vel=1.0, use_residual=False, gate_bias=-0.4, seed=0,
    cache=None, save_model=None,
    # train-split supervision (optional — falls back to val if absent)
    train_cam_json=None, train_split=None, train_cache=None,
)


def load_config(path: str) -> dict:
    mod = _load_module(path, 'cf_config')
    cfg = dict(DEFAULTS)
    cfg.update(mod.CONFIG)
    for req in ('version', 'dataroot', 'split', 'cam_json', 'out'):
        if req not in cfg:
            raise KeyError(f'config missing required key: {req}')
    return cfg


def main():
    p = argparse.ArgumentParser(
        description='Train CenterFusion gate on train split, evaluate on val')
    p.add_argument('--config', required=True)
    p.add_argument('--rebuild-cache', action='store_true')
    p.add_argument('--no-eval', action='store_true')
    p.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    args = p.parse_args()

    os.chdir(_ROOT)
    config = load_config(args.config)
    torch.manual_seed(config['seed'])
    os.makedirs(config['out'], exist_ok=True)
    print(json.dumps({k: v for k, v in config.items() if k != 'frustum'}, indent=2))

    from nuscenes.nuscenes import NuScenes
    print(f'Loading nuScenes {config["version"]} from {config["dataroot"]}...')
    nusc = NuScenes(version=config['version'], dataroot=config['dataroot'], verbose=False)

    # --- Val records (inference target for evaluation) ---
    val_records = build_or_load_records(nusc, config, rebuild=args.rebuild_cache, with_gt=True)

    # --- Training records ---
    train_cam_json = config.get('train_cam_json')
    if train_cam_json:
        train_cfg = dict(config)
        train_cfg.update({
            'cam_json': train_cam_json,
            'split': config['train_split'],
            'cache': config.get('train_cache'),
        })
        print(f'\nBuilding train records ({config["train_split"]})...')
        train_records = build_or_load_records(
            nusc, train_cfg, rebuild=args.rebuild_cache, with_gt=True)
        diag_label = 'val'
    else:
        print('[WARN] No train_cam_json — training on val (val eval will be biased).')
        train_records = val_records
        diag_label = 'val in-sample'

    arrays = assemble_arrays(train_records)
    print(f'Training rows (radar-associated & GT-matched): {len(arrays["weight"])}')

    full_mask = np.ones(len(arrays['weight']), dtype=bool)
    model = train_model(arrays, full_mask, config, args.device)

    # --- Gate diagnostics on val ---
    val_results, val_diag = infer_records(val_records, model, args.device)
    np.save(os.path.join(config['out'], 'gate_diagnostics_val.npy'), val_diag)
    report_gate_diagnostics(val_diag, label=diag_label)

    # --- Save model ---
    if config.get('save_model'):
        os.makedirs(os.path.dirname(config['save_model']), exist_ok=True)
        torch.save({'state_dict': model.state_dict(),
                    'config': {k: config[k] for k in
                               ('hidden_dim', 'use_residual', 'gate_bias')},
                    'feature_dim': FEATURE_DIM}, config['save_model'])
        print(f'Saved model → {config["save_model"]}')

    # --- Evaluation on val ---
    if not args.no_eval:
        meta = {'use_camera': True, 'use_lidar': False, 'use_radar': True,
                'use_map': False, 'use_external': False}

        def _eval(results: dict, label: str) -> dict:
            path = os.path.join(config['out'], f'{label}_results.json')
            with open(path, 'w') as f:
                json.dump({'meta': meta, 'results': results}, f)
            print(f'\n>>> Evaluating: {label}')
            return evaluate(nusc, path, config['split'], config['out'], verbose=False)

        bias = '' if train_cam_json else ' [biased]'
        summaries = {
            'camera_only':  _eval(infer_records_fixed(val_records, 0.0, 0.0), 'camera_only'),
            'vanilla_hard': _eval(infer_records_fixed(val_records, 1.0, 1.0), 'vanilla_hard'),
            f'learned_gate{bias}': _eval(val_results, 'learned_gate'),
        }
        json.dump(summaries,
                  open(os.path.join(config['out'], 'comparison.json'), 'w'), indent=2)

        cols = ('mAP', 'NDS', 'mATE', 'mASE', 'mAOE', 'mAVE', 'mAAE')
        print(f'\n{"="*78}\n  CenterFusion comparison ({config["split"]})\n{"="*78}')
        print(f'  {"system":<24}' + ''.join(f'{c:>8}' for c in cols))
        for name, s in summaries.items():
            print(f'  {name:<24}' + ''.join(f'{s[c]:>8.4f}' for c in cols))
        print('=' * 78)


if __name__ == '__main__':
    main()
