"""
center_fusion.py — CenterFusion-style detection-level camera-radar fusion.

FCOS3D is frozen; radar's only job is refining the depth/velocity of existing
camera detections via frustum association, so it can never be gated off by a
camera optimiser. ``--mode rule_based`` hard-replaces with radar evidence
(gate g=1); ``--mode learned`` predicts g via a small uncertainty-gate MLP.

The frustum is built directly in ego coordinates rather than re-projected
onto an image (the FCOS3D JSON merges all 6 cameras and drops the
source-camera id): narrow azimuth window (camera is angularly reliable),
wide radial window (camera depth is not), height ignored (radar z is
unreliable).

Usage
-----
    # mini val
    python scripts/fusion/center_fusion.py --version v1.0-mini --dataroot data/nuscenes-mini \\
        --split mini_val \\
        --cam-json results/fcos3d_baseline/mini/detections/pred_instances_3d/results_nusc.json \\
        --out results/center_fusion/mini

    # full val
    python scripts/fusion/center_fusion.py --version v1.0-trainval --dataroot data/nuscenes \\
        --split val \\
        --cam-json results/fcos3d_baseline/full/detections/pred_instances_3d/results_nusc.json \\
        --out results/center_fusion/full
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from pyquaternion import Quaternion

_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_ROOT))

from scripts.radar.radar_preprocess import (
    get_radar_pointcloud,
    OUT_X, OUT_Y, OUT_VX, OUT_VY, OUT_RCS,
)
# Reuse — do NOT modify late_fusion.py; import its helpers only.
from scripts.fusion.late_fusion import (
    load_camera_detections,
    get_ego_pose,
    global_to_ego,
    ego_to_global,
    DETECTION_CLASSES,
)

# ---------------------------------------------------------------------------
# Feature engineering (shared with models/radar_refinement.py)
# ---------------------------------------------------------------------------
# Normalisation scales — keep features ~O(1) for a stable MLP.
RANGE_NORM = 50.0  # m   (nuScenes eval range ≈ 50 m)
SPEED_NORM = 15.0  # m/s
RCS_NORM = 20.0  # dBsm

# Feature-vector width: 7 camera scalars + 7 radar scalars + 10 class one-hot.
# Mirrored by FEATURE_DIM in models/radar_refinement.py; the trainer asserts
# the two agree. Defined here too so this module stays torch/mmdet3d-free.
FEATURE_DIM = 7 + 7 + len(DETECTION_CLASSES)


def depth_var(cam_range: float, score: float) -> float:
    """Analytic monocular-depth-uncertainty proxy in [0, 1].

    No depth-uncertainty field exists in the FCOS3D JSON, so we derive one.
    Two physical priors, blended:
      * monocular depth error grows with range → variance ∝ range²
      * a low detection score signals an unreliable box overall
    The gate MLP also sees raw range and score, so this is a convenience prior
    that makes the "gate conditioned on depth_var" narrative literal; the MLP
    is free to recombine the raw signals.
    """
    r_term = float(np.clip(cam_range / RANGE_NORM, 0.0, 1.0)) ** 2
    s_term = float(np.clip(1.0 - score, 0.0, 1.0))
    return float(np.clip(0.5 * r_term + 0.5 * s_term, 0.0, 1.0))


def build_feature_row(det: dict, cam_center_ego: np.ndarray, assoc: dict) -> np.ndarray:
    """Assemble the per-detection feature vector (frame-independent scalars).

    Order MUST match CAM_SCALAR_FEATURES + RADAR_SCALAR_FEATURES + class
    one-hot in models/radar_refinement.py. ``assoc`` must be non-None (the
    model only ever sees radar-associated detections).
    """
    cx, cy = float(cam_center_ego[0]), float(cam_center_ego[1])
    cam_range = float(np.hypot(cx, cy))
    score = float(np.clip(det['detection_score'], 0.0, 1.0))
    w, l, h = (float(s) for s in det['size'])
    speed = float(np.hypot(det['velocity'][0], det['velocity'][1]))

    cam_feats = [
        score,
        cam_range / RANGE_NORM,
        depth_var(cam_range, score),
        np.log(max(w, 1e-3)),
        np.log(max(l, 1e-3)),
        np.log(max(h, 1e-3)),
        speed / SPEED_NORM,
    ]

    radar_range = float(assoc['radar_range'])
    radar_feats = [
        radar_range / RANGE_NORM,
        (radar_range - cam_range) / RANGE_NORM,
        float(assoc['radar_vx']) / SPEED_NORM,
        float(assoc['radar_vy']) / SPEED_NORM,
        float(assoc['mean_rcs']) / RCS_NORM,
        float(assoc['range_std']) / RANGE_NORM,
        float(np.log1p(assoc['n_pts'])),
    ]

    onehot = [0.0] * len(DETECTION_CLASSES)
    name = det['detection_name']
    if name in DETECTION_CLASSES:
        onehot[DETECTION_CLASSES.index(name)] = 1.0

    row = np.asarray(cam_feats + radar_feats + onehot, dtype=np.float32)
    assert row.shape[0] == FEATURE_DIM, (row.shape[0], FEATURE_DIM)
    return row


# ---------------------------------------------------------------------------
# Frustum association configuration
# ---------------------------------------------------------------------------
@dataclass
class FrustumConfig:
    """Geometry of the per-detection radar association frustum.

    Args:
        nsweeps:        Radar sweeps to accumulate (denser cloud).
        radial_frac:    Half-depth of the radial window as a fraction of the
                        camera range. Monocular depth error grows with range,
                        so the window widens with range.
        min_radial:     Floor on the radial half-window (m), for close objects.
        lateral_margin: Extra half-width (m) added to the box footprint before
                        converting to an angular window — absorbs azimuth error.
        min_angle_deg:  Floor on the azimuth half-window (deg), so tiny / very
                        distant boxes still subtend a usable angular slice.
        max_radial:     Cap on the radial half-window (m), to stop a far box
                        from swallowing the whole scene.
    """

    nsweeps: int = 6
    radial_frac: float = 0.25
    min_radial: float = 2.5
    max_radial: float = 15.0
    lateral_margin: float = 1.0
    min_angle_deg: float = 3.0


# ---------------------------------------------------------------------------
# Radar polar precompute + frustum association
# ---------------------------------------------------------------------------
def _wrap_angle(a: np.ndarray | float) -> np.ndarray | float:
    """Wrap angle(s) to [-π, π]."""
    return (a + np.pi) % (2.0 * np.pi) - np.pi


def radar_polar(pts: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Per-point BEV range and azimuth for a (6, M) ego-frame cloud.

    Returns:
        (range, azimuth) each shape (M,). range = sqrt(x²+y²), azimuth in rad.
    """
    px, py = pts[OUT_X], pts[OUT_Y]
    rng = np.sqrt(px ** 2 + py ** 2)
    az = np.arctan2(py, px)
    return rng, az


def associate_frustum(
        cam_center_ego: np.ndarray,  # (3,) [x, y, z] ego frame
        size: list[float],  # [w, l, h]
        pts: np.ndarray,  # (6, M) ego frame
        rng: np.ndarray,  # (M,) precomputed point ranges
        az: np.ndarray,  # (M,) precomputed point azimuths
        cfg: FrustumConfig,
) -> dict | None:
    """Associate radar points falling inside a detection's frustum.

    Returns aggregated radar features, or ``None`` if no point is inside.
    """
    cx, cy = float(cam_center_ego[0]), float(cam_center_ego[1])
    r_cam = float(np.hypot(cx, cy))
    if r_cam < 1e-3:
        return None  # detection essentially at the ego origin — ill-defined ray

    theta_cam = float(np.arctan2(cy, cx))

    w, l = float(size[0]), float(size[1])
    half_extent = 0.5 * max(w, l) + cfg.lateral_margin
    dtheta = max(np.arctan2(half_extent, r_cam),
                 np.deg2rad(cfg.min_angle_deg))
    dr = float(np.clip(cfg.radial_frac * r_cam, cfg.min_radial, cfg.max_radial))

    ang_diff = np.abs(_wrap_angle(az - theta_cam))
    mask = (ang_diff <= dtheta) & (rng >= r_cam - dr) & (rng <= r_cam + dr)

    n = int(mask.sum())
    if n == 0:
        return None

    rr = rng[mask]
    return {
        'n_pts': n,
        'radar_range': float(np.median(rr)),  # robust depth estimate
        'range_std': float(np.std(rr)),
        'radar_vx': float(pts[OUT_VX, mask].mean()),
        'radar_vy': float(pts[OUT_VY, mask].mean()),
        'mean_rcs': float(pts[OUT_RCS, mask].mean()),
    }


# ---------------------------------------------------------------------------
# Refinement (gated formula; gate=1.0 → vanilla hard replacement)
# ---------------------------------------------------------------------------
def apply_refinement(
        det: dict,
        cam_center_ego: np.ndarray,
        refined_range: float,
        refined_vel_ego: np.ndarray,
        ego_t: np.ndarray,
        ego_r: Quaternion,
) -> dict:
    """Write a refined range + ego velocity back into a detection dict.

    Model-agnostic geometry shared by the rule-based and learned paths. Depth
    is corrected along the camera ray (azimuth preserved): we rescale the ego
    (x, y) so the BEV range matches ``refined_range`` while keeping the camera's
    reliable bearing. z is left untouched (radar z is unreliable).
    """
    cx, cy, cz = float(cam_center_ego[0]), float(cam_center_ego[1]), float(cam_center_ego[2])
    r_cam = float(np.hypot(cx, cy))

    scale = refined_range / r_cam if r_cam > 1e-3 else 1.0
    fused_center_ego = np.array([cx * scale, cy * scale, cz])
    fused_center_global = ego_to_global(fused_center_ego, ego_t, ego_r)

    fused_vel_ego = np.array([refined_vel_ego[0], refined_vel_ego[1], 0.0])
    fused_vel_global = ego_r.rotate(fused_vel_ego)

    out = dict(det)
    out['translation'] = fused_center_global.tolist()
    out['velocity'] = [float(fused_vel_global[0]), float(fused_vel_global[1])]
    return out


def refine_detection(
        det: dict,
        cam_center_ego: np.ndarray,
        assoc: dict,
        ego_t: np.ndarray,
        ego_r: Quaternion,
        gate: float = 1.0,
) -> dict:
    """Rule-based refinement of one detection with a fixed gate.

        refined_range = cam_range + g·(radar_range − cam_range)
        refined_vel   = cam_vel   + g·(radar_vel   − cam_vel)

    With g=1.0 this is vanilla hard replacement.
    """
    cx, cy = float(cam_center_ego[0]), float(cam_center_ego[1])
    r_cam = float(np.hypot(cx, cy))

    refined_range = r_cam + gate * (assoc['radar_range'] - r_cam)

    cam_vel_global = np.array(det['velocity'] + [0.0])
    cam_vel_ego = ego_r.inverse.rotate(cam_vel_global)
    radar_vel_ego = np.array([assoc['radar_vx'], assoc['radar_vy'], 0.0])
    refined_vel_ego = (cam_vel_ego + gate * (radar_vel_ego - cam_vel_ego))[:2]

    return apply_refinement(det, cam_center_ego, refined_range, refined_vel_ego,
                            ego_t, ego_r)


# ---------------------------------------------------------------------------
# Per-sample fusion
# ---------------------------------------------------------------------------
def fuse_sample(
        nusc,
        sample_token: str,
        cam_detections: list[dict],
        cfg: FrustumConfig,
        gate: float = 1.0,
) -> tuple[list[dict], dict]:
    """Frustum-associate and refine all detections in one sample.

    Detections with no associated radar point are kept as the raw camera
    detection (radar has nothing to say about them).
    """
    stats = {'n_cam': len(cam_detections), 'n_radar_pts': 0, 'n_assoc': 0}
    if not cam_detections:
        return [], stats

    pts = get_radar_pointcloud(nusc, sample_token, nsweeps=cfg.nsweeps)
    stats['n_radar_pts'] = int(pts.shape[1])
    if pts.shape[1] == 0:
        return list(cam_detections), stats

    ego_t, ego_r = get_ego_pose(nusc, sample_token)
    rng, az = radar_polar(pts)

    fused: list[dict] = []
    for det in cam_detections:
        center_ego = global_to_ego(np.array(det['translation']), ego_t, ego_r)
        assoc = associate_frustum(center_ego, det['size'], pts, rng, az, cfg)
        if assoc is None:
            fused.append(dict(det))
        else:
            stats['n_assoc'] += 1
            fused.append(refine_detection(det, center_ego, assoc, ego_t, ego_r, gate))

    return fused, stats


# ---------------------------------------------------------------------------
# Record extraction + GT supervision (drives the learned-gate trainer)
# ---------------------------------------------------------------------------
def scene_name_of(nusc, sample_token: str) -> str:
    """Scene name for a sample (stored in records for diagnostics and ablations)."""
    sample = nusc.get('sample', sample_token)
    return nusc.get('scene', sample['scene_token'])['name']


def get_gt_boxes_ego(nusc, sample_token: str, ego_t: np.ndarray,
                     ego_r: Quaternion) -> list[dict]:
    """Ground-truth boxes for a sample, in the ego frame, mapped to the 10
    nuScenes detection classes. Velocity comes from ``nusc.box_velocity``
    (global, may be NaN at sequence ends → zeroed)."""
    from nuscenes.eval.detection.utils import category_to_detection_name

    sample = nusc.get('sample', sample_token)
    boxes = []
    for ann_token in sample['anns']:
        ann = nusc.get('sample_annotation', ann_token)
        name = category_to_detection_name(ann['category_name'])
        if name is None:
            continue
        center_ego = global_to_ego(np.array(ann['translation']), ego_t, ego_r)
        vel_global = np.nan_to_num(nusc.box_velocity(ann_token), nan=0.0)
        vel_ego = ego_r.inverse.rotate(vel_global)
        boxes.append({
            'name': name,
            'range': float(np.hypot(center_ego[0], center_ego[1])),
            'az': float(np.arctan2(center_ego[1], center_ego[0])),
            'vel_ego': np.array([float(vel_ego[0]), float(vel_ego[1])], np.float32),
        })
    return boxes


def match_detection_to_gt(
        cam_center_ego: np.ndarray,
        det_name: str,
        gt_boxes: list[dict],
        ang_thresh_deg: float = 6.0,
        range_thresh: float = 20.0,
) -> dict | None:
    """Match a detection to a same-class GT box for depth/velocity targets.

    Matching is strict in azimuth (camera-reliable) and lenient in range, so a
    true positive displaced by monocular depth error still matches its GT and
    yields the correct target range. Returns the matched GT or ``None``.
    """
    cx, cy = float(cam_center_ego[0]), float(cam_center_ego[1])
    theta, r = np.arctan2(cy, cx), np.hypot(cx, cy)
    ang_thresh = np.deg2rad(ang_thresh_deg)

    best, best_key = None, np.inf
    for gt in gt_boxes:
        if gt['name'] != det_name:
            continue
        dtheta = abs(_wrap_angle(gt['az'] - theta))
        dr = abs(gt['range'] - r)
        if dtheta <= ang_thresh and dr <= range_thresh:
            key = dtheta + 0.01 * dr  # azimuth-first, range tiebreak
            if key < best_key:
                best_key, best = key, gt
    return best


def extract_sample_records(
        nusc,
        sample_token: str,
        cam_detections: list[dict],
        cfg: FrustumConfig,
        with_gt: bool = False,
) -> dict:
    """Build per-detection records for one sample.

    Drives both the trainer (features + GT targets) and learned inference
    (per-detection raw values for the refinement formula). Runs the expensive
    radar association exactly once per sample.

    Returns a sample record::

        {
          'sample_token', 'scene_name',
          'ego_t': (3,), 'ego_q': (4,) wxyz,
          'dets': [ { 'det', 'cam_center_ego', 'has_radar',
                      # present iff has_radar:
                      'feat', 'cam_range', 'radar_range',
                      'cam_vel_ego', 'radar_vel_ego',
                      # present iff with_gt and a GT matched:
                      'has_target', 'target_range', 'target_vel_ego', 'score' },
                    ... ] }
    """
    ego_t, ego_r = get_ego_pose(nusc, sample_token)
    rec = {
        'sample_token': sample_token,
        'scene_name': scene_name_of(nusc, sample_token),
        'ego_t': np.asarray(ego_t, np.float64),
        'ego_q': np.asarray(ego_r.elements, np.float64),
        'dets': [],
    }
    if not cam_detections:
        return rec

    pts = get_radar_pointcloud(nusc, sample_token, nsweeps=cfg.nsweeps)
    have_pts = pts.shape[1] > 0
    rng, az = radar_polar(pts) if have_pts else (None, None)
    gt_boxes = get_gt_boxes_ego(nusc, sample_token, ego_t, ego_r) if with_gt else []

    for det in cam_detections:
        center_ego = global_to_ego(np.array(det['translation']), ego_t, ego_r)
        assoc = (associate_frustum(center_ego, det['size'], pts, rng, az, cfg)
                 if have_pts else None)
        d = {'det': det, 'cam_center_ego': center_ego.astype(np.float32),
             'has_radar': assoc is not None}

        if assoc is not None:
            cam_range = float(np.hypot(center_ego[0], center_ego[1]))
            cam_vel_global = np.array(det['velocity'] + [0.0])
            cam_vel_ego = ego_r.inverse.rotate(cam_vel_global)[:2]
            d.update(
                feat=build_feature_row(det, center_ego, assoc),
                cam_range=cam_range,
                radar_range=float(assoc['radar_range']),
                cam_vel_ego=cam_vel_ego.astype(np.float32),
                radar_vel_ego=np.array([assoc['radar_vx'], assoc['radar_vy']],
                                       np.float32),
            )
            if with_gt:
                gt = match_detection_to_gt(center_ego, det['detection_name'], gt_boxes)
                d['has_target'] = gt is not None
                if gt is not None:
                    d['target_range'] = float(gt['range'])
                    d['target_vel_ego'] = gt['vel_ego']
                    d['score'] = float(det['detection_score'])

        rec['dets'].append(d)

    return rec


def infer_records_fixed(records: list[dict], gate_depth: float = 1.0,
                        gate_vel: float = 1.0) -> dict:
    """Apply a FIXED gate to cached records (torch-free).

    ``gate=0`` reproduces camera-only; ``gate=1`` is vanilla hard replacement.
    Lets the camera-only / vanilla baselines reuse the same cached records as
    the learned model, so every row of the comparison table comes from one
    association pass and is directly comparable.
    """
    results = {}
    for rec in records:
        ego_t = rec['ego_t']
        ego_r = Quaternion(rec['ego_q'])
        out_dets = []
        for d in rec['dets']:
            if d.get('has_radar'):
                cam_range, radar_range = d['cam_range'], d['radar_range']
                refined_range = cam_range + gate_depth * (radar_range - cam_range)
                refined_vel = (d['cam_vel_ego']
                               + gate_vel * (d['radar_vel_ego'] - d['cam_vel_ego']))
                out_dets.append(apply_refinement(
                    d['det'], d['cam_center_ego'], refined_range, refined_vel,
                    ego_t, ego_r))
            else:
                out_dets.append(dict(d['det']))
        results[rec['sample_token']] = out_dets
    return results


# ---------------------------------------------------------------------------
# Run over a split
# ---------------------------------------------------------------------------
def split_sample_tokens(nusc, split: str) -> list[str]:
    """Ordered sample tokens for an eval split ('mini_val' or 'val')."""
    from nuscenes.utils.splits import create_splits_scenes
    split_scenes = create_splits_scenes()[split]
    tokens = []
    for scene in nusc.scene:
        if scene['name'] in split_scenes:
            tok = scene['first_sample_token']
            while tok:
                tokens.append(tok)
                tok = nusc.get('sample', tok)['next']
    return tokens


def run_fusion(
        nusc,
        cam_json_path: str,
        split: str,
        out_dir: str,
        cfg: FrustumConfig,
        gate: float = 1.0,
        verbose: bool = True,
) -> str:
    """Run CenterFusion over a split and write a nuScenes submission JSON."""
    os.makedirs(out_dir, exist_ok=True)

    cam_results = load_camera_detections(cam_json_path)
    tokens = split_sample_tokens(nusc, split)
    if verbose:
        total_cam = sum(len(v) for v in cam_results.values())
        print(f'Loaded {total_cam} camera detections across {len(cam_results)} samples')
        print(f'Running CenterFusion (mode=rule_based, gate={gate}) on '
              f'{len(tokens)} samples ({split})...')
        print(f'  nsweeps={cfg.nsweeps}, radial_frac={cfg.radial_frac}, '
              f'min_radial={cfg.min_radial}, max_radial={cfg.max_radial}, '
              f'lateral_margin={cfg.lateral_margin}, min_angle={cfg.min_angle_deg}deg')

    tot_cam = tot_pts = tot_assoc = 0
    results = {}
    for i, tok in enumerate(tokens):
        cam_dets = cam_results.get(tok, [])
        fused, stats = fuse_sample(nusc, tok, cam_dets, cfg, gate=gate)
        results[tok] = fused
        tot_cam += stats['n_cam']
        tot_pts += stats['n_radar_pts']
        tot_assoc += stats['n_assoc']
        if verbose and (i + 1) % 50 == 0:
            print(f'  {i + 1}/{len(tokens)} samples processed...')

    submission = {
        'meta': {
            'use_camera': True, 'use_lidar': False, 'use_radar': True,
            'use_map': False, 'use_external': False,
        },
        'results': results,
    }
    results_path = os.path.join(out_dir, 'results.json')
    with open(results_path, 'w') as f:
        json.dump(submission, f)

    if verbose:
        assoc_rate = tot_assoc / max(tot_cam, 1) * 100
        print('\nAssociation statistics:')
        print(f'  Camera detections:      {tot_cam}')
        print(f'  Radar points (total):   {tot_pts}')
        print(f'  Detections w/ radar:    {tot_assoc} ({assoc_rate:.1f}% refined)')
        print(f'Wrote results → {results_path}')

    return results_path


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------
def evaluate(nusc, results_path: str, split: str, out_dir: str,
             verbose: bool = True) -> dict:
    """Run the official nuScenes detection evaluation on fused results."""
    from nuscenes.eval.detection.config import config_factory
    from nuscenes.eval.detection.evaluate import NuScenesEval

    cfg = config_factory('detection_cvpr_2019')
    evaluator = NuScenesEval(
        nusc, config=cfg, result_path=results_path,
        eval_set=split, output_dir=out_dir, verbose=verbose,
    )
    metrics, _ = evaluator.evaluate()

    summary = {
        'mAP': metrics.mean_ap,
        'NDS': metrics.nd_score,
        'mATE': metrics.tp_errors.get('trans_err', float('nan')),
        'mASE': metrics.tp_errors.get('scale_err', float('nan')),
        'mAOE': metrics.tp_errors.get('orient_err', float('nan')),
        'mAVE': metrics.tp_errors.get('vel_err', float('nan')),
        'mAAE': metrics.tp_errors.get('attr_err', float('nan')),
    }
    for cls in DETECTION_CLASSES:
        summary[f'{cls}_AP'] = metrics.mean_dist_aps.get(cls, float('nan'))

    metrics_path = os.path.join(out_dir, 'metrics.json')
    with open(metrics_path, 'w') as f:
        json.dump(summary, f, indent=2)

    if verbose:
        print(f'\n{"=" * 60}')
        print('  CenterFusion Results')
        print(f'{"=" * 60}')
        print(f'  mAP : {summary["mAP"]:.4f}')
        print(f'  NDS : {summary["NDS"]:.4f}')
        print(f'  mATE: {summary["mATE"]:.4f}  (translation error — radar depth)')
        print(f'  mASE: {summary["mASE"]:.4f}  (scale error)')
        print(f'  mAOE: {summary["mAOE"]:.4f}  (orientation error)')
        print(f'  mAVE: {summary["mAVE"]:.4f}  (velocity error — radar Doppler)')
        print(f'  mAAE: {summary["mAAE"]:.4f}  (attribute error)')
        print(f'{"=" * 60}')
        print(f'Metrics saved → {metrics_path}')

    return summary


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser(
        description='CenterFusion-style camera-radar detection refinement')
    p.add_argument('--version', default='v1.0-mini')
    p.add_argument('--dataroot', default='data/nuscenes-mini')
    p.add_argument('--split', default='mini_val')
    p.add_argument('--cam-json', required=True,
                   help='FCOS3D detections JSON (nuScenes submission format)')
    p.add_argument('--out', default='results/center_fusion/mini')
    p.add_argument('--mode', default='rule_based', choices=['rule_based'],
                   help='rule_based: fixed gate (vanilla). learned: added in Stage 2.')
    p.add_argument('--gate', type=float, default=1.0,
                   help='Fixed gate g for rule_based mode (1.0 = hard replace).')
    p.add_argument('--nsweeps', type=int, default=6)
    p.add_argument('--radial-frac', type=float, default=0.25)
    p.add_argument('--min-radial', type=float, default=2.5)
    p.add_argument('--max-radial', type=float, default=15.0)
    p.add_argument('--lateral-margin', type=float, default=1.0)
    p.add_argument('--min-angle-deg', type=float, default=3.0)
    p.add_argument('--no-eval', action='store_true',
                   help='Write results.json but skip NuScenesEval.')
    return p.parse_args()


if __name__ == '__main__':
    os.chdir(_ROOT)
    from nuscenes.nuscenes import NuScenes

    args = parse_args()
    cfg = FrustumConfig(
        nsweeps=args.nsweeps,
        radial_frac=args.radial_frac,
        min_radial=args.min_radial,
        max_radial=args.max_radial,
        lateral_margin=args.lateral_margin,
        min_angle_deg=args.min_angle_deg,
    )

    print(f'Loading nuScenes {args.version} from {args.dataroot}...')
    nusc = NuScenes(version=args.version, dataroot=args.dataroot, verbose=False)

    results_path = run_fusion(
        nusc, cam_json_path=args.cam_json, split=args.split,
        out_dir=args.out, cfg=cfg, gate=args.gate, verbose=True,
    )

    if not args.no_eval:
        evaluate(nusc, results_path=results_path, split=args.split,
                 out_dir=args.out, verbose=True)
