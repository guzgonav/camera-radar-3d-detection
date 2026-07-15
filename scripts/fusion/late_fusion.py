"""
late_fusion.py - Camera-radar late fusion for nuScenes object 3D detection

Merges independent camera (FCOS3D) and radar (DBSCAN clustering) detections
via Hungarian matching in BEV (ego frame): camera contributes class/size/
orientation, radar contributes depth (time-of-flight) and velocity (Doppler).
Should improve mATE/mAVE/NDS while leaving mASE/mAOE/mAAE (camera-driven)
roughly unchanged.

Usage
------

    # Run on mini val (requires camera detections JSON from fcos3d_mini_extract.py)
    python scripts/fusion/late_fusion.py --version v1.0-mini --dataroot data/nuscenes-mini \\
        --split mini_val --cam-json results/fcos3d_baseline/mini/detections/pred_instances_3d/results_nusc.json \\
        --out results/late_fusion/mini

    # Run on full val     # Run on mini val (requires camera detections JSON from fcos3d_full_extract.py)
    python scripts/fusion/late_fusion.py --version v1.0-trainval --dataroot data/nuscenes \\
        --split val --cam-json results/fcos3d_baseline/full/detections/pred_instances_3d/results_nusc.json \\
        --out results/late_fusion/full
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
from pyquaternion import Quaternion
from scipy.optimize import linear_sum_assignment
from sklearn.cluster import DBSCAN

_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_ROOT))

from scripts.radar.radar_preprocess import (
    get_radar_pointcloud,
    OUT_X, OUT_Y, OUT_Z, OUT_VX, OUT_VY, OUT_RCS,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Hungarian matching: maximum BEV distance (meters) to accept a match
# A car is aprox 4.5m long. Camera depth errors at 30-50m range can be 2-4m
# 5m is generous enough for true matches, tight enough to avoid cross-object
DEFAULT_MATCH_THRESH = 5.0

# DBSCAN parameters (same as radar_detect.py)
DEFAULT_EPS = 2.5
DEFAULT_MIN_SAMPLES = 2

# Detection classes accepted by nuScenes evaluator
DETECTION_CLASSES = [
    'car', 'truck', 'bus', 'trailer', 'construction_vehicle',
    'pedestrian', 'motorcycle', 'bicycle', 'traffic_cone', 'barrier',
]

# ---------------------------------------------------------------------------
# Load camera detections
# ---------------------------------------------------------------------------
def load_camera_detections(json_path: str) -> dict[str, list[dict]]:
    """
    Load camera detections from a nuScenes submission JSON.

    Args:
        json_path: path to results_nusc.json

    Returns:
        Dict mapping sample_token -> list of detection dicts.
        Each detection has: sample_token, translation, size, rotation
        velocity, detection_name, detection_score, attribute_name
    """
    with open(json_path) as f:
        data = json.load(f)

    assert 'results' in data, f'Expected "results" key in {json_path}'
    return data['results']

# ---------------------------------------------------------------------------
# Coordinate frame helpers
# ---------------------------------------------------------------------------
# TODO -> Check if this should be extracted into a helpers script
# TODO -> This function might need their own unit tests
def get_ego_pose(nusc, sample_token: str) -> tuple[np.ndarray, Quaternion]:
    """
    Get the ego pose (translation, rotation) for a sample

    Uses LIDAR_TOP's ego_pose as the reference frame - standard because LIDAR_TOP
    timestamp defines the keyframe

    Returns:
        (ego_t, ego_r): translation (3, ) and rotation quaternion
    """
    sample = nusc.get('sample', sample_token)
    lidar_sd = nusc.get('sample_data', sample['data']['LIDAR_TOP'])
    ep = nusc.get('ego_pose', lidar_sd['ego_pose_token'])
    ego_t = np.array(ep['translation'])
    ego_r = Quaternion(ep ['rotation'])
    return ego_t, ego_r

def global_to_ego(point_global: np.ndarray, ego_t: np.ndarray, ego_r: Quaternion) -> np.ndarray:
    """Convert a 3D point from global frame to ego frame."""
    return ego_r.inverse.rotate(point_global - ego_t)

def ego_to_global(point_ego: np.ndarray, ego_t: np.ndarray, ego_r: Quaternion) -> np.ndarray:
    """Convert a 3D point from ego frame to global frame."""
    return ego_r.rotate(point_ego) + ego_t

# ---------------------------------------------------------------------------
# Radar clustering (simplified from radar_detect.py — no class assignment)
# ---------------------------------------------------------------------------
def cluster_radar(
    pts: np.ndarray,
    eps: float = DEFAULT_EPS,
    min_samples: int = DEFAULT_MIN_SAMPLES,
) -> list[dict]:
    """
    Cluster radar points in BEV and extract center + velocity per cluster
    No class assignment here - the camera provides the class.

    Args:
        pts:         (6, M) radar point cloud in ego_frame
        eps:         DBSCAN neighbourhood radius (meters in BEV)
        min_samples: DBSCAN minimum cluster size

    Returns:
        List of dicts with keys:
            center_ego: [x, y, z] in ego frame
            velocity_ego: [vx, vy] compensated velocity in ego frame
            n_pts: number of points in cluster
            mean_rcs: mean radar cross section (dBsm)
    """
    if pts.shape[1] == 0:
        return []

    bev_xy = pts[[OUT_X, OUT_Y], :].T # (M, 2)
    labels = DBSCAN(eps=eps, min_samples=min_samples).fit_predict(bev_xy)

    # TODO -> understand this
    clusters = []
    for cid in np.unique(labels):
        if cid == -1: # noise
            continue
        mask = labels == cid
        cluster_pts = pts[:, mask]

        clusters.append({
            'center_ego': [
                float(cluster_pts[OUT_X].mean()),
                float(cluster_pts[OUT_Y].mean()),
                0.0, # radar z is unreliable -> use 0 (ground level)
            ],
            'velocity_ego': [
                float(cluster_pts[OUT_VX].mean()),
                float(cluster_pts[OUT_VY].mean()),
            ],
            'n_pts': int(mask.sum()),
            'mean_rcs': float(cluster_pts[OUT_RCS].mean()),
        })

    return clusters

# ---------------------------------------------------------------------------
# Per-sample fusion
# ---------------------------------------------------------------------------
def fuse_sample(
    nusc,
    sample_token: str,
    cam_detections: list[dict],
    nsweeps: int = 6,
    eps: float = DEFAULT_EPS,
    min_samples: int = DEFAULT_MIN_SAMPLES,
    match_thresh: float = DEFAULT_MATCH_THRESH,
) -> list[dict]:
    """
    Run late fusion on a single sample

    Steps:
        1. Get radar point cloud and cluster it
        2. Convert camera detections from global to ego frame
        3. Build BEV cost matrix (Euclidean distance)
        4. Hungarian matching with distance gate
        5. Merge matched pairs: camera class/size/orient + radar depth/velocity
        6. Keep unmatched camera detections as-is
        7. Discard unmatched radar clusters (radar-only detections deteriorate precision)
        8. Convert everything back to ego frame

    Args:
        nusc: NuScenes instance
        sample_token: Sample to process
        cam_detections: List of camera detections for this sample (global frame)
        nsweeps: Number of radar sweeps to accumulate
        eps: DBSCAN eps
        min_samples: DBSCAN min_samples
        match_thresh: Max BEV distance (m) to accept a Hungarian match

    Returns:
        list of (fused_detections, stats) where:
            fused_detections: list of detection dicts in nuScenes format (global frame)
            stats: dict with n_cam, n_radar, n_matched
    """

    # --- Step 1: radar clustering ---
    pts = get_radar_pointcloud(nusc, sample_token, nsweeps=nsweeps)
    radar_clusters = cluster_radar(pts, eps=eps, min_samples=min_samples)

    stats = {'n_cam': len(cam_detections), 'n_radar': len(radar_clusters), 'n_matched': 0}

    # If no camera detections, nothing to fuse
    if not cam_detections:
        return [], stats

    # If no radar clusters, return camera detections unchanged
    if not radar_clusters:
        return cam_detections, stats

    # --- Step 2: convert camera detections to ego frame ---
    ego_t, ego_r = get_ego_pose(nusc, sample_token)

    cam_centers_ego = []
    for det in cam_detections:
        center_global = np.array(det['translation'])
        center_ego = global_to_ego(center_global, ego_t, ego_r)
        cam_centers_ego.append(center_ego)

    # --- Step 3: build BEV cost matrix ---
    n_cam = len(cam_detections)
    n_radar = len(radar_clusters)
    cost = np.full((n_cam, n_radar), fill_value=1e6)

    for i, cam_ego in enumerate(cam_centers_ego):
        for j, radar in enumerate(radar_clusters):
            dx = cam_ego[0] - radar['center_ego'][0]
            dy = cam_ego[1] - radar['center_ego'][1]
            cost[i, j] = np.sqrt(dx**2 + dy**2)

    # --- Step 4: Hungarian matching with distance gate ---
    cam_idx, rad_idx = linear_sum_assignment(cost)

    matched_pairs = []  # list of (cam_idx, radar_idx, bev_distance)
    for cam_i, rad_j in zip(cam_idx, rad_idx):
        if cost[cam_i, rad_j] < match_thresh:
            matched_pairs.append((cam_i, rad_j, cost[cam_i, rad_j]))

    matched_cam_set = {cam_i for cam_i, _, _ in matched_pairs}
    stats['n_matched'] = len(matched_pairs)

    # --- Step 5 & 6: merge matched, keep unmatched camera ---
    fused_detections = []

    for i, detection in enumerate(cam_detections):
        # Copy the original detection - modify the matched ones
        fused = dict(detection)

        if i in matched_cam_set:
            # Find the matched radar cluster and match distance
            _, matched_rad_idx, bev_dist = next(
                pair for pair in matched_pairs if pair[0] == i
            )
            radar_cluster = radar_clusters[matched_rad_idx]

            # Blending weight: closer BEV match → more radar influence
            # 1.0 at dist=0 (full radar), 0.0 at dist=match_thresh (full camera)
            radar_weight = max(0.0, 1.0 - bev_dist / match_thresh)

            # --- Depth fusion: blend camera and radar range ---
            # Camera: good at azimuth (pixel columns), unreliable depth
            # Radar: good at range (ToF), unreliable azimuth
            # Strategy: keep camera azimuth, blend ranges by match confidence
            cam_ego = cam_centers_ego[i]
            radar_ego = np.array(radar_cluster['center_ego'])

            cam_range = np.sqrt(cam_ego[0]**2 + cam_ego[1]**2)
            radar_range = np.sqrt(radar_ego[0]**2 + radar_ego[1]**2)
            fused_range = radar_weight * radar_range + (1 - radar_weight) * cam_range

            if cam_range > 1e-3:  # avoid division by zero for very close detections
                scale = fused_range / cam_range
                fused_x = cam_ego[0] * scale
                fused_y = cam_ego[1] * scale
            else:
                fused_x = radar_ego[0]
                fused_y = radar_ego[1]

            # Keep camera z (radar z is unreliable)
            fused_center_ego = np.array([fused_x, fused_y, cam_ego[2]])

            # Convert back to global
            fused_center_global = ego_to_global(fused_center_ego, ego_t, ego_r)
            fused['translation'] = fused_center_global.tolist()

            # --- Velocity fusion: blend camera and radar in ego frame ---
            # Convert camera velocity (global) → ego for blending
            cam_vel_global = np.array(detection['velocity'] + [0.0])
            cam_vel_ego = ego_r.inverse.rotate(cam_vel_global)

            radar_vel_ego = np.array([
                radar_cluster['velocity_ego'][0],
                radar_cluster['velocity_ego'][1],
                0.0,
            ])

            fused_vel_ego = radar_weight * radar_vel_ego + (1 - radar_weight) * cam_vel_ego
            fused_vel_global = ego_r.rotate(fused_vel_ego)
            fused['velocity'] = [float(fused_vel_global[0]), float(fused_vel_global[1])]

        fused_detections.append(fused)

    return fused_detections, stats

# ---------------------------------------------------------------------------
# Run over a full split
# ---------------------------------------------------------------------------
def run_fusion(
    nusc,
    cam_json_path: str,
    split: str,
    out_dir: str,
    nsweeps: int = 6,
    eps: float = DEFAULT_EPS,
    min_samples: int = DEFAULT_MIN_SAMPLES,
    match_thresh: float = DEFAULT_MATCH_THRESH,
    verbose: bool = True,
) -> str:
    """
    Run late fusion over all samples in a split.

    Args:
        nusc:          NuScenes instance
        cam_json_path: Path to camera detections JSON
        split:         Eval split ('mini_val' or 'val')
        out_dir:       Output directory for results.json and metrics.json
        nsweeps:       Radar sweeps to accumulate
        eps:           DBSCAN eps
        min_samples:   DBSCAN min_samples
        match_thresh:  Hungarian matching distance threshold
        verbose:       Print progress

    Returns:
        Path to the written results.json
    """
    os.makedirs(out_dir, exist_ok=True)

    # Load camera detections
    cam_results = load_camera_detections(cam_json_path)
    if verbose:
        total_cam = sum(len(v) for v in cam_results.values())
        print(f'Loaded {total_cam} camera detections across {len(cam_results)} samples')

    # Get sample tokens for the requested split
    from nuscenes.utils.splits import create_splits_scenes
    split_scenes = create_splits_scenes()[split]
    split_tokens = []
    for scene in nusc.scene:
        if scene['name'] in split_scenes:
            token = scene['first_sample_token']
            while token:
                split_tokens.append(token)
                token = nusc.get('sample', token)['next']

    if verbose:
        print(f'Running late fusion on {len(split_tokens)} samples ({split} split)...')
        print(f'  match_thresh={match_thresh}m, nsweeps={nsweeps}, '
              f'eps={eps}, min_samples={min_samples}')

    # Track matching statistics
    total_cam_dets = 0
    total_rad_clusters = 0
    total_matches = 0

    results = {}
    for i, sample_token in enumerate(split_tokens):
        cam_dets = cam_results.get(sample_token, [])

        fused, stats = fuse_sample(
            nusc, sample_token, cam_dets,
            nsweeps=nsweeps, eps=eps, min_samples=min_samples,
            match_thresh=match_thresh,
        )
        results[sample_token] = fused

        total_cam_dets += stats['n_cam']
        total_rad_clusters += stats['n_radar']
        total_matches += stats['n_matched']

        if verbose and (i + 1) % 50 == 0:
            print(f'  {i+1}/{len(split_tokens)} samples processed...')

    # nuScenes submission format
    submission = {
        'meta': {
            'use_camera': True,
            'use_lidar': False,
            'use_radar': True,
            'use_map': False,
            'use_external': False,
        },
        'results': results,
    }

    results_path = os.path.join(out_dir, 'results.json')
    with open(results_path, 'w') as f:
        json.dump(submission, f)

    total_fused = sum(len(v) for v in results.values())
    if verbose:
        match_rate = total_matches / max(total_cam_dets, 1) * 100
        print(f'\nFusion statistics:')
        print(f'  Camera detections: {total_cam_dets}')
        print(f'  Radar clusters:    {total_rad_clusters}')
        print(f'  Matched pairs:     {total_matches} ({match_rate:.1f}% of camera dets)')
        print(f'  Fused detections:  {total_fused}')
        print(f'Wrote results → {results_path}')

    return results_path

# ---------------------------------------------------------------------------
# Evaluation (same pattern as radar_detect.py)
# ---------------------------------------------------------------------------

def evaluate(nusc, results_path: str, split: str, out_dir: str, verbose: bool = True) -> dict:
    """
    Run the official nuScenes detection evaluation on fused results.

    Args:
        nusc:         NuScenes instance
        results_path: Path to results.json
        split:        Eval split ('mini_val' or 'val')
        out_dir:      Directory to write metrics.json
        verbose:      Print evaluation details

    Returns:
        Dict with mAP, NDS, and all per-class metrics
    """
    from nuscenes.eval.detection.config import config_factory
    from nuscenes.eval.detection.evaluate import NuScenesEval

    cfg = config_factory('detection_cvpr_2019')

    evaluator = NuScenesEval(
        nusc,
        config=cfg,
        result_path=results_path,
        eval_set=split,
        output_dir=out_dir,
        verbose=verbose,
    )

    metrics, metric_data_list = evaluator.evaluate()

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
        print(f'\n{"="*60}')
        print(f'  Late Fusion Results')
        print(f'{"="*60}')
        print(f'  mAP : {summary["mAP"]:.4f}')
        print(f'  NDS : {summary["NDS"]:.4f}')
        print(f'  mATE: {summary["mATE"]:.4f}  (translation error, lower=better)')
        print(f'  mASE: {summary["mASE"]:.4f}  (scale error)')
        print(f'  mAOE: {summary["mAOE"]:.4f}  (orientation error)')
        print(f'  mAVE: {summary["mAVE"]:.4f}  (velocity error — radar strength)')
        print(f'  mAAE: {summary["mAAE"]:.4f}  (attribute error)')
        print(f'{"="*60}')
        print(f'Metrics saved → {metrics_path}')

    return summary

# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description='Camera-radar late fusion for nuScenes 3D detection')
    p.add_argument('--version', default='v1.0-mini',
                   help='nuScenes version (v1.0-mini or v1.0-trainval)')
    p.add_argument('--dataroot', default='data/nuscenes-mini',
                   help='Path to nuScenes data root')
    p.add_argument('--split', default='mini_val',
                   help='Evaluation split: mini_val or val')
    p.add_argument('--cam-json', required=True,
                   help='Path to camera detections JSON (nuScenes submission format)')
    p.add_argument('--out', default='results/late_fusion/mini',
                   help='Output directory for results.json and metrics.json')
    p.add_argument('--nsweeps', type=int, default=6,
                   help='Number of radar sweeps to accumulate')
    p.add_argument('--eps', type=float, default=DEFAULT_EPS,
                   help='DBSCAN eps parameter (meters)')
    p.add_argument('--min-samples', type=int, default=DEFAULT_MIN_SAMPLES,
                   help='DBSCAN min_samples parameter')
    p.add_argument('--match-thresh', type=float, default=DEFAULT_MATCH_THRESH,
                   help='Hungarian matching distance threshold (meters)')
    return p.parse_args()


if __name__ == '__main__':
    os.chdir(Path(__file__).resolve().parent.parent.parent)

    from nuscenes.nuscenes import NuScenes

    args = parse_args()

    print(f'Loading nuScenes {args.version} from {args.dataroot}...')
    nusc = NuScenes(version=args.version, dataroot=args.dataroot, verbose=False)

    results_path = run_fusion(
        nusc,
        cam_json_path=args.cam_json,
        split=args.split,
        out_dir=args.out,
        nsweeps=args.nsweeps,
        eps=args.eps,
        min_samples=args.min_samples,
        match_thresh=args.match_thresh,
        verbose=True,
    )

    evaluate(nusc, results_path=results_path, split=args.split,
             out_dir=args.out, verbose=True)

