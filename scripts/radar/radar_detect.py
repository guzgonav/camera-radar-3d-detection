"""
radar_detect.py - Radar-only 3D object detector for nuScenes

Classical (not learned) radar-only detection via DBSCAN clustering in BEV,
evaluated with the official nuScenes metrics. A typical sample has only
~100-300 radar points spread across a 100m x 100m area — too sparse for a
learned pillar/voxel detector to generalize without radar-specific
pretraining, so clustering is the honest baseline for what raw radar alone
can do (mAP ~0.03-0.10 is expected and intentional: the point is to
establish the floor that fusion must exceed, not to be good). Expect lower
mATE/mAVE than camera-only (radar measures range and Doppler velocity
directly) but poor AOE (no orientation estimate) and missed low-RCS classes
like pedestrians/cyclists.

Usage
-----
    # Run on mini val
    python scripts/radar/radar_detect.py --version v1.0-mini --dataroot data/nuscenes-mini \\
        --split val --out results/radar_baseline/mini

    # Run on full val
    python scripts/radar/radar_detect.py --version v1.0-trainval --dataroot data/nuscenes \\
        --split val --out results/radar_baseline/full
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
from pkg_resources import split_sections
from sklearn.cluster import DBSCAN

_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_ROOT))

from scripts.radar.radar_preprocess import get_radar_pointcloud, iterate_samples, OUT_X, OUT_Y, OUT_Z, OUT_VX, OUT_VY, OUT_RCS

# ---------------------------------------------------------------------------
# Class priors
# ---------------------------------------------------------------------------

# Size priors (width, length, height) in meters, derived from nuScenes annotation
# statistics. Used because radar clusters have no shape information.
# Source: nuScenes devkit annotation statistics on the train split.
#TODO -> review this
CLASS_SIZE_PRIORS = {
    'car':                    (1.93, 4.62, 1.72),
    'truck':                  (2.51, 6.93, 2.84),
    'bus':                    (2.96, 11.19, 3.49),
    'trailer':                (2.87, 12.29, 3.87),
    'construction_vehicle':   (2.63, 6.37, 3.19),
    'pedestrian':             (0.67, 0.73, 1.77),
    'motorcycle':             (0.96, 2.11, 1.47),
    'bicycle':                (0.60, 1.70, 1.28),
    'traffic_cone':           (0.41, 0.41, 1.07),
    'barrier':                (2.53, 0.50, 0.98),
}

# Required attribute name per class for nuScenes submission
# The devkit requires a valid attribute even if the detector can't estimate it
# We always pick the most common attribute for each class
CLASS_DEFAULT_ATTRIBUTE = {
    'car':                  'vehicle.moving',
    'truck':                'vehicle.moving',
    'bus':                  'vehicle.moving',
    'trailer':              'vehicle.stopped',
    'construction_vehicle': 'vehicle.stopped',
    'pedestrian':           'pedestrian.moving',
    'motorcycle':           'cycle.without_rider',
    'bicycle':              'cycle.without_rider',
    'traffic_cone':         '',
    'barrier':              '',
}

# Valid detection class names accepted by the nuScenes evaluator
DETECTION_CLASSES = list(CLASS_SIZE_PRIORS.keys())

# ---------------------------------------------------------------------------
# Class assignment heuristic
# ---------------------------------------------------------------------------

def assign_class (cluster_pts: np.ndarray) -> tuple[str, float]:
    """
Assign a detection class and confidence score to a radar cluster.

    This is a simple heuristic based on:
      - RCS (higher → larger / more metallic object → vehicle)
      - Cluster size (number of points and spatial extent)
      - Speed (moving vs stationary)

    A learned classifier would do better, but this is sufficient for a
    radar-only baseline. The heuristic is intentionally conservative —
    we only confidently assign 'car' or 'truck' for large, strong clusters.
    Everything else gets 'car' as a catch-all with a low score.

    Args:
        cluster_pts: (6, K) array for one cluster — [x,y,z,vx,vy,rcs]

    Returns:
        (class_name, score) tuple
    """
    n_pts = cluster_pts.shape[1]
    mean_rcs = float(cluster_pts[OUT_RCS].mean())
    speed = float(np.sqrt(cluster_pts[OUT_VX]**2 + cluster_pts[OUT_VY]**2).mean())

    # Spatial extent in BEV (bounding box diagonal)
    x_range = float(cluster_pts[OUT_X].max() - cluster_pts[OUT_X].min())
    y_range = float(cluster_pts[OUT_Y].max() - cluster_pts[OUT_Y].min())
    bev_diag = float(np.sqrt(x_range**2 + y_range**2))

    # Heuristic rules (tuned on mini visualizations)
    # TODO -> tune this heuristics

    # Large cluster with high RCS -> truck or bus
    if n_pts >= 8 and mean_rcs > 15 and bev_diag > 5:
        return 'truck', 0.5

    # Medium cluster, reasonable RCS -> car
    if n_pts >= 3 and mean_rcs > 5:
        return 'car', 0.4

    # Very small cluster with any motion -> possible pedestrian / cyclist
    # (low confidence as radar barely sees them)
    if n_pts <= 3 and speed > 0.3:
        return 'pedestrian', 0.2

    # Default fallback
    return 'car', 0.25

# ---------------------------------------------------------------------------
# Per-sample detection
# ---------------------------------------------------------------------------
def detect_sample(
    nusc,
    sample_token: str,
    nsweeps: int = 6,
    dbscan_eps: float = 2.5,
    dbscan_min_samples: int = 2,
    max_detections: int = 500,
) -> list[dict]:
    """
    Run the radar-only detector on a single sample.

    Steps:
      1. Get the preprocessed radar point cloud (6, M) in ego frame
      2. Cluster in BEV using DBSCAN
      3. For each cluster, generate one detection dict
      4. Convert from ego frame to global frame (required by nuScenes eval)

    DBSCAN parameters
    -----------------
    eps=2.5m means two points within 2.5m of each other belong to the same
    cluster. This roughly matches the footprint of a car (4.6 × 1.9m).
    min_samples=2 means a cluster needs at least 2 points (avoids single
    isolated noise points becoming detections).

    Coordinate frames
    -----------------
    radar_preprocess outputs points in the EGO frame.
    The nuScenes evaluator expects detections in the GLOBAL frame.
    We apply the ego_pose transform to convert.

    Args:
        nusc:               NuScenes instance
        sample_token:       Sample to process
        nsweeps:            Passed to get_radar_pointcloud
        dbscan_eps:         DBSCAN neighborhood radius (meters in BEV)
        dbscan_min_samples: DBSCAN minimum cluster size
        max_detections:     Cap on detections per sample (eval requirement)

    Returns:
        List of detection dicts in nuScenes submission format:
        {
            'sample_token': str,
            'translation':  [x, y, z],        # global frame (m)
            'size':         [w, l, h],         # meters
            'rotation':     [w, x, y, z],      # quaternion (heading unknown → identity)
            'velocity':     [vx, vy],          # m/s in global frame
            'detection_name': str,
            'detection_score': float,
            'attribute_name': str,
        }
    """
    # Step 1: Get radar point cloud in ego frame
    pts = get_radar_pointcloud(nusc, sample_token, nsweeps=nsweeps) # (6, M)
    if pts.shape[1] == 0:
        return []

    # Step 2: DBSCAN clustering in BEV (x, y only)
    # We cluster in 2D because radar z is unreliable (radar measures in a near-horizontal
    # plane, so z is mostly the sensor height, not the object height).
    bev_xy = pts[[OUT_X, OUT_Y], :].T  # (M, 2)
    labels = DBSCAN(eps=dbscan_eps, min_samples=dbscan_min_samples).fit_predict(bev_xy)

    # labels == -1 means noise (no cluster assigned) - skip those points
    cluster_ids = [l for l in np.unique(labels) if l != -1]
    if not cluster_ids:
        return []

    # Get ego pose for ego -> global transform
    sample = nusc.get('sample', sample_token)
    # Use LIDAR_TOP's ego_pose as the reference
    lidar_sd = nusc.get('sample_data', sample['data']['LIDAR_TOP'])
    ep = nusc.get('ego_pose', lidar_sd['ego_pose_token'])
    ego_t = np.array(ep['translation']) # (3, ) global position of ego
    # import here to avoid top-level dependency
    from pyquaternion import Quaternion
    ego_r = Quaternion(ep['rotation'])

    detections = []

    for cid in cluster_ids[:max_detections]:
        mask = labels == cid
        cluster_pts = pts[:, mask] # (6, K)

        # Step 3: Generate detection from cluster

        # Center: mean position of cluster points in ego frame
        cx_ego = float(cluster_pts[OUT_X].mean())
        cy_ego = float(cluster_pts[OUT_Y].mean())
        # Z: Radar z is unreliable - use fixed ground-relative height
        # The nuScenes annotation center z for a car is ~0.0 to 0.5m above ground
        # Ego frame z ≈ is roughly ground level
        cz_ego = 0.0

        # Velocity: mean compensated velocity from cluster points
        vx_ego = float(cluster_pts[OUT_VX].mean())
        vy_ego = float(cluster_pts[OUT_VY].mean())

        # Class + score from heuristic
        class_name, score = assign_class(cluster_pts)
        w, l, h = CLASS_SIZE_PRIORS[class_name]

        # Orientation: radar gives no heading information
        # Using identity quaternion (heading = 0 -> facing forward along x-axis)
        # This hurts AOE (orientation error) - that's expected for radar-only
        rotation_ego = [1.0, 0.0, 0.0, 0.0] # quaternion (w, x, y, z)

        # Step 4: ego frame -> global frame
        # Translation: rotate the ego-frame center, then add ego global position
        center_ego = np.array([cx_ego, cy_ego, cz_ego])
        center_global = ego_r.rotate(center_ego) + ego_t

        # Velocity: rotate velocity vector from ego to global frame
        vel_ego = np.array([vx_ego, vy_ego, 0.0])
        vel_global = ego_r.rotate(vel_ego)

        # Rotation: combine ego heading with detection heading
        # Since detection rotation is identity in ego frame, global rotation = ego rotation
        rot_global = ego_r  # Quaternion
        rotation_global = [rot_global.w, rot_global.x, rot_global.y, rot_global.z]

        detections.append({
            'sample_token': sample_token,
            'translation': center_global.tolist(),
            'size': [w, l, h],
            'rotation': rotation_global,
            'velocity': [float(vel_global[0]), float(vel_global[1])],
            'detection_name': class_name,
            'detection_score': score,
            'attribute_name': CLASS_DEFAULT_ATTRIBUTE[class_name],
        })

    return detections

# ---------------------------------------------------------------------------
# Run over a full split and write submission JSON
# ---------------------------------------------------------------------------
def run_detection(
    nusc,
    split: str,
    out_dir: str,
    nsweeps: int = 6,
    dbscan_eps: float = 2.5,
    dbscan_min_samples: int = 2,
    verbose: bool = True,
) -> str:
    """
    Run the radar-only detector over all samples in a split and write
    the nuScenes submission JSON to `out_dir/results.json`.

    Args:
        nusc:    NuScenes instance
        split:   Split name, e.g. 'val' or 'mini_val'
        out_dir: Directory to write results.json and metrics.json
        nsweeps: Radar sweeps to accumulate
        verbose: Print progress

    Returns:
        Path to the written results.json
    """
    os.makedirs(out_dir, exist_ok=True)

    # Collect any sample tokens for the requested split
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
        print(f'Running radar detector on {len(split_tokens)} samples ({split} split)...')

    results = {}
    for i, sample_token in enumerate(split_tokens):
        dets = detect_sample(
            nusc, sample_token,
            nsweeps=nsweeps,
            dbscan_eps=dbscan_eps,
            dbscan_min_samples=dbscan_min_samples,
        )
        results[sample_token] = dets

        if verbose and (i + 1) % 50 == 0:
            print(f'  {i+1}/{len(split_tokens)} samples processed...')

    # nuScenes submission format requires a top-level 'results' dict and a 'meta' dict
    submission = {
        'meta': {
            'use_camera':   False,
            'use_lidar':    False,
            'use_radar':    True,
            'use_map':      False,
            'use_external': False,
        },
        'results': results,
    }

    results_path = os.path.join(out_dir, 'results.json')
    with open(results_path, 'w') as f:
        json.dump(submission, f)

    total_dets = sum(len(v) for v in results.values())
    if verbose:
        print(f'Wrote {total_dets} detections across {len(results)} samples → {results_path}')

    return results_path

# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def evaluate(nusc, results_path: str, split: str, out_dir: str, verbose: bool = True) -> dict:
    """
    Run the official nuScenes detection evaluation on a results file.

    Args:
        nusc:         NuScenes instance
        results_path: Path to the results.json written by run_detection
        split:        Eval split ('val' or 'mini_val')
        out_dir:      Directory to write metrics.json
        verbose:      Print evaluation details

    Returns:
        Dict with mAP and NDS (and all per-class metrics)
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

    # Flatten to a simple dict matching the format used by fcos3d_baseline
    summary = {
        'mAP': metrics.mean_ap,
        'NDS': metrics.nd_score,
        'mATE': metrics.tp_errors.get('trans_err', float('nan')),
        'mASE': metrics.tp_errors.get('scale_err', float('nan')),
        'mAOE': metrics.tp_errors.get('orient_err', float('nan')),
        'mAVE': metrics.tp_errors.get('vel_err', float('nan')),
        'mAAE': metrics.tp_errors.get('attr_err', float('nan')),
    }

    # Per-class AP
    for cls in DETECTION_CLASSES:
        summary[f'{cls}_AP'] = metrics.mean_dist_aps.get(cls, float('nan'))

    metrics_path = os.path.join(out_dir, 'metrics.json')
    with open(metrics_path, 'w') as f:
        json.dump(summary, f, indent=2)

    if verbose:
        print(f'\n{"="*50}')
        print(f'  mAP : {summary["mAP"]:.4f}')
        print(f'  NDS : {summary["NDS"]:.4f}')
        print(f'  mATE: {summary["mATE"]:.4f}  (translation error, lower=better)')
        print(f'  mAVE: {summary["mAVE"]:.4f}  (velocity error — radar strength)')
        print(f'{"="*50}')
        print(f'Metrics saved → {metrics_path}')

    return summary


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description='Radar-only 3D detection baseline for nuScenes')
    p.add_argument('--version',   default='v1.0-mini',
                   help='nuScenes version (v1.0-mini or v1.0-trainval)')
    p.add_argument('--dataroot',  default='data/nuscenes-mini',
                   help='Path to nuScenes data root')
    p.add_argument('--split',     default='mini_val',
                   help='Evaluation split: mini_val or val')
    p.add_argument('--out',       default='results/radar_baseline/mini',
                   help='Output directory for results.json and metrics.json')
    p.add_argument('--nsweeps',   type=int, default=6,
                   help='Number of radar sweeps to accumulate')
    p.add_argument('--eps',       type=float, default=2.5,
                   help='DBSCAN eps parameter (meters)')
    p.add_argument('--min-samples', type=int, default=2,
                   help='DBSCAN min_samples parameter')
    return p.parse_args()

# TODO -> tune DBSCAN and radar sweeps to get a better result

if __name__ == '__main__':
    import os
    from pathlib import Path

    # Change to project root when running as a script
    os.chdir(Path(__file__).resolve().parent.parent.parent)

    from nuscenes.nuscenes import NuScenes

    args = parse_args()

    print(f'Loading nuScenes {args.version} from {args.dataroot}...')
    nusc = NuScenes(version=args.version, dataroot=args.dataroot, verbose=False)

    # Run detection over all samples in the split
    results_path = run_detection(
        nusc,
        split=args.split,
        out_dir=args.out,
        nsweeps=args.nsweeps,
        dbscan_eps=args.eps,
        dbscan_min_samples=args.min_samples,
        verbose=True,
    )

    # Evaluate with official nuScenes metrics
    evaluate(nusc, results_path=results_path, split=args.split, out_dir=args.out, verbose=True)
