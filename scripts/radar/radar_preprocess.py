"""
radar_preprocess.py — Radar preprocessing pipeline for nuScenes.

This module is the foundation for all radar-related work in this project:
imported by ``radar_detect.py`` (radar-only baseline), ``late_fusion.py``,
and the BEV/center/rpp fusion models. ``filter_radar_points()`` removes
invalid/clutter points, ``accumulate_sweeps()`` stacks past sweeps
ego-motion-compensated, and ``get_radar_pointcloud()`` merges all 5 sensors
into a compact ``(6, M)`` array ``[x, y, z, vx_comp, vy_comp, rcs]`` in the
ego frame.

Coordinate frames: ``radar_sensor_frame -> ego_frame -> global_frame`` (via
``calibrated_sensor`` extrinsics, then ``ego_pose``). Multi-sweep
accumulation chains ``past_sensor -> past_ego -> global -> current_ego``.

The 18 raw fields (in order)
-----------------------------
 0  x               position in sensor frame (m)
 1  y               position in sensor frame (m)
 2  z               position in sensor frame (m)
 3  dyn_prop        motion classification (0=moving,1=stationary,2=oncoming,
                    3=stat_candidate,4=unknown,5=crossing_stat,
                    6=crossing_mov,7=stopped)
 4  id              point ID from radar hardware
 5  rcs             radar cross section (dBsm) — higher = stronger reflector
 6  vx              raw velocity X (includes ego motion, m/s)
 7  vy              raw velocity Y (includes ego motion, m/s)
 8  vx_comp         compensated velocity X — ego motion removed (m/s) ← use this
 9  vy_comp         compensated velocity Y — ego motion removed (m/s) ← use this
10  is_quality_valid  quality flag (1=valid)
11  ambig_state     Doppler ambiguity (3=valid velocity)
12  x_rms           x position uncertainty (lookup table index)
13  y_rms           y position uncertainty (lookup table index)
14  invalid_state   validity flag — 0=valid, anything else=bad   ← filter first
15  pdh0            false detection probability
16  vx_rms          vx uncertainty (lookup table index)
17  vy_rms          vy uncertainty (lookup table index)
"""

from __future__ import annotations

import numpy as np
from pyquaternion import Quaternion
from nuscenes.nuscenes import NuScenes
from nuscenes.utils.data_classes import RadarPointCloud


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

RADAR_CHANNELS = [
    'RADAR_FRONT',
    'RADAR_FRONT_LEFT',
    'RADAR_FRONT_RIGHT',
    'RADAR_BACK_LEFT',
    'RADAR_BACK_RIGHT',
]

# dyn_prop values to KEEP.
# We drop 4 (unknown) and 3 (stationary candidate) by default —
# both are low-confidence classifications that add more noise than signal.
# Adjust this if your analysis shows they contain useful returns.
DEFAULT_VALID_DYN_PROPS = (0, 1, 2, 5, 6, 7)

# Default RCS threshold in dBsm.
# Points below this are likely ground reflections or multipath clutter.
# Typical vehicles: 5–25 dBsm. Pedestrians: 0–5 dBsm. Noise: < -5 dBsm.
DEFAULT_MIN_RCS = 0

# Number of past sweeps to accumulate (including the current one).
# 6 is the nuScenes community standard (used by CenterFusion, etc.).
# More sweeps → denser cloud but older data. Diminishing returns > 6.
DEFAULT_NSWEEPS = 6

# Output field indices in the compact (6, M) representation
OUT_X = 0
OUT_Y = 1
OUT_Z = 2
OUT_VX = 3  # compensated velocity X (m/s)
OUT_VY = 4  # compensated velocity Y (m/s)
OUT_RCS = 5  # radar cross section (dBsm)

# Extra rows present only in the v2 layout (see get_radar_pointcloud(layout='v2')).
# Rows 0-5 are identical to v1 so downstream code that indexes positions
# (0,1) and velocities (3,4) — e.g. the BEV flip/rotation transforms —
# works unchanged on both layouts.
OUT_DT = 6    # sweep age in seconds relative to the LIDAR keyframe (0 = newest)
OUT_DYN = 7   # raw dyn_prop motion classification (0..7, stored as float)

# v2 cache defaults (week 17, radar-primary detector): keep every return the
# hardware itself doesn't flag as invalid, and let the *network* learn clutter
# rejection from rcs/dyn_prop/dt features instead of thresholding them away.
# Measured cost of the v1 filters on the val coverage ceiling: car 73.4->62.6%,
# pedestrian 48.0->32.2% (scripts/radar/radar_coverage_oracle.py, 2026-07-02).
V2_NSWEEPS = 7
V2_MIN_RCS = -1e9
V2_VALID_DYN_PROPS = tuple(range(8))

# ---------------------------------------------------------------------------
# Validity and clutter filtering
# ---------------------------------------------------------------------------

def filter_radar_points(
    points: np.ndarray,
    min_rcs: float = DEFAULT_MIN_RCS,
    valid_dyn_props: tuple[int, ...] = DEFAULT_VALID_DYN_PROPS,
) -> np.ndarray:
    """
    Filter a raw (18, N) radar point array.

    Three sequential filters are applied:

    1. **Validity filter** (invalid_state == 0)
       This is the hardware's own validity flag. Any point with
       invalid_state != 0 is known bad — remove it unconditionally.

    2. **Dynamic property filter** (dyn_prop in valid_dyn_props)
       The radar firmware classifies each detection's motion state.
       We drop categories that are typically clutter or low-confidence.

    3. **RCS filter** (rcs >= min_rcs)
       Low radar cross-section → weak reflector → likely clutter.
       Vehicles produce 5–25 dBsm. Values below -5 dBsm are almost
       always noise.

    Args:
        points:          (18, N) array from RadarPointCloud.points
        min_rcs:         Minimum RCS in dBsm. Points below are discarded.
        valid_dyn_props: Tuple of dyn_prop integer values to keep.

    Returns:
        (18, M) filtered array where M <= N.
    """
    # Field indices
    IDX_INVALID_STATE = 14
    IDX_DYN_PROP = 3
    IDX_MIN_RCS = 5

    # Build boolean mask -> True = keep
    mask_valid = points[IDX_INVALID_STATE] == 0
    mask_dyn = np.isin(points[IDX_DYN_PROP].astype(int), valid_dyn_props)
    mask_rcs = points[IDX_MIN_RCS] >= min_rcs

    mask = mask_valid & mask_dyn & mask_rcs
    return points[:, mask]

# ---------------------------------------------------------------------------
# Multi-sweep accumulation (one sensor at a time)
# ---------------------------------------------------------------------------
def accumulate_sweeps(
    nusc: NuScenes,
    sample_data_token: str,
    nsweeps: int = DEFAULT_NSWEEPS,
    min_rcs: float = DEFAULT_MIN_RCS,
    valid_dyn_props: tuple[int, ...] = DEFAULT_VALID_DYN_PROPS,
    ref_ego_pose_token: str | None = None,
    with_dt: bool = False,
    ref_timestamp_us: int | None = None,
) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
    """
    Accumulate `nsweeps` radar sweeps for a **single sensor channel**,
    transforming all past points into the reference ego frame.

    Why accumulate?
    ---------------
    A single radar frame has ~5–30 points. That is far too sparse for
    clustering or feature extraction. By stacking the last N frames we
    get ~100–300 points with essentially no added latency (the past sweeps
    are already on disk from earlier in the sequence).

    Ego-motion compensation
    -----------------------
    The ego vehicle moves between sweeps, so naively stacking past points
    would place them in the wrong position. We compensate by chaining three
    transforms for each past sweep:

        past_sensor_frame
            → past_ego_frame       (calibrated_sensor extrinsics — constant)
            → global_frame         (past ego_pose)
            → reference_ego_frame  (inverse of reference ego_pose)

    This correctly places all historical points in the reference frame.

    Note: dynamic objects (other cars, pedestrians) will appear as trails
    because they ALSO moved between sweeps. This is expected and acceptable
    for detection — the most recent sweep dominates the cluster center.

    Args:
        nusc:              NuScenes instance.
        sample_data_token: Token of the CURRENT radar sample_data record.
        nsweeps:           Number of sweeps to accumulate (includes current).
        min_rcs:           Passed to filter_radar_points.
        valid_dyn_props:   Passed to filter_radar_points.
        ref_ego_pose_token: ego_pose token to use as the output reference
            frame. Pass the LIDAR_TOP keyframe's ego_pose_token so all five
            radars land in the same frame as GT boxes. Defaults to the
            radar's own ego_pose_token (legacy behaviour).
        with_dt:   If True, also return a (M,) array of per-point sweep ages
            in seconds (ref_timestamp_us minus the sweep's timestamp).
        ref_timestamp_us: Reference timestamp (microseconds) for the dt
            computation. Pass the LIDAR keyframe's timestamp so dt is
            consistent across the five radars. Defaults to the current
            radar sample_data's own timestamp.

    Returns:
        (18, M) array — accumulated, filtered points in reference ego frame.
        Returns a (18, 0) empty array if no valid points exist.
        If ``with_dt`` is True, returns a tuple ``(points, dt)`` instead.
    """
    current_sd = nusc.get('sample_data', sample_data_token)
    # Use the LIDAR keyframe ego pose as the reference frame so all five
    # radar sensors are consistent with GT box coordinates. Falling back to
    # the radar's own ego pose is kept for callers that don't have the token.
    ref_token = ref_ego_pose_token or current_sd['ego_pose_token']
    current_ep = nusc.get('ego_pose', ref_token)
    current_ego_t = np.array(current_ep['translation'])
    current_ego_r = Quaternion(current_ep['rotation'])

    # Get the sensor's calibration (constant for all sweeps of this sensor)
    # The sensor is rigidly mounted on the vehicle, so extrinsics never change
    cs = nusc.get('calibrated_sensor', current_sd['calibrated_sensor_token'])
    sensor_t = np.array(cs['translation'])
    sensor_r = Quaternion(cs['rotation'])

    accumulated = []
    dts = []
    ref_ts = (ref_timestamp_us if ref_timestamp_us is not None
              else current_sd['timestamp'])
    sd_token = sample_data_token

    for _ in range(nsweeps):
        if not sd_token:
            break # Reached the beginning of the sequence

        sd = nusc.get('sample_data', sd_token)

        # Load raw point cloud for this sweep
        pc = RadarPointCloud.from_file(nusc.get_sample_data_path(sd['token']))

        # Apply filters
        pc.points = filter_radar_points(
            pc.points, min_rcs=min_rcs, valid_dyn_props=valid_dyn_props
        )

        if pc.points.shape[1] > 0:
            # Step A: sensor frame -> past ego frame
            # (Apply rigid sensor mount: rotate then translate)
            pc.rotate(sensor_r.rotation_matrix)
            pc.translate(sensor_t)

            # Step B: past ego frame -> global frame
            # (Apply the ego_pose recorded at this sweep's timestamp
            sweep_ep = nusc.get('ego_pose', sd['ego_pose_token'])
            sweep_ego_t = np.array(sweep_ep['translation'])
            sweep_ego_r = Quaternion(sweep_ep['rotation'])
            pc.rotate(sweep_ego_r.rotation_matrix)
            pc.translate(sweep_ego_t)

            # Step C: global frame -> current ego frame
            # (Undo the current ego_pose: translate first, then rotate inverse)
            pc.translate(-current_ego_t)
            pc.rotate(current_ego_r.inverse.rotation_matrix)

            # Rotate velocity fields through the same chain.
            # pc.rotate() only transforms positions (rows 0-2).
            # Compensated velocities vx_comp (row 8) and vy_comp (row 9)
            # must be rotated manually so all sensors share the ego frame.
            R_vel = (current_ego_r.inverse.rotation_matrix
                     @ sweep_ego_r.rotation_matrix
                     @ sensor_r.rotation_matrix)
            v3 = np.vstack([pc.points[[8, 9], :],
                            np.zeros((1, pc.points.shape[1]))])
            v3 = R_vel @ v3
            pc.points[8] = v3[0]
            pc.points[9] = v3[1]

            accumulated.append(pc.points)
            if with_dt:
                dts.append(np.full(
                    pc.points.shape[1],
                    (ref_ts - sd['timestamp']) * 1e-6,
                    dtype=np.float64,
                ))

        # Follow the 'prev' link to the previous sweep of this sensor
        sd_token = sd['prev']

    if accumulated:
        pts = np.concatenate(accumulated, axis=1)
        if with_dt:
            return pts, np.concatenate(dts)
        return pts
    if with_dt:
        return np.zeros((18, 0)), np.zeros((0,))
    return np.zeros((18, 0))

# ---------------------------------------------------------------------------
# Merge all 5 sensors → compact output
# ---------------------------------------------------------------------------
def get_radar_pointcloud(
    nusc: NuScenes,
    sample_token: str,
    nsweeps: int = DEFAULT_NSWEEPS,
    min_rcs: float = DEFAULT_MIN_RCS,
    valid_dyn_props: tuple[int, ...] = DEFAULT_VALID_DYN_PROPS,
    layout: str = 'v1',
) -> np.ndarray:
    """
    Full radar preprocessing pipeline for a single nuScenes sample.

    Runs the complete pipeline for all 5 radar sensors and returns a
    compact, ready-to-use point cloud in the ego vehicle frame.

    Args:
        nusc:          NuScenes instance.
        sample_token:  Token of the keyframe sample.
        nsweeps:       Number of sweeps to accumulate per sensor.
        min_rcs:       RCS filter threshold (dBsm).
        valid_dyn_props: dyn_prop values to keep.
        layout:        'v1' (default) or 'v2'. v2 appends two rows the
            radar-primary detector needs: per-point sweep age ``dt`` (s,
            relative to the LIDAR keyframe timestamp) and the raw
            ``dyn_prop`` motion class as a *feature*. Rows 0-5 are
            identical between layouts.

    Returns:
        (6, M) float32 array with rows:
            [0] x        — forward (m)
            [1] y        — left (m)
            [2] z        — up (m)
            [3] vx_comp  — compensated velocity X (m/s)
            [4] vy_comp  — compensated velocity Y (m/s)
            [5] rcs      — radar cross section (dBsm)
        For ``layout='v2'``, an (8, M) array with two extra rows:
            [6] dt       — sweep age (s), 0 for the newest sweep
            [7] dyn_prop — motion classification (0..7, as float)

        All coordinates are in the **LIDAR keyframe ego frame** of the given sample.
        Returns shape (6, 0) / (8, 0) if no valid points exist after filtering.
    """
    if layout not in ('v1', 'v2'):
        raise ValueError(f"layout must be 'v1' or 'v2', got {layout!r}")
    n_rows = 6 if layout == 'v1' else 8

    sample = nusc.get('sample', sample_token)
    # Use the LIDAR_TOP keyframe ego pose as the common reference frame so
    # all five radars and GT boxes (LiDARInstance3DBoxes) share the same frame.
    lidar_sd = nusc.get('sample_data', sample['data']['LIDAR_TOP'])
    lidar_ego_pose_token = lidar_sd['ego_pose_token']

    all_18field = []
    all_dt = []

    for channel in RADAR_CHANNELS:
        sd_token = sample['data'][channel]
        pts = accumulate_sweeps(
            nusc, sd_token,
            nsweeps=nsweeps,
            min_rcs=min_rcs,
            valid_dyn_props=valid_dyn_props,
            ref_ego_pose_token=lidar_ego_pose_token,
            with_dt=(layout == 'v2'),
            ref_timestamp_us=lidar_sd['timestamp'],
        )
        if layout == 'v2':
            pts, dt = pts
            all_dt.append(dt)
        all_18field.append(pts)

    if not all_18field:
        return np.zeros((n_rows, 0), dtype=np.float32)

    pts_18 = np.concatenate(all_18field, axis=1)  # (18, M)

    if pts_18.shape[1] == 0:
        return np.zeros((n_rows, 0), dtype=np.float32)

    # Extract the 6 fields we care about for detection and fusion.
    # Fields 8 and 9 are the EGO-COMPENSATED velocities — use these as recommended in docs,
    # not the raw vx/vy (fields 6/7) which include ego vehicle motion.
    rows = [
        pts_18[0],  # x
        pts_18[1],  # y
        pts_18[2],  # z
        pts_18[8],  # vx_comp
        pts_18[9],  # vy_comp
        pts_18[5],  # rcs
    ]
    if layout == 'v2':
        rows.append(np.concatenate(all_dt))  # dt
        rows.append(pts_18[3])               # dyn_prop
    return np.stack(rows, axis=0).astype(np.float32)  # (n_rows, M)

# ---------------------------------------------------------------------------
# Convenience: iterate over all samples
# ---------------------------------------------------------------------------

def iterate_samples(nusc: NuScenes):
    """
    Yield all sample tokens in the dataset, in scene order
    Usage:
        for token in iterate_samples(nusc):
            pts = get_radar_pointcloud(nusc, token)
    """
    for scene in nusc.scene:
        token = scene['first_sample_token']
        while token:
            yield token
            sample = nusc.get('sample', token)
            token = sample['next']

# ---------------------------------------------------------------------------
# Quick test (run as a script: python scripts/radar/radar_preprocess.py)
# ---------------------------------------------------------------------------
if __name__ == '__main__':
    import os
    from pathlib import Path

    # Find the project root
    root = Path(__file__).resolve().parent.parent.parent
    os.chdir(root)

    print('Loading nuScenes mini...')
    nusc = NuScenes(version='v1.0-mini', dataroot='data/nuscenes-mini', verbose=False)

    # Use the 5th sample of scene 0 so there are previous sweeps available
    # The first sample has no 'prev' links, so all nsweeps would give the same count
    sample_token = nusc.scene[0]['first_sample_token']
    for _ in range (5):
        sample_token = nusc.get('sample', sample_token)['next']
    print(f' Sample token (5th in scene 0): {sample_token}')

    for nsweeps in [1, 3, 6]:
        pts = get_radar_pointcloud(nusc, sample_token, nsweeps=nsweeps)
        speed = np.sqrt(pts[OUT_VX]**2 + pts[OUT_VY]**2)
        print(
            f'  nsweeps={nsweeps}: {pts.shape[1]:4d} points | '
            f'rcs mean={pts[OUT_RCS].mean():5.1f} dBsm | '
            f'speed mean={speed.mean():4.1f} m/s  max={speed.max():5.1f} m/s'
        )

    print('\nSmoke test passed.')






