"""
nuscenes_radar_dataset.py — mmdet3d-compatible nuScenes dataset that
also yields a per-sample radar BEV tensor for the BEV fusion model.

Two pieces are registered here:

1. ``LoadRadarBEV`` (transform, registered with mmdet3d's TRANSFORMS)
   Reads the cached ``<radar_bev_dir>/<sample_token>.npy`` file written
   by ``scripts/radar/precompute_radar.py`` and rasterises the (6, M) ego-frame
   point cloud into a (Cr, Hb, Wb) tensor with channels:

       [count, mean vx, mean vy, mean rcs, max rcs]

   This is cheap pure-NumPy work — moving it into a transform keeps the
   dataset class itself tiny and lets the rasterisation parameters be
   set via the config.

2. ``NuScenesRadarDataset`` (dataset, registered with DATASETS)
   Subclass of mmdet3d's ``NuScenesDataset``. Adds the resolved
   ``radar_npy_path`` to every parsed data info dict so the
   ``LoadRadarBEV`` transform can find it. No other behaviour changes —
   GT loading, metric class, etc. all stay identical, so the
   ``configs/fcos3d_full.py`` pipeline is reusable.

Coordinate convention: the cached cloud is in the **LIDAR_TOP ego frame
at sample timestamp** (see ``scripts/radar/radar_preprocess.py``). The BEV
grid is centred on the ego with x forward, y left:

    col = floor((x + range_m) / res)
    row = floor((y + range_m) / res)
"""

from __future__ import annotations

import os
from typing import Any, Optional

import numpy as np

from mmdet3d.datasets.nuscenes_dataset import NuScenesDataset
from mmdet3d.evaluation.metrics.nuscenes_metric import NuScenesMetric
from mmdet3d.registry import DATASETS, METRICS, TRANSFORMS

from scripts.radar.radar_preprocess import OUT_X, OUT_Y, OUT_VX, OUT_VY, OUT_RCS


# Channel layout of the rasterised radar BEV tensor.
RADAR_CHANNELS_OUT = ('count', 'mean_vx', 'mean_vy', 'mean_rcs', 'max_rcs')
RADAR_NUM_CHANNELS = len(RADAR_CHANNELS_OUT)


def rasterise_radar_pointcloud(
    pts_6xM: np.ndarray,
    range_m: float = 51.2,
    res: float = 0.8,
) -> np.ndarray:
    """Rasterise a (6, M) ego-frame radar cloud to a (Cr, Hb, Wb) BEV tensor.

    Args:
        pts_6xM: Output of ``get_radar_pointcloud()`` — float32 array with
            rows ``[x, y, z, vx_comp, vy_comp, rcs]``.
        range_m: Half-extent of the BEV grid in metres (square grid).
        res:     Cell size in metres.

    Returns:
        ``(RADAR_NUM_CHANNELS, Hb, Wb)`` float32 tensor. Channels:
            0 — count of returns per cell
            1 — mean vx_comp per cell (0 if cell empty)
            2 — mean vy_comp per cell (0 if cell empty)
            3 — mean rcs per cell (0 if cell empty)
            4 — max rcs per cell (0 if cell empty)
    """
    hb = wb = int(round(2 * range_m / res))
    out = np.zeros((RADAR_NUM_CHANNELS, hb, wb), dtype=np.float32)

    if pts_6xM.size == 0 or pts_6xM.shape[1] == 0:
        return out

    x = pts_6xM[OUT_X]
    y = pts_6xM[OUT_Y]
    vx = pts_6xM[OUT_VX]
    vy = pts_6xM[OUT_VY]
    rcs = pts_6xM[OUT_RCS]

    col = np.floor((x + range_m) / res).astype(np.int64)
    row = np.floor((y + range_m) / res).astype(np.int64)
    keep = (col >= 0) & (col < wb) & (row >= 0) & (row < hb)
    if not np.any(keep):
        return out

    col = col[keep]; row = row[keep]
    vx = vx[keep]; vy = vy[keep]; rcs = rcs[keep]

    # Flat index for fast accumulation.
    flat = row * wb + col

    counts = np.bincount(flat, minlength=hb * wb).astype(np.float32)
    sum_vx = np.bincount(flat, weights=vx, minlength=hb * wb).astype(np.float32)
    sum_vy = np.bincount(flat, weights=vy, minlength=hb * wb).astype(np.float32)
    sum_rcs = np.bincount(flat, weights=rcs, minlength=hb * wb).astype(np.float32)

    nonzero = counts > 0
    mean_vx = np.zeros_like(counts)
    mean_vy = np.zeros_like(counts)
    mean_rcs = np.zeros_like(counts)
    mean_vx[nonzero] = sum_vx[nonzero] / counts[nonzero]
    mean_vy[nonzero] = sum_vy[nonzero] / counts[nonzero]
    mean_rcs[nonzero] = sum_rcs[nonzero] / counts[nonzero]

    # max rcs per cell — bincount only handles sums, so do a manual loop
    # via np.maximum.at, which is fine for typical M ~ a few hundred.
    max_rcs = np.zeros(hb * wb, dtype=np.float32)
    np.maximum.at(max_rcs, flat, rcs.astype(np.float32))

    out[0] = counts.reshape(hb, wb)
    out[1] = mean_vx.reshape(hb, wb)
    out[2] = mean_vy.reshape(hb, wb)
    out[3] = mean_rcs.reshape(hb, wb)
    out[4] = max_rcs.reshape(hb, wb)
    return out


# Maximum number of radar returns retained per sample for the raw-point
# branch (depth supervision + Phase B splat-back). Typical accumulated
# clouds are 100-300 returns; the 99th percentile sits well under 1024.
# Samples above the cap are randomly subsampled; samples below get
# zero-padded with a False entry in radar_points_mask.
MAX_RADAR_POINTS = 1024


@TRANSFORMS.register_module()
class LoadRadarPoints:
    """Load the raw radar point cloud cached at ``radar_npy_path`` and pack
    it into ``results`` as a fixed-shape padded tensor plus a boolean
    validity mask, ready to stack through ``Det3DDataPreprocessor``.

    Supports both cache layouts (``n_rows=6`` for v1, ``n_rows=8`` for the
    week-17 v2 cache with dt + dyn_prop) and, optionally, concatenation of
    the offline painted camera features cached at ``radar_paint_path``
    (a (K, M) float16 array, point-aligned with the v2 cache — see
    ``scripts/radar/paint_radar_cache.py``). Painted rows are appended AFTER the
    geometry rows, so the BEV flip/rotation transforms (which index rows
    0/1 and 3/4) stay valid.

    Outputs:
        results['radar_points']      — (n_rows [+ paint_rows], max_points)
                                       float32, LIDAR keyframe ego frame.
        results['radar_points_mask'] — (max_points,) bool, True at real
                                       returns, False at padding.

    Mirrors ``LoadRadarBEV`` for the failure-mode handling: optionally
    falls back to an empty (all-False mask) tensor when the cache file
    is absent.
    """

    def __init__(
        self,
        max_points: int = MAX_RADAR_POINTS,
        n_rows: int = 6,
        paint: bool = False,
        paint_rows: int = 14,
        paint_shuffle: bool = False,
        to_lidar_frame: bool = False,
        allow_missing: bool = False,
    ) -> None:
        self.max_points = int(max_points)
        self.n_rows = int(n_rows)
        self.paint = bool(paint)
        self.paint_rows = int(paint_rows) if paint else 0
        # Ablation switch: permute the paint columns ACROSS POINTS within
        # each sample (fixed per-sample seed). Detaches camera semantics
        # from radar geometry while preserving both marginals — the
        # camera-content load-bearing test (scripts/ablation/shuffle_rpp.py).
        self.paint_shuffle = bool(paint_shuffle)
        # The cache is in the LIDAR-keyframe *ego* frame, but mmdet3d GT
        # boxes (and the camera extrinsics since the week-7 inv(lidar2cam)
        # fix) are in the LIDAR *sensor* frame — offset by lidar2ego,
        # a ~90° yaw + ~0.94 m on nuScenes. Measured through this very
        # pipeline (2026-07-03): car GT footprint coverage 25.2 % without
        # the transform vs 83.5 % with it. Every consumer that matches
        # radar points against gt_bboxes_3d MUST set this to True.
        # (Default False only to preserve the legacy v4 phase-B configs.)
        self.to_lidar_frame = bool(to_lidar_frame)
        self.allow_missing = bool(allow_missing)

    @property
    def _total_rows(self) -> int:
        return self.n_rows + self.paint_rows

    def _empty(self) -> tuple[np.ndarray, np.ndarray]:
        return (
            np.zeros((self._total_rows, self.max_points), dtype=np.float32),
            np.zeros((self.max_points,), dtype=bool),
        )

    def __call__(self, results: dict) -> dict:
        path = results.get('radar_npy_path', None)
        if path is None or not os.path.isfile(path):
            if self.allow_missing:
                pts, mask = self._empty()
                results['radar_points'] = pts
                results['radar_points_mask'] = mask
                return results
            raise FileNotFoundError(
                f'Radar point cache missing: {path!r}. '
                f'Run scripts/radar/precompute_radar.py first.')

        raw = np.load(path)  # (n_rows, M) float32
        if raw.ndim != 2 or raw.shape[0] != self.n_rows:
            raise ValueError(
                f'Expected ({self.n_rows}, M) radar points; got shape '
                f'{raw.shape} from {path!r}')

        if self.to_lidar_frame and raw.shape[1] > 0:
            l2e = (results.get('lidar_points') or {}).get('lidar2ego')
            if l2e is None:
                raise KeyError(
                    "results['lidar_points']['lidar2ego'] missing — "
                    'to_lidar_frame needs the standard nuScenes info dict.')
            T = np.asarray(l2e, dtype=np.float32)
            R, t = T[:3, :3], T[:3, 3]
            raw = raw.copy()
            raw[:3] = R.T @ (raw[:3] - t[:, None])          # ego -> lidar
            vel = np.vstack([raw[3:5], np.zeros((1, raw.shape[1]),
                                                dtype=raw.dtype)])
            raw[3:5] = (R.T @ vel)[:2]

        if self.paint:
            paint_path = results.get('radar_paint_path', None)
            if paint_path is None or not os.path.isfile(paint_path):
                raise FileNotFoundError(
                    f'Painted-feature cache missing: {paint_path!r}. '
                    f'Run scripts/radar/paint_radar_cache.py first.')
            paint = np.load(paint_path).astype(np.float32)  # (K, M)
            if paint.shape != (self.paint_rows, raw.shape[1]):
                raise ValueError(
                    f'Paint/points mismatch: paint {paint.shape} vs points '
                    f'{raw.shape} (expected ({self.paint_rows}, '
                    f'{raw.shape[1]})) for {paint_path!r}')
            if self.paint_shuffle and paint.shape[1] > 1:
                seed = hash(results.get('sample_idx', 0)) & 0xFFFFFFFF
                perm = np.random.RandomState(seed).permutation(paint.shape[1])
                paint = paint[:, perm]
            raw = np.concatenate([raw, paint], axis=0)

        M = raw.shape[1]
        pts = np.zeros((self._total_rows, self.max_points), dtype=np.float32)
        mask = np.zeros((self.max_points,), dtype=bool)

        if M > 0:
            if M > self.max_points:
                # Randomly subsample. Use numpy's default RNG so the
                # choice is reproducible across worker reseeding.
                idx = np.random.choice(M, self.max_points, replace=False)
                raw = raw[:, idx]
                M = self.max_points
            pts[:, :M] = raw.astype(np.float32, copy=False)
            mask[:M] = True

        results['radar_points'] = pts
        results['radar_points_mask'] = mask
        return results

    def __repr__(self) -> str:
        return (f'{type(self).__name__}(max_points={self.max_points}, '
                f'n_rows={self.n_rows}, paint={self.paint})')


@TRANSFORMS.register_module()
class LoadRadarBEV:
    """mmcv-style transform that loads the cached radar `.npy` for a
    sample and writes the rasterised BEV tensor under ``results['radar_bev']``.

    Expects each input ``results`` dict to carry ``'radar_npy_path'``
    (added by ``NuScenesRadarDataset.parse_data_info``).
    """

    def __init__(
        self,
        bev_range: float = 51.2,
        bev_res: float = 0.8,
        to_lidar_frame: bool = False,
        allow_missing: bool = False,
    ) -> None:
        self.bev_range = float(bev_range)
        self.bev_res = float(bev_res)
        # See LoadRadarPoints.to_lidar_frame: the cache is in the ego
        # frame, GT/camera are in the LIDAR sensor frame (~90° yaw +
        # ~0.94 m apart). Every v3x run rasterised the radar UNALIGNED
        # (measured car GT coverage 25 % vs 84 % aligned, 2026-07-03).
        # Default False preserves the historical configs; set True for
        # any new or re-run BEV fusion training.
        self.to_lidar_frame = bool(to_lidar_frame)
        self.allow_missing = bool(allow_missing)
        # Pre-compute grid size once — used for the empty fallback.
        self._hb = self._wb = int(round(2 * self.bev_range / self.bev_res))

    def __call__(self, results: dict) -> dict:
        path = results.get('radar_npy_path', None)
        if path is None or not os.path.isfile(path):
            if self.allow_missing:
                results['radar_bev'] = np.zeros(
                    (RADAR_NUM_CHANNELS, self._hb, self._wb), dtype=np.float32)
                return results
            raise FileNotFoundError(
                f'Radar BEV cache missing: {path!r}. '
                f'Run scripts/radar/precompute_radar.py first.')

        pts = np.load(path)  # (6, M) float32
        if self.to_lidar_frame and pts.shape[1] > 0:
            l2e = (results.get('lidar_points') or {}).get('lidar2ego')
            if l2e is None:
                raise KeyError(
                    "results['lidar_points']['lidar2ego'] missing — "
                    'to_lidar_frame needs the standard nuScenes info dict.')
            T = np.asarray(l2e, dtype=np.float32)
            R, t = T[:3, :3], T[:3, 3]
            pts = pts.copy()
            pts[:3] = R.T @ (pts[:3] - t[:, None])
            vel = np.vstack([pts[3:5], np.zeros((1, pts.shape[1]),
                                                dtype=pts.dtype)])
            pts[3:5] = (R.T @ vel)[:2]
        results['radar_bev'] = rasterise_radar_pointcloud(
            pts, range_m=self.bev_range, res=self.bev_res)
        return results

    def __repr__(self) -> str:
        return (f'{type(self).__name__}'
                f'(bev_range={self.bev_range}, bev_res={self.bev_res})')


@DATASETS.register_module()
class NuScenesRadarDataset(NuScenesDataset):
    """nuScenes dataset that also exposes a per-sample cached radar
    point cloud path. Identical to ``NuScenesDataset`` for everything
    else (GT loading, eval).

    Args:
        radar_bev_dir: Directory containing ``<sample_token>.npy`` files
            written by ``scripts/radar/precompute_radar.py``. Resolved against
            ``data_root`` if relative.
        radar_paint_dir: Optional directory containing the painted-feature
            ``<sample_token>.npy`` files written by
            ``scripts/radar/paint_radar_cache.py`` (point-aligned with the radar
            cache in ``radar_bev_dir``). When set, every data info also
            carries ``radar_paint_path``.
        allow_missing_radar: If True, silently fall back to a zero
            radar tensor when the cache file is absent. Useful only for
            smoke tests; the default raises.
        Other kwargs forwarded to ``NuScenesDataset``.
    """

    def __init__(
        self,
        *args,
        radar_bev_dir: str = 'radar_bev',
        radar_paint_dir: Optional[str] = None,
        allow_missing_radar: bool = False,
        **kwargs,
    ) -> None:
        self._radar_bev_dir = radar_bev_dir
        self._radar_paint_dir = radar_paint_dir
        self._allow_missing_radar = allow_missing_radar
        super().__init__(*args, **kwargs)

    def _resolve_dir(self, d: str) -> str:
        if os.path.isabs(d):
            return d
        return os.path.join(self.data_root or '', d)

    def parse_data_info(self, info: dict) -> Optional[dict[str, Any]]:  # type: ignore[override]
        data_info = super().parse_data_info(info)
        if data_info is None:
            return None
        token = info.get('token') or info.get('sample_idx')
        radar_dir = self._resolve_dir(self._radar_bev_dir)
        radar_path = os.path.join(radar_dir, f'{token}.npy')
        if (not os.path.isfile(radar_path)) and not self._allow_missing_radar:
            # Don't raise here — many MMEngine code paths probe info dicts
            # before training starts. Surface the error in LoadRadarBEV
            # instead, where the missing file actually matters.
            pass
        data_info['radar_npy_path'] = radar_path
        if self._radar_paint_dir is not None:
            paint_dir = self._resolve_dir(self._radar_paint_dir)
            data_info['radar_paint_path'] = os.path.join(
                paint_dir, f'{token}.npy')
        return data_info


@METRICS.register_module()
class SubsetNuScenesMetric(NuScenesMetric):
    """NuScenesMetric that evaluates correctly on an arbitrary subset of val samples.

    The official NuScenesEval asserts pred_tokens == gt_tokens (full val split).
    This subclass temporarily patches ``nuscenes.eval.detection.evaluate.load_gt``
    to return only the GT entries whose sample tokens appear in the predictions,
    making the evaluation work on any filtered pkl (e.g. the 50 hardest scenes).
    """

    def _evaluate_single(
        self,
        result_path: str,
        classes=None,
        result_name: str = 'pred_instances_3d',
    ):
        import json
        import nuscenes.eval.detection.evaluate as _nus_eval_mod
        from nuscenes.eval.common.data_classes import EvalBoxes

        with open(result_path) as _f:
            pred_tokens = set(json.load(_f)['results'].keys())

        _orig_load_gt = _nus_eval_mod.load_gt

        def _subset_load_gt(nusc, eval_set, box_cls, verbose=False):
            all_gt = _orig_load_gt(nusc, eval_set, box_cls, verbose=False)
            subset = EvalBoxes()
            for tok in sorted(pred_tokens):
                subset.add_boxes(tok, all_gt.boxes.get(tok, []))
            return subset

        _nus_eval_mod.load_gt = _subset_load_gt
        try:
            return super()._evaluate_single(
                result_path, classes=classes, result_name=result_name)
        finally:
            _nus_eval_mod.load_gt = _orig_load_gt
