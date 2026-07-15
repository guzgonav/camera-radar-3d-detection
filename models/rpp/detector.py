"""
RPPDetector — radar-primary painted-pillar detector (week 17).

The role inversion of the v3x BEV fusion line: radar points are the
geometric substrate (there is no camera-only path for the optimiser to
route through — no radar, no input), and the camera contributes only as
frozen, offline-painted per-point features. Every trainable parameter is
downstream of the radar cloud.

Inputs (``batch_inputs_dict``)
------------------------------
- ``radar_points``      : (B, C, N) padded points. Rows 0..7 = the v2 cache
                          geometry [x, y, z, vx, vy, rcs, dt, dyn_prop];
                          rows 8.. = painted camera features (optional).
- ``radar_points_mask`` : (B, N) bool validity mask.

Ablation switches
-----------------
- ``paint_zero``     : zero the painted rows at inference — the true
                       radar-only ablation (camera contribution test).
- ``paint_dropout``  : per-sample Bernoulli zeroing of the painted rows at
                       train time. Keeps the paint-zero ablation
                       in-distribution (the v3f modality-dropout lesson) and
                       regularises against paint over-reliance.
Radar-shuffle is a data-level ablation (permute cache files across samples,
see scripts/ablation/shuffle_rpp.py) — the model needs no switch for it.
"""

from __future__ import annotations

from typing import Dict, List, Optional

import torch
from torch import Tensor

from mmdet3d.models.data_preprocessors import Det3DDataPreprocessor
from mmdet3d.models.detectors.base import Base3DDetector
from mmdet3d.registry import MODELS
from mmdet3d.structures import Det3DDataSample


@MODELS.register_module()
class RPPDataPreprocessor(Det3DDataPreprocessor):
    """Det3DDataPreprocessor for an image-free pipeline.

    The parent's ``simple_process`` only knows imgs/points; this override
    casts the batch to the model device (in mmdet3d 1.4 that happens inside
    ``collate_data``, which we bypass) and stacks the per-sample radar
    tensors into a batch.
    """

    RADAR_KEYS = ('radar_points', 'radar_points_mask')

    def simple_process(self, data: dict, training: bool = False) -> dict:
        data = self.cast_data(data)
        inputs = data['inputs']
        batch: Dict[str, Tensor] = {}
        for key in self.RADAR_KEYS:
            val = inputs.get(key, None)
            if val is None:
                raise KeyError(
                    f'{key!r} missing from inputs — check that the pipeline '
                    'runs LoadRadarPoints and PackBEVFusionInputs.')
            batch[key] = torch.stack(val, dim=0) if isinstance(val, list) else val
        return {'inputs': batch, 'data_samples': data.get('data_samples')}


@MODELS.register_module()
class RPPDetector(Base3DDetector):
    """Radar-primary painted-pillar detector.

    Args:
        pillar_encoder:  ``RadarPillarEncoder`` config.
        bbox_head:       ``CenterHead`` config (CenterPoint-style).
        bev_backbone:    Optional ``BEVBackbone`` config (identity-init
                         residual trunk; expands the head's receptive field).
        geo_channels:    Number of geometry rows in ``radar_points`` (8 for
                         the v2 cache).
        paint_channels:  Number of painted rows appended after the geometry
                         (0 = unpainted radar-only model).
        paint_dropout:   Train-time per-sample probability of zeroing the
                         painted rows.
        paint_zero:      If True, always zero the painted rows (camera-off
                         ablation at eval).
    """

    def __init__(
        self,
        pillar_encoder: dict,
        bbox_head: dict,
        bev_backbone: Optional[dict] = None,
        geo_channels: int = 8,
        paint_channels: int = 0,
        paint_dropout: float = 0.0,
        paint_zero: bool = False,
        train_cfg: Optional[dict] = None,
        test_cfg: Optional[dict] = None,
        init_cfg: Optional[dict] = None,
        data_preprocessor: Optional[dict] = None,
    ) -> None:
        super().__init__(
            init_cfg=init_cfg, data_preprocessor=data_preprocessor)

        self.pillar_encoder = MODELS.build(pillar_encoder)
        self.bev_backbone = MODELS.build(bev_backbone) if bev_backbone else None

        if train_cfg is not None:
            bbox_head = {**bbox_head, 'train_cfg': train_cfg}
        if test_cfg is not None:
            bbox_head = {**bbox_head, 'test_cfg': test_cfg}
        self.bbox_head = MODELS.build(bbox_head)

        self.geo_channels = int(geo_channels)
        self.paint_channels = int(paint_channels)
        self.paint_dropout = float(paint_dropout)
        self.paint_zero = bool(paint_zero)

        expected = self.geo_channels + self.paint_channels
        if self.pillar_encoder.in_channels != expected:
            raise ValueError(
                f'pillar_encoder.in_channels={self.pillar_encoder.in_channels} '
                f'!= geo_channels + paint_channels = {expected}')

    # ------------------------------------------------------------------
    # Feature extraction
    # ------------------------------------------------------------------
    def _apply_paint_ablation(self, points: Tensor) -> Tensor:
        """Zero the painted rows: always (paint_zero), or per-sample with
        probability ``paint_dropout`` at train time."""
        if self.paint_channels == 0:
            return points
        g = self.geo_channels
        if self.paint_zero:
            points = points.clone()
            points[:, g:g + self.paint_channels] = 0.0
            return points
        if self.training and self.paint_dropout > 0.0:
            B = points.shape[0]
            drop = (torch.rand(B, device=points.device)
                    < self.paint_dropout)                      # (B,)
            if drop.any():
                points = points.clone()
                points[drop, g:g + self.paint_channels] = 0.0
        return points

    def extract_feat(self, batch_inputs_dict: Dict[str, Tensor]) -> Tensor:
        points = batch_inputs_dict['radar_points']         # (B, C, N)
        mask = batch_inputs_dict['radar_points_mask']      # (B, N)
        points = self._apply_paint_ablation(points)
        bev = self.pillar_encoder(points, mask)            # (B, C', H, W)
        if self.bev_backbone is not None:
            # fp32 fence, same rationale as BEVFusionDetector: the
            # zero-init bn2 makes early-iter gradients vanish under fp16.
            with torch.cuda.amp.autocast(enabled=False):
                bev = self.bev_backbone(bev.float())
        return bev

    # ------------------------------------------------------------------
    # loss / predict / tensor
    # ------------------------------------------------------------------
    def loss(
        self,
        batch_inputs_dict: Dict[str, Tensor],
        batch_data_samples: List[Det3DDataSample],
        **kwargs,
    ) -> Dict[str, Tensor]:
        bev = self.extract_feat(batch_inputs_dict)
        # Head + loss in fp32: mmdet3d's clip_sigmoid eps (1e-4) is below
        # the fp16 spacing at 1.0, which lets saturated logits reach
        # log(0) in GaussianFocalLoss (same fence as BEVFusionDetector).
        with torch.cuda.amp.autocast(enabled=False):
            return self.bbox_head.loss(
                [bev.float()], batch_data_samples, **kwargs)

    def predict(
        self,
        batch_inputs_dict: Dict[str, Tensor],
        batch_data_samples: List[Det3DDataSample],
        **kwargs,
    ) -> List[Det3DDataSample]:
        bev = self.extract_feat(batch_inputs_dict)
        with torch.cuda.amp.autocast(enabled=False):
            results_list_3d = self.bbox_head.predict(
                [bev.float()], batch_data_samples, **kwargs)
        # nuScenes evaluator hard-limits 500 boxes per sample.
        _MAX = 500
        for i, res in enumerate(results_list_3d):
            if len(res) > _MAX:
                _, keep = res.scores_3d.topk(_MAX)
                results_list_3d[i] = res[keep]
        return self.add_pred_to_datasample(
            batch_data_samples, data_instances_3d=results_list_3d)

    def _forward(
        self,
        batch_inputs_dict: Dict[str, Tensor],
        batch_data_samples: Optional[List[Det3DDataSample]] = None,
        **kwargs,
    ):
        return self.extract_feat(batch_inputs_dict)
