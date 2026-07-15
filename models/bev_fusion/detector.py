"""
BEVFusionDetector — top-level glue for camera + radar mid-level BEV fusion.

Subclasses ``Base3DDetector`` (the mmdet3d 1.4 abstract base) and
implements the three required entry points:

    forward(mode='loss')    -> dict of losses (training)
    forward(mode='predict') -> list[Det3DDataSample] (inference)
    forward(mode='tensor')  -> raw BEV feature (debug)

Inputs (``batch_inputs_dict``)
------------------------------
- ``imgs``       : (B, N, 3, H, W) — multi-view images.
- ``radar_bev``  : (B, Cr, Hb, Wb) — rasterised radar BEV tensor.
- ``cam2img``    : (B, N, 3, 3) — per-camera intrinsics.
- ``cam2ego``    : (B, N, 4, 4) — per-camera cam→ego extrinsics.

The intrinsics + extrinsics live in the inputs dict because they're
sample-specific *but not* per-pixel — the data preprocessor leaves them
untouched.

GT supervision
--------------
The detection head (``CenterHead``) handles target assignment and loss
computation natively. We just hand it the fused BEV feature and the
``batch_data_samples``.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple, Union

import torch
import torch.nn as nn
from torch import Tensor

from mmdet3d.models.detectors.base import Base3DDetector
from mmdet3d.registry import MODELS
from mmdet3d.structures import Det3DDataSample


@MODELS.register_module()
class BEVFusionDetector(Base3DDetector):
    """Camera + radar BEV fusion detector.

    Args:
        img_backbone:       mmdet/mmcv backbone config (e.g. ResNet-50).
        img_neck:           FPN config (output of the image branch).
        view_transform:     ``LSSViewTransform`` config.
        radar_encoder:      ``RadarBEVEncoder`` config.
        fusion_neck:        ``BEVFusionNeck`` config.
        bbox_head:          ``CenterHead`` config (CenterPoint-style head).
        img_feat_idx:       Which scale of the FPN output to use as
                            input to the view transform.
        radar_branch_zero:  If True, multiply the radar feature by 0
                            before fusion. Used for the "camera-only
                            ablation in full training" sanity check
                            (Verification Checklist item #6).
        train_cfg / test_cfg: forwarded to ``bbox_head``.
    """

    def __init__(
        self,
        img_backbone: dict,
        img_neck: dict,
        view_transform: dict,
        radar_encoder: dict,
        fusion_neck: dict,
        bbox_head: dict,
        bev_backbone: Optional[dict] = None,
        radar_splat_back: Optional[dict] = None,
        enable_conditioning: bool = False,
        img_feat_idx: int = 0,
        radar_branch_zero: bool = False,
        depth_loss_weight: float = 0.0,
        depth_loss_warmup_iters: int = 1000,
        train_cfg: Optional[dict] = None,
        test_cfg: Optional[dict] = None,
        init_cfg: Optional[dict] = None,
        data_preprocessor: Optional[dict] = None,
    ) -> None:
        super().__init__(
            init_cfg=init_cfg, data_preprocessor=data_preprocessor)

        self.img_backbone = MODELS.build(img_backbone)
        self.img_neck = MODELS.build(img_neck)
        self.view_transform = MODELS.build(view_transform)
        self.radar_encoder = MODELS.build(radar_encoder)
        self.fusion_neck = MODELS.build(fusion_neck)
        self.bev_backbone = MODELS.build(bev_backbone) if bev_backbone else None
        # Phase B: radar splat-back module. Active only when
        # ``enable_conditioning`` is True. The view_transform must be
        # configured with ``radar_feat_channels`` matching the splat-back
        # output channel count (== ``radar_encoder.out_channels``).
        self.radar_splat_back = (
            MODELS.build(radar_splat_back) if radar_splat_back else None)
        self.enable_conditioning = bool(enable_conditioning)

        # Forward train/test cfg into the head (CenterPoint convention).
        if train_cfg is not None:
            bbox_head = {**bbox_head, 'train_cfg': train_cfg}
        if test_cfg is not None:
            bbox_head = {**bbox_head, 'test_cfg': test_cfg}
        self.bbox_head = MODELS.build(bbox_head)

        self.img_feat_idx = int(img_feat_idx)
        self.radar_branch_zero = bool(radar_branch_zero)
        self.depth_loss_weight = float(depth_loss_weight)
        self.depth_loss_warmup_iters = int(depth_loss_warmup_iters)
        # Iter counter for the depth-loss linear warmup. Buffer (not a
        # parameter) so DDP keeps it in sync; non-persistent so it
        # restarts from 0 on resume — the warmup is short and
        # restarting is harmless.
        self.register_buffer(
            '_depth_loss_iter',
            torch.zeros((), dtype=torch.long),
            persistent=False,
        )

        # Cache of the most recent per-camera depth softmax. ``loss()``
        # consumes it; ``predict()`` ignores it. Stashed instead of
        # threaded through return values to keep ``extract_feat``'s
        # public signature stable for callers that just want the BEV.
        self._last_cam_depth: Optional[Tensor] = None

    # ------------------------------------------------------------------
    # Feature extraction
    # ------------------------------------------------------------------
    def extract_img_feat(self, imgs: Tensor) -> Tensor:
        """Run a (B, N, C, H, W) image stack through backbone + neck and
        return the FPN scale at ``self.img_feat_idx`` reshaped to
        (B, N, C_feat, Hf, Wf)."""
        B, N, C, H, W = imgs.shape
        x = imgs.view(B * N, C, H, W)
        feats = self.img_backbone(x)
        feats = self.img_neck(feats)
        # img_neck always returns a tuple of multi-scale tensors.
        feat = feats[self.img_feat_idx]  # (B*N, C', Hf, Wf)
        Cf, Hf, Wf = feat.shape[1:]
        return feat.view(B, N, Cf, Hf, Wf)

    def extract_feat(
        self,
        batch_inputs_dict: Dict[str, Tensor],
    ) -> Tuple[Tensor, Tensor, Tensor]:
        """Run all three branches and return the fused BEV feature.

        Returns:
            fused_bev: (B, C_fuse, Hb, Wb)
            cam_bev:   (B, C_cam,  Hb, Wb) — kept for ablation hooks
            rad_bev:   (B, C_rad,  Hb, Wb)
        """
        imgs = batch_inputs_dict['imgs']            # (B, N, 3, H, W)
        radar_bev = batch_inputs_dict['radar_bev']  # (B, Cr, Hb, Wb)
        cam2img = batch_inputs_dict['cam2img']      # (B, N, 3, 3)
        cam2ego = batch_inputs_dict['cam2ego']      # (B, N, 4, 4)
        radar_points = batch_inputs_dict.get('radar_points', None)
        radar_mask = batch_inputs_dict.get('radar_points_mask', None)

        # 1. Per-camera image features.
        img_feat = self.extract_img_feat(imgs)

        # 2. Radar -> BEV (must run before view_transform for Phase B so
        #    its output can be splatted back into per-camera image space).
        rad_bev = self.radar_encoder(radar_bev)
        if self.radar_branch_zero:
            rad_bev = rad_bev * 0.0  # ablation: keep BN running stats live

        # 3. Phase B: splat the encoded radar BEV back to per-camera image
        #    cells. ``radar_feat`` is concat'd with the FPN feature before
        #    the LSS depth net. Off when ``enable_conditioning=False``
        #    (Phase A behaviour) or when radar points aren't in the batch.
        #    Run in fp32 — same autocast fence as compute_depth_loss: the
        #    splat-back inverts cam2ego and runs a per-cell scatter_add
        #    of radar features, both safer outside autocast.
        radar_feat = None
        if (self.enable_conditioning
                and self.radar_splat_back is not None
                and radar_points is not None
                and radar_mask is not None):
            with torch.cuda.amp.autocast(enabled=False):
                radar_feat = self.radar_splat_back(
                    rad_bev.float(), radar_points, radar_mask,
                    cam2img, cam2ego)

        # 4. Camera -> BEV. The view transform also returns the per-camera
        #    depth softmax used by the Phase A depth loss.
        cam_bev, cam_depth = self.view_transform(
            img_feat, cam2img, cam2ego, radar_feat=radar_feat)
        self._last_cam_depth = cam_depth

        # 5. Fuse + optional BEV backbone. The trunk runs in fp32: with
        # zero-init bn2.weight the gradient through bn2 is exactly 0 at
        # iter 0, which under autocast triggers fp16 underflow and lets
        # the dynamic loss-scale climb until activations overflow in the
        # backward recompute (worsened by `use_reentrant=False`
        # checkpointing). Forcing fp32 here costs little — three small
        # 256-ch ResBlocks — and removes the loss-scale feedback loop.
        fused = self.fusion_neck(cam_bev, rad_bev)
        if self.bev_backbone is not None:
            with torch.cuda.amp.autocast(enabled=False):
                fused = self.bev_backbone(fused.float())
        return fused, cam_bev, rad_bev

    # ------------------------------------------------------------------
    # forward / loss / predict
    # ------------------------------------------------------------------
    def loss(
        self,
        batch_inputs_dict: Dict[str, Tensor],
        batch_data_samples: List[Det3DDataSample],
        **kwargs,
    ) -> Dict[str, Tensor]:
        fused_bev, _, _ = self.extract_feat(batch_inputs_dict)
        # CenterHead expects a list of feature levels — wrap in a list.
        # Disable autocast for the head: mmdet3d's clip_sigmoid clamps to
        # 1-1e-4, but in fp16 that rounds to 1.0 (fp16 spacing at 1.0 is
        # ~9.8e-4 > 1e-4), so saturated logits leak through and produce
        # log(0) = -inf in GaussianFocalLoss. Running head + loss in fp32
        # lets the eps actually clamp.
        with torch.cuda.amp.autocast(enabled=False):
            losses = self.bbox_head.loss(
                [fused_bev.float()], batch_data_samples, **kwargs)

        # Phase A: radar-projected depth supervision. Linear warmup of the
        # weight from 0 → ``depth_loss_weight`` over the first
        # ``depth_loss_warmup_iters`` training iters keeps early training
        # stable while the depth net is still random-init.
        if (self.training
                and self.depth_loss_weight > 0.0
                and self._last_cam_depth is not None
                and 'radar_points' in batch_inputs_dict
                and 'radar_points_mask' in batch_inputs_dict):
            with torch.cuda.amp.autocast(enabled=False):
                loss_depth, n_targets = self.view_transform.compute_depth_loss(
                    self._last_cam_depth,
                    batch_inputs_dict['cam2img'],
                    batch_inputs_dict['cam2ego'],
                    batch_inputs_dict['radar_points'],
                    batch_inputs_dict['radar_points_mask'],
                )
            warmup = 1.0
            if self.depth_loss_warmup_iters > 0:
                t = self._depth_loss_iter.float()
                warmup = (t / float(self.depth_loss_warmup_iters)).clamp(max=1.0)
            losses['loss_depth'] = self.depth_loss_weight * warmup * loss_depth
            losses['depth_n_targets'] = n_targets.detach().float()
            self._depth_loss_iter += 1

        # Drop the cached softmax to free memory between steps.
        self._last_cam_depth = None
        return losses

    def predict(
        self,
        batch_inputs_dict: Dict[str, Tensor],
        batch_data_samples: List[Det3DDataSample],
        **kwargs,
    ) -> List[Det3DDataSample]:
        fused_bev, _, _ = self.extract_feat(batch_inputs_dict)
        with torch.cuda.amp.autocast(enabled=False):
            results_list_3d = self.bbox_head.predict(
                [fused_bev.float()], batch_data_samples, **kwargs)
        # nuScenes evaluator hard-limits 500 boxes per sample.
        # Keep top-scoring boxes if the head returns more.
        _MAX = 500
        for i, res in enumerate(results_list_3d):
            if len(res) > _MAX:
                _, keep = res.scores_3d.topk(_MAX)
                results_list_3d[i] = res[keep]
        self._last_cam_depth = None
        return self.add_pred_to_datasample(
            batch_data_samples, data_instances_3d=results_list_3d)

    def _forward(
        self,
        batch_inputs_dict: Dict[str, Tensor],
        batch_data_samples: Optional[List[Det3DDataSample]] = None,
        **kwargs,
    ):
        fused_bev, _, _ = self.extract_feat(batch_inputs_dict)
        self._last_cam_depth = None
        return fused_bev
