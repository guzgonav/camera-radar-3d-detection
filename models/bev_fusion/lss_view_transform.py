"""
LSS (Lift-Splat-Shoot) view transform — pure PyTorch, no CUDA ops.

Given per-camera image features ``(B, N, C_in, Hf, Wf)`` and the
intrinsics + cam2ego extrinsics for each camera, produces a BEV feature
``(B, C_out, Hb, Wb)`` in the **ego frame**: a 1x1-conv depth_net predicts
a per-pixel depth softmax and a linear feature, their outer product lifts
each ``(u, v, d)`` cell to a 3D point via the frustum geometry, then voxel
pooling scatter-adds into the ego BEV grid. The frustum geometry only
depends on intrinsics + image size + depth discretisation (constant across
a training run), so it's built once and reused; per-sample ego-frame
coordinates come from chaining cam2ego.

Coordinate convention
---------------------
- Camera frame: x right, y down, z forward (standard pinhole).
- Ego frame: x forward, y left, z up (nuScenes ego/LIDAR_TOP frame).
- BEV grid: cell (row, col) <-> ego (y, x) via
      col = floor((x + range_m) / res)
      row = floor((y + range_m) / res)
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from mmdet3d.registry import MODELS


@MODELS.register_module()
class LSSViewTransform(nn.Module):
    """LSS view transform.

    Args:
        in_channels:   Number of input feature channels per camera (after FPN).
        out_channels:  Number of output feature channels lifted into 3D.
        feat_stride:   Downsampling factor between input image and feature map
                       (e.g. 8 if FPN P3 / image stride is 8).
        image_hw:      (H, W) of the **input image** to the backbone.
        depth_bins:    (d_min, d_max, n_bins) for the categorical depth grid.
                       Bins are linearly spaced in metres.
        bev_range:     Half-extent of the BEV grid (square) in metres.
        bev_res:       BEV cell size in metres.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int = 64,
        feat_stride: int = 8,
        image_hw: tuple[int, int] = (256, 704),
        depth_bins: tuple[float, float, int] = (1.0, 60.0, 64),
        bev_range: float = 51.2,
        bev_res: float = 0.8,
        radar_feat_channels: int = 0,
    ) -> None:
        super().__init__()

        self.in_channels = int(in_channels)
        self.out_channels = int(out_channels)
        self.feat_stride = int(feat_stride)
        self.image_hw = (int(image_hw[0]), int(image_hw[1]))
        self.bev_range = float(bev_range)
        self.bev_res = float(bev_res)
        # Width of the per-camera radar feature concatenated with the FPN
        # feature *before* the depth net. 0 disables Phase B conditioning
        # (the v3b code path).
        self.radar_feat_channels = int(radar_feat_channels)

        d_min, d_max, n_bins = depth_bins
        self.d_min = float(d_min)
        self.d_max = float(d_max)
        self.n_bins = int(n_bins)
        self.register_buffer(
            'depth_centres',
            torch.linspace(self.d_min, self.d_max, self.n_bins),
            persistent=False,
        )

        # Depth net: 1x1 conv emitting (D + C_out) channels per pixel.
        # Phase B: the input width grows by ``radar_feat_channels`` to
        # accept the splatted-back radar feature concatenated with FPN.
        self.depth_net = nn.Conv2d(
            self.in_channels + self.radar_feat_channels,
            self.n_bins + self.out_channels,
            kernel_size=1, padding=0)

        # Normalize per-ray feat before the lift. voxel_pooling sums K rays
        # per BEV cell — unbounded feat × variable K produces extreme
        # outliers that downstream BN can't tame (sparse outliers don't
        # dominate batch variance). Keeping per-ray feat unit-scale keeps
        # the post-splat output O(K) ≈ O(5–10).
        self.feat_norm = nn.BatchNorm2d(self.out_channels)

        # Feature map size derived from image size + stride.
        self.feat_hw = (
            self.image_hw[0] // self.feat_stride,
            self.image_hw[1] // self.feat_stride,
        )
        self.bev_hw = (
            int(round(2 * self.bev_range / self.bev_res)),
            int(round(2 * self.bev_range / self.bev_res)),
        )

        # Frustum points in **camera frame** — (D, Hf, Wf, 3).
        # Reused for every camera (only intrinsics differ between cameras).
        self.register_buffer(
            'frustum',
            self._build_frustum(),
            persistent=False,
        )

    def _build_frustum(self) -> torch.Tensor:
        """Build a (D, Hf, Wf, 3) tensor of (u, v, d) points in pixel
        coordinates and depth metres. The actual back-projection uses
        the intrinsics provided per-sample in ``forward``.
        """
        Hf, Wf = self.feat_hw
        ds = self.depth_centres.view(-1, 1, 1).expand(self.n_bins, Hf, Wf)

        # Sample pixel centres of the *input image* corresponding to the
        # feature-map grid (account for stride).
        u_lin = torch.linspace(
            0.5, self.image_hw[1] - 0.5, Wf,
            dtype=torch.float32)
        v_lin = torch.linspace(
            0.5, self.image_hw[0] - 0.5, Hf,
            dtype=torch.float32)
        vv, uu = torch.meshgrid(v_lin, u_lin, indexing='ij')
        uu = uu.expand(self.n_bins, Hf, Wf)
        vv = vv.expand(self.n_bins, Hf, Wf)

        return torch.stack([uu, vv, ds], dim=-1)  # (D, Hf, Wf, 3)

    # ------------------------------------------------------------------
    # Geometry: image (u, v, d) -> ego (x, y, z)
    # ------------------------------------------------------------------
    def get_geometry(
        self,
        cam2img: torch.Tensor,   # (B, N, 3, 3) or (B, N, 4, 4)
        cam2ego: torch.Tensor,   # (B, N, 4, 4)
    ) -> torch.Tensor:
        """Transform the frustum points into the ego frame.

        Returns:
            ``(B, N, D, Hf, Wf, 3)`` in ego frame.
        """
        B, N = cam2img.shape[:2]
        D, Hf, Wf, _ = self.frustum.shape

        if cam2img.shape[-1] == 4:
            cam2img = cam2img[..., :3, :3]  # (B, N, 3, 3)

        # frustum: (D, Hf, Wf, 3) -> expand to (B, N, D, Hf, Wf, 3)
        pts = self.frustum.view(1, 1, D, Hf, Wf, 3).expand(B, N, D, Hf, Wf, 3)

        # Convert (u, v, d) -> (u*d, v*d, d). Note 'd' is along the
        # camera optical axis (z). pinhole back-projection:
        # X_cam = K^-1 . [u*d, v*d, d]^T
        u, v, d = pts[..., 0], pts[..., 1], pts[..., 2]
        pix = torch.stack([u * d, v * d, d], dim=-1)  # (..., 3)

        K_inv = torch.linalg.inv(cam2img)  # (B, N, 3, 3)
        X_cam = torch.einsum('bnij,bndhwj->bndhwi', K_inv, pix)

        # Camera -> ego (rotation + translation)
        R = cam2ego[..., :3, :3]   # (B, N, 3, 3)
        t = cam2ego[..., :3, 3]    # (B, N, 3)
        X_ego = torch.einsum('bnij,bndhwj->bndhwi', R, X_cam) + \
            t.view(B, N, 1, 1, 1, 3)
        return X_ego

    # ------------------------------------------------------------------
    # Voxel pooling: scatter (C, D, Hf, Wf) features into a BEV grid
    # ------------------------------------------------------------------
    def voxel_pooling(
        self,
        feats: torch.Tensor,   # (B, N, C, D, Hf, Wf)
        geom: torch.Tensor,    # (B, N, D, Hf, Wf, 3) in ego frame
    ) -> torch.Tensor:
        """Mean-pool features into the ego BEV grid.

        Sum-pooling makes per-cell magnitude scale with the number of
        frustum rays K landing in the cell — K varies from 0 to ~50 across
        the BEV (multi-camera overlap, near-camera depth fan), producing
        sparse outliers downstream BN can't tame. Dividing by K (mean
        pooling) removes that scale dependence; cells without rays stay
        zero.
        """
        B, N, C, D, Hf, Wf = feats.shape
        Hb, Wb = self.bev_hw

        x = geom[..., 0]
        y = geom[..., 1]
        col = torch.floor((x + self.bev_range) / self.bev_res).long()
        row = torch.floor((y + self.bev_range) / self.bev_res).long()
        valid = (col >= 0) & (col < Wb) & (row >= 0) & (row < Hb)

        M = N * D * Hf * Wf
        feats_bcm = feats.permute(0, 2, 1, 3, 4, 5).reshape(B, C, M)

        flat_idx = (row * Wb + col).reshape(B, M)
        valid = valid.reshape(B, M)
        sentinel = Hb * Wb
        flat_idx = torch.where(valid, flat_idx, torch.full_like(flat_idx, sentinel))

        out = feats.new_zeros(B, C, Hb * Wb + 1)
        # Per-batch ray count for each BEV cell — sentinel cell counts get
        # discarded along with the sentinel feature column.
        counts = feats.new_zeros(B, 1, Hb * Wb + 1)
        ones = feats.new_ones(B, 1, M)
        for b in range(B):
            out[b].index_add_(1, flat_idx[b], feats_bcm[b])
            counts[b].index_add_(1, flat_idx[b], ones[b])
        out = out[:, :, :Hb * Wb].view(B, C, Hb, Wb)
        counts = counts[:, :, :Hb * Wb].view(B, 1, Hb, Wb)
        # Avoid divide-by-zero in cells with no rays — those stay at 0.
        return out / counts.clamp(min=1.0)

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------
    def forward(
        self,
        x: torch.Tensor,                                    # (B, N, C_in, Hf, Wf)
        cam2img: torch.Tensor,                              # (B, N, 3, 3) or (B, N, 4, 4)
        cam2ego: torch.Tensor,                              # (B, N, 4, 4)
        radar_feat: Optional[torch.Tensor] = None,          # (B, N, C_radar, Hf, Wf), Phase B
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Lift per-camera features to a shared ego BEV feature map.

        Returns:
            bev:   ``(B, C_out, Hb, Wb)``
            depth: ``(B, N, D, Hf, Wf)`` — softmaxed depth distribution,
                   used by the Phase A radar-projected depth supervision.
        """
        B, N, C_in, Hf, Wf = x.shape
        assert C_in == self.in_channels, \
            f'in_channels mismatch: got {C_in} vs configured {self.in_channels}'
        assert (Hf, Wf) == self.feat_hw, \
            f'feature size mismatch: got {(Hf, Wf)} vs expected {self.feat_hw}'

        # Depth net + split. Optionally concat radar conditioning (Phase B).
        x_in = x.view(B * N, C_in, Hf, Wf)
        if self.radar_feat_channels > 0:
            if radar_feat is None:
                raise ValueError(
                    'radar_feat_channels > 0 but no radar_feat passed to forward().')
            assert radar_feat.shape[:2] == (B, N) and \
                radar_feat.shape[2] == self.radar_feat_channels, \
                f'radar_feat shape mismatch: got {tuple(radar_feat.shape)}'
            r_in = radar_feat.reshape(B * N, self.radar_feat_channels, Hf, Wf)
            x_in = torch.cat([x_in, r_in], dim=1)

        depth_feat = self.depth_net(x_in)  # (B*N, D + C_out, Hf, Wf)
        depth_logits = depth_feat[:, :self.n_bins]              # (B*N, D, Hf, Wf)
        feat = depth_feat[:, self.n_bins:]                       # (B*N, C_out, Hf, Wf)
        feat = F.relu(self.feat_norm(feat), inplace=True)

        depth = F.softmax(depth_logits, dim=1)                  # (B*N, D, Hf, Wf)

        # Outer product: feat (..,C,1,..) * depth (..,1,D,..) -> (..,C,D,..)
        x_lifted = feat.unsqueeze(2) * depth.unsqueeze(1)
        x_lifted = x_lifted.view(B, N, self.out_channels, self.n_bins, Hf, Wf)

        geom = self.get_geometry(cam2img, cam2ego)
        bev = self.voxel_pooling(x_lifted, geom)

        depth_per_cam = depth.view(B, N, self.n_bins, Hf, Wf)
        return bev, depth_per_cam

    # Convenience for unit tests / notebook
    def forward_with_explicit_depth(
        self,
        feat: torch.Tensor,        # (B, N, C_out, Hf, Wf)
        depth: torch.Tensor,       # (B, N, D, Hf, Wf), already softmaxed
        cam2img: torch.Tensor,
        cam2ego: torch.Tensor,
    ) -> torch.Tensor:
        """Bypass the depth net — useful for sanity tests where we want
        to inject a one-hot depth distribution and verify the splat."""
        B, N, C, Hf, Wf = feat.shape
        x_lifted = feat.unsqueeze(3) * depth.unsqueeze(2)
        x_lifted = x_lifted.view(B, N, C, depth.shape[2], Hf, Wf)
        geom = self.get_geometry(cam2img, cam2ego)
        return self.voxel_pooling(x_lifted, geom)

    # ------------------------------------------------------------------
    # Phase A: radar-projected depth supervision
    # ------------------------------------------------------------------
    @staticmethod
    def _default_z_smear() -> tuple[float, ...]:
        """Single ego-z sample at 1.0 m — roughly the centroid of vehicle
        and pedestrian heights radar plausibly returns from. The five-
        sample variant tested at Gate 3 produced an irreducible L1 floor
        ~8 m because each radar return became 5 supervised cells, but
        only one of those cells aligned with vehicle-height image
        content — the other four hit road, sky, or foreground objects
        with very different ground-truth depths. A single sample at the
        most-likely elevation removes that within-cell ambiguity at the
        cost of ~5x fewer supervised cells per sample (~300 vs ~1500),
        still well above Gate 1's 80 % non-empty bar.
        """
        return (1.0,)

    def compute_depth_loss(
        self,
        depth_softmax: torch.Tensor,       # (B, N, D, Hf, Wf), already softmaxed
        cam2img: torch.Tensor,             # (B, N, 3, 3) or (B, N, 4, 4)
        cam2ego: torch.Tensor,             # (B, N, 4, 4)
        radar_points: torch.Tensor,        # (B, 6, M)
        radar_mask: torch.Tensor,          # (B, M) bool
        z_smear: Optional[tuple[float, ...]] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Sparse L1 on the LSS depth softmax against radar-projected depth.

        Each valid radar return (x, y, z=row[2] is unreliable elevation,
        treated as 0) is smeared over a few ego-z samples in [0, 2 m] —
        radar carries no elevation, so we hypothesise a vertical column.
        Each (x, y, z_smear) point is projected into every camera; pixels
        landing in-image with positive depth become supervised feature
        cells (uf, vf) at stride ``feat_stride`` with target depth = Z.

        Per (batch, camera, cell) we use the **closest** target depth when
        multiple radar points project to the same cell — closer reflectors
        dominate, matching the physics of occlusion.

        The loss is L1 on the **expected depth** (sum over softmaxed
        bins × bin centres), which tolerates the noise of the
        radar→image projection (radar has no true elevation, the
        projected pixel is approximate).

        Returns:
            loss:       scalar L1 averaged over supervised cells.
            n_targets:  scalar long, number of supervised cells in the
                        batch — useful as a probe sanity-check.
        """
        if z_smear is None:
            z_smear = self._default_z_smear()

        depth_softmax = depth_softmax.float()
        B, N, D, Hf, Wf = depth_softmax.shape
        M = radar_points.shape[2]
        stride = self.feat_stride
        H_img, W_img = self.image_hw
        n_z = len(z_smear)

        if cam2img.shape[-1] == 4:
            cam2img = cam2img[..., :3, :3]

        device = depth_softmax.device
        cam2img_f = cam2img.float()
        cam2ego_f = cam2ego.float()

        # Expected depth (B, N, Hf, Wf): Σ_d softmax · centre_d.
        centres = self.depth_centres.to(device).view(1, 1, D, 1, 1)
        expected_depth = (depth_softmax * centres).sum(dim=2)

        z_smear_t = torch.tensor(z_smear, device=device, dtype=torch.float32)

        # Build (B, 4, M*n_z) homogeneous ego coords with z varied across
        # the smear samples. Repeat radar_mask alongside.
        x_e = radar_points[:, 0, :].float()                      # (B, M)
        y_e = radar_points[:, 1, :].float()
        x_rep = x_e.unsqueeze(2).expand(B, M, n_z).reshape(B, -1)
        y_rep = y_e.unsqueeze(2).expand(B, M, n_z).reshape(B, -1)
        z_rep = z_smear_t.view(1, 1, n_z).expand(B, M, n_z).reshape(B, -1)
        ones = torch.ones_like(x_rep)
        pts_e = torch.stack([x_rep, y_rep, z_rep, ones], dim=1)  # (B, 4, M*n_z)

        mask_e = radar_mask.bool().unsqueeze(2).expand(B, M, n_z).reshape(B, -1)

        # Ego → cam via inverse extrinsics (cam2ego is the cam→ego transform
        # the rest of the module uses; the LSS frustum side already builds
        # this once per forward but in (D, Hf, Wf) grid form, not what we
        # need here — invert directly).
        T_ec = torch.linalg.inv(cam2ego_f)                       # (B, N, 4, 4)
        pts_c = torch.einsum('bnij,bjk->bnik', T_ec, pts_e)      # (B, N, 4, M*n_z)
        Z = pts_c[:, :, 2, :]                                    # (B, N, M*n_z)

        pts_img = torch.einsum('bnij,bnjk->bnik', cam2img_f,
                               pts_c[:, :, :3, :])               # (B, N, 3, M*n_z)
        Z_safe = Z.clamp(min=1e-6)
        u = pts_img[:, :, 0, :] / Z_safe
        v = pts_img[:, :, 1, :] / Z_safe

        in_img = (u >= 0) & (u < W_img) & (v >= 0) & (v < H_img)
        in_z = (Z >= self.d_min) & (Z <= self.d_max)
        valid = in_img & in_z & mask_e.unsqueeze(1)              # (B, N, M*n_z)

        uf = (u / stride).long().clamp(0, Wf - 1)
        vf = (v / stride).long().clamp(0, Hf - 1)
        flat_idx = (vf * Wf + uf).reshape(B * N, -1)             # (B*N, M*n_z)

        Z_flat = Z.reshape(B * N, -1)
        valid_flat = valid.reshape(B * N, -1)

        # Push invalid Z to +inf so per-cell scatter-amin ignores them.
        inf = torch.tensor(float('inf'), device=device, dtype=torch.float32)
        Z_masked = torch.where(valid_flat, Z_flat, inf.expand_as(Z_flat))

        T = torch.full((B * N, Hf * Wf), float('inf'),
                       device=device, dtype=torch.float32)
        T = T.scatter_reduce(1, flat_idx, Z_masked,
                             reduce='amin', include_self=True)
        cell_valid = torch.isfinite(T)                            # (B*N, Hf*Wf)
        n_targets = cell_valid.sum().to(torch.long)

        E = expected_depth.reshape(B * N, -1)
        diff = (E - T).abs()
        # Where cell_valid is False, T is +inf → diff is +inf; mask to 0
        # so the sum is finite.
        diff = torch.where(cell_valid, diff, torch.zeros_like(diff))

        if n_targets.item() == 0:
            # No coverage at all — return a 0 that still depends on the
            # depth softmax (so backward is well-defined and a no-op).
            return E.sum() * 0.0, n_targets

        loss = diff.sum() / n_targets.float().clamp(min=1.0)
        return loss, n_targets
