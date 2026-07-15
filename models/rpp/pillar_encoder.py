"""
RadarPillarEncoder — painted radar points → BEV feature map.

The radar-primary counterpart of the camera LSS lift: instead of learning
where to place camera features in BEV (the v3x trap), the geometry is given
directly by the radar returns and only the per-point feature extraction is
learned. PointPillars-style, with two departures suited to radar:

- **No pillar grouping / max-points-per-pillar cap.** Radar clouds are three
  orders of magnitude sparser than LiDAR (~1.5 k points/sample), so all valid
  points across the batch are processed as one flat (ΣM, C) tensor and
  max-reduced straight into the BEV grid. No padding waste, and BatchNorm
  statistics are computed over real points only (the padded-zeros pollution
  that PointPillars tolerates would be ~30 % here).
- **Scatter is amax, empty cells are exactly 0.** The MLP ends in ReLU, so
  features are non-negative and an amax-scatter onto a zero-initialised grid
  is consistent: empty pillar == silent feature, same convention as
  ``rasterise_radar_pointcloud``.

Grid convention (must match CenterHead's target assignment and the existing
radar rasteriser): col = floor((x + range)/res) → W axis, row =
floor((y + range)/res) → H axis.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor

from mmdet3d.registry import MODELS


@MODELS.register_module()
class RadarPillarEncoder(nn.Module):
    """Encode a padded radar point batch into a dense BEV feature map.

    Args:
        in_channels:   Rows of the input point tensor (8 for the v2 cache,
                       8 + K when painted camera features are appended).
        feat_channels: Widths of the per-point MLP layers.
        out_channels:  Channels of the output BEV map (a 3×3 conv after the
                       scatter mixes neighbouring pillars and lifts the
                       channel count for the BEV backbone / head).
        bev_range:     Half-extent of the BEV grid (m).
        bev_res:       Cell size (m).
    """

    def __init__(
        self,
        in_channels: int = 8,
        feat_channels: tuple[int, ...] = (64, 64),
        out_channels: int = 128,
        bev_range: float = 51.2,
        bev_res: float = 0.4,
    ) -> None:
        super().__init__()
        self.in_channels = int(in_channels)
        self.bev_range = float(bev_range)
        self.bev_res = float(bev_res)
        self.bev_size = int(round(2 * self.bev_range / self.bev_res))
        self.out_channels = int(out_channels)

        # +2: offsets of the point from its pillar center (dx_c, dy_c) —
        # sub-cell position information the scatter would otherwise destroy.
        layers: list[nn.Module] = []
        prev = self.in_channels + 2
        for width in feat_channels:
            layers += [
                nn.Linear(prev, width, bias=False),
                nn.BatchNorm1d(width),
                nn.ReLU(inplace=True),
            ]
            prev = width
        self.point_mlp = nn.Sequential(*layers)
        self.point_channels = prev

        self.bev_conv = nn.Sequential(
            nn.Conv2d(self.point_channels, self.out_channels, 3, padding=1,
                      bias=False),
            nn.BatchNorm2d(self.out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, points: Tensor, mask: Tensor) -> Tensor:
        """
        Args:
            points: (B, C, N) padded point tensor, rows 0/1 = ego x/y.
            mask:   (B, N) bool, True at real returns.

        Returns:
            (B, out_channels, bev_size, bev_size) BEV feature map.
        """
        B, C, N = points.shape
        H = W = self.bev_size
        device, dtype = points.device, points.dtype

        # Pillar indices from ego x/y; drop padding and out-of-range points.
        x, y = points[:, 0], points[:, 1]                        # (B, N)
        col = torch.floor((x + self.bev_range) / self.bev_res).long()
        row = torch.floor((y + self.bev_range) / self.bev_res).long()
        valid = (mask & (col >= 0) & (col < W)
                 & (row >= 0) & (row < H))                       # (B, N)

        bev_flat = torch.zeros(
            B * H * W, self.point_channels, device=device, dtype=dtype)

        if valid.any():
            b_idx = (torch.arange(B, device=device)
                     .unsqueeze(1).expand(B, N))[valid]          # (ΣM,)
            feats = points.permute(0, 2, 1)[valid]               # (ΣM, C)
            col_v, row_v = col[valid], row[valid]

            # Offsets from the pillar center, in metres.
            cx = (col_v.to(dtype) + 0.5) * self.bev_res - self.bev_range
            cy = (row_v.to(dtype) + 0.5) * self.bev_res - self.bev_range
            feats = torch.cat(
                [feats,
                 (feats[:, 0] - cx).unsqueeze(1),
                 (feats[:, 1] - cy).unsqueeze(1)], dim=1)        # (ΣM, C+2)

            feats = self.point_mlp(feats)                        # (ΣM, F) ≥ 0

            flat_idx = b_idx * (H * W) + row_v * W + col_v       # (ΣM,)
            # Under autocast the MLP emits fp16 while bev_flat is fp32 and
            # scatter_reduce_ requires matching dtypes — upcast (cheap, and
            # keeps the empty-batch path allocation independent of AMP).
            bev_flat.scatter_reduce_(
                0,
                flat_idx.unsqueeze(1).expand(-1, self.point_channels),
                feats.to(bev_flat.dtype),
                reduce='amax',
                include_self=True,
            )

        bev = (bev_flat
               .view(B, H, W, self.point_channels)
               .permute(0, 3, 1, 2)
               .contiguous())                                    # (B, F, H, W)
        return self.bev_conv(bev)

    def extra_repr(self) -> str:
        return (f'in_channels={self.in_channels}, '
                f'bev={self.bev_size}x{self.bev_size}@{self.bev_res}m')
