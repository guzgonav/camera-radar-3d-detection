"""
RadarGatedFusion — radar-conditioned attention fusion neck.

Replaces the v1/v2 plain-concat ``BEVFusionNeck``. Builds a structural
prior into the network: camera-BEV features should be *boosted* in
cells where radar fires.

Mechanism
---------

    attn = sigmoid(small_conv(radar_bev))    # (B, 1, Hb, Wb) ∈ (0, 1)
    cam_boosted = camera_bev * (1 + α · attn)
    fused = ConvBlock(cat([cam_boosted, radar_bev]))

At init the small_conv is randomly initialised → sigmoid output ≈ 0.5,
so the camera features get a uniform 1 + 0.5 · α boost in every cell.
With ``α = 1.0`` (default), this is a 1.5× boost everywhere — equivalent
to a fixed scaling that the head can absorb. During training, the
attention head learns to concentrate the boost where radar evidence
fires, supplying a per-cell radar-conditioned signal to the head's
heatmap and velocity outputs.

Why a residual gate (``1 + α · attn``) and not a direct gate (``attn``)
or a sigmoid in [0, 1]:
- Direct gate would zero out camera features in radar-empty cells
  (~90 % of the BEV grid for typical scenes). Camera works fine
  alone — we don't want to wipe it.
- Plain attention (multiply by [0, 1]) shrinks features below the head's
  expected magnitude. Residual ``1 + …`` keeps the camera magnitude as
  the floor and lets radar push it up.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from mmdet3d.registry import MODELS


@MODELS.register_module()
class RadarGatedFusion(nn.Module):
    """Radar-gated fusion neck.

    Args:
        camera_channels:  Channel count of the camera-BEV feature.
        radar_channels:   Channel count of the radar-BEV feature.
        out_channels:     Channels emitted to the detection head.
        hidden_channels:  Width of the fusion conv (default = out_channels).
        boost_alpha:      Maximum multiplicative boost factor; the
                          per-cell boost is ``1 + boost_alpha · attn``,
                          so attn=1 → camera scaled by ``1 + α``.
                          Default 1.0 → up to 2× boost.
    """

    def __init__(
        self,
        camera_channels: int,
        radar_channels: int,
        out_channels: int = 256,
        hidden_channels: int | None = None,
        boost_alpha: float = 1.0,
    ) -> None:
        super().__init__()
        in_ch = int(camera_channels) + int(radar_channels)
        hidden = int(hidden_channels or out_channels)

        # Spatial attention from radar — scalar mask per BEV cell.
        # Width chosen as max(8, hidden//4) so we always have at least
        # a small bottleneck even with a small hidden dimension.
        attn_hidden = max(8, hidden // 4)
        self.spatial_attn = nn.Sequential(
            nn.Conv2d(int(radar_channels), attn_hidden, kernel_size=1, bias=False),
            nn.BatchNorm2d(attn_hidden),
            nn.ReLU(inplace=True),
            nn.Conv2d(attn_hidden, 1, kernel_size=1),
            nn.Sigmoid(),
        )

        # Fusion conv on concat(boosted_camera, radar).
        self.fuse = nn.Sequential(
            nn.Conv2d(in_ch, hidden, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(hidden),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, int(out_channels), kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(int(out_channels)),
            nn.ReLU(inplace=True),
        )

        self.boost_alpha = float(boost_alpha)
        self.out_channels = int(out_channels)

    def forward(
        self,
        camera_bev: torch.Tensor,   # (B, Cc, Hb, Wb)
        radar_bev: torch.Tensor,    # (B, Cr, Hb, Wb)
    ) -> torch.Tensor:
        """Boost camera by radar attention, concat with radar, mix."""
        if camera_bev.shape[-2:] != radar_bev.shape[-2:]:
            raise ValueError(
                f'BEV grid mismatch: camera {tuple(camera_bev.shape[-2:])} '
                f'vs radar {tuple(radar_bev.shape[-2:])}')

        attn = self.spatial_attn(radar_bev)            # (B, 1, Hb, Wb)
        cam_boosted = camera_bev * (1.0 + self.boost_alpha * attn)
        x = torch.cat([cam_boosted, radar_bev], dim=1)
        return self.fuse(x)
