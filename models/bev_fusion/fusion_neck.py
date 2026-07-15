"""
BEVFusionNeck — concat camera BEV + radar BEV along the channel axis,
then 2 × Conv-BN-ReLU 3×3 to mix them.

Output channel count is decoupled from either branch's input count, so
the head sees a fixed feature width regardless of how the camera/radar
encoders are configured.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from mmdet3d.registry import MODELS


@MODELS.register_module()
class BEVFusionNeck(nn.Module):
    """Concatenate camera + radar BEV features and mix.

    Args:
        camera_channels: Channel count of the camera-BEV feature.
        radar_channels:  Channel count of the radar-BEV feature.
        out_channels:    Channels emitted to the detection head.
        hidden_channels: Width of the intermediate conv (default = out_channels).
    """

    def __init__(
        self,
        camera_channels: int,
        radar_channels: int,
        out_channels: int = 256,
        hidden_channels: int = None,
    ) -> None:
        super().__init__()
        in_ch = int(camera_channels) + int(radar_channels)
        hidden_channels = int(hidden_channels or out_channels)

        self.fuse = nn.Sequential(
            nn.Conv2d(in_ch, hidden_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(hidden_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )
        self.out_channels = int(out_channels)

    def forward(
        self,
        camera_bev: torch.Tensor,   # (B, Cc, Hb, Wb)
        radar_bev: torch.Tensor,    # (B, Cr, Hb, Wb)
    ) -> torch.Tensor:
        """Concat along channel axis then mix.

        Both inputs must already be on the same BEV grid.
        """
        if camera_bev.shape[-2:] != radar_bev.shape[-2:]:
            raise ValueError(
                f'BEV grid mismatch: camera {tuple(camera_bev.shape[-2:])} '
                f'vs radar {tuple(radar_bev.shape[-2:])}')
        x = torch.cat([camera_bev, radar_bev], dim=1)
        return self.fuse(x)
