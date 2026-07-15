"""
RadarBEVEncoder — small conv refiner over the rasterised radar BEV tensor.

The dataset transform (``LoadRadarBEV``) does the heavy lifting: it
already produces a ``(Cr=5, Hb, Wb)`` tensor with channels
``[count, mean vx, mean vy, mean rcs, max rcs]`` in the ego frame.

This module's job is therefore minimal — keep the network's view of the
radar branch lightweight and avoid washing out the physical signals
that we paid the cost to compute. Three 3×3 convolutions with a small
hidden width are sufficient.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from mmdet3d.registry import MODELS


@MODELS.register_module()
class RadarBEVEncoder(nn.Module):
    """Small CNN refiner for the rasterised radar BEV tensor.

    Args:
        in_channels:   Number of input channels (default 5 — see
                       ``datasets.nuscenes_radar_dataset.RADAR_NUM_CHANNELS``).
        hidden_channels: Width of the intermediate convs.
        out_channels:  Output channels — concatenated with the camera BEV
                       feature in the fusion neck.
    """

    def __init__(
        self,
        in_channels: int = 5,
        hidden_channels: int = 32,
        out_channels: int = 32,
    ) -> None:
        super().__init__()
        self.in_channels = int(in_channels)
        self.out_channels = int(out_channels)

        self.net = nn.Sequential(
            nn.Conv2d(in_channels, hidden_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(hidden_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels, hidden_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(hidden_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels, out_channels, kernel_size=1, padding=0),
        )

    def forward(self, radar_bev: torch.Tensor) -> torch.Tensor:
        """``radar_bev``: (B, Cr, Hb, Wb) -> (B, out_channels, Hb, Wb)."""
        return self.net(radar_bev)
