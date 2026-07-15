"""
BEVBackbone — small ResNet-style trunk on the fused BEV feature.

v3b fed ``RadarGatedFusion`` output directly into ``CenterHead``. Without a
BEV-side trunk the head's effective receptive field is just ``share_conv``
(3x3) plus the per-task heads (3x3) — five BEV cells = 2 m at 0.4 m grid,
smaller than a truck. This module is a thin stack of standard stride-1
residual blocks (matching BEVDet/BEVFusion's recipe); with
``channels=256`` and ``num_blocks=3`` the receptive field expands to ~13
BEV cells (~5 m), enough for the head to see surrounding context when
assigning heatmap peaks and orientation regressions.

Since pretrained weights are inherited from v3b for everything except this
module, a random-init backbone would briefly garble the head's input
distribution. To avoid that, the second BatchNorm in each block is
zero-initialised, so each block emits ``f(x) + identity ≈ identity`` at
iteration 0 and the backbone is functionally invisible until training
nudges it (the same trick used in ResNet and BERT residuals). A 1x1
``proj`` layer is omitted since in/out channels already match, and keeping
it would break the zero-residual identity property at init.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.utils.checkpoint as checkpoint

from mmdet3d.registry import MODELS


class BasicBlock(nn.Module):
    """Stride-1 ResNet basic block with zero-init residual.

    Standard ``conv1 → bn1 → relu → conv2 → bn2 → +identity → relu``,
    but ``bn2`` starts at gamma=0 so the residual branch is silenced
    at init: ``out = relu(0 + identity) = identity`` (since the input
    has already passed a ReLU upstream and is non-negative).
    """

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(
            channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(
            channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)
        self.relu = nn.ReLU(inplace=True)

        # Zero-residual init — kill the residual branch at iter 0.
        nn.init.zeros_(self.bn2.weight)
        nn.init.zeros_(self.bn2.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = out + identity
        return self.relu(out)


@MODELS.register_module()
class BEVBackbone(nn.Module):
    """Small BEV-side trunk consumed by the detection head.

    Args:
        channels:           Channel count of the BEV feature in and out.
        num_blocks:         Number of residual blocks at stride 1.
        use_checkpointing:  If True, recompute the residual stack during
                            backward instead of caching activations.
                            Halves the saved-activation footprint of the
                            backbone (essential for fitting bs=8 in a
                            24 GB GPU) at the cost of ~25 % extra
                            forward compute per iteration.
    """

    def __init__(
        self,
        channels: int = 256,
        num_blocks: int = 3,
        use_checkpointing: bool = True,
    ) -> None:
        super().__init__()
        self.blocks = nn.Sequential(
            *[BasicBlock(int(channels)) for _ in range(int(num_blocks))]
        )
        self.out_channels = int(channels)
        self.use_checkpointing = bool(use_checkpointing)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.use_checkpointing and self.training and x.requires_grad:
            return checkpoint.checkpoint(
                self.blocks, x, use_reentrant=False)
        return self.blocks(x)
