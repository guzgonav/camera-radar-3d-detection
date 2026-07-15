"""
radar_refinement.py — CenterFusion-style detection-level refinement MLP
with camera-uncertainty-conditioned radar gates.

This is the *positive contribution* of the thesis, built after the BEV-fusion
chapter (v3c/v3e/v3f) showed that end-to-end joint training lets the optimiser
gate radar off (shuffle-ΔNDS ≈ 0). The structural fix:

    FCOS3D is frozen — we never train through it. Radar has an *exclusive*
    job (refine the depth and velocity of existing camera detections) and so
    cannot be gated off by a camera optimiser, because there is no camera
    optimiser in this stage. Only this small MLP is trainable.

Original contribution — the uncertainty gate
---------------------------------------------
A per-detection scalar ``g = σ(MLP_gate(features))`` decides how much radar to
trust. When the camera is confident (close range, high score, low
``depth_var``) the gate learns ``g → 0`` and the detection is left alone; when
the camera is structurally weak (far / occluded / low score) ``g → 1`` and
radar takes over. This operationalises the BEV failure analysis: trust radar
exactly where the monocular depth trunk is foggy.

Two gates, not one
------------------
The mini fixed-gate sweep (see PLANS / center_fusion results) showed depth and
velocity want *different* trust levels — depth is best near g≈0.25, velocity
near g≈0.75. A single scalar is forced to compromise, so the MLP emits two
gates from a shared uncertainty trunk:

    g_depth = σ(head_depth(trunk(x)))   # depth-uncertainty gate (the headline)
    g_vel   = σ(head_vel(trunk(x)))     # velocity-trust gate

Refinement formula (matches the pipeline diagram)
-------------------------------------------------
    refined_range = cam_range + g_depth · (radar_range − cam_range) + Δdepth
    refined_vel   = cam_vel   + g_vel   · (radar_vel   − cam_vel)   + Δvel

``Δ*`` are small learned residuals, disabled with ``use_residual=False`` to
recover the clean, fully-interpretable gated-interpolation form.

A detection with **no associated radar point** is never passed to this module
— the pipeline keeps it as the raw camera detection. So ``forward`` always
sees detections that have radar evidence.

This is a plain ``torch.nn.Module`` (NOT an mmdet3d registered model): it is
driven by ``scripts/fusion/center_fusion.py`` / ``scripts/train/train_center_fusion.py``,
not by an mmengine Runner.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# Feature layout — canonical definition shared with the pipeline.
# scripts/fusion/center_fusion.py:build_feature_row MUST produce this exact order.
# ---------------------------------------------------------------------------
NUM_CLASSES = 10  # len(DETECTION_CLASSES); asserted against the pipeline.

# Camera features (per detection):
#   score            detection confidence in [0, 1]
#   cam_range_norm   sqrt(x²+y²) / RANGE_NORM       (ego BEV range, normalised)
#   depth_var        analytic monocular-depth-uncertainty proxy in [0, 1]
#   log_size_w/l/h   log box dims (scale-stable)
#   cam_speed_norm   |camera velocity| / SPEED_NORM
CAM_SCALAR_FEATURES = (
    'score', 'cam_range_norm', 'depth_var',
    'log_size_w', 'log_size_l', 'log_size_h', 'cam_speed_norm',
)
# Radar features (per detection, from frustum-associated points):
#   radar_range_norm  median associated radar range / RANGE_NORM
#   range_residual    (radar_range − cam_range) / RANGE_NORM   (signed)
#   radar_vx/vy_norm  mean compensated radar velocity / SPEED_NORM (ego)
#   mean_rcs_norm     mean RCS / RCS_NORM
#   range_std_norm    std of associated radar ranges / RANGE_NORM
#   log_n_points      log1p(n associated radar points)
RADAR_SCALAR_FEATURES = (
    'radar_range_norm', 'range_residual', 'radar_vx_norm', 'radar_vy_norm',
    'mean_rcs_norm', 'range_std_norm', 'log_n_points',
)

FEATURE_DIM = len(CAM_SCALAR_FEATURES) + len(RADAR_SCALAR_FEATURES) + NUM_CLASSES


@dataclass
class RefinementOutput:
    """Bundle returned by :meth:`RadarRefinementMLP.forward`."""

    refined_range: torch.Tensor   # (N,)   fused BEV range, metres
    refined_vel: torch.Tensor     # (N, 2) fused ego-frame velocity, m/s
    gate_depth: torch.Tensor      # (N,)   depth gate g_depth ∈ (0, 1)
    gate_vel: torch.Tensor        # (N,)   velocity gate g_vel ∈ (0, 1)
    delta_depth: torch.Tensor     # (N,)   learned depth residual, metres
    delta_vel: torch.Tensor       # (N, 2) learned velocity residual, m/s


class RadarRefinementMLP(nn.Module):
    """Per-detection radar refinement with separate depth / velocity gates.

    Args:
        input_dim:    Feature-vector width. Defaults to :data:`FEATURE_DIM`.
        hidden_dim:   Width of the two hidden layers.
        use_residual: If ``True`` the network emits learned residuals
                      ``Δdepth, Δvx, Δvy``. If ``False`` (default) the
                      refinement is the pure gated interpolation
                      ``cam + g·(radar − cam)`` — fully interpretable.
        delta_scale:  Multiplier on the residual head (metres / m·s⁻¹), keeps
                      residuals small so the gate carries the signal.
        gate_bias:    Initial bias on both gate heads. A slightly negative
                      value (default −0.4 → g≈0.4) starts the model cautious,
                      biased toward keeping the camera rather than overwriting.
    """

    def __init__(
        self,
        input_dim: int = FEATURE_DIM,
        hidden_dim: int = 64,
        use_residual: bool = False,
        delta_scale: float = 2.0,
        gate_bias: float = -0.4,
    ) -> None:
        super().__init__()
        self.input_dim = int(input_dim)
        self.use_residual = bool(use_residual)
        self.delta_scale = float(delta_scale)

        self.trunk = nn.Sequential(
            nn.Linear(self.input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(inplace=True),
        )
        # Two uncertainty gates from the shared trunk.
        self.gate_depth_head = nn.Linear(hidden_dim, 1)
        self.gate_vel_head = nn.Linear(hidden_dim, 1)
        # Residual head — [Δdepth, Δvx, Δvy].
        self.delta_head = nn.Linear(hidden_dim, 3)

        self._init_weights(gate_bias)

    def _init_weights(self, gate_bias: float) -> None:
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        # Residual head starts at zero → early training is a pure gated blend.
        nn.init.zeros_(self.delta_head.weight)
        nn.init.zeros_(self.delta_head.bias)
        # Cautious gate init: default toward keeping the camera.
        nn.init.constant_(self.gate_depth_head.bias, gate_bias)
        nn.init.constant_(self.gate_vel_head.bias, gate_bias)

    def forward(
        self,
        feats: torch.Tensor,        # (N, input_dim)
        cam_range: torch.Tensor,    # (N,)   ego BEV range, metres
        radar_range: torch.Tensor,  # (N,)   associated radar range, metres
        cam_vel: torch.Tensor,      # (N, 2) ego-frame camera velocity, m/s
        radar_vel: torch.Tensor,    # (N, 2) ego-frame radar velocity, m/s
    ) -> RefinementOutput:
        """Apply the gated refinement formula to a batch of detections."""
        if feats.shape[-1] != self.input_dim:
            raise ValueError(
                f'feature dim mismatch: got {feats.shape[-1]}, '
                f'expected {self.input_dim}')

        h = self.trunk(feats)
        g_depth = torch.sigmoid(self.gate_depth_head(h)).squeeze(-1)   # (N,)
        g_vel = torch.sigmoid(self.gate_vel_head(h)).squeeze(-1)       # (N,)

        if self.use_residual:
            delta = self.delta_head(h) * self.delta_scale              # (N, 3)
        else:
            delta = torch.zeros(feats.shape[0], 3, device=feats.device,
                                dtype=feats.dtype)
        delta_depth = delta[:, 0]
        delta_vel = delta[:, 1:]

        refined_range = cam_range + g_depth * (radar_range - cam_range) + delta_depth
        refined_vel = cam_vel + g_vel.unsqueeze(-1) * (radar_vel - cam_vel) + delta_vel

        return RefinementOutput(
            refined_range=refined_range,
            refined_vel=refined_vel,
            gate_depth=g_depth,
            gate_vel=g_vel,
            delta_depth=delta_depth,
            delta_vel=delta_vel,
        )
