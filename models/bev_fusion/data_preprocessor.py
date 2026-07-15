"""
BEVFusionDataPreprocessor — extends ``Det3DDataPreprocessor`` so the
extra inputs we pack in (``radar_bev``, ``cam2img``, ``cam2ego``) are
batched, moved to the model's device, and forwarded into the model.

The parent only knows about ``imgs`` and ``points``; everything else
gets dropped. We wrap ``simple_process`` to add our keys back after the
parent has done the image normalisation.
"""

from __future__ import annotations

from typing import Dict, List, Sequence, Union

import torch

from mmdet3d.models.data_preprocessors import Det3DDataPreprocessor
from mmdet3d.registry import MODELS


def _stack_per_sample(
    tensors: Sequence[torch.Tensor],
    device: torch.device,
) -> torch.Tensor:
    """Stack a list of (per-sample) tensors of identical shape into a
    batched tensor and move to ``device``."""
    return torch.stack([t.to(device) for t in tensors], dim=0)


@MODELS.register_module()
class BEVFusionDataPreprocessor(Det3DDataPreprocessor):
    """Det3DDataPreprocessor that also forwards radar_bev, cam2img, cam2ego."""

    EXTRA_KEYS = (
        'radar_bev', 'cam2img', 'cam2ego', 'radar_points', 'radar_points_mask',
    )

    def simple_process(self, data: dict, training: bool = False) -> dict:
        # Pull the per-sample extras out before the parent strips them.
        # ``data['inputs']`` is a dict of lists at this point.
        extras = {}
        for key in self.EXTRA_KEYS:
            val = data['inputs'].get(key, None)
            if val is not None:
                extras[key] = val

        out = super().simple_process(data, training=training)

        if extras:
            device = next(iter(out['inputs'].values())).device \
                if out['inputs'] else torch.device('cpu')
            for key, val in extras.items():
                if isinstance(val, list):
                    out['inputs'][key] = _stack_per_sample(val, device)
                elif isinstance(val, torch.Tensor):
                    # Already a stacked tensor (e.g. via default_collate).
                    out['inputs'][key] = val.to(device)
                else:
                    raise TypeError(
                        f'Unexpected type for {key}: {type(val).__name__}')

        return out
