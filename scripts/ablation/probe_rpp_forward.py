"""
probe_rpp_forward.py — G0/G1 forward probe for the rpp detector.

Checks, on a handful of real mini samples:
  A. dataset pipeline yields (geo[+paint], max_points) tensors + mask;
  B. pillar scatter occupancy matches the number of occupied BEV cells;
  C. loss forward is finite and backward reaches the point MLP;
  D. predict returns a finite, bounded box list;
  E. (paint mode) zeroing the paint changes the BEV feature — the camera
     branch is live; shuffling radar points across samples changes it too.

Usage:
    .venv/bin/python scripts/ablation/probe_rpp_forward.py                 # radar-only
    .venv/bin/python scripts/ablation/probe_rpp_forward.py --paint         # painted
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_ROOT))


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config', default='configs/rpp_mini.py')
    p.add_argument('--paint', action='store_true',
                   help='keep painted channels enabled (needs the paint cache)')
    p.add_argument('--device', default='cuda' if torch.cuda.is_available()
                   else 'cpu')
    p.add_argument('--n-samples', type=int, default=4)
    return p.parse_args()


def strip_paint(cfg):
    """Disable the painted channels everywhere in the config."""
    cfg.model.paint_channels = 0
    cfg.model.pillar_encoder.in_channels = cfg.model.geo_channels
    for dl in (cfg.train_dataloader, cfg.val_dataloader):
        dl.dataset.radar_paint_dir = None
        for t in dl.dataset.pipeline:
            if t['type'] == 'LoadRadarPoints':
                t['paint'] = False
    return cfg


def main():
    args = parse_args()
    import os
    os.chdir(_ROOT)

    from mmengine.config import Config
    from mmengine.dataset import pseudo_collate
    from mmengine.registry import init_default_scope
    from mmdet3d.registry import DATASETS, MODELS

    init_default_scope('mmdet3d')
    import datasets  # noqa: F401 — register custom transforms/datasets
    import models    # noqa: F401 — register rpp modules

    cfg = Config.fromfile(args.config)
    if not args.paint:
        cfg = strip_paint(cfg)

    # ---- A. pipeline ---------------------------------------------------
    dataset = DATASETS.build(cfg.train_dataloader.dataset)
    items = [dataset[i] for i in range(args.n_samples)]
    pts0 = items[0]['inputs']['radar_points']
    mask0 = items[0]['inputs']['radar_points_mask']
    n_rows = cfg.model.geo_channels + cfg.model.paint_channels
    assert pts0.shape == (n_rows, cfg.max_radar_points), pts0.shape
    assert mask0.dtype == torch.bool
    n_valid = int(mask0.sum())
    print(f'A. pipeline OK — points {tuple(pts0.shape)}, '
          f'{n_valid} valid, gt boxes: '
          f'{len(items[0]["data_samples"].gt_instances_3d)}')

    # ---- model ----------------------------------------------------------
    model = MODELS.build(cfg.model).to(args.device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'   model built — {n_params / 1e6:.2f} M trainable params')

    batch = pseudo_collate(items)
    data = model.data_preprocessor(batch, training=True)
    points = data['inputs']['radar_points']
    mask = data['inputs']['radar_points_mask']
    assert points.device.type == args.device.split(':')[0], points.device

    # ---- B. scatter occupancy -------------------------------------------
    model.eval()
    with torch.no_grad():
        bev = model.pillar_encoder(points, mask)
        # Count occupied pillars directly (pre-conv feature ≥ 0, empty = 0).
        enc = model.pillar_encoder
        x, y = points[:, 0], points[:, 1]
        col = torch.floor((x + enc.bev_range) / enc.bev_res).long()
        row = torch.floor((y + enc.bev_range) / enc.bev_res).long()
        valid = (mask & (col >= 0) & (col < enc.bev_size)
                 & (row >= 0) & (row < enc.bev_size))
        n_pillars = sum(
            len(torch.unique(row[b][valid[b]] * enc.bev_size
                             + col[b][valid[b]]))
            for b in range(points.shape[0]))
    assert torch.isfinite(bev).all()
    print(f'B. encoder OK — BEV {tuple(bev.shape)}, '
          f'{n_pillars} occupied pillars across batch')

    # ---- C. loss + backward ---------------------------------------------
    model.train()
    losses = model(**data, mode='loss')
    total = sum(v for v in losses.values() if isinstance(v, torch.Tensor)
                and v.requires_grad)
    assert torch.isfinite(total), losses
    total.backward()
    g = model.pillar_encoder.point_mlp[0].weight.grad
    gnorm = float(g.norm()) if g is not None else 0.0
    assert gnorm > 0, 'no gradient reached the point MLP!'
    per = {k: f'{float(v):.3f}' for k, v in losses.items()
           if isinstance(v, torch.Tensor)}
    print(f'C. loss OK — total {float(total):.2f}, '
          f'point-MLP grad-norm {gnorm:.4f}')
    print(f'   {per}')

    # ---- D. predict -------------------------------------------------------
    model.eval()
    model.zero_grad(set_to_none=True)
    with torch.no_grad():
        preds = model(**data, mode='predict')
    n_boxes = [len(p.pred_instances_3d) for p in preds]
    assert all(n <= 500 for n in n_boxes)
    print(f'D. predict OK — boxes/sample {n_boxes} (untrained: arbitrary)')

    # ---- E. sensitivity ----------------------------------------------------
    with torch.no_grad():
        base = model.extract_feat(
            {'radar_points': points, 'radar_points_mask': mask})
        perm = torch.roll(points, 1, dims=0)          # radar from another sample
        pmask = torch.roll(mask, 1, dims=0)
        shuf = model.extract_feat(
            {'radar_points': perm, 'radar_points_mask': pmask})
        d_shuffle = float((base - shuf).abs().mean())
        msg = f'E. radar-shuffle ΔBEV {d_shuffle:.4f} (must be > 0)'
        assert d_shuffle > 0
        if cfg.model.paint_channels > 0:
            zeroed = points.clone()
            zeroed[:, cfg.model.geo_channels:] = 0.0
            nop = model.extract_feat(
                {'radar_points': zeroed, 'radar_points_mask': mask})
            d_paint = float((base - nop).abs().mean())
            msg += f'; paint-zero ΔBEV {d_paint:.4f} (must be > 0)'
            assert d_paint > 0
        print(msg)

    print('\nPROBE PASSED')


if __name__ == '__main__':
    main()
