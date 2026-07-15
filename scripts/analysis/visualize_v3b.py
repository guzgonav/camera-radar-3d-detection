"""
visualize_v3b.py — Visualize BEVFusion v3b predictions on a single val sample.

Projects predicted 3D boxes onto all 6 cameras and a BEV top-down view.

Usage:
    .venv/bin/python scripts/analysis/visualize_v3b.py
    .venv/bin/python scripts/analysis/visualize_v3b.py --idx 5 --score-thr 0.3
    .venv/bin/python scripts/analysis/visualize_v3b.py --idx 10 --out-dir results/vis_v3b
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from PIL import Image

_ROOT = Path(__file__).resolve().parent.parent.parent
os.chdir(_ROOT)
sys.path.insert(0, str(_ROOT))

import torch
from mmengine.config import Config
from mmengine.dataset import pseudo_collate
from mmengine.registry import init_default_scope
from mmengine.runner import load_checkpoint
from mmdet3d.registry import DATASETS, MODELS

import datasets  # noqa: F401  registers NuScenesRadarDataset + transforms
import models    # noqa: F401  registers BEVFusionDetector + components

CONFIG = 'configs/bev_fusion_full_v3b.py'
CKPT = (
    'work_dirs/bev_fusion_full_v3b/'
    'best_NuScenes metric_pred_instances_3d_NuScenes_NDS_epoch_24.pth'
)
DATA_ROOT = 'data/nuscenes'
BEV_RANGE = 51.2

CLASS_NAMES = [
    'car', 'truck', 'construction_vehicle', 'bus', 'trailer',
    'barrier', 'motorcycle', 'bicycle', 'pedestrian', 'traffic_cone',
]
CLASS_COLORS = [
    '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
    '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf',
]

CAM_ORDER = [
    'CAM_FRONT_LEFT', 'CAM_FRONT',      'CAM_FRONT_RIGHT',
    'CAM_BACK_LEFT',  'CAM_BACK',       'CAM_BACK_RIGHT',
]

# 3D box edge pairs (mmdet3d LiDARInstance3DBoxes corner layout)
#      6 ── 5
#     /|   /|   (top face: 4-5-6-7)
#    7 ── 4 |
#    | 2 ─|─ 1
#    |/   |/    (bottom face: 0-1-2-3)
#    3 ── 0
BOX_EDGES = [
    (0, 1), (1, 2), (2, 3), (3, 0),  # bottom face
    (4, 5), (5, 6), (6, 7), (7, 4),  # top face
    (0, 4), (1, 5), (2, 6), (3, 7),  # pillars
]


def project_corners(corners_lidar: np.ndarray, lidar2cam: np.ndarray,
                    cam2img: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return (8,2) pixel coords and (8,) depths for box corners."""
    ones = np.ones((8, 1), dtype=np.float32)
    pts_h = np.hstack([corners_lidar.astype(np.float32), ones])  # (8,4)
    pts_cam = (lidar2cam @ pts_h.T).T                            # (8,4)
    depths = pts_cam[:, 2]
    # Perspective division; guard against zero depth
    uv1 = (cam2img @ pts_cam[:, :3].T).T                        # (8,3)
    depth_safe = np.where(np.abs(depths) > 1e-6, depths, 1e-6)
    uv = uv1[:, :2] / depth_safe[:, None]                       # (8,2) pixels
    return uv, depths


def draw_boxes_on_camera(ax, boxes_corners, labels, lidar2cam, cam2img, img_hw):
    """Draw all predicted box edges on a camera axes."""
    H, W = img_hw
    for bi in range(len(boxes_corners)):
        uv, depths = project_corners(boxes_corners[bi], lidar2cam, cam2img)
        # Keep edges only when at least one endpoint is in front of camera
        color = CLASS_COLORS[int(labels[bi]) % len(CLASS_COLORS)]
        for i, j in BOX_EDGES:
            if depths[i] <= 0 and depths[j] <= 0:
                continue
            # Rough visibility check: skip if both points are far off-screen
            pts = np.array([[uv[i, 0], uv[i, 1]], [uv[j, 0], uv[j, 1]]])
            if ((pts[:, 0] < -W) | (pts[:, 0] > 2 * W) |
                    (pts[:, 1] < -H) | (pts[:, 1] > 2 * H)).all():
                continue
            ax.plot(pts[:, 0], pts[:, 1], color=color, linewidth=1.2, alpha=0.9)


def draw_bev(ax, boxes_corners, labels):
    """Draw box footprints in BEV (top-down, x=forward, y=left)."""
    ax.set_facecolor('#111111')
    ax.set_xlim(-BEV_RANGE, BEV_RANGE)
    ax.set_ylim(-BEV_RANGE, BEV_RANGE)
    ax.set_aspect('equal')
    ax.set_xlabel('x (m)  →  forward', fontsize=7)
    ax.set_ylabel('y (m)  →  left', fontsize=7)
    ax.set_title('BEV (ego frame)', fontsize=9)
    ax.plot(0, 0, 'w^', markersize=10, zorder=5)

    for bi in range(len(boxes_corners)):
        corners = boxes_corners[bi]  # (8,3)
        color = CLASS_COLORS[int(labels[bi]) % len(CLASS_COLORS)]
        # Bottom face: corners 0-3 form the ground footprint
        bot = corners[:4, :2]  # (4,2) x,y of bottom corners
        # Sort by angle to close the polygon correctly
        cx, cy = bot[:, 0].mean(), bot[:, 1].mean()
        angles = np.arctan2(bot[:, 1] - cy, bot[:, 0] - cx)
        order = np.argsort(angles)
        poly = np.vstack([bot[order], bot[order[0]]])  # close loop
        ax.plot(poly[:, 0], poly[:, 1], color=color, linewidth=1.2, alpha=0.9)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--idx', type=int, default=0,
                        help='Val dataset index (default: 0)')
    parser.add_argument('--score-thr', type=float, default=0.25,
                        help='Score threshold for displaying boxes (default: 0.25)')
    parser.add_argument('--out-dir', default='results/vis_v3b',
                        help='Output directory (default: results/vis_v3b)')
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    init_default_scope('mmdet3d')
    cfg = Config.fromfile(CONFIG)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    print(f'Building val dataset ...')
    dataset = DATASETS.build(cfg.test_dataloader.dataset)
    info = dataset.get_data_info(args.idx)
    item = dataset[args.idx]

    print(f'Loading checkpoint: {CKPT}')
    model = MODELS.build(cfg.model).to(device).eval()
    load_checkpoint(model, CKPT, map_location=device)

    print('Running inference ...')
    batch = pseudo_collate([item])
    with torch.no_grad():
        data = model.data_preprocessor(batch, training=False)
        preds = model.predict(data['inputs'], data['data_samples'])

    pred = preds[0]
    scores = pred.pred_instances_3d.scores_3d.cpu().numpy()
    labels = pred.pred_instances_3d.labels_3d.cpu().numpy()
    corners_all = pred.pred_instances_3d.bboxes_3d.corners.cpu().numpy()  # (N,8,3)

    keep = scores >= args.score_thr
    boxes_corners = corners_all[keep]
    labels_k = labels[keep]
    scores_k = scores[keep]
    print(f'Sample idx={args.idx}: {keep.sum()} / {len(scores)} boxes above '
          f'score threshold {args.score_thr}')
    for cls_id in np.unique(labels_k):
        n = int((labels_k == cls_id).sum())
        print(f'  {CLASS_NAMES[cls_id]}: {n}')

    # Load original resolution camera images + raw matrices from dataset info
    cam_imgs, cam_l2c, cam_k = {}, {}, {}
    for cam in CAM_ORDER:
        ci = info['images'][cam]
        img_file = Path(DATA_ROOT) / 'samples' / cam / Path(ci['img_path']).name
        cam_imgs[cam] = np.array(Image.open(img_file))
        cam_l2c[cam] = np.array(ci['lidar2cam'], dtype=np.float32)   # (4,4)
        cam_k[cam] = np.array(ci['cam2img'], dtype=np.float32)        # (3,3)

    # Layout: 2×3 cameras + 1 BEV column
    fig = plt.figure(figsize=(24, 9))
    gs = fig.add_gridspec(2, 4, hspace=0.04, wspace=0.04,
                          left=0.01, right=0.99, top=0.92, bottom=0.08)

    cam_positions = [(0, 0), (0, 1), (0, 2), (1, 0), (1, 1), (1, 2)]
    for ci_idx, cam in enumerate(CAM_ORDER):
        row, col = cam_positions[ci_idx]
        ax = fig.add_subplot(gs[row, col])
        ax.imshow(cam_imgs[cam])
        ax.set_title(cam.replace('CAM_', ''), fontsize=8, pad=2)
        ax.axis('off')
        H, W = cam_imgs[cam].shape[:2]
        draw_boxes_on_camera(ax, boxes_corners, labels_k,
                             cam_l2c[cam], cam_k[cam], (H, W))

    bev_ax = fig.add_subplot(gs[:, 3])
    draw_bev(bev_ax, boxes_corners, labels_k)

    # Legend
    handles = [mpatches.Patch(color=CLASS_COLORS[i], label=CLASS_NAMES[i])
               for i in range(len(CLASS_NAMES))]
    fig.legend(handles=handles, loc='lower center', ncol=5, fontsize=8,
               bbox_to_anchor=(0.38, 0.01))

    token = info.get('token', f'idx{args.idx}')
    fig.suptitle(
        f'BEVFusion v3b  |  val sample {args.idx}  |  '
        f'token: {token[:20]}…  |  score ≥ {args.score_thr}',
        fontsize=10,
    )

    out_file = out_dir / f'sample_{args.idx:04d}.png'
    fig.savefig(out_file, dpi=120, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved → {out_file}')


if __name__ == '__main__':
    main()
