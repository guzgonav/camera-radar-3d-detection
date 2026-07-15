"""
paint_radar_cache.py — offline painted-feature cache for the rpp detector.

For every sample: project every v2-cache radar point into the 6 cameras at
3 pillar heights (radar elevation is unreliable), run the FROZEN FCOS3D
once per image, and sample its per-pixel head outputs at each projected
location. Per point, keep the (camera, height, FPN level) sample with the
highest centerness. Output one point-aligned ``(14, M)`` float16 array per
sample:

    [0:10]  class scores (sigmoid) — nuScenes 10 classes, FCOS3D order
    [10]    centerness (sigmoid)
    [11]    predicted metric depth / 50
    [12]    (predicted depth − point's camera-frame depth) / 50
    [13]    valid-projection flag (0 = point painted with zeros)

This is the ONLY place the camera stack ever runs: training reads the
cache and never touches an image.

Frames: radar cache is in the LIDAR-keyframe EGO frame; pillar heights are
set in ego z (0 = road at ego), then points go ego → lidar (inv lidar2ego)
→ camera (lidar2cam, the ego-motion-correct matrix from the info pkl).

Usage:
    # mini smoke + cache
    .venv/bin/python scripts/radar/paint_radar_cache.py \
        --dataroot data/nuscenes-mini --ann nuscenes_infos_train.pkl \
        --radar-dir radar_pts_v2 --out radar_paint_v1
    # full (run per split; resume-safe)
    .venv/bin/python scripts/radar/paint_radar_cache.py \
        --dataroot data/nuscenes --ann nuscenes_infos_train.pkl \
        --radar-dir radar_pts_v2 --out radar_paint_v1
"""
from __future__ import annotations

import argparse
import json
import pickle
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F

_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_ROOT))

CAMS = ('CAM_FRONT', 'CAM_FRONT_RIGHT', 'CAM_FRONT_LEFT',
        'CAM_BACK', 'CAM_BACK_LEFT', 'CAM_BACK_RIGHT')
# FCOS3D caffe-BGR normalisation (mean subtracted, std = 1).
MEAN_BGR = np.array([103.530, 116.280, 123.675], dtype=np.float32)
PAINT_ROWS = 14
FPN_STRIDES = (8, 16, 32, 64, 128)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dataroot', default='data/nuscenes')
    p.add_argument('--ann', default='nuscenes_infos_train.pkl',
                   help='info pkl (relative to dataroot)')
    p.add_argument('--radar-dir', default='radar_pts_v2',
                   help='v2 radar cache dir (relative to dataroot)')
    p.add_argument('--out', default='radar_paint_v1',
                   help='output dir (relative to dataroot)')
    p.add_argument('--fcos3d-config', default='configs/fcos3d_full.py')
    p.add_argument('--checkpoint', default=(
        'checkpoints/fcos3d_r101_caffe_fpn_gn-head_dcn_2x8_1x_'
        'nus-mono3d_finetune_20210717_095645-8d806dc2.pth'))
    p.add_argument('--heights', type=float, nargs='+', default=[0.0, 0.8, 1.6],
                   help='pillar sample heights in ego z (m)')
    p.add_argument('--levels', type=int, nargs='+', default=[0, 1, 2],
                   help='FPN level indices to sample (strides 8/16/32)')
    p.add_argument('--min-cam-depth', type=float, default=0.5)
    p.add_argument('--fp32', action='store_true',
                   help='disable autocast (fallback)')
    p.add_argument('--limit', type=int, default=0,
                   help='process only the first N samples (smoke test)')
    p.add_argument('--overwrite', action='store_true')
    return p.parse_args()


class ImageSet(torch.utils.data.Dataset):
    """Loads + normalises + pads the 6 camera images of one sample."""

    def __init__(self, data_list, dataroot: Path):
        self.data_list = data_list
        self.dataroot = dataroot

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, i):
        info = self.data_list[i]
        imgs, calib = [], []
        for cam in CAMS:
            entry = info['images'][cam]
            path = self.dataroot / 'samples' / cam / Path(entry['img_path']).name
            img = cv2.imread(str(path))  # BGR, (900, 1600, 3)
            if img is None:
                raise FileNotFoundError(path)
            h, w = img.shape[:2]
            ph = (32 - h % 32) % 32
            pw = (32 - w % 32) % 32
            img = img.astype(np.float32) - MEAN_BGR
            if ph or pw:
                img = np.pad(img, ((0, ph), (0, pw), (0, 0)))
            imgs.append(torch.from_numpy(img).permute(2, 0, 1))
            l2c = np.eye(4, dtype=np.float32)
            l2c_raw = np.asarray(entry['lidar2cam'], dtype=np.float32)
            l2c[:l2c_raw.shape[0], :l2c_raw.shape[1]] = l2c_raw
            K = np.asarray(entry['cam2img'], dtype=np.float32)[:3, :3]
            calib.append((l2c, K, float(h), float(w)))
        return i, torch.stack(imgs), calib


def collate(batch):  # one sample at a time; keep calib as python objects
    return batch[0]


@torch.no_grad()
def paint_sample(model, imgs, calib, pts, l2e, args, device):
    """Return the (14, M) paint array for one sample."""
    M = pts.shape[1]
    paint = np.zeros((PAINT_ROWS, M), dtype=np.float32)
    if M == 0:
        return paint

    # Forward the 6 images once.
    imgs = imgs.to(device, non_blocking=True)
    with torch.autocast('cuda', enabled=not args.fp32):
        feats = model.backbone(imgs)
        feats = model.neck(feats)
        cls_scores, bbox_preds, _, _, centernesses = model.bbox_head(feats)

    # Candidate 3D points: (n_heights, 3, M) in ego, then -> lidar.
    R, t = l2e[:3, :3], l2e[:3, 3]
    # Selection uses the standard FCOS score max_cls × centerness.
    # Centerness ALONE is unusable: it is only supervised at foreground
    # locations, so on background pixels it is unconstrained noise
    # (measured: near-flat σ≈0.32 everywhere, and argmax-centerness
    # selection gave 13 m median depth residuals).
    best_sel = np.full(M, -1.0, dtype=np.float32)
    n_h = len(args.heights)
    cand_ego = np.repeat(pts[None, :3, :], n_h, axis=0)
    for hi, h in enumerate(args.heights):
        cand_ego[hi, 2, :] = h
    cand_lidar = np.einsum('ij,hjm->him', R.T, cand_ego - t[:, None])

    pad_h, pad_w = imgs.shape[-2], imgs.shape[-1]

    for ci in range(len(CAMS)):
        l2c, K, img_h, img_w = calib[ci]
        # lidar -> cam for all heights at once: (n_h, 3, M)
        p_cam = (np.einsum('ij,hjm->him', l2c[:3, :3], cand_lidar)
                 + l2c[:3, 3][None, :, None])
        z = p_cam[:, 2, :]
        with np.errstate(divide='ignore', invalid='ignore'):
            u = K[0, 0] * p_cam[:, 0, :] / z + K[0, 2]
            v = K[1, 1] * p_cam[:, 1, :] / z + K[1, 2]
        ok = ((z > args.min_cam_depth) & (u >= 0) & (u < img_w)
              & (v >= 0) & (v < img_h))                     # (n_h, M)
        if not ok.any():
            continue
        hi_idx, m_idx = np.nonzero(ok)
        uu = torch.from_numpy(u[hi_idx, m_idx]).to(device).float()
        vv = torch.from_numpy(v[hi_idx, m_idx]).to(device).float()
        # normalised grid coords over the PADDED image extent
        grid = torch.stack(
            [uu / pad_w * 2 - 1, vv / pad_h * 2 - 1],
            dim=-1).view(1, 1, -1, 2)                       # (1,1,P,2)

        # Sample each requested level; keep per-candidate best FCOS score.
        n_cand = uu.shape[0]
        best_l_sel = torch.full((n_cand,), -1.0, device=device)
        best_l_ctr = torch.zeros((n_cand,), device=device)
        best_l_cls = torch.zeros((10, n_cand), device=device)
        best_l_dep = torch.zeros((n_cand,), device=device)
        for lvl in args.levels:
            cls_l = cls_scores[lvl][ci:ci + 1].float()      # (1,10,H,W)
            ctr_l = centernesses[lvl][ci:ci + 1].float()    # (1,1,H,W)
            dep_l = bbox_preds[lvl][ci:ci + 1, 2:3].float() # (1,1,H,W) metres
            s_cls = F.grid_sample(cls_l, grid, align_corners=False)[0, :, 0]
            s_ctr = F.grid_sample(ctr_l, grid, align_corners=False)[0, 0, 0]
            s_dep = F.grid_sample(dep_l, grid, align_corners=False)[0, 0, 0]
            s_cls = s_cls.sigmoid()
            s_ctr = s_ctr.sigmoid()
            s_sel = s_cls.max(dim=0).values * s_ctr         # FCOS det score
            upd = s_sel > best_l_sel
            best_l_sel = torch.where(upd, s_sel, best_l_sel)
            best_l_ctr = torch.where(upd, s_ctr, best_l_ctr)
            best_l_cls[:, upd] = s_cls[:, upd]
            best_l_dep = torch.where(upd, s_dep, best_l_dep)

        sel_np = best_l_sel.cpu().numpy()
        ctr_np = best_l_ctr.cpu().numpy()
        cls_np = best_l_cls.cpu().numpy()
        dep_np = best_l_dep.cpu().numpy()
        z_np = z[hi_idx, m_idx]

        # Reduce candidates -> points: keep best FCOS score per point.
        for j in range(n_cand):
            m = m_idx[j]
            if sel_np[j] > best_sel[m]:
                best_sel[m] = sel_np[j]
                paint[0:10, m] = cls_np[:, j]
                paint[10, m] = ctr_np[j]
                paint[11, m] = dep_np[j] / 50.0
                paint[12, m] = (dep_np[j] - z_np[j]) / 50.0
                paint[13, m] = 1.0

    return paint


def main():
    args = parse_args()
    import os
    os.chdir(_ROOT)
    dataroot = Path(args.dataroot)
    out_dir = dataroot / args.out
    out_dir.mkdir(parents=True, exist_ok=True)
    radar_dir = dataroot / args.radar_dir

    with open(dataroot / args.ann, 'rb') as f:
        data_list = pickle.load(f)['data_list']
    if args.limit:
        data_list = data_list[:args.limit]
    todo = [d for d in data_list
            if args.overwrite or not (out_dir / f"{d['token']}.npy").exists()]
    print(f'{len(todo)}/{len(data_list)} samples to paint -> {out_dir}')
    if not todo:
        return

    from mmdet3d.apis import init_model
    device = 'cuda:0'
    model = init_model(args.fcos3d_config, args.checkpoint, device=device)
    model.eval()

    loader = torch.utils.data.DataLoader(
        ImageSet(todo, dataroot), batch_size=1, num_workers=2,
        collate_fn=collate, pin_memory=True)

    t0 = time.time()
    n_pts = n_painted = 0
    for k, (i, imgs, calib) in enumerate(loader):
        info = todo[i]
        pts = np.load(radar_dir / f"{info['token']}.npy")
        l2e = np.asarray(info['lidar_points']['lidar2ego'], dtype=np.float32)
        paint = paint_sample(model, imgs, calib, pts, l2e, args, device)
        np.save(out_dir / f"{info['token']}.npy", paint.astype(np.float16))
        n_pts += pts.shape[1]
        n_painted += int((paint[13] > 0).sum())
        if (k + 1) % 100 == 0:
            r = (k + 1) / (time.time() - t0)
            eta = (len(todo) - k - 1) / r / 3600
            print(f'  [{k + 1:5d}/{len(todo)}] {r:4.2f} samples/s, '
                  f'ETA {eta:.1f} h, painted {n_painted / max(n_pts, 1):.1%}',
                  flush=True)

    meta = dict(vars(args))
    meta['painted_fraction'] = n_painted / max(n_pts, 1)
    with open(out_dir / '_meta.json', 'w') as f:
        json.dump(meta, f, indent=2, default=str)
    print(f'Done: {len(todo)} samples, '
          f'painted fraction {n_painted / max(n_pts, 1):.1%}')


if __name__ == '__main__':
    main()
