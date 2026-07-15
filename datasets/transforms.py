"""
Custom transforms for the BEV fusion pipeline.

Exported transforms:

- ``CollectCameraExtrinsics`` — packs per-camera intrinsics (3x3) and
  cam2ego extrinsics (4x4) into stacked tensors ``cam2img`` (N,3,3)
  and ``cam2ego`` (N,4,4).

- ``ResizeMultiViewImage`` — resizes multi-view images and rescales
  intrinsics.

- ``BEVHorizontalFlip`` — randomly mirrors the BEV scene along the
  y-axis (left/right), consistently updating images, intrinsics,
  extrinsics, GT boxes, and radar BEV.

- ``BEVGlobalRotation`` — random z-axis rotation applied to GT boxes,
  camera extrinsics, and radar BEV (spatial + velocity channels).

- ``PackBEVFusionInputs`` — extends ``Pack3DDetInputs`` to also forward
  ``radar_bev``, ``cam2img``, and ``cam2ego`` into the model's
  ``inputs`` dict.
"""

from __future__ import annotations

import numpy as np
import torch

from mmcv.transforms.base import BaseTransform
from mmdet3d.registry import TRANSFORMS
from mmdet3d.datasets.transforms.formating import Pack3DDetInputs


@TRANSFORMS.register_module()
class CollectCameraExtrinsics(BaseTransform):
    """Stack per-camera intrinsics + cam2ego matrices into tensors.

    Reads from ``results['images'][CAM]['cam2img']`` and
    ``results['images'][CAM]['cam2ego']`` (the format mmdet3d 1.x writes
    when parsing nuScenes infos). Adds:

        results['cam2img']  -> (N, 3, 3) float32
        results['cam2ego']  -> (N, 4, 4) float32

    Args:
        cam_order: list of CAM channel names in fixed order. Default is
            the standard nuScenes ordering.
    """

    DEFAULT_ORDER = (
        'CAM_FRONT', 'CAM_FRONT_RIGHT', 'CAM_FRONT_LEFT',
        'CAM_BACK', 'CAM_BACK_LEFT', 'CAM_BACK_RIGHT',
    )

    def __init__(self, cam_order=None):
        self.cam_order = tuple(cam_order or self.DEFAULT_ORDER)

    def transform(self, results: dict) -> dict:
        images = results.get('images')
        if not images:
            raise KeyError(
                "CollectCameraExtrinsics requires results['images'] populated "
                'with per-CAM dicts (use mmdet3d nuScenes info loader).')

        cam2img = []
        cam2ego = []
        for cam in self.cam_order:
            entry = images[cam]
            K = np.asarray(entry['cam2img'], dtype=np.float32)
            if K.shape == (4, 4):
                K = K[:3, :3]
            cam2img.append(K)
            # Use inv(lidar2cam) instead of cam2ego so camera frustum points
            # land in the LIDAR keyframe frame, matching GT box coordinates.
            # cam2ego uses the camera's own timestamp ego pose (10-50 ms off),
            # causing up to ~0.75 m drift at urban speeds.
            L = np.asarray(entry['lidar2cam'], dtype=np.float32)
            if L.shape == (3, 4):
                L_ = np.eye(4, dtype=np.float32)
                L_[:3, :4] = L
                L = L_
            cam2ego.append(np.linalg.inv(L))

        results['cam2img'] = np.stack(cam2img, axis=0)
        results['cam2ego'] = np.stack(cam2ego, axis=0)
        return results

    def __repr__(self) -> str:
        return f'{type(self).__name__}(cam_order={self.cam_order})'


@TRANSFORMS.register_module()
class ResizeMultiViewImage(BaseTransform):
    """Resize all multi-view images and rescale intrinsics in lockstep.

    Operates on ``results['img']`` (a list of HxWx3 ndarrays produced by
    ``LoadMultiViewImageFromFiles``) and on ``results['cam2img']`` if
    present (otherwise it walks ``results['images'][CAM]['cam2img']``).
    """

    def __init__(self, size: tuple[int, int]):
        # size is (H, W) — same convention as image_hw in LSS.
        self.size = (int(size[0]), int(size[1]))

    def transform(self, results: dict) -> dict:
        import cv2

        imgs = results.get('img')
        if imgs is None:
            return results
        H_new, W_new = self.size

        new_imgs = []
        scales = []  # (sx, sy) per image
        for img in imgs:
            H_old, W_old = img.shape[:2]
            sx = W_new / float(W_old)
            sy = H_new / float(H_old)
            new_imgs.append(cv2.resize(img, (W_new, H_new), interpolation=cv2.INTER_LINEAR))
            scales.append((sx, sy))

        results['img'] = new_imgs
        results['img_shape'] = [(H_new, W_new) for _ in new_imgs]
        results['ori_shape'] = results.get('ori_shape', [(H_new, W_new) for _ in new_imgs])
        if 'pad_shape' in results:
            results['pad_shape'] = [(H_new, W_new) for _ in new_imgs]

        # Rescale intrinsics. Cope with both stacked tensor and per-cam dict.
        if 'cam2img' in results:
            K = np.asarray(results['cam2img']).astype(np.float32)
            for i, (sx, sy) in enumerate(scales):
                K[i, 0, 0] *= sx
                K[i, 0, 2] *= sx
                K[i, 1, 1] *= sy
                K[i, 1, 2] *= sy
            results['cam2img'] = K
        return results

    def __repr__(self) -> str:
        return f'{type(self).__name__}(size={self.size})'


@TRANSFORMS.register_module()
class BEVHorizontalFlip(BaseTransform):
    """Randomly mirror the BEV scene along the y-axis with probability p.

    Consistent augmentation across all modalities:
      - Flips each camera image horizontally and updates cx: cx → W − cx.
      - Applies cam2ego_new = diag(1,−1,1,1) @ cam2ego @ diag(−1,1,1,1),
        reflecting the ego y-axis and the camera x-axis simultaneously.
      - Swaps FRONT_LEFT↔FRONT_RIGHT and BACK_LEFT↔BACK_RIGHT (indices
        1↔2 and 4↔5 in the standard 6-camera ordering).
      - Calls LiDARInstance3DBoxes.flip('horizontal') on GT boxes.
      - Mirrors radar_bev along the row axis (row = y in our BEV layout)
        and negates the vy channel (channel 2).
      - Negates y and vy of raw radar_points (rows 1 and 4) when present;
        the validity mask is unchanged.
    """

    # [FRONT, FR, FL, BACK, BL, BR] → swap 1↔2 and 4↔5
    _SWAP = [0, 2, 1, 3, 5, 4]

    # cam2ego_new[i,j] = F_EGO[i] * cam2ego[i,j] * F_CAM[j]
    _F_EGO = np.array([1., -1., 1., 1.], dtype=np.float32)
    _F_CAM = np.array([-1., 1., 1., 1.], dtype=np.float32)

    def __init__(self, prob: float = 0.5):
        self.prob = prob

    def transform(self, results: dict) -> dict:
        if np.random.random() >= self.prob:
            return results

        import cv2

        imgs = results.get('img', [])
        if imgs:
            W = imgs[0].shape[1]
            results['img'] = [cv2.flip(img, 1) for img in imgs]
            K = results['cam2img'].copy()      # (N, 3, 3)
            K[:, 0, 2] = W - K[:, 0, 2]
            results['cam2img'] = K[self._SWAP]

        # Camera-free pipelines (rpp) carry no extrinsics — skip cleanly.
        if 'cam2ego' in results:
            M = results['cam2ego'].copy()      # (N, 4, 4)
            M = (M
                 * self._F_EGO[np.newaxis, :, np.newaxis]
                 * self._F_CAM[np.newaxis, np.newaxis, :])
            results['cam2ego'] = M[self._SWAP]

        if 'gt_bboxes_3d' in results:
            results['gt_bboxes_3d'].flip('horizontal')

        bev = results.get('radar_bev')
        if bev is not None:
            if isinstance(bev, np.ndarray):
                bev = bev[:, ::-1, :].copy()
                bev[2] *= -1                   # negate vy (y-velocity mirrors)
            else:
                bev = bev.flip(-2).clone()
                bev[2] *= -1
            results['radar_bev'] = bev

        pts = results.get('radar_points')
        if pts is not None:
            # Rows: [x, y, z, vx, vy, rcs]. Mirror along ego y-axis: y→-y, vy→-vy.
            if isinstance(pts, np.ndarray):
                pts = pts.copy()
                pts[1] *= -1
                pts[4] *= -1
            else:
                pts = pts.clone()
                pts[1] *= -1
                pts[4] *= -1
            results['radar_points'] = pts

        return results

    def __repr__(self) -> str:
        return f'{type(self).__name__}(prob={self.prob})'


@TRANSFORMS.register_module()
class BEVGlobalRotation(BaseTransform):
    """Random z-axis rotation for camera+radar BEV fusion.

    Samples θ ~ U(rot_range) and consistently applies it to:
      - GT boxes: LiDARInstance3DBoxes.rotate(θ) (CCW in ego x-y plane).
      - Camera extrinsics: cam2ego_new = R_z(θ) @ cam2ego.
      - Radar BEV: spatial rotation by θ CCW + rotation of velocity
        channels (vx, vy) to the new frame.

    The radar BEV spatial rotation uses scipy.ndimage.rotate with
    angle = −θ_deg (scipy's (row,col) CCW equals world (x,y) CW, so
    the minus sign recovers the correct CCW world rotation).
    """

    def __init__(self, rot_range=(-0.3927, 0.3927)):  # ±π/8 radians
        self.rot_range = rot_range

    def transform(self, results: dict) -> dict:
        from scipy.ndimage import rotate as ndimage_rotate

        angle = float(np.random.uniform(*self.rot_range))
        cos_a, sin_a = float(np.cos(angle)), float(np.sin(angle))

        if 'gt_bboxes_3d' in results and len(results['gt_bboxes_3d']) > 0:
            results['gt_bboxes_3d'].rotate(angle)

        R_z = np.array([
            [cos_a, -sin_a, 0., 0.],
            [sin_a,  cos_a, 0., 0.],
            [0.,     0.,    1., 0.],
            [0.,     0.,    0., 1.],
        ], dtype=np.float32)
        # R_z @ (N,4,4): numpy broadcasts the leading batch dim correctly.
        # Camera-free pipelines (rpp) carry no extrinsics — skip cleanly.
        if 'cam2ego' in results:
            results['cam2ego'] = (R_z @ results['cam2ego']).astype(np.float32)

        bev = results.get('radar_bev')
        if bev is not None:
            angle_deg = float(np.degrees(angle))
            bev_np = bev.numpy() if isinstance(bev, torch.Tensor) else bev

            # Spatial rotation. scipy CCW in (row,col) = world CW → negate.
            rotated = np.stack([
                ndimage_rotate(ch, -angle_deg, reshape=False, order=1, cval=0.)
                for ch in bev_np
            ], axis=0).astype(np.float32)

            # Rotate velocity vectors (ch1=vx, ch2=vy) to the new ego frame.
            vx, vy = rotated[1].copy(), rotated[2].copy()
            rotated[1] = cos_a * vx - sin_a * vy
            rotated[2] = sin_a * vx + cos_a * vy

            results['radar_bev'] = (
                torch.from_numpy(rotated)
                if isinstance(bev, torch.Tensor) else rotated
            )

        pts = results.get('radar_points')
        if pts is not None:
            is_tensor = isinstance(pts, torch.Tensor)
            arr = pts.numpy() if is_tensor else pts.copy()
            x, y = arr[0].copy(), arr[1].copy()
            arr[0] = cos_a * x - sin_a * y
            arr[1] = sin_a * x + cos_a * y
            vx, vy = arr[3].copy(), arr[4].copy()
            arr[3] = cos_a * vx - sin_a * vy
            arr[4] = sin_a * vx + cos_a * vy
            results['radar_points'] = (
                torch.from_numpy(arr) if is_tensor else arr
            )

        return results

    def __repr__(self) -> str:
        return f'{type(self).__name__}(rot_range={self.rot_range})'


@TRANSFORMS.register_module()
class PackBEVFusionInputs(Pack3DDetInputs):
    """``Pack3DDetInputs`` extended with the BEV-fusion-specific keys.

    Adds ``radar_bev`` (Cr, Hb, Wb), ``cam2img`` (N, 3, 3) and
    ``cam2ego`` (N, 4, 4) to the model's ``inputs`` dict so they are
    auto-moved to GPU by the data preprocessor and visible to the
    detector's forward.

    These extra keys are *not* listed in ``self.keys`` (the parent
    raises ``NotImplementedError`` for unknown keys). Instead the
    subclass injects them into ``inputs`` after the parent finishes.
    """

    BEV_FUSION_INPUT_KEYS = (
        'radar_bev', 'cam2img', 'cam2ego', 'radar_points', 'radar_points_mask',
    )
    # Boolean keys that must keep dtype=bool through the cast below — the
    # mask is a validity flag, not a numeric tensor.
    _BOOL_KEYS = ('radar_points_mask',)

    def pack_single_results(self, results: dict) -> dict:
        # Cast our extras to torch tensors before delegating, so the
        # data preprocessor downstream sees consistent types.
        for key in self.BEV_FUSION_INPUT_KEYS:
            if key in results and not isinstance(results[key], torch.Tensor):
                if key in self._BOOL_KEYS:
                    arr = np.asarray(results[key])
                    arr = arr.astype(bool, copy=False)
                    results[key] = torch.from_numpy(np.ascontiguousarray(arr))
                else:
                    arr = np.asarray(results[key], dtype=np.float32)
                    results[key] = torch.from_numpy(np.ascontiguousarray(arr))

        packed = super().pack_single_results(results)

        # Parent only forwards keys it lists in INPUTS_KEYS. Inject ours.
        inputs = packed.get('inputs', {})
        for key in self.BEV_FUSION_INPUT_KEYS:
            if key in results:
                inputs[key] = results[key]
        packed['inputs'] = inputs
        return packed
