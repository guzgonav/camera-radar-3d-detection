# v3b — full 24-epoch retrain WITH the radar frame fix.
#
# The radar cache was rasterised in the nuScenes ego frame while GT/camera
# geometry are in the LIDAR_TOP sensor frame (~90° yaw + ~1 m offset).
# Setting LoadRadarBEV(to_lidar_frame=True) re-expresses the raster in the
# LIDAR frame, which flips radar from inert to load-bearing (shuffle-ΔNDS
# +0.1142 at full convergence, see the BEV fusion chapter). This is the
# reference v3b checkpoint used for the hard-scenes / error-by-distance
# comparisons.
#
# Exactly ONE thing changes relative to bev_fusion_full_v3b.py: both
# LoadRadarBEV calls (train + val/test pipelines) get to_lidar_frame=True.
# Everything else — 24 epochs, EMA, save-best, val_interval=2, score
# thresholds, CBGS-off — is inherited unchanged.

_base_ = './bev_fusion_full_v3b.py'

bev_range = 51.2
bev_res = 0.4
image_hw = (256, 704)

class_names = [
    'car', 'truck', 'construction_vehicle', 'bus', 'trailer',
    'barrier', 'motorcycle', 'bicycle', 'pedestrian', 'traffic_cone',
]

data_root = 'data/nuscenes/'
radar_bev_dir = 'radar_bev'
metainfo = dict(version='v1.0-trainval', classes=class_names)

data_prefix = dict(
    CAM_FRONT='samples/CAM_FRONT',
    CAM_FRONT_RIGHT='samples/CAM_FRONT_RIGHT',
    CAM_FRONT_LEFT='samples/CAM_FRONT_LEFT',
    CAM_BACK='samples/CAM_BACK',
    CAM_BACK_LEFT='samples/CAM_BACK_LEFT',
    CAM_BACK_RIGHT='samples/CAM_BACK_RIGHT',
    pts='samples/LIDAR_TOP',
    sweeps='sweeps/LIDAR_TOP',
)

# ------------------------------------------------------------------------
# Pipelines — identical to v3b except to_lidar_frame=True.
# (_base_ underscore-prefixed pipeline vars are not exported; redeclared.)
# ------------------------------------------------------------------------
_train_pipeline_aug = [
    dict(type='LoadMultiViewImageFromFiles', to_float32=True),
    dict(type='CollectCameraExtrinsics'),
    dict(type='ResizeMultiViewImage', size=image_hw),
    dict(type='LoadRadarBEV', bev_range=bev_range, bev_res=bev_res,
         to_lidar_frame=True),                    # ← THE fix
    dict(type='LoadAnnotations3D', with_bbox_3d=True, with_label_3d=True),
    dict(type='MultiViewWrapper',
         transforms=[dict(type='PhotoMetricDistortion3D')]),
    dict(type='BEVHorizontalFlip', prob=0.5),
    dict(type='BEVGlobalRotation', rot_range=(-0.3927, 0.3927)),
    dict(type='PackBEVFusionInputs',
         keys=['img', 'gt_bboxes_3d', 'gt_labels_3d'],
         meta_keys=('sample_idx', 'box_mode_3d', 'box_type_3d')),
]

_test_pipeline = [
    dict(type='LoadMultiViewImageFromFiles', to_float32=True),
    dict(type='CollectCameraExtrinsics'),
    dict(type='ResizeMultiViewImage', size=image_hw),
    dict(type='LoadRadarBEV', bev_range=bev_range, bev_res=bev_res,
         to_lidar_frame=True),                    # ← THE fix
    dict(type='PackBEVFusionInputs',
         keys=['img'],
         meta_keys=('sample_idx', 'box_mode_3d', 'box_type_3d')),
]

# Same train dataset as v3b (unwrapped CBGS, NuScenesRadarDataset), just the
# pipeline swapped in above.
train_dataloader = dict(
    dataset=dict(
        _delete_=True,
        type='NuScenesRadarDataset',
        data_root=data_root,
        ann_file='nuscenes_infos_train.pkl',
        radar_bev_dir=radar_bev_dir,
        pipeline=_train_pipeline_aug,
        metainfo=metainfo,
        modality=dict(use_camera=True, use_lidar=False),
        data_prefix=data_prefix,
        test_mode=False,
        box_type_3d='LiDAR',
        indices=None,
    ),
)

val_dataloader = dict(dataset=dict(pipeline=_test_pipeline))
test_dataloader = val_dataloader

work_dir = 'work_dirs/bev_fusion_full_v3b_framefix'
