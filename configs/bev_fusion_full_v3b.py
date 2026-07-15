# BEV fusion — full training config v3b (= v3 minus CBGS).
#
# v3 measured 2.6 s/iter and CBGS oversamples to ~15.3k iters/epoch
# (4.3× the base nuScenes train set, dominated by ~17× oversample on
# construction_vehicle). 24 epochs at that rate ≈ 10 days, blowing the
# wallclock budget. v3b drops the CBGSDataset wrapper, taking iters back
# to ~3.5k/epoch and total wallclock to ~2.5 days for 24 epochs.
#
# Tradeoff: rare-class APs (construction_vehicle, trailer) take a hit
# vs CBGS-on. The architecture-level v3 changes already address most of
# what CBGS was added to fix in v2 (0.4 m BEV resolves small classes,
# wider channels and FCOS3D init give richer features, RadarGatedFusion
# concentrates radar evidence). Expected loss vs v3 final: a few points
# on the rarest two classes; everything else unchanged.
#
# Everything in v3 is inherited unchanged: model, optimizer, schedule,
# warmup, EMA, save-best, val_interval, etc. Only the train dataloader
# dataset block is replaced.

_base_ = './bev_fusion_full_v3.py'

# _base_ does not export underscore-prefixed names, so the pipeline used
# by the train dataset must be redeclared here.
bev_range = 51.2
bev_res = 0.4
image_hw = (256, 704)

class_names = [
    'car', 'truck', 'construction_vehicle', 'bus', 'trailer',
    'barrier', 'motorcycle', 'bicycle', 'pedestrian', 'traffic_cone',
]

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

_train_pipeline_aug = [
    dict(type='LoadMultiViewImageFromFiles', to_float32=True),
    dict(type='CollectCameraExtrinsics'),
    dict(type='ResizeMultiViewImage', size=image_hw),
    dict(type='LoadRadarBEV', bev_range=bev_range, bev_res=bev_res),
    dict(type='LoadAnnotations3D', with_bbox_3d=True, with_label_3d=True),
    dict(type='MultiViewWrapper',
         transforms=[dict(type='PhotoMetricDistortion3D')]),
    dict(type='BEVHorizontalFlip', prob=0.5),
    dict(type='BEVGlobalRotation', rot_range=(-0.3927, 0.3927)),
    dict(type='PackBEVFusionInputs',
         keys=['img', 'gt_bboxes_3d', 'gt_labels_3d'],
         meta_keys=('sample_idx', 'box_mode_3d', 'box_type_3d')),
]

# Replace the train dataset: unwrap CBGS, use NuScenesRadarDataset directly.
# _delete_=True wipes the inherited CBGSDataset config wholesale.
train_dataloader = dict(
    dataset=dict(
        _delete_=True,
        type='NuScenesRadarDataset',
        data_root='data/nuscenes/',
        ann_file='nuscenes_infos_train.pkl',
        radar_bev_dir='radar_bev',
        pipeline=_train_pipeline_aug,
        metainfo=dict(version='v1.0-trainval', classes=class_names),
        modality=dict(use_camera=True, use_lidar=False),
        data_prefix=data_prefix,
        test_mode=False,
        box_type_3d='LiDAR',
        indices=None,
    ),
)

# Stricter inference thresholds to avoid > 500 boxes per sample at epoch 8+.
# Both bbox_coder (decode-time filtering) and test_cfg need adjustment.
model = dict(
    bbox_head=dict(
        bbox_coder=dict(score_threshold=0.15)  # 0.05 → 0.15 (decode-time filter)
    ),
    test_cfg=dict(score_threshold=0.15)  # also update test-time threshold
)

work_dir = 'work_dirs/bev_fusion_full_v3b'
