# BEV fusion — full training config (v1.0-trainval, 12 epochs).
#
# Inherits the architecture from the mini config and overrides the data
# paths, schedule, and AMP/optimiser settings for the real run.

_base_ = './bev_fusion_mini.py'

# ------------------------------------------------------------------------
# Data: full v1.0-trainval split
# ------------------------------------------------------------------------
data_root = 'data/nuscenes/'
radar_bev_dir = 'radar_bev'

metainfo = dict(version='v1.0-trainval', classes={{_base_.class_names}})

# Augmented pipeline for the full run.  The mini base has no augmentation
# (intentional — it is an overfit gate); here we add the three standard
# BEV augmentations before the full 12-epoch training run.
_train_pipeline_aug = [
    dict(type='LoadMultiViewImageFromFiles', to_float32=True),
    dict(type='CollectCameraExtrinsics'),
    dict(type='ResizeMultiViewImage', size={{_base_.image_hw}}),
    dict(type='LoadRadarBEV',
         bev_range={{_base_.bev_range}}, bev_res={{_base_.bev_res}}),
    dict(type='LoadAnnotations3D', with_bbox_3d=True, with_label_3d=True),
    # Colour jitter applied independently to each camera with shared params.
    dict(type='MultiViewWrapper',
         transforms=[dict(type='PhotoMetricDistortion3D')]),
    # Geometric augmentations (consistent across cameras, GT, and radar BEV).
    dict(type='BEVHorizontalFlip', prob=0.5),
    dict(type='BEVGlobalRotation', rot_range=(-0.3927, 0.3927)),
    dict(type='PackBEVFusionInputs',
         keys=['img', 'gt_bboxes_3d', 'gt_labels_3d'],
         meta_keys=('sample_idx', 'box_mode_3d', 'box_type_3d')),
]

train_dataloader = dict(
    batch_size=4,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=dict(
        type='NuScenesRadarDataset',
        data_root=data_root,
        ann_file='nuscenes_infos_train.pkl',
        radar_bev_dir=radar_bev_dir,
        pipeline=_train_pipeline_aug,
        metainfo=metainfo,
        modality=dict(use_camera=True, use_lidar=False),
        test_mode=False,
        box_type_3d='LiDAR',
        indices=None,  # use the whole train split
    ),
)

val_dataloader = dict(
    batch_size=1,
    num_workers=4,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type='NuScenesRadarDataset',
        data_root=data_root,
        ann_file='nuscenes_infos_val.pkl',
        radar_bev_dir=radar_bev_dir,
        pipeline={{_base_.test_pipeline}},
        metainfo=metainfo,
        modality=dict(use_camera=True, use_lidar=False),
        test_mode=True,
        box_type_3d='LiDAR',
    ),
)
test_dataloader = val_dataloader

val_evaluator = dict(
    type='NuScenesMetric',
    data_root=data_root,
    ann_file=data_root + 'nuscenes_infos_val.pkl',
    metric='bbox',
)
test_evaluator = val_evaluator

# ------------------------------------------------------------------------
# Schedule: 12 epochs, FP16, AdamW + cosine
# ------------------------------------------------------------------------
train_cfg = dict(_delete_=True, type='EpochBasedTrainLoop', max_epochs=12, val_interval=2)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')

optim_wrapper = dict(
    type='AmpOptimWrapper',  # FP16 mixed precision
    loss_scale='dynamic',
    optimizer=dict(type='AdamW', lr=4e-4, weight_decay=0.01),
    clip_grad=dict(max_norm=35, norm_type=2),
)
param_scheduler = [
    dict(
        type='LinearLR', start_factor=1e-3,
        by_epoch=False, begin=0, end=1000),
    dict(
        type='CosineAnnealingLR',
        T_max=12, eta_min=1e-6,
        begin=0, end=12, by_epoch=True),
]

default_hooks = dict(
    timer=dict(type='IterTimerHook'),
    logger=dict(type='LoggerHook', interval=50),
    param_scheduler=dict(type='ParamSchedulerHook'),
    checkpoint=dict(
        type='CheckpointHook', interval=2,
        by_epoch=True, max_keep_ckpts=3,
        save_best='NuScenes metric/pred_instances_3d_NuScenes/NDS',
        rule='greater'),
    sampler_seed=dict(type='DistSamplerSeedHook'),
)
log_processor = dict(type='LogProcessor', window_size=50, by_epoch=True)

work_dir = 'work_dirs/bev_fusion_full'

env_cfg = dict(
    cudnn_benchmark=True,
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0),
    dist_cfg=dict(backend='nccl'),
)
