# BEV fusion — mini config for plumbing / overfit sanity.
#
# 8 train samples from the mini split, 200 iterations. NOT for real
# numbers — its only purpose is to catch dimension/wiring bugs before
# committing to a multi-day full-training run (Step 7 in the plan).

custom_imports = dict(
    imports=['datasets', 'models'],
    allow_failed_imports=False)

# ------------------------------------------------------------------------
# Geometry
# ------------------------------------------------------------------------
bev_range = 51.2
bev_res = 0.8
bev_size = int(2 * bev_range / bev_res)              # 128
image_hw = (256, 704)
feat_stride = 16  # ResNet stage C4 stride (out_indices=(2,))
n_depth_bins = 64

class_names = [
    'car', 'truck', 'construction_vehicle', 'bus', 'trailer',
    'barrier', 'motorcycle', 'bicycle', 'pedestrian', 'traffic_cone',
]
metainfo = dict(version='v1.0-mini', classes=class_names)

# Channel budgets
img_neck_out_channels = 64
cam_bev_channels = 32
radar_bev_channels = 32
fused_channels = 128

point_cloud_range = [-bev_range, -bev_range, -5.0, bev_range, bev_range, 3.0]

# ------------------------------------------------------------------------
# Model
# ------------------------------------------------------------------------
model = dict(
    type='BEVFusionDetector',
    data_preprocessor=dict(
        type='BEVFusionDataPreprocessor',
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        bgr_to_rgb=True,
        pad_size_divisor=32,
    ),
    img_backbone=dict(
        type='mmdet.ResNet',
        depth=50,
        num_stages=3,
        strides=(1, 2, 2),
        dilations=(1, 1, 1),
        out_indices=(2, ),  # only the C4 stage feeds the FPN
        frozen_stages=1,
        norm_cfg=dict(type='BN', requires_grad=True),
        norm_eval=True,
        style='pytorch',
        init_cfg=dict(type='Pretrained', checkpoint='torchvision://resnet50'),
    ),
    img_neck=dict(
        type='mmdet.FPN',
        in_channels=[1024],
        out_channels=img_neck_out_channels,
        num_outs=1,
    ),
    view_transform=dict(
        type='LSSViewTransform',
        in_channels=img_neck_out_channels,
        out_channels=cam_bev_channels,
        feat_stride=feat_stride,
        image_hw=image_hw,
        depth_bins=(1.0, 60.0, n_depth_bins),
        bev_range=bev_range,
        bev_res=bev_res,
    ),
    radar_encoder=dict(
        type='RadarBEVEncoder',
        in_channels=5,
        hidden_channels=32,
        out_channels=radar_bev_channels,
    ),
    fusion_neck=dict(
        type='BEVFusionNeck',
        camera_channels=cam_bev_channels,
        radar_channels=radar_bev_channels,
        out_channels=fused_channels,
    ),
    bbox_head=dict(
        type='CenterHead',
        in_channels=fused_channels,
        tasks=[
            dict(num_class=1, class_names=['car']),
            dict(num_class=2, class_names=['truck', 'construction_vehicle']),
            dict(num_class=2, class_names=['bus', 'trailer']),
            dict(num_class=1, class_names=['barrier']),
            dict(num_class=2, class_names=['motorcycle', 'bicycle']),
            dict(num_class=2, class_names=['pedestrian', 'traffic_cone']),
        ],
        common_heads=dict(
            reg=(2, 2), height=(1, 2), dim=(3, 2), rot=(2, 2), vel=(2, 2)),
        share_conv_channel=64,
        bbox_coder=dict(
            type='CenterPointBBoxCoder',
            pc_range=[-bev_range, -bev_range],
            post_center_range=point_cloud_range,
            max_num=500,
            score_threshold=0.1,
            out_size_factor=1,
            voxel_size=[bev_res, bev_res],
            code_size=9,
        ),
        separate_head=dict(
            type='SeparateHead', init_bias=-2.19, final_kernel=3),
        loss_cls=dict(type='mmdet.GaussianFocalLoss', reduction='mean'),
        loss_bbox=dict(
            type='mmdet.L1Loss', reduction='mean', loss_weight=0.25),
        norm_bbox=True,
    ),
    train_cfg=dict(
        grid_size=[bev_size, bev_size, 1],
        voxel_size=[bev_res, bev_res, 8.0],
        point_cloud_range=point_cloud_range,
        out_size_factor=1,
        dense_reg=1,
        gaussian_overlap=0.1,
        max_objs=500,
        min_radius=1,
        code_weights=[1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.2, 0.2],
    ),
    test_cfg=dict(
        post_center_limit_range=point_cloud_range,
        max_per_img=500,
        max_pool_nms=False,
        min_radius=[4, 12, 10, 1, 0.85, 0.175],
        score_threshold=0.1,
        pc_range=[-bev_range, -bev_range],
        out_size_factor=1,
        voxel_size=[bev_res, bev_res],
        nms_type='rotate',
        pre_max_size=1000,
        post_max_size=83,
        nms_thr=0.2,
    ),
)

# ------------------------------------------------------------------------
# Data
# ------------------------------------------------------------------------
data_root = 'data/nuscenes-mini/'
radar_bev_dir = 'radar_bev'
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

train_pipeline = [
    dict(type='LoadMultiViewImageFromFiles', to_float32=True),
    dict(type='CollectCameraExtrinsics'),
    dict(type='ResizeMultiViewImage', size=image_hw),
    dict(type='LoadRadarBEV', bev_range=bev_range, bev_res=bev_res),
    dict(
        type='LoadAnnotations3D',
        with_bbox_3d=True,
        with_label_3d=True,
    ),
    dict(
        type='PackBEVFusionInputs',
        keys=['img', 'gt_bboxes_3d', 'gt_labels_3d'],
        meta_keys=('sample_idx', 'box_mode_3d', 'box_type_3d'),
    ),
]

test_pipeline = [
    dict(type='LoadMultiViewImageFromFiles', to_float32=True),
    dict(type='CollectCameraExtrinsics'),
    dict(type='ResizeMultiViewImage', size=image_hw),
    dict(type='LoadRadarBEV', bev_range=bev_range, bev_res=bev_res),
    dict(
        type='PackBEVFusionInputs',
        keys=['img'],
        meta_keys=('sample_idx', 'box_mode_3d', 'box_type_3d'),
    ),
]

# Tiny "8 samples" overfit dataset is created via indices_first_n in the
# dataset class; mmdet3d 1.4 doesn't expose that, so we use sampler.indices.
train_dataloader = dict(
    batch_size=2,
    num_workers=0,
    persistent_workers=False,
    sampler=dict(type='InfiniteSampler', shuffle=True),
    dataset=dict(
        type='NuScenesRadarDataset',
        data_root=data_root,
        ann_file='nuscenes_infos_train.pkl',
        radar_bev_dir=radar_bev_dir,
        pipeline=train_pipeline,
        metainfo=metainfo,
        modality=dict(use_camera=True, use_lidar=False),
        data_prefix=data_prefix,
        test_mode=False,
        box_type_3d='LiDAR',
        # Take only the first 8 samples for the overfit gate.
        indices=8,
    ),
)
val_dataloader = dict(
    batch_size=1,
    num_workers=0,
    persistent_workers=False,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type='NuScenesRadarDataset',
        data_root=data_root,
        ann_file='nuscenes_infos_val.pkl',
        radar_bev_dir=radar_bev_dir,
        pipeline=test_pipeline,
        metainfo=metainfo,
        modality=dict(use_camera=True, use_lidar=False),
        data_prefix=data_prefix,
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
# Schedule
# ------------------------------------------------------------------------
train_cfg = dict(type='IterBasedTrainLoop', max_iters=200, val_interval=200)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')

optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(type='AdamW', lr=2e-4, weight_decay=0.01),
    clip_grad=dict(max_norm=35, norm_type=2),
)
param_scheduler = [
    dict(type='LinearLR', start_factor=1.0, by_epoch=False, begin=0, end=10),
]

# ------------------------------------------------------------------------
# Runtime
# ------------------------------------------------------------------------
default_scope = 'mmdet3d'
default_hooks = dict(
    timer=dict(type='IterTimerHook'),
    logger=dict(type='LoggerHook', interval=20),
    param_scheduler=dict(type='ParamSchedulerHook'),
    checkpoint=dict(type='CheckpointHook', interval=200, by_epoch=False),
    sampler_seed=dict(type='DistSamplerSeedHook'),
)
env_cfg = dict(
    cudnn_benchmark=False,
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0),
    dist_cfg=dict(backend='nccl'),
)
log_processor = dict(type='LogProcessor', window_size=20, by_epoch=False)
log_level = 'INFO'
load_from = None
resume = False
work_dir = 'work_dirs/bev_fusion_mini'

vis_backends = [dict(type='LocalVisBackend')]
visualizer = dict(
    type='Det3DLocalVisualizer', vis_backends=vis_backends, name='visualizer')

randomness = dict(seed=0, deterministic=False)
