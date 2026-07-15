# BEV fusion — full training config v3 (v1.0-trainval, 24 epochs).
#
# Major v3 changes vs v2, fixing why v1/v2 underperformed:
#
#   - Backbone:  R-50 (torchvision) → R-101 caffe + DCN, FCOS3D init
#   - Norm:      RGB ImageNet → BGR caffe (required by FCOS3D weights)
#   - FPN:       single-scale [1024] → multi-scale [1024, 2048] (C4+C5)
#   - Channels:  fpn 64/cam 32/fused 128 → 256/80/256
#   - BEV res:   0.8 m (128×128) → 0.4 m (256×256)
#   - Fusion:    plain concat → RadarGatedFusion (radar attn boosts cam)
#   - Vel loss:  code_weights[8:10] = 0.2 → 1.0 (radar IS the vel signal)
#   - Decoder:   post_max_size 83→200, score_thr 0.10→0.05,
#                min_radius rescaled for 0.4 m BEV
#   - CBGS:      added (rare-class balance)
#   - EMA:       added (free +1-2 NDS)
#   - Save best: NDS only → NDS *and* mAP
#   - Warmup:    1k iters/sf=1e-3 → 2k iters/sf=1e-4
#   - LR:        8e-4 → 2e-4 (R-101+DCN sensitivity)
#   - Workers:   4/persist=True → 2/persist=False (RAM pressure on host)

_base_ = './bev_fusion_mini.py'

# ------------------------------------------------------------------------
# Geometry — overrides for 0.4 m BEV
# ------------------------------------------------------------------------
bev_range = 51.2
bev_res = 0.4
bev_size = int(2 * bev_range / bev_res)              # 256
image_hw = (256, 704)
feat_stride = 16
n_depth_bins = 64

class_names = [
    'car', 'truck', 'construction_vehicle', 'bus', 'trailer',
    'barrier', 'motorcycle', 'bicycle', 'pedestrian', 'traffic_cone',
]

# Channel budgets (much wider than v1/v2)
img_neck_out_channels = 256
cam_bev_channels = 80
radar_bev_channels = 32
fused_channels = 256

point_cloud_range = [-bev_range, -bev_range, -5.0, bev_range, bev_range, 3.0]

# ------------------------------------------------------------------------
# Data
# ------------------------------------------------------------------------
data_root = 'data/nuscenes/'
radar_bev_dir = 'radar_bev'
metainfo = dict(version='v1.0-trainval', classes=class_names)

# nuScenes stores camera frames under samples/CAM_<VIEW>/; the .pkl only
# carries bare filenames, so the dataset needs these prefixes to resolve
# them. The base mini config sets this, but v3's _delete_=True on the
# train dataset wipes the inherited value — pass it again explicitly.
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

# FCOS3D R-101 caffe-style checkpoint — provides depth-aware features
# (substitute for LIDAR depth supervision; thesis is camera+radar only).
FCOS3D_R101_CKPT = (
    'checkpoints/fcos3d_r101_caffe_fpn_gn-head_dcn_2x8_1x_'
    'nus-mono3d_finetune_20210717_095645-8d806dc2.pth'
)

# ------------------------------------------------------------------------
# Model
# ------------------------------------------------------------------------
model = dict(
    type='BEVFusionDetector',
    # BGR caffe normalisation — required by FCOS3D R-101 weights.
    data_preprocessor=dict(
        type='BEVFusionDataPreprocessor',
        mean=[103.530, 116.280, 123.675],
        std=[1.0, 1.0, 1.0],
        bgr_to_rgb=False,
        pad_size_divisor=32,
    ),
    img_backbone=dict(
        type='mmdet.ResNet',
        depth=101,
        num_stages=4,
        strides=(1, 2, 2, 2),                         # standard ResNet — C2/C3/C4/C5
        dilations=(1, 1, 1, 1),
        out_indices=(2, 3),                           # C4 + C5 → FPN
        frozen_stages=1,
        norm_cfg=dict(type='BN', requires_grad=False),
        norm_eval=True,
        style='caffe',
        dcn=dict(type='DCNv2', deform_groups=1, fallback_on_stride=False),
        stage_with_dcn=(False, False, True, True),    # DCN on layer3, layer4
        init_cfg=dict(
            type='Pretrained',
            checkpoint=FCOS3D_R101_CKPT,
            prefix='backbone.',                       # only load backbone.* keys
        ),
    ),
    img_neck=dict(
        type='mmdet.FPN',
        in_channels=[1024, 2048],                     # C4, C5
        out_channels=img_neck_out_channels,
        num_outs=2,
    ),
    view_transform=dict(
        type='LSSViewTransform',
        in_channels=img_neck_out_channels,
        out_channels=cam_bev_channels,
        feat_stride=feat_stride,                      # FPN[0] is at C4 stride (16)
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
        type='RadarGatedFusion',
        camera_channels=cam_bev_channels,
        radar_channels=radar_bev_channels,
        out_channels=fused_channels,
        hidden_channels=fused_channels,
        boost_alpha=1.0,                              # radar can up to 2× cam features
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
        share_conv_channel=128,                       # 64 → 128 (matches wider input)
        bbox_coder=dict(
            type='CenterPointBBoxCoder',
            pc_range=[-bev_range, -bev_range],
            post_center_range=point_cloud_range,
            max_num=500,
            score_threshold=0.05,                     # 0.10 → 0.05
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
        grid_size=[bev_size, bev_size, 1],            # 256×256×1
        voxel_size=[bev_res, bev_res, 8.0],
        point_cloud_range=point_cloud_range,
        out_size_factor=1,
        dense_reg=1,
        gaussian_overlap=0.1,
        max_objs=500,
        min_radius=1,                                 # 1 cell × 0.4 m = 0.4 m floor
        # All ten code dims weighted 1.0. Velocity (last two) lifted from
        # the LIDAR default of 0.2 — radar is the velocity sensor here,
        # the loss must care about it.
        code_weights=[1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
    ),
    test_cfg=dict(
        post_center_limit_range=point_cloud_range,
        max_per_img=500,
        max_pool_nms=False,
        # NMS radii in head-output cells. At 0.4 m/cell with out_size_factor=1
        # these match the canonical CenterPoint *physical* radii
        # (originally [4, 12, 10, 1, 0.85, 0.175] cells at 0.6 m physical).
        # Per-task: car / truck+cv / bus+trailer / barrier / moto+bike / ped+cone.
        min_radius=[6.0, 18.0, 15.0, 1.5, 1.275, 0.2625],
        score_threshold=0.05,                         # 0.10 → 0.05
        pc_range=[-bev_range, -bev_range],
        out_size_factor=1,
        voxel_size=[bev_res, bev_res],
        nms_type='rotate',
        pre_max_size=1000,
        post_max_size=200,                            # 83 → 200
        nms_thr=0.2,
    ),
)

# ------------------------------------------------------------------------
# Pipelines
# ------------------------------------------------------------------------
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

_test_pipeline = [
    dict(type='LoadMultiViewImageFromFiles', to_float32=True),
    dict(type='CollectCameraExtrinsics'),
    dict(type='ResizeMultiViewImage', size=image_hw),
    dict(type='LoadRadarBEV', bev_range=bev_range, bev_res=bev_res),
    dict(type='PackBEVFusionInputs',
         keys=['img'],
         meta_keys=('sample_idx', 'box_mode_3d', 'box_type_3d')),
]

# ------------------------------------------------------------------------
# Dataloaders — CBGS wrap for class-balanced sampling
# ------------------------------------------------------------------------
train_dataloader = dict(
    batch_size=8,                                     # measured 6.2 GB peak at bs=4 → ~12 GB at bs=8
    num_workers=2,                                    # halve to relieve RAM pressure
    persistent_workers=False,                         # release per-epoch caches
    sampler=dict(_delete_=True, type='DefaultSampler', shuffle=True),
    # _delete_=True wipes the inherited NuScenesRadarDataset args from
    # bev_fusion_mini.py — otherwise mmengine deep-merges its data_root /
    # ann_file / pipeline / etc. up onto CBGSDataset, which then errors
    # on __init__(data_root=...).
    dataset=dict(
        _delete_=True,
        type='CBGSDataset',                           # class-balanced sampling
        dataset=dict(
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
    ),
)

val_dataloader = dict(
    batch_size=1,
    num_workers=2,
    persistent_workers=False,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type='NuScenesRadarDataset',
        data_root=data_root,
        ann_file='nuscenes_infos_val.pkl',
        radar_bev_dir=radar_bev_dir,
        pipeline=_test_pipeline,
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
# Schedule — 24 epochs, FP16, AdamW + cosine, backbone × 0.1
# ------------------------------------------------------------------------
train_cfg = dict(_delete_=True, type='EpochBasedTrainLoop',
                 max_epochs=24, val_interval=2)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')

optim_wrapper = dict(
    type='AmpOptimWrapper',                           # FP16 mixed precision
    loss_scale='dynamic',
    optimizer=dict(type='AdamW', lr=4e-4, weight_decay=0.01),  # bs=8 → 2× the bs=4 LR
    clip_grad=dict(max_norm=35, norm_type=2),
    paramwise_cfg=dict(custom_keys={
        'img_backbone': dict(lr_mult=0.1, decay_mult=1.0),
    }),
)

# Longer, slower warmup → kills the iter-50 inf-loss spike seen in v2.
param_scheduler = [
    dict(type='LinearLR', start_factor=1e-4,
         by_epoch=False, begin=0, end=2000),
    dict(type='CosineAnnealingLR',
         T_max=24, eta_min=1e-6,
         begin=0, end=24, by_epoch=True),
]

# ------------------------------------------------------------------------
# Hooks — save best on BOTH NDS and mAP; EMA for free NDS lift
# ------------------------------------------------------------------------
default_hooks = dict(
    timer=dict(type='IterTimerHook'),
    logger=dict(type='LoggerHook', interval=50),
    param_scheduler=dict(type='ParamSchedulerHook'),
    checkpoint=dict(
        type='CheckpointHook', interval=2,
        by_epoch=True, max_keep_ckpts=3,
        save_best=[
            'NuScenes metric/pred_instances_3d_NuScenes/NDS',
            'NuScenes metric/pred_instances_3d_NuScenes/mAP',
        ],
        rule=['greater', 'greater'],
    ),
    sampler_seed=dict(type='DistSamplerSeedHook'),
)

custom_hooks = [
    dict(type='EMAHook', ema_type='mmdet.ExpMomentumEMA',
         momentum=0.0002, update_buffers=True, priority=49),
]

log_processor = dict(type='LogProcessor', window_size=50, by_epoch=True)

work_dir = 'work_dirs/bev_fusion_full_v3'

env_cfg = dict(
    cudnn_benchmark=True,
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0),
    dist_cfg=dict(backend='nccl'),
)
