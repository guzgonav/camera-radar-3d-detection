# rpp — radar-primary painted-pillar detector, mini config.
#
# Purpose: plumbing probe + G2 mini-overfit gate (8 train samples, 200
# iters). NOT for real numbers.
#
# The pipeline is image-free: the camera enters ONLY through the offline
# painted-feature cache (scripts/radar/paint_radar_cache.py), so training never
# loads or forwards an image. Head / grid / NMS blocks are carried over
# from the v3 line (bev_fusion_full_v3.py + the v3b score-threshold fix) —
# same 0.4 m BEV the CenterHead configuration was already tuned for.

custom_imports = dict(
    imports=['datasets', 'models'],
    allow_failed_imports=False)

# ------------------------------------------------------------------------
# Geometry
# ------------------------------------------------------------------------
bev_range = 51.2
bev_res = 0.4
bev_size = int(2 * bev_range / bev_res)              # 256
point_cloud_range = [-bev_range, -bev_range, -5.0, bev_range, bev_range, 3.0]

class_names = [
    'car', 'truck', 'construction_vehicle', 'bus', 'trailer',
    'barrier', 'motorcycle', 'bicycle', 'pedestrian', 'traffic_cone',
]
metainfo = dict(version='v1.0-mini', classes=class_names)

# ------------------------------------------------------------------------
# Channel budgets
# ------------------------------------------------------------------------
geo_channels = 8        # v2 cache: [x, y, z, vx, vy, rcs, dt, dyn_prop]
paint_channels = 14     # painted camera features (paint_radar_cache.py)
bev_channels = 128
max_radar_points = 2048

# ------------------------------------------------------------------------
# Model
# ------------------------------------------------------------------------
model = dict(
    type='RPPDetector',
    data_preprocessor=dict(type='RPPDataPreprocessor'),
    geo_channels=geo_channels,
    paint_channels=paint_channels,
    paint_dropout=0.15,
    pillar_encoder=dict(
        type='RadarPillarEncoder',
        in_channels=geo_channels + paint_channels,
        feat_channels=(64, 64),
        out_channels=bev_channels,
        bev_range=bev_range,
        bev_res=bev_res,
    ),
    bev_backbone=dict(
        type='BEVBackbone',
        channels=bev_channels,
        num_blocks=4,
        # Params are tiny but fp32 BEV activations at 256² are not:
        # bs16 without checkpointing OOMed the 24 GB card (G3, 2026-07-03).
        use_checkpointing=True,
    ),
    bbox_head=dict(
        type='CenterHead',
        in_channels=bev_channels,
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
            score_threshold=0.15,    # v3b fix: avoid >500 boxes/sample
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
        min_radius=1,                # 1 cell × 0.4 m
        # Velocity weighted 1.0 (v3 rationale: radar IS the velocity
        # sensor) — doubly true here, Doppler is a direct input feature.
        code_weights=[1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
    ),
    test_cfg=dict(
        post_center_limit_range=point_cloud_range,
        max_per_img=500,
        max_pool_nms=False,
        # Per-task NMS radii in 0.4 m cells (v3 values):
        # car / truck+cv / bus+trailer / barrier / moto+bike / ped+cone.
        min_radius=[6.0, 18.0, 15.0, 1.5, 1.275, 0.2625],
        score_threshold=0.15,
        pc_range=[-bev_range, -bev_range],
        out_size_factor=1,
        voxel_size=[bev_res, bev_res],
        nms_type='rotate',
        pre_max_size=1000,
        post_max_size=200,
        nms_thr=0.2,
    ),
)

# ------------------------------------------------------------------------
# Data — image-free pipelines
# ------------------------------------------------------------------------
data_root = 'data/nuscenes-mini/'
radar_pts_dir = 'radar_pts_v2'
radar_paint_dir = 'radar_paint_v1'

# Camera prefixes are still required by the NuScenesDataset info parser
# (paths are recorded in the info dicts) but no image is ever opened.
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
    dict(type='LoadRadarPoints',
         max_points=max_radar_points, n_rows=geo_channels,
         paint=True, paint_rows=paint_channels,
         # Cache is ego-frame; GT/camera are LIDAR-frame. MANDATORY —
         # without it the radar is ~90° misaligned (the v3x bug).
         to_lidar_frame=True),
    dict(type='LoadAnnotations3D', with_bbox_3d=True, with_label_3d=True),
    dict(type='BEVHorizontalFlip', prob=0.5),
    dict(type='BEVGlobalRotation', rot_range=(-0.3927, 0.3927)),
    dict(type='PackBEVFusionInputs',
         keys=['gt_bboxes_3d', 'gt_labels_3d'],
         meta_keys=('sample_idx', 'box_mode_3d', 'box_type_3d')),
]

test_pipeline = [
    dict(type='LoadRadarPoints',
         max_points=max_radar_points, n_rows=geo_channels,
         paint=True, paint_rows=paint_channels,
         # Cache is ego-frame; GT/camera are LIDAR-frame. MANDATORY —
         # without it the radar is ~90° misaligned (the v3x bug).
         to_lidar_frame=True),
    dict(type='PackBEVFusionInputs',
         keys=[],
         meta_keys=('sample_idx', 'box_mode_3d', 'box_type_3d')),
]

train_dataloader = dict(
    batch_size=2,
    num_workers=0,
    persistent_workers=False,
    sampler=dict(type='InfiniteSampler', shuffle=True),
    dataset=dict(
        type='NuScenesRadarDataset',
        data_root=data_root,
        ann_file='nuscenes_infos_train.pkl',
        radar_bev_dir=radar_pts_dir,
        radar_paint_dir=radar_paint_dir,
        pipeline=train_pipeline,
        metainfo=metainfo,
        modality=dict(use_camera=True, use_lidar=False),
        data_prefix=data_prefix,
        test_mode=False,
        box_type_3d='LiDAR',
        # First 8 samples only — the G2 overfit gate.
        indices=8,
    ),
)
val_dataloader = dict(
    batch_size=4,
    num_workers=2,
    persistent_workers=False,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type='NuScenesRadarDataset',
        data_root=data_root,
        ann_file='nuscenes_infos_val.pkl',
        radar_bev_dir=radar_pts_dir,
        radar_paint_dir=radar_paint_dir,
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
# Schedule — overfit gate
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
    cudnn_benchmark=True,
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0),
    dist_cfg=dict(backend='nccl'),
)
log_processor = dict(type='LogProcessor', window_size=20, by_epoch=False)
log_level = 'INFO'
load_from = None
resume = False
work_dir = 'work_dirs/rpp_mini'

vis_backends = [dict(type='LocalVisBackend')]
visualizer = dict(
    type='Det3DLocalVisualizer', vis_backends=vis_backends, name='visualizer')

randomness = dict(seed=0, deterministic=False)
