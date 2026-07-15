# rpp — radar-primary painted-pillar detector, full training config
# (v1.0-trainval, 24 epochs).
#
# Inherits the model / head / pipeline from rpp_mini.py; overrides data
# roots, dataloaders, and schedule. The model is tiny (~5 M trainable
# params) and the pipeline is image-free, so epochs are dataloading-bound:
# batch 16 fits trivially in 24 GB.

_base_ = './rpp_mini.py'

# ------------------------------------------------------------------------
# Data
# ------------------------------------------------------------------------
data_root = 'data/nuscenes/'
metainfo = dict(version='v1.0-trainval', classes={{_base_.class_names}})

train_dataloader = dict(
    batch_size=8,   # 16 OOMed: fp32 head/loss activations at 256² dominate
    num_workers=4,
    persistent_workers=True,
    sampler=dict(_delete_=True, type='DefaultSampler', shuffle=True),
    dataset=dict(
        data_root=data_root,
        metainfo=metainfo,
        indices=None,               # full train split
    ),
)
val_dataloader = dict(
    batch_size=8,
    num_workers=4,
    dataset=dict(
        data_root=data_root,
        metainfo=metainfo,
    ),
)
test_dataloader = val_dataloader

val_evaluator = dict(
    data_root=data_root,
    ann_file=data_root + 'nuscenes_infos_val.pkl',
)
test_evaluator = val_evaluator

# ------------------------------------------------------------------------
# Schedule — 24 epochs, AMP, AdamW + cosine
# ------------------------------------------------------------------------
train_cfg = dict(_delete_=True, type='EpochBasedTrainLoop',
                 max_epochs=24, val_interval=2)

optim_wrapper = dict(
    _delete_=True,
    type='AmpOptimWrapper',          # fp32 fences live inside RPPDetector
    loss_scale='dynamic',
    optimizer=dict(type='AdamW', lr=2e-4, weight_decay=0.01),
    clip_grad=dict(max_norm=35, norm_type=2),
)

param_scheduler = [
    dict(type='LinearLR', start_factor=1e-3,
         by_epoch=False, begin=0, end=1000),
    dict(type='CosineAnnealingLR',
         T_max=24, eta_min=1e-6,
         begin=0, end=24, by_epoch=True),
]

default_hooks = dict(
    logger=dict(type='LoggerHook', interval=50),
    checkpoint=dict(
        type='CheckpointHook', interval=2,
        by_epoch=True, max_keep_ckpts=3,
        save_best=[
            'NuScenes metric/pred_instances_3d_NuScenes/NDS',
            'NuScenes metric/pred_instances_3d_NuScenes/mAP',
        ],
        rule=['greater', 'greater'],
    ),
)

custom_hooks = [
    dict(type='EMAHook', ema_type='mmdet.ExpMomentumEMA',
         momentum=0.0002, update_buffers=True, priority=49),
]

log_processor = dict(type='LogProcessor', window_size=50, by_epoch=True)
work_dir = 'work_dirs/rpp_full'
