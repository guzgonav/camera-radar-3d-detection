# rpp — G3 probe config: 4 epochs, EMA off, cosine over 4.
#
# Mirrors the v3e/v3f gate-3 protocol so the ablation deltas
# (paint-zero / paint-shuffle / radar-shuffle) are measured on an
# equal-footing short run before committing to the 24-epoch schedule.
# Absolute NDS from this config is NOT comparable to full runs — only
# the deltas matter.

_base_ = './rpp_full.py'

train_cfg = dict(max_epochs=4, val_interval=4)

param_scheduler = [
    dict(type='LinearLR', start_factor=1e-3,
         by_epoch=False, begin=0, end=1000),
    dict(type='CosineAnnealingLR',
         T_max=4, eta_min=1e-6,
         begin=0, end=4, by_epoch=True),
]

# EMA off for the probe (v3f gate-3 protocol).
custom_hooks = []

# _delete_ wipes the inherited save_best/rule pair wholesale — setting
# save_best=None alone leaves rule=['greater','greater'] behind and
# CheckpointHook then crashes on len(None).
default_hooks = dict(
    checkpoint=dict(
        _delete_=True,
        type='CheckpointHook', interval=1,
        by_epoch=True, max_keep_ckpts=4,
    ),
)

work_dir = 'work_dirs/rpp_full_gate3'
