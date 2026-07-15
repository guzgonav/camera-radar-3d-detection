# CenterFusion gate — full config (v1.0-trainval).
#
# MLP trained on the train split (GT supervision), evaluated on val.
# Train and val are disjoint, so val NDS is fully leakage-free.
CONFIG = dict(
    version='v1.0-trainval',
    dataroot='data/nuscenes',
    split='val',
    cam_json='results/fcos3d_baseline/full/detections/pred_instances_3d/results_nusc.json',
    out='results/center_fusion/full_learned',
    cache='results/center_fusion/cache/records_full_val.pkl',

    # Train-split supervision.
    train_split='train',
    train_cam_json='results/fcos3d_baseline/train/detections/pred_instances_3d/results_nusc.json',
    train_cache='results/center_fusion/cache/records_full_train.pkl',

    frustum=dict(nsweeps=6, radial_frac=0.25, min_radial=2.5,
                 max_radial=15.0, lateral_margin=1.0, min_angle_deg=3.0),

    epochs=300,
    lr=1e-3,
    weight_decay=1e-4,
    hidden_dim=64,
    lambda_vel=1.0,
    use_residual=False,
    gate_bias=-0.4,
    seed=0,
    save_model='results/center_fusion/full_learned/gate_mlp_final.pt',
)
