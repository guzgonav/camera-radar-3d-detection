# CenterFusion gate — mini config (v1.0-mini, 2 val scenes).
#
# Smoke test of the training/inference plumbing. No test split in v1.0-mini,
# so test inference is skipped (test_cam_json absent).
CONFIG = dict(
    version='v1.0-mini',
    dataroot='data/nuscenes-mini',
    split='mini_val',
    cam_json='results/fcos3d_baseline/mini/detections/pred_instances_3d/results_nusc.json',
    out='results/center_fusion/mini_learned',
    cache='results/center_fusion/cache/records_mini_val.pkl',

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
    save_model='results/center_fusion/mini_learned/gate_mlp_final.pt',
)
