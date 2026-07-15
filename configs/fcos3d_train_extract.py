# FCOS3D config for extracting detections on the nuScenes train split.
#
# Output: results/fcos3d_baseline/train/detections/pred_instances_3d/results_nusc.json
_base_ = 'fcos3d_full.py'

val_dataloader = dict(dataset=dict(ann_file='nuscenes_infos_train.pkl'))
test_dataloader = val_dataloader

val_evaluator = dict(
    ann_file='data/nuscenes/nuscenes_infos_train.pkl',
    format_only=True,
    jsonfile_prefix='results/fcos3d_baseline/train/detections',
)
test_evaluator = dict(
    ann_file='data/nuscenes/nuscenes_infos_train.pkl',
    format_only=True,
    jsonfile_prefix='results/fcos3d_baseline/train/detections',
)
