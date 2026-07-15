# FCOS3D config for extracting detections to nuScenes JSON (mini split).
#
# Inherits fcos3d_mini.py and sets format_only=True so that test.py
# writes the predictions as a nuScenes submission JSON instead of
# running the full evaluation.
#
# Output: results/fcos3d_baseline/mini/detections/pred_instances_3d/results_nusc.json
_base_ = 'fcos3d_mini.py'

val_evaluator = dict(
    format_only=True,
    jsonfile_prefix='results/fcos3d_baseline/mini/detections',
)
test_evaluator = dict(
    format_only=True,
    jsonfile_prefix='results/fcos3d_baseline/mini/detections',
)
