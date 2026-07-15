#!/bin/bash

# Runs FCOS3D inference on the nuScenes full val split (v1.0-trainval, 150 scenes).
# Same pipeline as run_inference.sh but using the full dataset PKLs.
#
# 1. Loads configs/fcos3d_full.py (inherits trainval defaults)
# 2. Loads the pretrained FCOS3D checkpoint
# 3. Iterates over the full val split — 150 scenes, each with 6 cameras
# 4. Computes official mAP and NDS metrics via NuScenesMetric
# 5. Saves everything (logs, PKLs, optional visualizations) to work_dirs/fcos3d_full/

set -e

VISUALIZE=${1:-false}  # usage: bash run_inference_full.sh [true|false]

MIM=.venv/lib/python3.10/site-packages/mmdet3d/.mim

SHOW_ARGS=""
if [ "$VISUALIZE" = true ]; then
    SHOW_ARGS="--show-dir work_dirs/fcos3d_full/vis --task mono_det --cfg-options default_hooks.visualization.score_thr=0.3"
fi

PYTHONPATH=$MIM .venv/bin/python $MIM/tools/test.py \
    configs/fcos3d_full.py \
    checkpoints/fcos3d_r101_caffe_fpn_gn-head_dcn_2x8_1x_nus-mono3d_finetune_20210717_095645-8d806dc2.pth \
    --work-dir work_dirs/fcos3d_full \
    $SHOW_ARGS
