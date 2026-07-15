#!/bin/bash

#  test.py — MMDetection3D's inference runner. It:
#  1. Loads the config (fcos3d_mini.py) to know the model architecture and data pipeline
#  2. Loads the checkpoint (pretrained FCOS3D weights, 211MB)
#  3. Iterates over the mini val split — 2 scenes, each with 6 cameras — feeding each image through the network
#  4. For each image, FCOS3D predicts 3D bounding boxes (class, position, size, orientation, velocity) directly from the 2D image pixels, no depth sensor needed
#  5. After all images are processed, runs the NuScenes evaluator which compares predictions against ground truth annotations and computes the official metrics: mAP and NDS
#  6. Saves everything to work_dirs/fcos3d_mini/

set -e

# Whether to save predicted bounding boxes overlaid on images
VISUALIZE=${1:-false}  # usage: bash run_inference.sh [true|false]

MIM=.venv/lib/python3.10/site-packages/mmdet3d/.mim

SHOW_ARGS=""
if [ "$VISUALIZE" = true ]; then
    SHOW_ARGS="--show-dir work_dirs/fcos3d_mini/vis --task mono_det --cfg-options default_hooks.visualization.score_thr=0.3"
fi

PYTHONPATH=$MIM .venv/bin/python $MIM/tools/test.py \
    configs/fcos3d_mini.py \
    checkpoints/fcos3d_r101_caffe_fpn_gn-head_dcn_2x8_1x_nus-mono3d_finetune_20210717_095645-8d806dc2.pth \
    --work-dir work_dirs/fcos3d_mini \
    $SHOW_ARGS
