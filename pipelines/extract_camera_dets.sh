#!/bin/bash

# extract_camera_dets.sh — Extract FCOS3D detections to nuScenes JSON
#
# Runs FCOS3D inference and saves the raw per-sample detections as a
# nuScenes submission JSON (instead of computing metrics).
# This JSON is needed by late_fusion.py.
#
# Usage:
#   bash pipelines/extract_camera_dets.sh          # mini split (default)
#   bash pipelines/extract_camera_dets.sh full      # full val split

set -e

SPLIT=${1:-mini}

if [ "$SPLIT" = "full" ]; then
    CONFIG="configs/fcos3d_full_extract.py"
    WORKDIR="work_dirs/fcos3d_full"
elif [ "$SPLIT" = "train" ]; then
    CONFIG="configs/fcos3d_train_extract.py"
    WORKDIR="work_dirs/fcos3d_train"
else
    CONFIG="configs/fcos3d_mini_extract.py"
    WORKDIR="work_dirs/fcos3d_mini"
fi

MIM=.venv/lib/python3.10/site-packages/mmdet3d/.mim
CHECKPOINT=checkpoints/fcos3d_r101_caffe_fpn_gn-head_dcn_2x8_1x_nus-mono3d_finetune_20210717_095645-8d806dc2.pth

echo "Extracting FCOS3D detections ($SPLIT split)..."

PYTHONPATH=$MIM .venv/bin/python $MIM/tools/test.py \
    "$CONFIG" \
    "$CHECKPOINT" \
    --work-dir "$WORKDIR"
