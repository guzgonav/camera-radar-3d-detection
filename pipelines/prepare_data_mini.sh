#!/bin/bash
# Prepares nuScenes mini data for MMDetection3D.
#
# Reads the raw nuScenes JSON metadata and generates two .pkl info files:
#   data/nuscenes/nuscenes_infos_train.pkl  (8 scenes)
#   data/nuscenes/nuscenes_infos_val.pkl    (2 scenes)
#
# These files are required by the dataloader and evaluator during inference
# and training. Only needs to be run once.
#
# Note: PYTHONPATH is set manually because mim run has a bug in this version
# that appends .py.py to the script name.
set -e

MIM=.venv/lib/python3.10/site-packages/mmdet3d/.mim

PYTHONPATH=$MIM .venv/bin/python $MIM/tools/create_data.py nuscenes \
    --root-path data/nuscenes-mini \
    --version v1.0-mini \
    --out-dir data/nuscenes-mini \
    --extra-tag nuscenes \
    --max-sweeps 10  # number of historical radar/lidar sweeps to aggregate per keyframe
