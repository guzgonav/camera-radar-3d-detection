#!/bin/bash
# Prepares nuScenes full (v1.0-trainval) data for MMDetection3D.
#
# Reads the raw nuScenes JSON metadata and generates two .pkl info files:
#   data/nuscenes/nuscenes_infos_train.pkl  (700 scenes)
#   data/nuscenes/nuscenes_infos_val.pkl    (150 scenes)
#
# These files are required by the dataloader and evaluator during inference
# and training. Only needs to be run once. Mini PKLs remain untouched under
# data/nuscenes-mini/.
#
# Note: PYTHONPATH is set manually because mim run has a bug in this version
# that appends .py.py to the script name.
set -e

MIM=.venv/lib/python3.10/site-packages/mmdet3d/.mim

echo ">>> Starting nuScenes full data preparation. This runs in two phases:"
echo ""
echo "    Phase 1: Iterates all samples and writes the raw PKLs."
echo "    Phase 2: Upgrades PKLs to MMDetection3D v2 format in-place."
echo "                       NuScenes tables will be loaded a second time"
echo ""

PYTHONPATH=$MIM .venv/bin/python $MIM/tools/create_data.py nuscenes \
    --root-path data/nuscenes \
    --version v1.0 \
    --out-dir data/nuscenes \
    --extra-tag nuscenes \
    --max-sweeps 0

echo ""
echo ">>> Phase 2/2 complete: PKLs upgraded to MMDetection3D v2 format."
echo "    Both phases finished successfully — safe to use the PKLs."
