#!/usr/bin/env bash
# run_center_fusion.sh — end-to-end CenterFusion (uncertainty-gate) pipeline.
#
# Trains the gate MLP on the train split, evaluates on val (leakage-free).
# FCOS3D is frozen throughout — only the small refinement MLP is trained.
#
#   ./pipelines/run_center_fusion.sh mini   # fast smoke test (2 val scenes)
#   ./pipelines/run_center_fusion.sh full   # real run (150 val scenes → test)
#
# The expensive radar association is cached on the first run; re-runs (e.g.
# hyper-parameter tweaks) reuse the cache and finish in seconds.
set -euo pipefail
cd "$(dirname "$0")/.."

PY=.venv/bin/python
SPLIT="${1:-mini}"

if [[ "$SPLIT" == "mini" ]]; then
    VERSION="v1.0-mini";      DATAROOT="data/nuscenes-mini"; EVALSPLIT="mini_val"
    CAMJSON="results/fcos3d_baseline/mini/detections/pred_instances_3d/results_nusc.json"
    CONFIG="configs/center_fusion_mini.py"
    OUT="results/center_fusion/mini"
elif [[ "$SPLIT" == "full" ]]; then
    VERSION="v1.0-trainval";  DATAROOT="data/nuscenes";      EVALSPLIT="val"
    CAMJSON="results/fcos3d_baseline/full/detections/pred_instances_3d/results_nusc.json"
    CONFIG="configs/center_fusion_full.py"
    OUT="results/center_fusion/full"
else
    echo "usage: $0 {mini|full}"; exit 1
fi

# The trainer builds the radar-association cache ONCE, then emits a
# three-way comparison: camera_only / vanilla_hard / learned_gate on val.
echo "###############################################################"
echo "# CenterFusion — train on train, evaluate on val"
echo "###############################################################"
$PY scripts/train/train_center_fusion.py --config "$CONFIG"

# Optional: standalone Stage-1 sweep / single fixed gate via the rule-based
# pipeline (re-runs association; handy for ad-hoc gate experiments):
#   $PY scripts/fusion/center_fusion.py --version "$VERSION" --dataroot "$DATAROOT" \
#       --split "$EVALSPLIT" --cam-json "$CAMJSON" --out "$OUT" --gate 0.25

echo
echo "Done. Comparison table + metrics written under the configured 'out' dir."
