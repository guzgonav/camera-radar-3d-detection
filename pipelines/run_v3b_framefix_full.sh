#!/usr/bin/env bash
# run_v3b_framefix_full.sh — full 24-epoch v3b retrain with the radar
# frame fix (LoadRadarBEV(to_lidar_frame=True)). Produces the reference
# v3b checkpoint for the hard-scenes / error-by-distance comparisons —
# do NOT use work_dirs/bev_fusion_full_v3b/epoch_24.pth (original,
# radar-inert) for that.

set -euo pipefail
cd "$(dirname "$0")/.."
PY=.venv/bin/python

CONFIG=configs/bev_fusion_full_v3b_framefix.py
WORKDIR=work_dirs/bev_fusion_full_v3b_framefix

$PY scripts/train/train_bev_fusion.py "$CONFIG" --work-dir "$WORKDIR" --resume

CKPT=$(find "$WORKDIR" -maxdepth 1 -name "best_*NDS*.pth" 2>/dev/null | sort | tail -1)
if [ -z "$CKPT" ] || [ ! -f "$CKPT" ]; then
    CKPT="$WORKDIR/latest.pth"
fi

$PY scripts/ablation/shuffle_bev_fusion.py \
    --config "$CONFIG" \
    --checkpoint "$CKPT"
