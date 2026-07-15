#!/bin/bash
# run_bev_fusion.sh — End-to-end BEV fusion pipeline.
#
# Stages:
#   1. Pre-compute radar BEV cache (one-time, idempotent).
#   2. Train BEVFusionDetector.
#   3. Evaluate the best checkpoint on the val split.
#
# Usage:
#   bash pipelines/run_bev_fusion.sh           # mini split (overfit sanity)
#   bash pipelines/run_bev_fusion.sh full      # full v1 (12 epochs, batch=4)
#   bash pipelines/run_bev_fusion.sh fullv2    # full v2 (24 epochs, batch=8)
#   bash pipelines/run_bev_fusion.sh fullv3    # full v3 (24 epochs, batch=8, R-101+FCOS3D, CBGS)
#   bash pipelines/run_bev_fusion.sh fullv3b   # full v3b (= v3 minus CBGS, ~2.5 days)
#
# For the reported v3b result (radar frame fix applied), use
# pipelines/run_v3b_framefix_full.sh instead.

set -e

SPLIT=${1:-mini}

if [ "$SPLIT" = "fullv3b" ]; then
    VERSION='v1.0-trainval'
    DATAROOT='data/nuscenes'
    CONFIG='configs/bev_fusion_full_v3b.py'
    WORKDIR='work_dirs/bev_fusion_full_v3b'
    RADAR_OUT="$DATAROOT/radar_bev"
elif [ "$SPLIT" = "fullv3" ]; then
    VERSION='v1.0-trainval'
    DATAROOT='data/nuscenes'
    CONFIG='configs/bev_fusion_full_v3.py'
    WORKDIR='work_dirs/bev_fusion_full_v3'
    RADAR_OUT="$DATAROOT/radar_bev"
elif [ "$SPLIT" = "fullv2" ]; then
    VERSION='v1.0-trainval'
    DATAROOT='data/nuscenes'
    CONFIG='configs/bev_fusion_full_v2.py'
    WORKDIR='work_dirs/bev_fusion_full_v2'
    RADAR_OUT="$DATAROOT/radar_bev"
elif [ "$SPLIT" = "full" ]; then
    VERSION='v1.0-trainval'
    DATAROOT='data/nuscenes'
    CONFIG='configs/bev_fusion_full.py'
    WORKDIR='work_dirs/bev_fusion_full'
    RADAR_OUT="$DATAROOT/radar_bev"
else
    VERSION='v1.0-mini'
    DATAROOT='data/nuscenes-mini'
    CONFIG='configs/bev_fusion_mini.py'
    WORKDIR='work_dirs/bev_fusion_mini'
    RADAR_OUT="$DATAROOT/radar_bev"
fi

# 1. Radar cache (idempotent — skipped per-file if it already exists).
echo "[1/3] Pre-computing radar BEV cache for $SPLIT..."
.venv/bin/python scripts/radar/precompute_radar.py \
    --version "$VERSION" \
    --dataroot "$DATAROOT" \
    --out "$RADAR_OUT"

# 2. Train.
echo "[2/3] Training BEV fusion ($SPLIT)..."
.venv/bin/python scripts/train/train_bev_fusion.py "$CONFIG" --work-dir "$WORKDIR" --resume

# 3. Eval.
# mmengine names the best checkpoint best_<key>_epoch_N.pth where <key> has
# '/' replaced by '_' but spaces preserved — the static 'best_NDS.pth' never
# matched.  Use a glob instead; fall back to latest if no best was saved.
CKPT=$(find "$WORKDIR" -maxdepth 1 -name "best_*NDS*.pth" 2>/dev/null | sort | tail -1)
if [ -z "$CKPT" ] || [ ! -f "$CKPT" ]; then
    CKPT="$WORKDIR/latest.pth"
fi
if [ ! -f "$CKPT" ] && [ -f "$WORKDIR/last_checkpoint" ]; then
    CKPT=$(cat "$WORKDIR/last_checkpoint")
fi
if [ -f "$CKPT" ]; then
    echo "[3/3] Evaluating checkpoint $CKPT..."
    OUT="results/bev_fusion/${SPLIT}"
    mkdir -p "$OUT"
    .venv/bin/python scripts/eval/eval_bev_fusion.py \
        "$CONFIG" "$CKPT" --work-dir "$OUT"
else
    echo "[3/3] No checkpoint at $CKPT — skipping eval. Resume training first."
fi
