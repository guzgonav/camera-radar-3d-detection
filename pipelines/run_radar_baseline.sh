#!/bin/bash

# run_radar_baseline.sh — Radar-only detection baseline for nuScenes
#
# Runs the classical radar pipeline (preprocessing → DBSCAN clustering → nuScenes eval)
# and writes results to results/radar_baseline/{mini|full}/.
#
# Usage:
#   bash run_radar_baseline.sh          # mini split (default)
#   bash run_radar_baseline.sh full     # full val split

set -e

SPLIT=${1:-mini}  # mini | full

if [ "$SPLIT" = "full" ]; then
    VERSION="v1.0-trainval"
    DATAROOT="data/nuscenes"
    EVAL_SPLIT="val"
    OUT="results/radar_baseline/full"
else
    VERSION="v1.0-mini"
    DATAROOT="data/nuscenes-mini"
    EVAL_SPLIT="mini_val"
    OUT="results/radar_baseline/mini"
fi

echo "Running radar-only baseline on nuScenes $SPLIT split..."

.venv/bin/python scripts/radar/radar_detect.py \
    --version "$VERSION" \
    --dataroot "$DATAROOT" \
    --split "$EVAL_SPLIT" \
    --out "$OUT"
