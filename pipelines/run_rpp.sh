#!/usr/bin/env bash
# run_rpp.sh — radar-primary painted-pillar detector (week 17).
#
# Stages:
#   ./pipelines/run_rpp.sh gate3          # 4-epoch probe + dual-shuffle ablations
#   ./pipelines/run_rpp.sh full           # 24-epoch training
#   ./pipelines/run_rpp.sh ablate <ckpt>  # ablations on any checkpoint
#
# Prerequisites (one-time):
#   scripts/radar/precompute_radar.py --layout v2   (radar cache, CPU)
#   scripts/radar/paint_radar_cache.py              (painted cache, GPU)

set -euo pipefail
cd "$(dirname "$0")/.."
PY=.venv/bin/python

STAGE="${1:-gate3}"

case "$STAGE" in
  gate3)
    $PY scripts/train/train_bev_fusion.py configs/rpp_full_gate3.py \
        --work-dir work_dirs/rpp_full_gate3 --resume
    $PY scripts/ablation/shuffle_rpp.py \
        --config configs/rpp_full_gate3.py \
        --checkpoint work_dirs/rpp_full_gate3/epoch_4.pth
    ;;
  full)
    $PY scripts/train/train_bev_fusion.py configs/rpp_full.py \
        --work-dir work_dirs/rpp_full --resume
    ;;
  ablate)
    CKPT="${2:?usage: run_rpp.sh ablate <checkpoint>}"
    $PY scripts/ablation/shuffle_rpp.py \
        --config configs/rpp_full.py --checkpoint "$CKPT"
    ;;
  *)
    echo "unknown stage: $STAGE (use gate3 | full | ablate)" >&2
    exit 1
    ;;
esac
