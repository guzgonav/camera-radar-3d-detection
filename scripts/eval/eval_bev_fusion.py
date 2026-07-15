"""
eval_bev_fusion.py — Wrapper around mmdet3d's tools/test.py that
imports our custom modules first.

Produces nuScenes-format results JSON + metric dict identical in shape
to what's emitted by ``run_inference_full.sh`` for FCOS3D, so the
existing four-way comparison code in the post-train notebook section
can ingest it without changes.

Usage:
    .venv/bin/python scripts/eval/eval_bev_fusion.py \\
        configs/bev_fusion_full.py \\
        work_dirs/bev_fusion_full/best_NDS.pth \\
        --work-dir results/bev_fusion/full
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent.parent
os.chdir(_ROOT)
sys.path.insert(0, str(_ROOT))

import datasets  # noqa: F401
import models    # noqa: F401

MIM_TOOLS = str(_ROOT / '.venv/lib/python3.10/site-packages/mmdet3d/.mim/tools')
sys.path.insert(0, MIM_TOOLS)

from test import main  # type: ignore


if __name__ == '__main__':
    main()
