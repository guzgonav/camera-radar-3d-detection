"""
train_bev_fusion.py — Thin wrapper around mmdet3d's tools/train.py that
imports the project's custom datasets and models so they're registered
before the runner builds the pipeline.

Usage:
    .venv/bin/python scripts/train/train_bev_fusion.py configs/bev_fusion_mini.py
    .venv/bin/python scripts/train/train_bev_fusion.py configs/bev_fusion_full.py \\
        --work-dir work_dirs/bev_fusion_full --resume
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent.parent
os.chdir(_ROOT)
sys.path.insert(0, str(_ROOT))

# Import side effects: register custom modules.
import datasets  # noqa: F401
import models    # noqa: F401

# Hand off to mmdet3d's training entrypoint.
MIM_TOOLS = str(_ROOT / '.venv/lib/python3.10/site-packages/mmdet3d/.mim/tools')
sys.path.insert(0, MIM_TOOLS)

from train import main  # type: ignore  # mmdet3d's tools/train.py


if __name__ == '__main__':
    main()
