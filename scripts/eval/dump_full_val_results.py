"""
dump_full_val_results.py — Thin wrapper around mmdet3d's tools/test.py that
persists a full-val results.json (nuScenes submission format) via
jsonfile_prefix, instead of the default temp-dir dump that gets deleted.

Needed for the JSON-based hard-scenes / error-by-distance path
(scripts/eval/eval_hard_scenes_json.py, scripts/analysis/error_by_distance.py) applied to
mmdet3d-Runner-based methods (v3b, rpp) — bringing them to the same
"terminates in a results.json" shape as late fusion / CenterFusion / T1 / T2.
scripts/eval/eval_hard_scenes.py (hard-scene-only, via SubsetNuScenesMetric) stays
the tool for the formal hard20/hard50 NDS/mAP table; this one is for the
full-val dump those two downstream scripts need.

Usage:
    .venv/bin/python scripts/eval/dump_full_val_results.py \\
        configs/bev_fusion_full_v3b_framefix.py \\
        "work_dirs/bev_fusion_full_v3b_framefix/best_NuScenes metric_pred_instances_3d_NuScenes_NDS_epoch_20.pth" \\
        --work-dir results/bev_fusion/fullv3b_framefix
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent.parent
os.chdir(_ROOT)
sys.path.insert(0, str(_ROOT))

# Import side effects: register custom datasets/models before the Runner builds.
import datasets  # noqa: F401,E402
import models    # noqa: F401,E402

MIM_TOOLS = str(_ROOT / '.venv/lib/python3.10/site-packages/mmdet3d/.mim/tools')
sys.path.insert(0, MIM_TOOLS)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('config')
    parser.add_argument('checkpoint')
    parser.add_argument('--work-dir', required=True,
                         help='Output dir; results land in '
                              '<work-dir>/detections/pred_instances_3d/results_nusc.json')
    args = parser.parse_args()

    jsonfile_prefix = os.path.join(args.work_dir, 'detections')

    sys.argv = [
        'test.py', args.config, args.checkpoint,
        '--work-dir', args.work_dir,
        '--cfg-options',
        f'val_evaluator.jsonfile_prefix={jsonfile_prefix}',
        f'test_evaluator.jsonfile_prefix={jsonfile_prefix}',
    ]

    from test import main as test_main  # mmdet3d tools/test.py
    test_main()


if __name__ == '__main__':
    main()
