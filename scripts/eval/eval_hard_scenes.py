#!/usr/bin/env python3
"""
eval_hard_scenes.py — Evaluate any mmdet3d model on the N most challenging val scenes.

Ranks every scene in classifications.json by a camera-difficulty composite
score, selects the hardest N that appear in the val split, writes a filtered
annotation pkl, then runs the standard nuScenes-metric evaluation.

Intended workflow — run once per model, reuse the same hard pkl:

    # 1. BEV fusion v3b (writes hard pkl the first time)
    .venv/bin/python scripts/eval/eval_hard_scenes.py \\
        configs/bev_fusion_full_v3b.py \\
        work_dirs/bev_fusion_full_v3b/epoch_20.pth \\
        --work-dir results/bev_fusion/hard20

    # 2. FCOS3D baseline (reuses the pkl written above — no NuScenes API call)
    .venv/bin/python scripts/eval/eval_hard_scenes.py \\
        configs/fcos3d_full.py \\
        checkpoints/fcos3d_r101_caffe_fpn_gn-head_dcn_2x8_1x_nus-mono3d_finetune_20210717_095645-8d806dc2.pth \\
        --work-dir results/fcos3d_baseline/hard20 \\
        --hard-pkl data/nuscenes/nuscenes_infos_hard20_val.pkl

Difficulty score (higher = harder for cameras, more radar benefit expected):
    score = camera_difficulty          # 2–9  (from classifications.json)
          + (10 - visibility)          # 1–8  (lower vis → harder)
          + 3 * (day_night == 'night') # 0/3
          + weather_penalty            # fog=3, rain=2, wet=1, else=0
"""
from __future__ import annotations

import argparse
import json
import os
import pickle
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent.parent
os.chdir(_ROOT)
sys.path.insert(0, str(_ROOT))

import datasets  # noqa: F401  — registers NuScenesRadarDataset + transforms
import models    # noqa: F401  — registers BEVFusionDetector + components

MIM_TOOLS = str(_ROOT / '.venv/lib/python3.10/site-packages/mmdet3d/.mim/tools')
sys.path.insert(0, MIM_TOOLS)


# ---------------------------------------------------------------------------
# Difficulty scoring
# ---------------------------------------------------------------------------

_WEATHER_PENALTY = {'fog': 3, 'rain': 2, 'wet': 1, 'cloudy': 0, 'clear': 0}


def _difficulty_score(info: dict) -> float:
    score  = info['camera_difficulty']                       # 2–9
    score += 10 - info['visibility']                         # 1–8
    score += 3 if info['day_night'] == 'night' else 0        # 0 or 3
    score += _WEATHER_PENALTY.get(info['weather'], 0)        # 0–3
    return float(score)


# ---------------------------------------------------------------------------
# Scene selection
# ---------------------------------------------------------------------------

def select_hard_scenes(
    classifications_path: str,
    val_pkl_path: str,
    n: int = 20,
) -> tuple[list[str], list[dict], dict]:
    """Return (scene_names, hard_infos, val_metainfo) for the n hardest val scenes.

    scene_names  — sorted list of nuScenes scene names (e.g. 'scene-0003')
    hard_infos   — data_list entries from the val pkl for those scenes
    val_metainfo — metainfo block from the val pkl (passed through to new pkl)
    """
    # 1. Score every scene from the classifications file.
    with open(classifications_path) as f:
        clf: dict[str, dict] = json.load(f)

    # Keys are "scene-XXXX.jpg" → strip .jpg to get the nuScenes scene name.
    scored: dict[str, float] = {
        name.removesuffix('.jpg'): _difficulty_score(info)
        for name, info in clf.items()
    }

    # 2. Load the val pkl and build a sample-token → scene-name map.
    with open(val_pkl_path, 'rb') as f:
        val_data = pickle.load(f)
    infos: list[dict] = val_data['data_list']

    print('Loading nuScenes API to map sample tokens → scene names …')
    from nuscenes.nuscenes import NuScenes
    nusc = NuScenes(
        version='v1.0-trainval',
        dataroot=str(_ROOT / 'data/nuscenes'),
        verbose=False,
    )
    token_to_scene: dict[str, str] = {}
    for scene in nusc.scene:
        tok = scene['first_sample_token']
        while tok:
            token_to_scene[tok] = scene['name']
            tok = nusc.get('sample', tok)['next']

    sample_scenes = [token_to_scene.get(info['token'], '') for info in infos]
    val_scene_names = {s for s in sample_scenes if s}

    # 3. Rank val scenes by difficulty and pick the top n.
    val_scored = {
        name: score for name, score in scored.items() if name in val_scene_names
    }
    ranked = sorted(val_scored, key=val_scored.__getitem__, reverse=True)
    hard_set = set(ranked[:n])

    print(f'\n=== Top {n} hardest val scenes (out of {len(val_scored)} scored) ===')
    print(f'  {"#":>3}  {"Scene":14}  {"Score":>5}  {"Diff":>4}  {"Vis":>3}  '
          f'{"Night":>5}  Weather')
    print('  ' + '-' * 60)
    for i, name in enumerate(ranked[:n], 1):
        info = clf[name + '.jpg']
        s = val_scored[name]
        night_flag = 'yes' if info['day_night'] == 'night' else 'no'
        print(f'  {i:3d}  {name:14}  {s:5.0f}  {info["camera_difficulty"]:4d}  '
              f'{info["visibility"]:3d}  {night_flag:>5}  {info["weather"]}')

    # 4. Collect the val-pkl entries that belong to the hard scenes.
    hard_infos = [
        info for info, scene in zip(infos, sample_scenes) if scene in hard_set
    ]
    print(f'\n  → {len(hard_set)} scenes, {len(hard_infos)} samples selected\n')

    return sorted(hard_set), hard_infos, val_data['metainfo']


# ---------------------------------------------------------------------------
# Filtered pkl
# ---------------------------------------------------------------------------

def write_filtered_pkl(
    hard_infos: list[dict],
    metainfo: dict,
    out_path: str,
) -> None:
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, 'wb') as f:
        pickle.dump({'metainfo': metainfo, 'data_list': hard_infos}, f)
    print(f'Wrote filtered pkl → {out_path}  ({len(hard_infos)} samples)')


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def run_eval(
    config: str,
    checkpoint: str,
    work_dir: str,
    hard_pkl: str,
) -> None:
    """Run mmdet3d test.py with the val dataset restricted to the hard pkl."""
    # Build absolute path so cfg-options work regardless of cwd.
    hard_pkl_abs = str(Path(hard_pkl).resolve())

    sys.argv = [
        'test.py',
        config,
        checkpoint,
        '--work-dir', work_dir,
        '--cfg-options',
        f'val_dataloader.dataset.ann_file={hard_pkl_abs}',
        f'test_dataloader.dataset.ann_file={hard_pkl_abs}',
        f'val_evaluator.ann_file={hard_pkl_abs}',
        f'test_evaluator.ann_file={hard_pkl_abs}',
        'val_evaluator.type=SubsetNuScenesMetric',
        'test_evaluator.type=SubsetNuScenesMetric',
    ]

    from test import main  # mmdet3d tools/test.py
    main()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description='Evaluate any mmdet3d model on the N most challenging val scenes.')
    parser.add_argument(
        'config',
        help='mmdet3d config (e.g. configs/bev_fusion_full_v3b.py or configs/fcos3d_full.py)')
    parser.add_argument(
        'checkpoint',
        help='Model checkpoint (.pth)')
    parser.add_argument(
        '--work-dir', default='results/bev_fusion/hard20',
        help='Output directory for predictions and metrics')
    parser.add_argument(
        '--n-scenes', type=int, default=20,
        help='Number of hardest scenes to evaluate on (default: 20)')
    parser.add_argument(
        '--classifications', default='classifications.json',
        help='Path to scene classifications JSON (default: classifications.json)')
    parser.add_argument(
        '--val-pkl', default='data/nuscenes/nuscenes_infos_val.pkl',
        help='Val annotation pkl (default: data/nuscenes/nuscenes_infos_val.pkl)')
    parser.add_argument(
        '--hard-pkl', default=None,
        help='Output path for the filtered pkl '
             '(default: data/nuscenes/nuscenes_infos_hard<N>_val.pkl). '
             'If the file already exists, scene selection is skipped '
             '(useful for re-running a second model on the same hard scenes).')
    args = parser.parse_args()

    hard_pkl = (
        args.hard_pkl
        or f'data/nuscenes/nuscenes_infos_hard{args.n_scenes}_val.pkl'
    )

    if Path(hard_pkl).exists():
        import pickle
        with open(hard_pkl, 'rb') as f:
            n_samples = len(pickle.load(f)['data_list'])
        print(f'Reusing existing hard pkl: {hard_pkl}  ({n_samples} samples)')
    else:
        _, hard_infos, metainfo = select_hard_scenes(
            args.classifications, args.val_pkl, n=args.n_scenes)
        write_filtered_pkl(hard_infos, metainfo, hard_pkl)

    run_eval(args.config, args.checkpoint, args.work_dir, hard_pkl)


if __name__ == '__main__':
    main()
