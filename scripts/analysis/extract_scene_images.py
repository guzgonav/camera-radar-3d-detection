#!/usr/bin/env python3
"""Extract the first image from each nuScenes scene and save to a directory."""

import sys
import os
from pathlib import Path
import shutil

# sys.executable → <project_root>/.venv/bin/python  (3 levels up = project root)
PROJECT_ROOT = Path(sys.executable).parent.parent.parent
os.chdir(PROJECT_ROOT)
print("Working dir:", Path.cwd())

from nuscenes.nuscenes import NuScenes
from PIL import Image

def extract_first_images(version='v1.0-mini', dataroot='data/nuscenes-mini', output_dir='data/scene_first_images'):
    """
    Extract the first camera image from each scene.

    Args:
        version: NuScenes version (e.g., 'v1.0-mini', 'v1.0-trainval')
        dataroot: Path to nuScenes dataset
        output_dir: Output directory to save images
    """
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {output_path}")

    # Load dataset
    nusc = NuScenes(version=version, dataroot=dataroot, verbose=False)
    scenes = nusc.scene
    print(f"Found {len(scenes)} scenes\n")

    # Extract first image from each scene
    for idx, scene in enumerate(scenes):
        scene_name = scene['name']
        first_sample_token = scene['first_sample_token']
        sample = nusc.get('sample', first_sample_token)

        # Find CAM_FRONT or first available camera
        cam_token = None
        cam_name = None

        for sensor, token in sample['data'].items():
            if 'CAM' in sensor:
                if sensor == 'CAM_FRONT':
                    cam_token = token
                    cam_name = sensor
                    break
                elif cam_token is None:  # Use first camera if CAM_FRONT not found
                    cam_token = token
                    cam_name = sensor

        if cam_token:
            cam_data = nusc.get('sample_data', cam_token)
            img_path = os.path.join(nusc.dataroot, cam_data['filename'])

            # Copy image to output directory with scene name
            output_img_path = output_path / f"{scene_name}.jpg"

            try:
                # Copy original image
                shutil.copy(img_path, output_img_path)
                print(f"[{idx+1}/{len(scenes)}] {scene_name} ({cam_name}) → {output_img_path.name}")
            except Exception as e:
                print(f"[{idx+1}/{len(scenes)}] {scene_name} - Error: {e}")
        else:
            print(f"[{idx+1}/{len(scenes)}] {scene_name} - No camera data found")

    print(f"\nDone! Extracted {len(list(output_path.glob('*.jpg')))} images to {output_path}")

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Extract first image from each nuScenes scene')
    parser.add_argument('--version', default='v1.0-mini', help='NuScenes version')
    parser.add_argument('--dataroot', default='data/nuscenes-mini', help='Path to nuScenes dataset')
    parser.add_argument('--output', default='data/scene_first_images', help='Output directory')

    args = parser.parse_args()
    extract_first_images(version=args.version, dataroot=args.dataroot, output_dir=args.output)
