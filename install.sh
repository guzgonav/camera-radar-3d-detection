#!/usr/bin/env bash
set -e

echo ">>> Creating virtual environment and installing dependencies..."
uv sync

echo ">>> Patching mmdet/mmdet3d hardcoded mmcv version cap..."
# mmdet and mmdet3d cap mmcv at <2.2.0, but the only prebuilt wheel for
# torch2.4.0 is 2.2.0. Bump the cap to 2.3.0 so the import check passes.
SITE=".venv/lib/python3.10/site-packages"
sed -i "s/mmcv_maximum_version = '2.2.0'/mmcv_maximum_version = '2.3.0'/" \
    "$SITE/mmdet/__init__.py" \
    "$SITE/mmdet3d/__init__.py"

echo ">>> Patching nuscenes_converter.py to skip LiDAR file existence check..."
# The PKL generator checks that each LiDAR .pcd.bin blob exists on disk.
# FCOS3D is camera-only and never reads LiDAR files, so this check is
# unnecessary and breaks PKL generation when LiDAR blobs are not downloaded.
sed -i 's/        mmengine\.check_file_exist(lidar_path)/        # mmengine.check_file_exist(lidar_path)  # skipped: LiDAR blobs not downloaded (camera-only pipeline)/' \
    "$SITE/mmdet3d/.mim/tools/dataset_converters/nuscenes_converter.py"

echo ">>> Patching create_data.py to skip ground truth database creation..."
# create_groundtruth_database() reads LiDAR .pcd.bin blobs to build a per-instance
# 3D crop database (nuscenes_dbinfos_train.pkl). This is only used for training-time
# copy-paste augmentation — not needed for inference. Skip it to avoid requiring
# LiDAR blobs.
sed -i 's/    create_groundtruth_database(dataset_name, root_path, info_prefix,/    # create_groundtruth_database(dataset_name, root_path, info_prefix,/' \
    "$SITE/mmdet3d/.mim/tools/create_data.py"
sed -i 's/                                f'"'"'{info_prefix}_infos_train.pkl'"'"')/    #                             f'"'"'{info_prefix}_infos_train.pkl'"'"')  # skipped: requires LiDAR blobs, only needed for training (not inference)/' \
    "$SITE/mmdet3d/.mim/tools/create_data.py"

echo ">>> Verifying installation..."
.venv/bin/python -c "
import torch, mmengine, mmcv, mmdet, mmdet3d
from mmcv.ops import nms
print('torch:', torch.__version__)
print('cuda:', torch.cuda.is_available())
print('mmengine:', mmengine.__version__)
print('mmcv:', mmcv.__version__)
print('mmcv ops: OK')
print('mmdet:', mmdet.__version__)
print('mmdet3d:', mmdet3d.__version__)
"

echo ">>> Done. Activate with: source .venv/bin/activate"
