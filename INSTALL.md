# Installation Guide

Environment for camera-radar fusion 3D object detection on nuScenes.

## System Requirements

- GPU: NVIDIA RTX 3090
- CUDA: 12.1
- OS: Linux (Ubuntu)
- Python: 3.10

## 1. Create the virtual environment

```bash
uv venv .venv --python 3.10
source .venv/bin/activate
```

## 2. Install PyTorch (cu121)

Defined in `pyproject.toml`. Install via:

```bash
uv sync
```

This installs:
- `torch==2.4.0+cu121`
- `torchvision==0.19.0`
- `torchaudio==2.4.0`
- `openmim>=0.3.9`

## 3. Install the MMDetection3D stack

Install each package in order using `uv pip`. Do **not** use bare `pip` in a uv-managed venv.

### mmengine

```bash
uv pip install mmengine
```

### mmcv

mmcv requires a prebuilt CUDA wheel. Install directly by URL — do **not** use `--extra-index-url` as uv will resolve the sdist from PyPI and attempt compilation.

```bash
uv pip install "https://download.openmmlab.com/mmcv/dist/cu121/torch2.4.0/mmcv-2.2.0-cp310-cp310-manylinux1_x86_64.whl"
```

> For other CUDA/torch combinations, browse: `https://download.openmmlab.com/mmcv/dist/{cuda}/{torch}/index.html`

### mmdet

```bash
uv pip install "mmdet==3.3.0"
```

### mmdet3d

```bash
uv pip install "mmdet3d==1.4.0"
```

## 4. Patch hardcoded version caps

mmdet and mmdet3d hardcode `mmcv_maximum_version = '2.2.0'` with a strict `<` check, but the only prebuilt mmcv wheel available for torch2.4.0 is 2.2.0. Bump the cap in both installed `__init__.py` files:

```bash
sed -i "s/mmcv_maximum_version = '2.2.0'/mmcv_maximum_version = '2.3.0'/" \
    .venv/lib/python3.10/site-packages/mmdet/__init__.py \
    .venv/lib/python3.10/site-packages/mmdet3d/__init__.py
```

> **Note:** This patch must be reapplied if mmdet or mmdet3d are reinstalled.

## 5. Patch LiDAR file existence check

The PKL generator verifies that each LiDAR `.pcd.bin` blob exists on disk. FCOS3D is camera-only and never reads these files, so the check is unnecessary and breaks PKL generation when LiDAR blobs are not downloaded. `install.sh` already applies this patch automatically. To apply it manually:

```bash
sed -i 's/        mmengine\.check_file_exist(lidar_path)/        # mmengine.check_file_exist(lidar_path)  # skipped: LiDAR blobs not downloaded (camera-only pipeline)/' \
    .venv/lib/python3.10/site-packages/mmdet3d/.mim/tools/dataset_converters/nuscenes_converter.py
```

> **Note:** This patch must be reapplied if mmdet3d is reinstalled.

Also patch `create_data.py` to skip ground truth database creation. `create_groundtruth_database()` reads LiDAR `.pcd.bin` blobs to build a per-instance 3D crop database used only for training-time copy-paste augmentation — not needed for inference. `install.sh` applies this patch automatically. To apply manually:

```bash
sed -i 's/    create_groundtruth_database(dataset_name, root_path, info_prefix,/    # create_groundtruth_database(dataset_name, root_path, info_prefix,/' \
    .venv/lib/python3.10/site-packages/mmdet3d/.mim/tools/create_data.py
sed -i "s/                                f'{info_prefix}_infos_train.pkl')/    #                             f'{info_prefix}_infos_train.pkl')  # skipped: requires LiDAR blobs, only needed for training (not inference)/" \
    .venv/lib/python3.10/site-packages/mmdet3d/.mim/tools/create_data.py
```

> **Note:** This patch must be reapplied if mmdet3d is reinstalled.

## 6. Set up nuScenes data

This project supports running both the mini and full nuScenes splits simultaneously. They live in separate directories:

```
data/
├── nuscenes-mini/          ← mini split (10 scenes, all sensor blobs)
│   ├── v1.0-mini/
│   ├── samples/
│   ├── sweeps/
│   ├── maps/
│   ├── nuscenes_infos_train.pkl
│   └── nuscenes_infos_val.pkl
└── nuscenes/               ← full split (1000 scenes, camera + radar blobs only)
    ├── v1.0-trainval/
    ├── v1.0-mini -> ../nuscenes-mini/v1.0-mini   ← symlink (see below)
    ├── samples/
    ├── sweeps/
    ├── maps/
    ├── nuscenes_infos_train.pkl
    └── nuscenes_infos_val.pkl
```

### Mini split

Download the nuScenes mini split from https://www.nuscenes.org/nuscenes and extract into `data/nuscenes-mini/`.

> **Important:** The nuScenes downloader may extract into a nested subdirectory. The expected layout is `data/nuscenes-mini/{v1.0-mini,samples,sweeps,maps}` — not `data/nuscenes-mini/v1.0-mini/{...}`.

Generate the PKL info files:

```bash
bash pipelines/prepare_data_mini.sh
```

This produces `data/nuscenes-mini/nuscenes_infos_train.pkl` and `data/nuscenes-mini/nuscenes_infos_val.pkl`.

### Full split

Download the nuScenes full split (camera + radar blobs; LiDAR blobs not required) and extract into `data/nuscenes/`.

MMDetection3D's `update_infos_to_v2.py` hardcodes `./data/nuscenes` as the data root and appends the version string (e.g. `v1.0-mini`) to build a path. Create a symlink so it can find the mini metadata from that hardcoded root:

```bash
ln -s /absolute/path/to/data/nuscenes-mini/v1.0-mini data/nuscenes/v1.0-mini
```

Generate the PKL info files:

```bash
bash pipelines/prepare_data_full.sh
```

This produces `data/nuscenes/nuscenes_infos_train.pkl` and `data/nuscenes/nuscenes_infos_val.pkl`.

## 7. Verify

```bash
python -c "
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
```

Expected output:
```
torch: 2.4.0+cu121
cuda: True
mmengine: 0.10.7
mmcv: 2.2.0
mmcv ops: OK
mmdet: 3.3.0
mmdet3d: 1.4.0
```

## Package summary

| Package        | Version      |
|----------------|--------------|
| Python         | 3.10         |
| torch          | 2.4.0+cu121  |
| torchvision    | 0.19.0       |
| torchaudio     | 2.4.0        |
| mmengine       | 0.10.7       |
| mmcv           | 2.2.0        |
| mmdet          | 3.3.0        |
| mmdet3d        | 1.4.0        |
| ultralytics    | >=8.0        |
| nuscenes-devkit| 1.1.11       |
| open3d         | 0.19.0       |
| numba          | 0.64.0       |
| numpy          | 2.2.6        |
| scipy          | 1.15.3       |
