# FCOS3D config override for nuScenes mini (v1.0-mini, 2 val scenes).
#
# Inherits everything from the installed FCOS3D finetune config
# (ResNet-101 + FPN + DCN, trained on full nuScenes mono3d) and overrides
# only the 3 things that differ for the mini split.
_base_ = '../.venv/lib/python3.10/site-packages/mmdet3d/.mim/configs/fcos3d/fcos3d_r101-caffe-dcn_fpn_head-gn_8xb2-1x_nus-mono3d_finetune.py'

class_names = [
    'car', 'truck', 'trailer', 'bus', 'construction_vehicle', 'bicycle',
    'motorcycle', 'pedestrian', 'traffic_cone', 'barrier'
]

# Without version='v1.0-mini', NuScenesMetric defaults to 'v1.0-trainval'
# and crashes because the mini PKL only contains 2 scenes.
metainfo = dict(version='v1.0-mini', classes=class_names)

val_dataloader = dict(dataset=dict(
    data_root='data/nuscenes-mini/',
    ann_file='nuscenes_infos_val.pkl',
    metainfo=metainfo))
test_dataloader = dict(dataset=dict(
    data_root='data/nuscenes-mini/',
    ann_file='nuscenes_infos_val.pkl',
    metainfo=metainfo))
val_evaluator = dict(data_root='data/nuscenes-mini/', ann_file='data/nuscenes-mini/nuscenes_infos_val.pkl')
test_evaluator = dict(data_root='data/nuscenes-mini/', ann_file='data/nuscenes-mini/nuscenes_infos_val.pkl')

# The finetune config hardcodes load_from='work_dirs/fcos3d_nus/latest.pth'
# which doesn't exist — clear it so the CLI --checkpoint argument is used.
load_from = None
