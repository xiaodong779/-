_base_ = [
    '../../_base_/models/my_net1.py', '../../_base_/default_runtime.py'
]

# ================= 1. 模型架构设置 (纯净版 DGPN Baseline) =================
model = dict(
    backbone=dict(
        type='ResNet_DGPN',       # 调用你原来的 DGPN 独立底座
        num_segments=16,          # 16 帧采样
        dgam_cfg=dict()),         # DGAM 动态融合配置接口
    test_cfg=dict(average_clips='prob', fcn_test=True)
)

# ================= 2. 数据集路径设置 =================
dataset_type = 'RawframeDataset'
data_root = '/root/lanyun-fs/data/Diving48_rawframes'
data_root_val = '/root/lanyun-fs/data/Diving48_rawframes'
ann_file_train = '/root/lanyun-fs/AMS-Net-main/data/diving48/annotations/diving48_train_rawframes.txt'
ann_file_val = '/root/lanyun-fs/AMS-Net-main/data/diving48/annotations/diving48_val_rawframes.txt'
ann_file_test = '/root/lanyun-fs/AMS-Net-main/data/diving48/annotations/diving48_val_rawframes.txt'

img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_bgr=False)

train_pipeline = [
    dict(type='SampleFrames', clip_len=1, frame_interval=1, num_clips=16),
    dict(type='RawFrameDecode'),
    dict(type='RandomResizedCrop'),
    dict(type='Resize', scale=(224, 224), keep_ratio=False),
    dict(type='Flip', flip_ratio=0.5),
    dict(type='ColorJitter', color_space_aug=True),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='FormatShape', input_format='NCHW'),
    dict(type='Collect', keys=['imgs', 'label'], meta_keys=[]),
    dict(type='ToTensor', keys=['imgs', 'label'])
]

val_pipeline = [
    dict(type='SampleFrames', clip_len=1, frame_interval=1, num_clips=16, test_mode=True),
    dict(type='RawFrameDecode'),
    dict(type='Resize', scale=(-1, 256)),
    dict(type='CenterCrop', crop_size=224),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='FormatShape', input_format='NCHW'),
    dict(type='Collect', keys=['imgs', 'label'], meta_keys=[]),
    dict(type='ToTensor', keys=['imgs'])
]

test_pipeline = [
    dict(type='SampleFrames', clip_len=1, frame_interval=1, num_clips=16, twice_sample=True, test_mode=True),
    dict(type='RawFrameDecode'),
    dict(type='Resize', scale=(-1, 256)),
    dict(type='ThreeCrop', crop_size=256),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='FormatShape', input_format='NCHW'),
    dict(type='Collect', keys=['imgs', 'label'], meta_keys=[]),
    dict(type='ToTensor', keys=['imgs'])
]

data = dict(
    videos_per_gpu=4,
    workers_per_gpu=8,
    train_dataloader=dict(drop_last=True),
    test_dataloader=dict(videos_per_gpu=2),
    train=dict(
        type=dataset_type, ann_file=ann_file_train, data_prefix=data_root,
        filename_tmpl='img_{:05}.jpg', pipeline=train_pipeline),
    val=dict(
        type=dataset_type, ann_file=ann_file_val, data_prefix=data_root_val,
        filename_tmpl='img_{:05}.jpg', pipeline=val_pipeline),
    test=dict(
        type=dataset_type, ann_file=ann_file_test, data_prefix=data_root_val,
        filename_tmpl='img_{:05}.jpg', pipeline=test_pipeline))

evaluation = dict(interval=1, metrics=['top_k_accuracy', 'mean_class_accuracy']) 

# ================= 3. 优化器与学习率 =================
optimizer = dict(type='SGD', lr=0.001, momentum=0.9, weight_decay=0.0005, nesterov=True)
optimizer_config = dict(grad_clip=dict(max_norm=20, norm_type=2))
lr_config = dict(policy='step', step=[45, 55])
total_epochs = 75

# ================= 4. 运行配置 =================
checkpoint_config = dict(interval=1, max_keep_ckpts=3) 
# 💥 已帮你恢复为昨天的权重保存路径
work_dir = './work_dirs/ams_r50_diving48_dgpn_dynamic_fusion'