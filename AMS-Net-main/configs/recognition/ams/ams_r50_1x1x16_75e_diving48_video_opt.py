# =========================================================
# 配置文件名: ams_r50_1x1x16_75e_diving48_video_opt.py
# 描述: 单 GPU 特化版 - 视频直读 + 严格学习率缩放
# =========================================================

_base_ = [
    '../../_base_/models/ams_r50.py', 
    '../../_base_/default_runtime.py'
]

# 1. 核心路径配置 (保持你的视频路径不变)
dataset_type = 'VideoDataset'
data_root = '/root/lanyun-fs/data/Diving48_rgb'
data_root_val = '/root/lanyun-fs/data/Diving48_rgb'
ann_file_train = '/root/lanyun-fs/AMS-Net-main/data/diving48/annotations/train.txt'
ann_file_val   = '/root/lanyun-fs/AMS-Net-main/data/diving48/annotations/val.txt'
ann_file_test  = '/root/lanyun-fs/AMS-Net-main/data/diving48/annotations/test.txt'

# 2. 模型配置
model = dict(backbone=dict(num_segments=16, gamma=1),
             neck=dict(gamma=1, temporal_modulation_cfg=dict(downsample_scales=(16, 16)),
             aux_head_cfg=dict(out_channels=48, loss_weight=0.5)),
             cls_head=dict(num_classes=48),
             test_cfg=dict(average_clips='prob', fcn_test=True))

img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_bgr=False)

# 3. 视频直读流水线
train_pipeline = [
    dict(type='DecordInit'),
    dict(type='SampleFrames', clip_len=1, frame_interval=1, num_clips=16),
    dict(type='DecordDecode'),
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
    dict(type='DecordInit'),
    dict(type='SampleFrames', clip_len=1, frame_interval=1, num_clips=16, test_mode=True),
    dict(type='DecordDecode'),
    dict(type='Resize', scale=(-1, 256)),
    dict(type='CenterCrop', crop_size=224),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='FormatShape', input_format='NCHW'),
    dict(type='Collect', keys=['imgs', 'label'], meta_keys=[]),
    dict(type='ToTensor', keys=['imgs'])
]

test_pipeline = [
    dict(type='DecordInit'),
    dict(type='SampleFrames', clip_len=1, frame_interval=1, num_clips=16, twice_sample=True, test_mode=True),
    dict(type='DecordDecode'),
    dict(type='Resize', scale=(-1, 256)),
    dict(type='ThreeCrop', crop_size=256),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='FormatShape', input_format='NCHW'),
    dict(type='Collect', keys=['imgs', 'label'], meta_keys=[]),
    dict(type='ToTensor', keys=['imgs'])
]

# 4. 单卡数据集配置 (1 GPU, 每次处理 4 个视频)
data = dict(
    videos_per_gpu=4,
    workers_per_gpu=8,
    train_dataloader=dict(drop_last=True),
    test_dataloader=dict(videos_per_gpu=2),
    train=dict(type=dataset_type, ann_file=ann_file_train, data_prefix=data_root, pipeline=train_pipeline),
    val=dict(type=dataset_type, ann_file=ann_file_val, data_prefix=data_root_val, pipeline=val_pipeline),
    test=dict(type=dataset_type, ann_file=ann_file_test, data_prefix=data_root_val, pipeline=test_pipeline))

evaluation = dict(interval=1, metrics=['top_k_accuracy', 'mean_class_accuracy']) 

# =========================================================
# 5. 单 GPU 专属优化器 (学习率已严格缩放)
# =========================================================
optimizer = dict(
    type='SGD', lr=0.001, momentum=0.9, weight_decay=0.0005, 
    nesterov=True) 
optimizer_config = dict(grad_clip=dict(max_norm=20, norm_type=2))

lr_config = dict(
    policy='step', 
    step=[45, 55],
    warmup='linear',         # 加入单卡预热
    warmup_iters=10,         
    warmup_ratio=0.1)        

total_epochs = 75
work_dir = './work_dirs/ams_r50_video_optimized'