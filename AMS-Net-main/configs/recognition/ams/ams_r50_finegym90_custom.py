_base_ = [
    '../../_base_/models/ams_r50_pro.py',
    '../../_base_/default_runtime.py'
]

# ================= 1. 模型架构设置 =================
model = dict(
    backbone=dict(num_segments=16),
    neck=dict(
        temporal_modulation_cfg=dict(downsample_scales=(16, 16)),
        aux_head_cfg=dict(
            out_channels=90,  # 匹配你清洗后的 90 类
            loss_weight=0.5,
            loss_cls=dict(
                type='LabelSmoothCrossEntropyLoss',
                label_smooth_eps=0.1
            )
        )),
    cls_head=dict(num_classes=90), # 匹配你清洗后的 90 类
    test_cfg=dict(average_clips='prob', fcn_test=True)
)

# ================= 2. 数据集路径与流水线设置 =================
dataset_type = 'VideoDataset'
# 视频文件所在的实际目录
data_root     = '/root/lanyun-fs/AMS-Net-main/data/finegym/subactions'
data_root_val = '/root/lanyun-fs/AMS-Net-main/data/finegym/subactions'

# --- 关键修改：使用你提供的最新路径 ---
ann_file_train = '/root/lanyun-fs/AMS-Net-main/gym99_refined_train.txt'
ann_file_val   = '/root/lanyun-fs/AMS-Net-main/data/finegym/gym99_refined_val.txt'
ann_file_test  = '/root/lanyun-fs/AMS-Net-main/data/finegym/gym99_refined_val.txt'

img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53],
    std=[58.395, 57.12, 57.375],
    to_bgr=False)

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

data = dict(
    videos_per_gpu=4,
    workers_per_gpu=4,
    train_dataloader=dict(drop_last=True),
    test_dataloader=dict(videos_per_gpu=2),
    train=dict(
        type=dataset_type,
        ann_file=ann_file_train,
        data_prefix=data_root,
        pipeline=train_pipeline),
    val=dict(
        type=dataset_type,
        ann_file=ann_file_val,
        data_prefix=data_root_val,
        pipeline=val_pipeline),
    test=dict(
        type=dataset_type,
        ann_file=ann_file_test,
        data_prefix=data_root_val,
        pipeline=test_pipeline))

# ================= 3. 评估与保存策略 =================
evaluation = dict(
    interval=1,
    metrics=['top_k_accuracy', 'mean_class_accuracy'],
    topk=(1, 5),
    save_best='top_k_accuracy',
    rule='greater'
)

# ================= 5. 优化器与学习率 =================
optimizer = dict(
    type='SGD', lr=0.001, momentum=0.9,
    weight_decay=0.0005, nesterov=True)

optimizer_config = dict(grad_clip=dict(max_norm=20, norm_type=2))

lr_config = dict(policy='step', step=[40, 60])
total_epochs = 80

# ================= 6. 运行配置 =================
work_dir = './work_dirs/ams_r50_finegym90_video_PAME_MTCM_Pro'
