_base_ = [
    '../../_base_/models/tsn_r50.py',
    '../../_base_/schedules/sgd_100e.py',
    '../../_base_/default_runtime.py'
]

# ================= 1. 模型架构设置 =================
model = dict(
    backbone=dict(num_segments=16),
    cls_head=dict(num_classes=48),
    test_cfg=dict(average_clips='prob')
)

# ================= 2. 数据集路径与流水线设置 =================
dataset_type = 'RawframeDataset'

data_root     = '/root/lanyun-fs/data/Diving48_rawframes'
data_root_val = '/root/lanyun-fs/data/Diving48_rawframes'

ann_file_train = '/root/lanyun-fs/AMS-Net-main/data/diving48/annotations/diving48_train_rawframes.txt'
ann_file_val   = '/root/lanyun-fs/AMS-Net-main/data/diving48/annotations/diving48_val_rawframes.txt'
ann_file_test  = '/root/lanyun-fs/AMS-Net-main/data/diving48/annotations/diving48_val_rawframes.txt'

img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53],
    std=[58.395, 57.12, 57.375],
    to_bgr=False)

train_pipeline = [
    dict(type='SampleFrames', clip_len=1, frame_interval=1, num_clips=16),
    dict(type='RawFrameDecode'),
    dict(type='Resize', scale=(-1, 256)),
    dict(
        type='MultiScaleCrop',
        input_size=224,
        scales=(1, 0.875, 0.75, 0.66),
        random_crop=False,
        max_wh_scale_gap=1),
    dict(type='Resize', scale=(224, 224), keep_ratio=False),
    dict(type='Flip', flip_ratio=0.5),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='FormatShape', input_format='NCHW'),
    dict(type='Collect', keys=['imgs', 'label'], meta_keys=[]),
    dict(type='ToTensor', keys=['imgs', 'label'])
]
val_pipeline = [
    dict(type='SampleFrames', clip_len=1, frame_interval=1,
         num_clips=16, test_mode=True),
    dict(type='RawFrameDecode'),
    dict(type='Resize', scale=(-1, 256)),
    dict(type='CenterCrop', crop_size=224),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='FormatShape', input_format='NCHW'),
    dict(type='Collect', keys=['imgs', 'label'], meta_keys=[]),
    dict(type='ToTensor', keys=['imgs'])
]
test_pipeline = [
    dict(type='SampleFrames', clip_len=1, frame_interval=1,
         num_clips=16, test_mode=True),
    dict(type='RawFrameDecode'),
    dict(type='Resize', scale=(-1, 256)),
    dict(type='CenterCrop', crop_size=224),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='FormatShape', input_format='NCHW'),
    dict(type='Collect', keys=['imgs', 'label'], meta_keys=[]),
    dict(type='ToTensor', keys=['imgs'])
]

data = dict(
    videos_per_gpu=4,
    workers_per_gpu=4,
    test_dataloader=dict(videos_per_gpu=1),
    train=dict(
        type=dataset_type,
        ann_file=ann_file_train,
        data_prefix=data_root,
        filename_tmpl='img_{:05}.jpg',
        pipeline=train_pipeline),
    val=dict(
        type=dataset_type,
        ann_file=ann_file_val,
        data_prefix=data_root_val,
        filename_tmpl='img_{:05}.jpg',
        pipeline=val_pipeline),
    test=dict(
        type=dataset_type,
        ann_file=ann_file_test,
        data_prefix=data_root_val,
        filename_tmpl='img_{:05}.jpg',
        pipeline=test_pipeline))

# ================= 3. 评估与保存策略 =================
evaluation = dict(
    interval=5,
    metrics=['top_k_accuracy', 'mean_class_accuracy'])

checkpoint_config = dict(interval=5, max_keep_ckpts=5)

# ================= 4. 优化器与学习率 =================
optimizer = dict(
    type='SGD',
    lr=0.000156,
    momentum=0.9,
    weight_decay=0.0001)

optimizer_config = dict(grad_clip=dict(max_norm=20, norm_type=2))

lr_config = dict(policy='step', step=[30, 60])

total_epochs = 75

# ================= 5. 运行配置 =================
work_dir = '/root/lanyun-fs/AMS-Net-main/work_dirs/tsn_r50_1x1x16_75e_diving48_rgb'

