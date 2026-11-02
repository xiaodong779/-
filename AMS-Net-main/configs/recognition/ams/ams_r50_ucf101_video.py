_base_ = [
    '../../_base_/models/ams_r50_pro.py',
    '../../_base_/default_runtime.py'
]

model = dict(
    backbone=dict(
        num_segments=16,
        norm_eval=False
    ),
    neck=dict(
        temporal_modulation_cfg=dict(downsample_scales=(16, 16)),
        aux_head_cfg=dict(
            out_channels=101,
            loss_weight=0.5,
            loss_cls=dict(
                type='LabelSmoothCrossEntropyLoss',
                label_smooth_eps=0.1
            )
        )
    ),
    cls_head=dict(
        num_classes=101,
        dropout_ratio=0.5
    ),
    test_cfg=dict(
        average_clips='prob',
        fcn_test=True
    )
)

dataset_type = 'VideoDataset'
data_root = '/root/lanyun-fs/AMS-Net-main/data/ucf101/videos/UCF-101'
data_root_val = '/root/lanyun-fs/AMS-Net-main/data/ucf101/videos/UCF-101'

ann_file_train = '/root/lanyun-fs/AMS-Net-main/data/ucf101/annotations/trainlist01_zero.txt'
ann_file_val = '/root/lanyun-fs/AMS-Net-main/data/ucf101/annotations/testlist01_with_label_zero.txt'
ann_file_test = '/root/lanyun-fs/AMS-Net-main/data/ucf101/annotations/testlist01_with_label_zero.txt'

img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53],
    std=[58.395, 57.12, 57.375],
    to_bgr=False
)

train_pipeline = [
    dict(type='DecordInit'),
    dict(type='SampleFrames', clip_len=1, frame_interval=1, num_clips=16),
    dict(type='DecordDecode'),
    dict(type='Resize', scale=(-1, 256)),
    dict(type='RandomResizedCrop'),
    dict(type='Resize', scale=(224, 224), keep_ratio=False),
    dict(type='Flip', flip_ratio=0.5),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='FormatShape', input_format='NCHW'),
    dict(type='Collect', keys=['imgs', 'label'], meta_keys=[]),
    dict(type='ToTensor', keys=['imgs', 'label'])
]

val_pipeline = [
    dict(type='DecordInit'),
    dict(type='SampleFrames', clip_len=1, frame_interval=1,
         num_clips=16, test_mode=True),
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
    dict(type='SampleFrames', clip_len=1, frame_interval=1,
         num_clips=16, twice_sample=True, test_mode=True),
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
    val_dataloader=dict(videos_per_gpu=2),
    test_dataloader=dict(videos_per_gpu=1),
    train=dict(
        type=dataset_type,
        ann_file=ann_file_train,
        data_prefix=data_root,
        pipeline=train_pipeline
    ),
    val=dict(
        type=dataset_type,
        ann_file=ann_file_val,
        data_prefix=data_root_val,
        pipeline=val_pipeline
    ),
    test=dict(
        type=dataset_type,
        ann_file=ann_file_test,
        data_prefix=data_root_val,
        pipeline=test_pipeline
    )
)

evaluation = dict(
    interval=1,
    metrics=['top_k_accuracy'],
    topk=(1, 5),
    save_best='top_k_accuracy',
    rule='greater'
)

checkpoint_config = dict(interval=1, max_keep_ckpts=5)

log_config = dict(
    interval=20,
    hooks=[
        dict(type='TextLoggerHook'),
        dict(type='TensorboardLoggerHook'),
    ]
)

optimizer = dict(
    type='SGD',
    lr=0.001,
    momentum=0.9,
    weight_decay=0.0005,
    nesterov=True
)

optimizer_config = dict(
    grad_clip=dict(max_norm=20, norm_type=2)
)

lr_config = dict(
    policy='step',
    step=[30, 45]
)

total_epochs = 60

work_dir = './work_dirs/ams_r50_ucf101_video_16f_split1_imagenet_v1'
