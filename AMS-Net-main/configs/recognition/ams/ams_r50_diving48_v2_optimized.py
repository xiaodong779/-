_base_ = ['../../_base_/default_runtime.py']

# === 模型定义 (参数写死在这里，绝对不会读错) ===
model = dict(
    type='Recognizer2D',
    backbone=dict(
        type='ResNetAMS',
        pretrained='torchvision://resnet50',
        depth=50,
        out_indices=(2, 3),
        CompetitiveFusion=True,
        temporal_block_indices=(0, 2), # 满血版双模块
        norm_eval=False),
    neck=dict(
        type='CSTP',
        flow_type='parallel',
        in_channels=(1024, 2048),
        out_channels=1024,
        spatial_modulation_cfg=dict(in_channels=(1024, 2048), out_channels=2048),
        
        # 【这里的 16 是修复报错的关键】
        # 16帧 / 16 = 1个输出 -> 匹配 batch_size 8
        temporal_modulation_cfg=dict(downsample_scales=(16, 16)), 
        
        upsample_cfg=dict(scale_factor=(1, 1, 1)),
        downsample_cfg=dict(downsample_scale=(1, 1, 1)),
        level_fusion_cfg=dict(
            in_channels=(1024, 1024),
            mid_channels=(1024, 1024),
            out_channels=2048,
            downsample_scales=((1, 1, 1), (1, 1, 1))),
        aux_head_cfg=dict(out_channels=48, loss_weight=0.5)),
    cls_head=dict(
        type='AMSHead',
        num_classes=48, 
        in_channels=2048,
        spatial_type='avg',
        consensus=dict(type='AvgConsensus', dim=1),
        dropout_ratio=0.5,
        init_std=0.01),
    train_cfg=None,
    test_cfg=dict(average_clips='prob', fcn_test=True)
)

# === 数据定义 ===
dataset_type = 'VideoDataset'
data_root = '/root/lanyun-fs/data/Diving48_rgb'
data_root_val = '/root/lanyun-fs/data/Diving48_rgb'
ann_file_train = '/root/lanyun-fs/AMS-Net-main/data/diving48/annotations/train_v2.txt'
ann_file_val   = '/root/lanyun-fs/AMS-Net-main/data/diving48/annotations/val_v2.txt'
ann_file_test  = '/root/lanyun-fs/AMS-Net-main/data/diving48/annotations/test_v2.txt'

img_norm_cfg = dict(mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_bgr=False)

train_pipeline = [
    dict(type='DecordInit'),
    dict(type='SampleFrames', clip_len=1, frame_interval=1, num_clips=16),
    dict(type='DecordDecode'),
    dict(type='Resize', scale=(256, 256), keep_ratio=False),
    dict(type='RandomCrop', size=224),
    dict(type='Flip', flip_ratio=0), 
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
    videos_per_gpu=8, 
    workers_per_gpu=4, 
    train_dataloader=dict(drop_last=True),
    test_dataloader=dict(videos_per_gpu=1),
    train=dict(type=dataset_type, ann_file=ann_file_train, data_prefix=data_root, pipeline=train_pipeline),
    val=dict(type=dataset_type, ann_file=ann_file_val, data_prefix=data_root_val, pipeline=val_pipeline),
    test=dict(type=dataset_type, ann_file=ann_file_test, data_prefix=data_root_val, pipeline=test_pipeline))

# === 训练策略 ===
evaluation = dict(interval=5, metrics=['top_k_accuracy', 'mean_class_accuracy'], save_best='top1_acc')
optimizer = dict(type='SGD', lr=0.001, momentum=0.9, weight_decay=0.0005, nesterov=True)
optimizer_config = dict(grad_clip=dict(max_norm=20, norm_type=2))
lr_config = dict(policy='step', step=[30, 60], warmup='linear', warmup_iters=500, warmup_ratio=0.001)
total_epochs = 75
checkpoint_config = dict(interval=5, max_keep_ckpts=3)
log_config = dict(interval=20, hooks=[dict(type='TextLoggerHook')])
# 使用新目录防止读到旧权重
work_dir = './work_dirs/ams_r50_diving48_v2_FINAL_RUN'
