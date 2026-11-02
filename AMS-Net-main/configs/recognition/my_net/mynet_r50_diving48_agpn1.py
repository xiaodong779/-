_base_ = [
    '../../_base_/models/my_net1.py', '../../_base_/default_runtime.py'
]

# ================= 1. 数据集路径设置 =================
dataset_type = 'RawframeDataset'
data_root = '/root/lanyun-fs/data/Diving48_rawframes'
data_root_val = '/root/lanyun-fs/data/Diving48_rawframes'
ann_file_train = '/root/lanyun-fs/AMS-Net-main/data/diving48/annotations/diving48_train_rawframes.txt'
ann_file_val = '/root/lanyun-fs/AMS-Net-main/data/diving48/annotations/diving48_val_rawframes.txt'
ann_file_test = '/root/lanyun-fs/AMS-Net-main/data/diving48/annotations/diving48_val_rawframes.txt'
# ==================================================

# 【极度关键】去除了原来针对 ams_r50 的特有冲突参数（如 gamma）。
# 因为你在 my_net1.py 里已经把 16帧、48分类 等参数配好了，这里只需保留测试模式配置即可！
model = dict(
    test_cfg=dict(average_clips='prob', fcn_test=True)
)

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
    dict(
        type='SampleFrames',
        clip_len=1,
        frame_interval=1,
        num_clips=16,
        test_mode=True),
    dict(type='RawFrameDecode'),
    dict(type='Resize', scale=(-1, 256)),
    dict(type='CenterCrop', crop_size=224),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='FormatShape', input_format='NCHW'),
    dict(type='Collect', keys=['imgs', 'label'], meta_keys=[]),
    dict(type='ToTensor', keys=['imgs'])
]

test_pipeline = [
    dict(
        type='SampleFrames',
        clip_len=1,
        frame_interval=1,
        num_clips=16,
        twice_sample=True,
        test_mode=True),
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

evaluation = dict(
    interval=1, metrics=['top_k_accuracy', 'mean_class_accuracy']) 

# ================= 2. 优化器与学习率设置 =================
# 遵循线性缩放法则，单卡训练设定 lr=0.001
optimizer = dict(
    type='SGD', lr=0.001, momentum=0.9, weight_decay=0.0005,
    nesterov=True)  
# ======================================================

optimizer_config = dict(grad_clip=dict(max_norm=20, norm_type=2))
# learning policy
lr_config = dict(policy='step', step=[45, 55])
total_epochs = 75

# ================= 3. 运行目录 =================
# 我帮你加了后缀，避免新模型覆盖掉你以前跑出来的原始权重！
work_dir = './work_dirs/ams_r50_1x1x16_75e_diving48_rawframes_agpm_fusion'