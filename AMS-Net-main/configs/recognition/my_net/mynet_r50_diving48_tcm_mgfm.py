_base_ = [
    '../../_base_/models/my_net1.py', '../../_base_/default_runtime.py'
]

# ================= 1. 究极轻量化高精度架构：PAME (物理提纯) + GRAM (时空节奏) =================
model = dict(
    type='Recognizer2D',
    backbone=dict(
        type='ResNet_ST',         
        pretrained='torchvision://resnet50',
        depth=50,
        num_segments=16,          
        norm_eval=False,
        # 💥 核心修正：显式指定只输出最后一层特征图 (Layer 4)，形状为 [N*T, 2048, 7, 7]
        # 这样返回的就是一个 Tensor 而不是 Tuple，彻底解决 size() 报错
        out_indices=(3, )         
    ),
    
    # 彻底砍掉 Neck，让 Backbone 直接对接 Head
    neck=None,
        
    cls_head=dict(
        type='TSMHead',            
        num_classes=48,            
        in_channels=2048,          # ResNet50 最后一层输出通道
        num_segments=16,           
        spatial_type='avg',
        consensus=dict(type='AvgConsensus', dim=1), 
        dropout_ratio=0.5,         
        init_std=0.01,
        is_shift=False             
    ),
    train_cfg=None,
    test_cfg=dict(average_clips='prob', fcn_test=False)
)

# ================= 2. 数据集路径设置 (保持你的原始路径) =================
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

# ================= 3. 优化器与学习率 (单卡 Warmup 版) =================
optimizer = dict(type='SGD', lr=0.001, momentum=0.9, weight_decay=0.0005, nesterov=True)
optimizer_config = dict(grad_clip=dict(max_norm=20, norm_type=2))

# 💥 增加 Warmup，防止新模块初始梯度过载
lr_config = dict(
    policy='step', 
    step=[45, 55], 
    warmup='linear', 
    warmup_iters=5, 
    warmup_ratio=0.1, 
    warmup_by_epoch=True
)
total_epochs = 75

# ================= 4. 运行配置 =================
checkpoint_config = dict(interval=1, max_keep_ckpts=3) 
work_dir = './work_dirs/ams_r50_1x1x16_75e_diving48_pame_gram'