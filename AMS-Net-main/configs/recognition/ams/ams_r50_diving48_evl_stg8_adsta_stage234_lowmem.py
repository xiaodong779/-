# 文件路径:
# /hy-tmp/AMS-Net-main/configs/recognition/ams/ams_r50_diving48_evl_stg8_adsta_stage234_bs24_lr12e-3_32g.py

_base_ = [
    '../../_base_/models/ams_r50_pro.py',
    '../../_base_/default_runtime.py'
]

# ================= 1. 模型架构设置 =================
# 8 帧版本：ResNet + ADSTA(stage2/3/4) + Frozen EVL + Stage3 STG
# ADSTA：插入主分支 ResNet stage 后，位置由 adsta_stages 控制
# STG：默认只在 layer3 后使用 Frozen EVL 特征进行引导
model = dict(
    backbone=dict(
        type='ResNetAMS_Pro',
        depth=50,
        pretrained='torchvision://resnet50',
        torchvision_pretrain=True,

        # 8 帧
        num_segments=8,

        # 只输出 layer4，避免 Recognizer2D 收到 tuple
        out_indices=(3,),

        # ================= ADSTA 设置 =================
        # 开启 stage2、stage3、stage4。
        # stage1 分辨率最大，先不开，避免显存压力过大。
        use_adsta=True,
        adsta_stages=(2, 3, 4),

        # 32G 显存版：保持 reduction=4，保留模块表达能力。
        # 如果仍然 OOM，可改成 8；如果显存仍有富余，可尝试 adsta_mamba_expand=2。
        adsta_reduction=4,
        adsta_sample_mode='density_244',

        # 32G 显存保守增强版：先保持 expand=1，主要通过 batch 提高显存利用率。
        # 如果 batch=24 稳定且显存还有明显富余，再改成 expand=2 或 videos_per_gpu=32。
        adsta_mamba_expand=1,
        adsta_mamba_d_conv=3,
        adsta_dropout=0.0,

        # ================= Frozen EVL + STG 设置 =================
        use_frozen_evl=True,
        use_stg=True,

        # 第一版只开 Stage3 STG
        use_stg3=True,
        use_stg4=False,

        # mmaction pipeline 是 ImageNet Normalize，所以这里内部会转成 CLIP Normalize
        evl_input_norm='imagenet',

        evl_cfg=dict(
            backbone_name='ViT-B/16-lnpre',
            backbone_type='clip',
            backbone_path='/hy-tmp/weights/ViT-B-16.pt',
            backbone_mode='freeze_fp32',
            return_layer=-1,
        ),

        stg_cfg=dict(
            reduction=4,
            num_heads=8,
            window_size=7,
            fusion_init=(0.5, 0.5),
            gate_init_bias=-2.0,
            lambda_init=0.01,
            lambda_max=0.2,
            attn_drop=0.0,
            proj_drop=0.0,
            qkv_bias=True,
        )
    ),

    # 先不要 neck
    neck=None,

    cls_head=dict(
        num_classes=48,
        in_channels=2048
    ),

    test_cfg=dict(
        average_clips='prob',
        fcn_test=True
    )
)

# ================= 2. 数据集路径与流水线设置 =================
dataset_type = 'RawframeDataset'

data_root = '/hy-tmp/Diving48_rawframes'
data_root_val = '/hy-tmp/Diving48_rawframes'

ann_file_train = '/hy-tmp/AMS-Net-main/data/diving48/annotations/diving48_train_rawframes.txt'
ann_file_val = '/hy-tmp/AMS-Net-main/data/diving48/annotations/diving48_val_rawframes.txt'
ann_file_test = '/hy-tmp/AMS-Net-main/data/diving48/annotations/diving48_val_rawframes.txt'

img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53],
    std=[58.395, 57.12, 57.375],
    to_bgr=False
)

# 训练：8 帧
# RandomResizedCrop 是对齐的，因为主分支和 Frozen EVL 分支共用同一个 pipeline 后的 imgs
train_pipeline = [
    dict(type='SampleFrames', clip_len=1, frame_interval=1, num_clips=8),
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

# 验证：8 帧，CenterCrop 224
val_pipeline = [
    dict(type='SampleFrames', clip_len=1, frame_interval=1,
         num_clips=8, test_mode=True),
    dict(type='RawFrameDecode'),
    dict(type='Resize', scale=(-1, 256)),
    dict(type='CenterCrop', crop_size=224),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='FormatShape', input_format='NCHW'),
    dict(type='Collect', keys=['imgs', 'label'], meta_keys=[]),
    dict(type='ToTensor', keys=['imgs'])
]

# 测试：先不要 ThreeCrop 256
# 因为 Frozen CLIP ViT-B/16 当前要求输入必须是 224x224
test_pipeline = [
    dict(type='SampleFrames', clip_len=1, frame_interval=1,
         num_clips=8, test_mode=True),
    dict(type='RawFrameDecode'),
    dict(type='Resize', scale=(-1, 256)),
    dict(type='CenterCrop', crop_size=224),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='FormatShape', input_format='NCHW'),
    dict(type='Collect', keys=['imgs', 'label'], meta_keys=[]),
    dict(type='ToTensor', keys=['imgs'])
]

data = dict(
    # 32G 显存版：由 8 提到 24，提高显存利用率。
    # 之前 batch=8 显存约 7.9G，batch=24 预计约 23~26G，比较稳。
    # 如果仍有明显富余，可以再试 32，同时 lr 改为 0.016。
    videos_per_gpu=24,
    workers_per_gpu=8,

    train_dataloader=dict(drop_last=True),

    # 验证/测试也适当提高，但仍保持保守，避免 eval OOM
    val_dataloader=dict(videos_per_gpu=4, workers_per_gpu=4),
    test_dataloader=dict(videos_per_gpu=4, workers_per_gpu=4),

    train=dict(
        type=dataset_type,
        ann_file=ann_file_train,
        data_prefix=data_root,
        filename_tmpl='img_{:05}.jpg',
        pipeline=train_pipeline
    ),

    val=dict(
        type=dataset_type,
        ann_file=ann_file_val,
        data_prefix=data_root_val,
        filename_tmpl='img_{:05}.jpg',
        pipeline=val_pipeline
    ),

    test=dict(
        type=dataset_type,
        ann_file=ann_file_test,
        data_prefix=data_root_val,
        filename_tmpl='img_{:05}.jpg',
        pipeline=test_pipeline
    )
)

# ================= 3. 评估与保存策略 =================
evaluation = dict(
    interval=1,
    metrics=['top_k_accuracy', 'mean_class_accuracy'],
    topk=(1, 5),
    save_best='top1_acc',
    rule='greater'
)

checkpoint_config = dict(
    interval=5,
    max_keep_ckpts=5
)

# ================= 4. 日志与可视化监控 =================
log_config = dict(
    interval=20,
    hooks=[
        dict(type='TextLoggerHook'),
        dict(type='TensorboardLoggerHook'),
    ]
)

# ================= 5. 优化器与学习率 =================
# 原 lowmem: videos_per_gpu=8, lr=0.004
# 现在: videos_per_gpu=24，是原来的 3 倍，所以 lr 按线性缩放到 0.012。
# 如果 videos_per_gpu=32，可把 lr 调到 0.016。
optimizer = dict(
    type='SGD',
    lr=0.012,
    momentum=0.9,
    weight_decay=0.0005,
    nesterov=True
)

optimizer_config = dict(
    grad_clip=dict(max_norm=20, norm_type=2)
)

lr_config = dict(
    policy='step',
    step=[45, 55]
)

total_epochs = 75

# ================= 6. 运行配置 =================
work_dir = './work_dirs/ams_r50_1x1x8_diving48_rgb_EVL_STG_ADSTA_stage234_bs24_lr12e-3_32g_v1'
log_level = 'INFO'
dist_params = dict(backend='nccl')
workflow = [('train', 1)]
opencv_num_threads = 0
mp_start_method = 'fork'
load_from = None
resume_from = None
