# 文件路径:
# /hy-tmp/AMS-Net-main/configs/recognition/ams/ams_r50_diving48_evl_stg8_adsta_stage1234_tsm_8f_bs8_lr4e-3_lowmem.py
#
# 与完整版 stage1234 系列对齐（即 8f dense 83.55% / 16f TTA 85.43% 的组件配置）:
#   - ADSTA ×4: adsta_stages=(1, 2, 3, 4)
#   - STG  ×3: use_stg2/3/4=True, use_stg1=False (stg234)
# 在此基础上, 主分支 ResNet 残差块内加 TSM 时移 (blockres, shift_div=8)。
# 训练超参与 stage1234_8f 原版对齐: bs8 + SGD lr4e-3 + step[45,55] + 75 epochs。

_base_ = [
    '../../_base_/models/ams_r50_pro.py',
    '../../_base_/default_runtime.py'
]

# ================= 1. 模型架构设置 =================
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

        # ================= TSM 时移（本实验新增） =================
        # blockres 模式: 残差分支做时移, shortcut 不动; shift_div=8 与 TSM 论文一致
        use_tsm=True,
        tsm_shift_div=8,

        # ================= ADSTA 设置（完整版: stage1-4 全开） =================
        use_adsta=True,
        adsta_stages=(1, 2, 3, 4),
        adsta_reduction=4,
        adsta_sample_mode='density_244',
        adsta_mamba_expand=1,
        adsta_mamba_d_conv=3,
        adsta_dropout=0.0,

        # ================= Frozen EVL + STG 设置（完整版: stg234） =================
        use_frozen_evl=True,
        use_stg=True,
        use_stg1=False,
        use_stg2=True,
        use_stg3=True,
        use_stg4=True,
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

# ================= 2. 数据集路径与流水线设置（与 stage1234 对齐） =================
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
    # 与 stage1234_8f 原版 lowmem 对齐: bs8 + lr4e-3
    videos_per_gpu=8,
    workers_per_gpu=8,

    train_dataloader=dict(drop_last=True),

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

# ================= 3. 评估与保存策略（与 stage1234 对齐） =================
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

# ================= 5. 优化器与学习率（与 stage1234_8f 原版对齐） =================
optimizer = dict(
    type='SGD',
    lr=0.004,
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
work_dir = './work_dirs/ams_r50_1x1x8_diving48_rgb_EVL_STG_ADSTA_stage1234_TSM_bs8_lr4e-3_32g_v1'
log_level = 'INFO'
dist_params = dict(backend='nccl')
workflow = [('train', 1)]
opencv_num_threads = 0
mp_start_method = 'fork'
load_from = None
resume_from = None
