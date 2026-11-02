# 文件路径:
# /hy-tmp/AMS-Net-main/configs/recognition/ams/ams_r50_3d_diving48_evl_stg8_adsta_stage1234_lowmem.py
#
# 版本说明:
#   3D ResNet SlowOnly + 双分支专家
#   上分支: Frozen EVL CLIP ViT-B/16 (空间专家，冻结)
#   下分支: AMSResNet3dSlowOnly (时序专家)
#   ADSTABlock + BWSTGBlock 在每个 stage(1/2/3/4)后均插入
#   stage1 ADSTA reduction=8 (显存控制)，其他 reduction=4

_base_ = [
    '../../_base_/models/ams_r50_pro.py',
    '../../_base_/default_runtime.py'
]

# ================= 1. 模型架构设置 =================
model = dict(
    type='Recognizer3D',  # ★ 2D → 3D

    backbone=dict(
        type='AMSResNet3dSlowOnlyPro',  # ★ 新 3D 主干
        depth=50,
        pretrained='torchvision://resnet50',
        pretrained2d=True,  # 2D 权重膨胀为 3D

        # 8 帧 3D 输入
        num_segments=8,

        # 只输出 layer4
        out_indices=(3,),

        # 3D 卷积参数（SlowOnly）
        conv1_kernel=(1, 7, 7),
        conv1_stride_t=1,
        pool1_stride_t=1,
        inflate=(0, 0, 1, 1),
        with_pool2=False,

        # ================= ADSTA: 每个 stage 后插入 =================
        use_adsta=True,
        adsta_stages=(1, 2, 3, 4),  # ★ 全 stage 插入
        adsta_reduction=4,
        adsta_reduction_s1=8,       # ★ stage1 单独用 reduction=8 控制显存
        adsta_sample_mode='density_244',
        adsta_mamba_expand=1,
        adsta_mamba_d_conv=3,
        adsta_dropout=0.0,

        # ================= Frozen EVL ViT 空间专家 =================
        use_frozen_evl=True,
        evl_input_norm='imagenet',

        evl_cfg=dict(
            backbone_name='ViT-B/16-lnpre',
            backbone_type='clip',
            backbone_path='/hy-tmp/weights/ViT-B-16.pt',
            backbone_mode='freeze_fp32',
            return_layer=-1,
        ),

        # ================= BW-STG: 每个 stage 后插入 =================
        use_stg=True,
        stg_stages=(1, 2, 3, 4),  # ★ 全 stage 双向引导
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
        ),
    ),

    neck=None,
    cls_head=dict(num_classes=48, in_channels=2048),

    # ★ 去掉 fcn_test（2D TSN 专用）
    # AMSHead 内部已用 consensus 对 num_segs 做 clip 平均, 框架 average_clip 需关闭,
    # 否则测试时会二次折叠 (4//8=0) 崩溃。
    test_cfg=dict(average_clips=None),
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

# ★ 训练 pipeline: NCHW → NCTHW
train_pipeline = [
    dict(type='SampleFrames', clip_len=1, frame_interval=1, num_clips=8),
    dict(type='RawFrameDecode'),
    dict(type='RandomResizedCrop'),
    dict(type='Resize', scale=(224, 224), keep_ratio=False),
    dict(type='Flip', flip_ratio=0.5),
    dict(type='ColorJitter', color_space_aug=True),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='FormatShape', input_format='NCTHW'),  # ★ 5D 输入
    dict(type='Collect', keys=['imgs', 'label'], meta_keys=[]),
    dict(type='ToTensor', keys=['imgs', 'label'])
]

# ★ 验证 pipeline
val_pipeline = [
    dict(type='SampleFrames', clip_len=1, frame_interval=1,
         num_clips=8, test_mode=True),
    dict(type='RawFrameDecode'),
    dict(type='Resize', scale=(-1, 256)),
    dict(type='CenterCrop', crop_size=224),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='FormatShape', input_format='NCTHW'),  # ★
    dict(type='Collect', keys=['imgs', 'label'], meta_keys=[]),
    dict(type='ToTensor', keys=['imgs'])
]

# ★ 测试 pipeline
test_pipeline = [
    dict(type='SampleFrames', clip_len=1, frame_interval=1,
         num_clips=8, test_mode=True),
    dict(type='RawFrameDecode'),
    dict(type='Resize', scale=(-1, 256)),
    dict(type='CenterCrop', crop_size=224),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='FormatShape', input_format='NCTHW'),  # ★
    dict(type='Collect', keys=['imgs', 'label'], meta_keys=[]),
    dict(type='ToTensor', keys=['imgs'])
]

data = dict(
    # ★ 3D 主干 + 全 stage ADSTA + BW-STG 显存大幅增加
    # 从 24 降至 8（如果 8 仍 OOM，可降至 4）
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

# ================= 4. 日志 =================
log_config = dict(
    interval=20,
    hooks=[
        dict(type='TextLoggerHook'),
        dict(type='TensorboardLoggerHook'),
    ]
)

# ================= 5. 优化器与学习率 =================
# batch=8 → lr=0.004（原 2D lowmem 基准）
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
# ★ 必须开启：benchmark=False 时 cuDNN 默认算法在 ADSTA 非常规 conv shape 上
#   触发 illegal memory access（V100 + torch1.11 实测），开启后自动选择正确算法
cudnn_benchmark = True

work_dir = './work_dirs/ams_r50_3d_1x1x8_diving48_rgb_EVL_STG_ADSTA_stage1234_bs8_lr4e-3_v1'
log_level = 'INFO'
dist_params = dict(backend='nccl')
workflow = [('train', 1)]
opencv_num_threads = 0
mp_start_method = 'fork'
load_from = None
resume_from = None
