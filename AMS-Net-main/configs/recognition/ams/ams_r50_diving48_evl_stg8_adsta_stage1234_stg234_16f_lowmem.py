model = dict(
    type='Recognizer2D',
    backbone=dict(
        type='ResNetAMS_Pro',
        pretrained='torchvision://resnet50',
        depth=50,
        out_indices=(3, ),
        norm_eval=False,
        num_segments=16,
        torchvision_pretrain=True,
        use_adsta=True,
        adsta_stages=(1, 2, 3, 4),
        adsta_reduction=4,
        adsta_sample_mode='density_244',
        adsta_mamba_expand=1,
        adsta_mamba_d_conv=3,
        adsta_dropout=0.0,
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
            return_layer=-1),
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
            qkv_bias=True)),
    neck=None,
    cls_head=dict(
        type='AMSHead',
        num_classes=48,
        in_channels=2048,
        spatial_type='avg',
        consensus=dict(type='AvgConsensus', dim=1),
        dropout_ratio=0.5,
        init_std=0.01,
        loss_cls=dict(
            type='LabelSmoothCrossEntropyLoss', label_smooth_eps=0.1)),
    train_cfg=None,
    test_cfg=dict(average_clips='prob', fcn_test=True))
checkpoint_config = dict(interval=5, max_keep_ckpts=5)
log_config = dict(
    interval=20,
    hooks=[dict(type='TextLoggerHook'),
           dict(type='TensorboardLoggerHook')])
dist_params = dict(backend='nccl')
log_level = 'INFO'
load_from = None
resume_from = None
workflow = [('train', 1)]
dataset_type = 'RawframeDataset'
data_root = '/hy-tmp/Diving48_rawframes'
data_root_val = '/hy-tmp/Diving48_rawframes'
ann_file_train = '/hy-tmp/AMS-Net-main/data/diving48/annotations/diving48_train_rawframes.txt'
ann_file_val = '/hy-tmp/AMS-Net-main/data/diving48/annotations/diving48_val_rawframes.txt'
ann_file_test = '/hy-tmp/AMS-Net-main/data/diving48/annotations/diving48_val_rawframes.txt'
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_bgr=False)
train_pipeline = [
    dict(type='SampleFrames', clip_len=1, frame_interval=1, num_clips=8),
    dict(type='RawFrameDecode'),
    dict(type='RandomResizedCrop'),
    dict(type='Resize', scale=(224, 224), keep_ratio=False),
    dict(type='Flip', flip_ratio=0.5),
    dict(type='ColorJitter', color_space_aug=True),
    dict(
        type='Normalize',
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        to_bgr=False),
    dict(type='FormatShape', input_format='NCHW'),
    dict(type='Collect', keys=['imgs', 'label'], meta_keys=[]),
    dict(type='ToTensor', keys=['imgs', 'label'])
]
val_pipeline = [
    dict(
        type='SampleFrames',
        clip_len=1,
        frame_interval=1,
        num_clips=8,
        test_mode=True),
    dict(type='RawFrameDecode'),
    dict(type='Resize', scale=(-1, 256)),
    dict(type='CenterCrop', crop_size=224),
    dict(
        type='Normalize',
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        to_bgr=False),
    dict(type='FormatShape', input_format='NCHW'),
    dict(type='Collect', keys=['imgs', 'label'], meta_keys=[]),
    dict(type='ToTensor', keys=['imgs'])
]
test_pipeline = [
    dict(
        type='SampleFrames',
        clip_len=1,
        frame_interval=1,
        num_clips=8,
        test_mode=True),
    dict(type='RawFrameDecode'),
    dict(type='Resize', scale=(-1, 256)),
    dict(type='CenterCrop', crop_size=224),
    dict(
        type='Normalize',
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        to_bgr=False),
    dict(type='FormatShape', input_format='NCHW'),
    dict(type='Collect', keys=['imgs', 'label'], meta_keys=[]),
    dict(type='ToTensor', keys=['imgs'])
]
data = dict(
    videos_per_gpu=8,
    workers_per_gpu=8,
    train_dataloader=dict(drop_last=True),
    val_dataloader=dict(videos_per_gpu=4, workers_per_gpu=4),
    test_dataloader=dict(videos_per_gpu=4, workers_per_gpu=4),
    train=dict(
        type='RawframeDataset',
        ann_file=
        '/hy-tmp/AMS-Net-main/data/diving48/annotations/diving48_train_rawframes.txt',
        data_prefix='/hy-tmp/Diving48_rawframes',
        filename_tmpl='img_{:05}.jpg',
        pipeline=[
            dict(
                type='SampleFrames',
                clip_len=1,
                frame_interval=1,
                num_clips=16),
            dict(type='RawFrameDecode'),
            dict(type='RandomResizedCrop'),
            dict(type='Resize', scale=(224, 224), keep_ratio=False),
            dict(type='Flip', flip_ratio=0.5),
            dict(type='ColorJitter', color_space_aug=True),
            dict(
                type='Normalize',
                mean=[123.675, 116.28, 103.53],
                std=[58.395, 57.12, 57.375],
                to_bgr=False),
            dict(type='FormatShape', input_format='NCHW'),
            dict(type='Collect', keys=['imgs', 'label'], meta_keys=[]),
            dict(type='ToTensor', keys=['imgs', 'label'])
        ]),
    val=dict(
        type='RawframeDataset',
        ann_file=
        '/hy-tmp/AMS-Net-main/data/diving48/annotations/diving48_val_rawframes.txt',
        data_prefix='/hy-tmp/Diving48_rawframes',
        filename_tmpl='img_{:05}.jpg',
        pipeline=[
            dict(
                type='SampleFrames',
                clip_len=1,
                frame_interval=1,
                num_clips=16,
                test_mode=True),
            dict(type='RawFrameDecode'),
            dict(type='Resize', scale=(-1, 256)),
            dict(type='CenterCrop', crop_size=224),
            dict(
                type='Normalize',
                mean=[123.675, 116.28, 103.53],
                std=[58.395, 57.12, 57.375],
                to_bgr=False),
            dict(type='FormatShape', input_format='NCHW'),
            dict(type='Collect', keys=['imgs', 'label'], meta_keys=[]),
            dict(type='ToTensor', keys=['imgs'])
        ]),
    test=dict(
        type='RawframeDataset',
        ann_file=
        '/hy-tmp/AMS-Net-main/data/diving48/annotations/diving48_val_rawframes.txt',
        data_prefix='/hy-tmp/Diving48_rawframes',
        filename_tmpl='img_{:05}.jpg',
        pipeline=[
            dict(
                type='SampleFrames',
                clip_len=1,
                frame_interval=1,
                num_clips=16,
                test_mode=True),
            dict(type='RawFrameDecode'),
            dict(type='Resize', scale=(-1, 256)),
            dict(type='CenterCrop', crop_size=224),
            dict(
                type='Normalize',
                mean=[123.675, 116.28, 103.53],
                std=[58.395, 57.12, 57.375],
                to_bgr=False),
            dict(type='FormatShape', input_format='NCHW'),
            dict(type='Collect', keys=['imgs', 'label'], meta_keys=[]),
            dict(type='ToTensor', keys=['imgs'])
        ]))
evaluation = dict(
    interval=1,
    metrics=['top_k_accuracy', 'mean_class_accuracy'],
    topk=(1, 5),
    save_best='top1_acc',
    rule='greater')
optimizer = dict(
    type='SGD', lr=0.004, momentum=0.9, weight_decay=0.0005, nesterov=True)
optimizer_config = dict(grad_clip=dict(max_norm=20, norm_type=2))
lr_config = dict(policy='step', step=[45, 55])
total_epochs = 75
work_dir = './work_dirs/ams_r50_1x1x16_diving48_rgb_EVL_STG_ADSTA_stage1234_stg234_bs8_lr4e-3_32g_v1'
opencv_num_threads = 0
mp_start_method = 'fork'
