# 继承你训练时的配置
_base_ = ['./ams_r50_diving48_v2_optimized.py']

# =========================================================
# 【核心修改】修复维度报错，开启 TenCrop SOTA 测试
# =========================================================
model = dict(
    test_cfg=dict(
        average_clips='prob',
        fcn_test=True
    )
)

test_pipeline = [
    dict(type='DecordInit'),
    dict(
        type='SampleFrames',
        clip_len=1,
        frame_interval=1,
        # 【必须改回 16】模型是按 16 帧训练的，这里必须是 16，否则报错！
        num_clips=16, 
        twice_sample=False,
        test_mode=True
    ),
    dict(type='DecordDecode'),
    dict(type='Resize', scale=(-1, 256)),
    
    # 【SOTA 秘技】使用 TenCrop (10重裁剪) 代替 ThreeCrop
    # 这会生成 10 个视图，大幅提高准确率，且不会破坏时间维度
    dict(type='TenCrop', crop_size=224), 
    
    dict(type='Normalize', 
         mean=[123.675, 116.28, 103.53], 
         std=[58.395, 57.12, 57.375], 
         to_bgr=False),
    dict(type='FormatShape', input_format='NCHW'),
    dict(type='Collect', keys=['imgs', 'label'], meta_keys=[]),
    dict(type='ToTensor', keys=['imgs'])
]

# 强制覆盖原本的 test_pipeline
data = dict(
    test=dict(pipeline=test_pipeline),
    test_dataloader=dict(videos_per_gpu=1) # TenCrop 显存占用大，必须设为 1
)