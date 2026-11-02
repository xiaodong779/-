# ================= 究极融合架构：AGPN (微观) + AMS (宏观) =================
model = dict(
    type='Recognizer2D',
    # 1. 替换底座：使用我们手工打造的装有 AGPM 模块的 ResNet
    backbone=dict(
        type='ResNet_AGPN',                 # <--- 缝合点 1：替换为你的新底座名字
        pretrained='torchvision://resnet50',
        depth=50,
        num_segments=16,                    # <--- 缝合点 2：AGPM 需要知道时间维度
        out_indices=(2, 3),                 # <--- 极其关键：吐出 Layer3 和 Layer4 给 CSTP
        norm_eval=False),
        
    # 2. 保留颈部：AMS-Net 核心的 CSTP 协同级联时空金字塔
    neck=dict(
        type='CSTP',
        flow_type='parallel',
        in_channels=(1024, 2048),           # 完美接住 Backbone 吐出的 1024 和 2048 通道
        out_channels=1024,
        spatial_modulation_cfg=dict(
            in_channels=(1024, 2048), out_channels=2048),
        # 注意：这里的 downsample_scales 必须和你的帧数 (16) 匹配
        temporal_modulation_cfg=dict(downsample_scales=(16, 16)), 
        upsample_cfg=dict(scale_factor=(1, 1, 1)),
        downsample_cfg=dict(downsample_scale=(1, 1, 1)),
        level_fusion_cfg=dict(
            in_channels=(1024, 1024),
            mid_channels=(1024, 1024),
            out_channels=2048,
            downsample_scales=((1, 1, 1), (1, 1, 1))),
        aux_head_cfg=dict(out_channels=48, loss_weight=0.5)), # Diving48 有 48 类
        
    # 3. 保留头部：AMS-Net 的专属分类头
    cls_head=dict(
        type='AMSHead',
        num_classes=48,                     # <--- 类别数必须改为 48
        in_channels=2048,
        spatial_type='avg',
        consensus=dict(type='AvgConsensus', dim=1),
        dropout_ratio=0.5,
        init_std=0.01),
        
    # 运行配置
    train_cfg=None,
    test_cfg=dict(average_clips='prob', fcn_test=False))