import torch.nn as nn
from ..builder import BACKBONES
from .resnet import ResNet, Bottleneck 
from ..common import TCM, MGFM 



# ==========================================================
# 微观：块内融合模块
# ==========================================================
class MGFM_Bottleneck(Bottleneck):
    def __init__(self, inplanes, planes, *args, **kwargs):
        self.with_mgfm = kwargs.pop('with_mgfm', False)
        self.num_segments = kwargs.pop('num_segments', 16)
        
        super(MGFM_Bottleneck, self).__init__(inplanes, planes, *args, **kwargs)

        if self.with_mgfm:
            self.mgfm_block = MGFM(in_channels=planes, num_segments=self.num_segments)

    def forward(self, x):
        def _inner_forward(x):
            identity = x
            
            out = self.conv1(x)
            if hasattr(self, 'norm1'):
                out = self.norm1(out)
            if hasattr(self, 'relu'):
                out = self.relu(out)

            if self.with_mgfm and hasattr(self, 'mgfm_block'):
                out = self.mgfm_block(out)

            out = self.conv2(out)
            if hasattr(self, 'norm2'):
                out = self.norm2(out)
            if hasattr(self, 'relu'):
                out = self.relu(out)
            
            out = self.conv3(out)
            if hasattr(self, 'norm3'):
                out = self.norm3(out)

            if self.downsample is not None:
                identity = self.downsample(x)

            out += identity
            return out

        out = _inner_forward(x)
        if hasattr(self, 'relu'):
            out = self.relu(out)
        return out

# ==========================================================
# 宏观：架构构建
# ==========================================================
@BACKBONES.register_module(force=True)
class ResNet_TCM_MGFM(ResNet):
    def __init__(self, num_segments=16, mgfm_stages=(False, True, True, True), **kwargs):
        self.stem_channels = 64
        
        # 💥 架构师终极修复：把 num_segments 存进 self 里面，让 Recognizer2D 能读到它！
        self.num_segments = num_segments
        
        super(ResNet_TCM_MGFM, self).__init__(**kwargs)
        
        def get_camouflaged_block(with_mgfm_flag):
            class CamouflagedBlock(MGFM_Bottleneck):
                expansion = 4
                def __init__(self, in_planes, planes, *args, **b_kwargs):
                    b_kwargs['with_mgfm'] = with_mgfm_flag
                    b_kwargs['num_segments'] = num_segments
                    super(CamouflagedBlock, self).__init__(in_planes, planes, *args, **b_kwargs)
            return CamouflagedBlock

        if hasattr(self, '_make_res_layer'):
            build_layer = self._make_res_layer
        elif hasattr(self, 'make_res_layer'):
            build_layer = self.make_res_layer
        else:
            from .resnet import make_res_layer
            build_layer = make_res_layer

        layer_kwargs = {}
        for attr in ['style', 'with_cp', 'conv_cfg', 'norm_cfg', 'dcn', 'plugins']:
            if hasattr(self, attr):
                layer_kwargs[attr] = getattr(self, attr)

        self.layer1 = build_layer(get_camouflaged_block(mgfm_stages[0]), 
                                  64, 64, self.stage_blocks[0], stride=1, **layer_kwargs)
        
        self.layer2 = build_layer(get_camouflaged_block(mgfm_stages[1]), 
                                  256, 128, self.stage_blocks[1], stride=2, **layer_kwargs)
        
        self.tcm = TCM(num_segments=num_segments, expansion=4, pos=2)

        self.layer3 = build_layer(get_camouflaged_block(mgfm_stages[2]), 
                                  512, 256, self.stage_blocks[2], stride=2, **layer_kwargs)
        
        self.layer4 = build_layer(get_camouflaged_block(mgfm_stages[3]), 
                                  1024, 512, self.stage_blocks[3], stride=2, **layer_kwargs)

    def forward(self, x):
        if getattr(self, 'deep_stem', False) and hasattr(self, 'stem'):
            x = self.stem(x)
        else:
            x = self.conv1(x)
            if hasattr(self, 'norm1'):
                x = self.norm1(x)
            if hasattr(self, 'relu'):
                x = self.relu(x)
            if hasattr(self, 'maxpool'):
                x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        
        if hasattr(self, 'tcm'):
            x = self.tcm(x)
        
        x = self.layer3(x)
        x = self.layer4(x)
        return x