import os
import sys
import torch
# Ensure repo root is on PYTHONPATH
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
# Create a minimal fake `mmcv` package to allow importing the project modules
import types
if 'mmcv' not in sys.modules:
    mmcv = types.ModuleType('mmcv')
    sys.modules['mmcv'] = mmcv
    # mmcv.cnn submodule
    mmcv.cnn = types.ModuleType('mmcv.cnn')
    sys.modules['mmcv.cnn'] = mmcv.cnn
    def kaiming_init(m):
        try:
            torch.nn.init.kaiming_normal_(m.weight)
        except Exception:
            pass
    def constant_init(tensor, val):
        try:
            torch.nn.init.constant_(tensor, val)
        except Exception:
            pass
    def normal_init(tensor, mean=0, std=0.01):
        try:
            torch.nn.init.normal_(tensor, mean=mean, std=std)
        except Exception:
            pass
    mmcv.cnn.kaiming_init = kaiming_init
    mmcv.cnn.constant_init = constant_init
    mmcv.cnn.normal_init = normal_init
    # Minimal ConvModule implementation
    import torch.nn as nn
    class ConvModule(nn.Module):
        def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, bias=False, conv_cfg=None, norm_cfg=None, act_cfg=None):
            super().__init__()
            self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, dilation=dilation, bias=bias)
            self.bn = nn.BatchNorm2d(out_channels)
            self.relu = nn.ReLU(inplace=True) if act_cfg is not None else None
            modules = [self.conv, self.bn]
            if self.relu is not None:
                modules.append(self.relu)
            self.module = nn.Sequential(*modules)
        def forward(self, x):
            return self.module(x)
    mmcv.cnn.ConvModule = ConvModule
    # runner and utils placeholders
    mmcv.runner = types.ModuleType('mmcv.runner')
    sys.modules['mmcv.runner'] = mmcv.runner
    def _load_checkpoint(path):
        raise RuntimeError('mmcv._load_checkpoint placeholder')
    def load_checkpoint(*args, **kwargs):
        raise RuntimeError('mmcv.load_checkpoint placeholder')
    mmcv.runner._load_checkpoint = _load_checkpoint
    mmcv.runner.load_checkpoint = load_checkpoint
    mmcv.utils = types.ModuleType('mmcv.utils')
    sys.modules['mmcv.utils'] = mmcv.utils
    mmcv.utils._BatchNorm = torch.nn.modules.batchnorm._BatchNorm
    # minimal helper to satisfy imports
    def collect_env():
        return {}
    mmcv.utils.collect_env = collect_env
    # minimal version helpers expected by mmaction.__init__
    def digit_version(v):
        # simple numeric parse
        return tuple(int(x) for x in str(v).split('.') if x.isdigit())
    mmcv.digit_version = digit_version
    mmcv.__version__ = '0.0.0'

from mmaction.models.backbones.resnet_st import ResNet_ST

def run_smoke():
    # Instantiate backbone without pretrained weights to avoid downloads
    model = ResNet_ST(depth=50, pretrained=None, num_segments=16)
    model.train()
    # Create dummy input: N=2, T=16 -> NT=32
    N = 2
    T = 16
    x = torch.randn(N * T, 3, 224, 224, requires_grad=True)
    out = model(x)
    print('Output shape:', out.shape if hasattr(out, 'shape') else type(out))
    loss = out.mean()
    loss.backward()
    print('Smoke test forward+backward OK')

if __name__ == '__main__':
    run_smoke()
