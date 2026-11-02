#!/usr/bin/env python

import argparse
import torch
import torch.nn as nn

from .weight_loaders import weight_loader_fn_dict
from .vision_transformer import (
    VisionTransformer2D,
    model_to_fp16,
    vit_presets,
)


class FrozenEVLViTBranch(nn.Module):
    """
    EVL 现成的 Frozen CLIP ViT-B/16 空间分支。

    输入:
        x: [B, 3, T, 224, 224]

    输出:
        patch_map: [B, 768, T, 14, 14]
    """

    def __init__(
        self,
        backbone_name='ViT-B/16-lnpre',
        backbone_type='clip',
        backbone_path='',
        backbone_mode='freeze_fp32',
        return_layer=-1,
    ):
        super().__init__()

        self.backbone_name = backbone_name
        self.backbone_type = backbone_type
        self.backbone_path = backbone_path
        self.backbone_mode = backbone_mode
        self.return_layer = return_layer

        backbone_config = self._create_backbone()

        self.feature_dim = backbone_config['feature_dim']
        self.input_size = backbone_config['input_size']
        self.patch_size = backbone_config['patch_size']
        self.spatial_size = tuple(
            x // y for x, y in zip(self.input_size, self.patch_size)
        )

        print('FrozenEVLViTBranch config:')
        print('  backbone_name:', self.backbone_name)
        print('  feature_dim:', self.feature_dim)
        print('  input_size:', self.input_size)
        print('  patch_size:', self.patch_size)
        print('  spatial_size:', self.spatial_size)

    def _create_backbone(self):
        assert self.backbone_name in vit_presets, \
            f'{self.backbone_name} not in vit_presets: {vit_presets.keys()}'

        assert self.backbone_mode in ['finetune', 'freeze_fp16', 'freeze_fp32']

        weight_loader_fn = weight_loader_fn_dict[self.backbone_type]
        state_dict = weight_loader_fn(self.backbone_path)

        backbone = VisionTransformer2D(
            return_all_features=True,
            **vit_presets[self.backbone_name]
        )

        backbone.load_state_dict(state_dict, strict=True)

        if self.backbone_mode == 'finetune':
            self.backbone = backbone
        else:
            backbone.eval()
            backbone.requires_grad_(False)

            if self.backbone_mode == 'freeze_fp16':
                model_to_fp16(backbone)

            # 和 EVL 原代码一致：放入 list，避免作为可训练参数注册
            self.backbone = [backbone]

        return vit_presets[self.backbone_name]

    def _get_backbone(self, x):
        if isinstance(self.backbone, list):
            self.backbone[0] = self.backbone[0].to(x.device)
            self.backbone[0].eval()
            return self.backbone[0]
        return self.backbone

    def train(self, mode=True):
        super().train(mode)
        if isinstance(self.backbone, list):
            self.backbone[0].eval()
        return self

    def forward(self, x: torch.Tensor):
        """
        x: [B, 3, T, 224, 224]
        """
        assert x.dim() == 5, f'Expected [B,3,T,H,W], got {x.shape}'

        backbone = self._get_backbone(x)

        B, C, T, H, W = x.shape
        assert C == 3
        assert (H, W) == tuple(self.input_size), \
            f'Expected input size {self.input_size}, got {(H, W)}'

        # [B, 3, T, H, W] -> [B*T, 3, H, W]
        frames = x.permute(0, 2, 1, 3, 4).flatten(0, 1).contiguous()

        # 如果 frozen backbone 是 fp16，输入也转 fp16
        param_dtype = next(backbone.parameters()).dtype
        frames = frames.to(dtype=param_dtype)

        if isinstance(self.backbone, list):
            with torch.no_grad():
                features = backbone(frames)
        else:
            features = backbone(frames)

        # 取最后一层 ViT token
        feat = features[self.return_layer]['out']  # [B*T, L, C]

        # [B*T, L, C] -> [B, T, L, C]
        feat = feat.float().view(B, T, *feat.shape[1:])

        # 去掉 cls token，只保留 patch token
        patch = feat[:, :, 1:, :]  # [B, T, 196, 768]

        Hs, Ws = self.spatial_size
        assert patch.size(2) == Hs * Ws, \
            f'Patch token mismatch: {patch.size(2)} vs {Hs}*{Ws}'

        # [B, T, 196, 768] -> [B, T, 14, 14, 768]
        patch = patch.view(B, T, Hs, Ws, self.feature_dim)

        # [B, T, H, W, C] -> [B, C, T, H, W]
        patch_map = patch.permute(0, 4, 1, 2, 3).contiguous()

        return patch_map


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--backbone_path', type=str, default='/hy-tmp/weights/ViT-B-16.pt')
    parser.add_argument('--num_frames', type=int, default=8)
    parser.add_argument(
        '--backbone_mode',
        type=str,
        default='freeze_fp32',
        choices=['freeze_fp32', 'freeze_fp16', 'finetune']
    )
    args = parser.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    model = FrozenEVLViTBranch(
        backbone_name='ViT-B/16-lnpre',
        backbone_type='clip',
        backbone_path=args.backbone_path,
        backbone_mode=args.backbone_mode,
        return_layer=-1,
    ).to(device)

    model.train()

    x = torch.randn(1, 3, args.num_frames, 224, 224).to(device)
    y = model(x)

    print('input shape:', tuple(x.shape))
    print('output shape:', tuple(y.shape))
    print('output requires_grad:', y.requires_grad)

    registered_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print('registered params:', registered_params)
    print('trainable params:', trainable_params)

    if isinstance(model.backbone, list):
        vit_total = sum(p.numel() for p in model.backbone[0].parameters())
        vit_trainable = sum(p.numel() for p in model.backbone[0].parameters() if p.requires_grad)
        print('frozen ViT total params:', vit_total)
        print('frozen ViT trainable params:', vit_trainable)


if __name__ == '__main__':
    main()
