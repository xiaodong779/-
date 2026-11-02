import torch
from torch import nn

from ..registry import RECOGNIZERS
from .base import BaseRecognizer


@RECOGNIZERS.register_module()
class Recognizer3D(BaseRecognizer):
    """3D recognizer model framework."""

    def forward_train(self, imgs, labels, **kwargs):
        """Defines the computation performed at every call when training.

        输入 imgs 为 6D [B, num_clips, C, T, H, W]（clip_len=8, num_clips=N）。
        reshape 为 5D [B*num_clips, C, T, H, W] 后传给 3D 主干。
        AMSHead 用 num_segs=None → 内部 AdaptiveAvgPool3d 对 T,H,W 平均池化，
        输出 [B*num_clips, num_classes]，每 clip 一个分数（时序已在 head 内池化）。
        """
        num_segs = imgs.shape[1]  # num_clips
        imgs = imgs.reshape((-1, ) + imgs.shape[2:])  # [B*num_clips, C, T, H, W]
        losses = dict()

        x = self.extract_feat(imgs)  # 主干输出 [B*num_clips, C, T, H, W]
        if self.with_neck:
            x, loss_aux = self.neck(x, labels.squeeze())
            losses.update(loss_aux)

        cls_score = self.cls_head(x, num_segs=None)
        gt_labels = labels.squeeze()
        loss_cls = self.cls_head.loss(cls_score, gt_labels, **kwargs)
        losses.update(loss_cls)

        return losses

    def _do_test(self, imgs):
        """Defines the computation performed at every call when evaluation,
        testing and gradcam."""
        num_segs = imgs.shape[1]  # num_clips（测试时 >1，用于多 clip 平均）
        imgs = imgs.reshape((-1, ) + imgs.shape[2:])  # [B*num_clips, C, T, H, W]

        if self.max_testing_views is not None:
            total_views = imgs.shape[0]
            assert num_segs == total_views, (
                'max_testing_views is only compatible '
                'with batch_size == 1')
            view_ptr = 0
            cls_scores = []
            while view_ptr < total_views:
                batch_imgs = imgs[view_ptr:view_ptr + self.max_testing_views]
                x = self.extract_feat(batch_imgs)
                if self.with_neck:
                    x, _ = self.neck(x)
                cls_score = self.cls_head(x, num_segs=None)
                cls_scores.append(cls_score)
                view_ptr += self.max_testing_views
            cls_score = torch.cat(cls_scores)
        else:
            x = self.extract_feat(imgs)
            if self.with_neck:
                x, _ = self.neck(x)
            cls_score = self.cls_head(x, num_segs=None)

        cls_score = self.average_clip(cls_score, num_segs)
        return cls_score

    def forward_test(self, imgs):
        """Defines the computation performed at every call when evaluation and
        testing."""
        return self._do_test(imgs).cpu().numpy()

    def forward_dummy(self, imgs, softmax=False):
        """Used for computing network FLOPs.

        See ``tools/analysis/get_flops.py``.

        Args:
            imgs (torch.Tensor): Input images.

        Returns:
            Tensor: Class score.
        """
        imgs = imgs.reshape((-1, ) + imgs.shape[2:])
        x = self.extract_feat(imgs)

        if self.with_neck:
            x, _ = self.neck(x)

        outs = self.cls_head(x)
        if softmax:
            outs = nn.functional.softmax(outs)
        return (outs, )

    def forward_gradcam(self, imgs):
        """Defines the computation performed at every call when using gradcam
        utils."""
        return self._do_test(imgs)
