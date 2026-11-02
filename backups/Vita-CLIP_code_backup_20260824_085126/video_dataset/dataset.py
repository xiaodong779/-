#!/usr/bin/env python

import os, sys
from typing import Dict, List, Optional
from PIL import Image
import numpy as np

import torch
from torchvision import transforms

from .transform import create_random_augment, random_resized_crop

class VideoDataset(torch.utils.data.Dataset):

    def __init__(
        self, list_path: str, data_root: str,
        num_spatial_views: int, num_temporal_views: int, random_sample: bool,
        num_frames: int, sampling_rate: int, spatial_size: int,
        mean: torch.Tensor, std: torch.Tensor,
        auto_augment: Optional[str] = None, interpolation: str = 'bicubic',
        mirror: bool = False,
        class_index_path: Optional[str] = None,
        label_offset: int = 1,
        num_classes: int = 101,
        train_scale: tuple = (0.08, 1.0),
    ):
        self.data_root = data_root
        self.interpolation = interpolation
        self.spatial_size = spatial_size
        self.train_scale = tuple(train_scale)

        self.mean, self.std = mean, std
        self.num_frames, self.sampling_rate = num_frames, sampling_rate
        self.label_offset = int(label_offset)
        self.num_classes = int(num_classes)

        if random_sample:
            assert num_spatial_views == 1 and num_temporal_views == 1
            self.random_sample = True
            self.mirror = mirror
            self.auto_augment = auto_augment
        else:
            assert auto_augment is None and not mirror
            self.random_sample = False
            self.num_temporal_views = num_temporal_views
            self.num_spatial_views = num_spatial_views

        # class name -> official numeric label (e.g. 1..101)
        self.class_to_label: Dict[str, int] = {}
        if class_index_path and os.path.isfile(class_index_path):
            with open(class_index_path) as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 2 and parts[0].isdigit():
                        self.class_to_label[parts[1]] = int(parts[0])

        self.data_list: List[str] = []
        self.labels: List[int] = []
        with open(list_path) as f:
            for raw_line in f:
                line = raw_line.strip()
                if not line:
                    continue
                parts = line.split()
                rel_path = parts[0]
                if len(parts) >= 2 and parts[1].lstrip('-').isdigit():
                    label = int(parts[1]) - self.label_offset
                else:
                    class_name = rel_path.split('/')[0]
                    label = self.class_to_label.get(class_name)
                    if label is None:
                        raise ValueError(
                            f"Unable to resolve a label for {rel_path!r} in "
                            f"{list_path}; provide a class_index_path."
                        )
                    label = label - self.label_offset
                self.data_list.append(rel_path)
                self.labels.append(label)


    def __len__(self):
        return len(self.data_list)


    def _frame_paths(self, rel_path: str) -> List[str]:
        # UCF101 raw-frame directories have no ``.avi`` suffix even though
        # ``testlist01.txt`` keeps the original video name.
        if rel_path.lower().endswith(".avi"):
            rel_path = rel_path[:-4]
        video_dir = os.path.join(self.data_root, rel_path)
        if not os.path.isdir(video_dir):
            raise FileNotFoundError(
                f"Raw-frame directory not found: {video_dir}"
            )
        frame_names = sorted(
            n for n in os.listdir(video_dir)
            if n.lower().endswith(('.jpg', '.jpeg', '.png'))
        )
        if not frame_names:
            raise FileNotFoundError(
                f"No image frames found in {video_dir}"
            )
        return [os.path.join(video_dir, n) for n in frame_names]


    def _load_frame(self, frame_path: str) -> torch.Tensor:
        with open(frame_path, 'rb') as f:
            img = Image.open(f).convert('RGB')
        return transforms.ToTensor()(img)  # [3, H, W]


    def __getitem__(self, idx):
        rel_path = self.data_list[idx]
        label = self.labels[idx]
        frame_paths = self._frame_paths(rel_path)

        if self.random_sample:
            frame_idx = self._random_sample_frame_idx(len(frame_paths))
            frames = torch.stack(
                [self._load_frame(frame_paths[x]) for x in frame_idx]
            )  # [T, 3, H, W]

            if self.auto_augment is not None:
                aug_transform = create_random_augment(
                    input_size=(frames.size(2), frames.size(3)),
                    auto_augment=self.auto_augment,
                    interpolation=self.interpolation,
                )
                frames = [
                    transforms.ToPILImage()(frames[i])
                    for i in range(frames.size(0))
                ]
                frames = aug_transform(frames)
                frames = torch.stack(
                    [transforms.ToTensor()(img) for img in frames]
                )  # [T, 3, H, W]

            frames = (
                frames - self.mean.view(1, 3, 1, 1)
            ) / self.std.view(1, 3, 1, 1)
            frames = frames.permute(1, 0, 2, 3)  # C, T, H, W
            frames = random_resized_crop(
                frames, self.spatial_size, self.spatial_size,
                scale=self.train_scale,
            )

        else:
            frames = torch.stack(
                [self._load_frame(p) for p in frame_paths]
            )  # [T, 3, H, W]
            frames = (
                frames - self.mean.view(1, 3, 1, 1)
            ) / self.std.view(1, 3, 1, 1)
            frames = frames.permute(1, 0, 2, 3)  # C, T, H, W

            if frames.size(-2) < frames.size(-1):
                new_width = frames.size(-1) * self.spatial_size // frames.size(-2)
                new_height = self.spatial_size
            else:
                new_height = frames.size(-2) * self.spatial_size // frames.size(-1)
                new_width = self.spatial_size
            frames = torch.nn.functional.interpolate(
                frames, size=(new_height, new_width),
                mode='bilinear', align_corners=False,
            )

            frames = self._generate_spatial_crops(frames)
            frames = sum([self._generate_temporal_crops(x) for x in frames], [])
            if len(frames) > 1:
                frames = torch.stack(frames)

        return frames, label


    def _generate_temporal_crops(self, frames):
        # frames: [C, T, H, W]
        if self.sampling_rate > 0:
            seg_len = (self.num_frames - 1) * self.sampling_rate + 1
            if frames.size(1) < seg_len:
                frames = torch.cat([frames, frames[:, -1:].repeat(1, seg_len - frames.size(1), 1, 1)], dim=1)
            slide_len = frames.size(1) - seg_len

            crops = []
            for i in range(self.num_temporal_views):
                if self.num_temporal_views == 1:
                    st = slide_len // 2
                else:
                    st = round(slide_len / (self.num_temporal_views - 1) * i)

                crops.append(frames[:, st: st + self.num_frames * self.sampling_rate: self.sampling_rate])
            return crops

        # TSN mode (sampling_rate < 0): deterministic, evenly spaced frames.
        total = frames.size(1)
        if total <= 0:
            raise RuntimeError("Cannot temporally sample an empty video.")
        crops = []
        for i in range(self.num_temporal_views):
            if self.num_temporal_views == 1:
                # Single view: uniformly cover the whole video (historical behavior).
                start = 0
                window = total
            else:
                # Multi-view: split the video into num_temporal_views segments and
                # sample each segment independently. This guarantees every view
                # contains motion, unlike linear offsets where the last view
                # degenerates to a single static frame (window=1).
                seg = total / self.num_temporal_views
                start = round(seg * i)
                end = round(seg * (i + 1))
                window = max(end - start, 1)
            indices = torch.linspace(
                start,
                start + window - 1,
                self.num_frames,
            ).long()
            crops.append(frames[:, indices])
        return crops


    def _generate_spatial_crops(self, frames):
        if self.num_spatial_views == 1:
            assert min(frames.size(-2), frames.size(-1)) >= self.spatial_size
            h_st = (frames.size(-2) - self.spatial_size) // 2
            w_st = (frames.size(-1) - self.spatial_size) // 2
            h_ed, w_ed = h_st + self.spatial_size, w_st + self.spatial_size
            return [frames[:, :, h_st: h_ed, w_st: w_ed]]

        elif self.num_spatial_views == 3:
            assert min(frames.size(-2), frames.size(-1)) == self.spatial_size
            crops = []
            margin = max(frames.size(-2), frames.size(-1)) - self.spatial_size
            for st in (0, margin // 2, margin):
                ed = st + self.spatial_size
                if frames.size(-2) > frames.size(-1):
                    crops.append(frames[:, :, st: ed, :])
                else:
                    crops.append(frames[:, :, :, st: ed])
            return crops

        else:
            raise NotImplementedError()


    def _random_sample_frame_idx(self, len):
        frame_indices = []

        if self.sampling_rate < 0: # tsn sample
            seg_size = (len - 1) / self.num_frames
            for i in range(self.num_frames):
                start, end = round(seg_size * i), round(seg_size * (i + 1))
                frame_indices.append(np.random.randint(start, end + 1))
        elif self.sampling_rate * (self.num_frames - 1) + 1 >= len:
            for i in range(self.num_frames):
                frame_indices.append(i * self.sampling_rate if i * self.sampling_rate < len else frame_indices[-1])
        else:
            start = np.random.randint(len - self.sampling_rate * (self.num_frames - 1))
            frame_indices = list(range(start, start + self.sampling_rate * self.num_frames, self.sampling_rate))

        return frame_indices


class DummyDataset(torch.utils.data.Dataset):

    def __init__(self, list_path: str, num_frames: int, num_views: int, spatial_size: int):
        with open(list_path) as f:
            self.len = len(f.read().splitlines())
        self.num_frames = num_frames
        self.num_views = num_views
        self.spatial_size = spatial_size

    def __len__(self):
        return self.len

    def __getitem__(self, _):
        shape = [3, self.num_frames, self.spatial_size, self.spatial_size]
        if self.num_views != 1:
            shape = [self.num_views] + shape
        return torch.zeros(shape), 0
