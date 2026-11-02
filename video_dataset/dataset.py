#!/usr/bin/env python

import os, sys
import random
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
        temporal_flip: bool = False,
        class_index_path: Optional[str] = None,
        label_offset: int = 1,
        num_classes: int = 101,
        train_scale: tuple = (0.08, 1.0),
        track_root: Optional[str] = None,
        tgl_max_tracks: int = 64,
    ):
        self.data_root = data_root
        self.interpolation = interpolation
        self.spatial_size = spatial_size
        self.train_scale = tuple(train_scale)
        self.mirror = bool(mirror)
        # TGL trajectory mode: when track_root is set, the dataset loads the
        # OFFLINE CoTracker3 caches (never runs a tracker) and switches to a
        # deterministic geometric pipeline (exact cached frames + pure resize
        # to the cached image size, no crop/flip) so the track coordinates
        # stay exactly aligned with the frames the ViT sees.
        self.track_root = track_root
        self.tgl_max_tracks = int(tgl_max_tracks)
        # Temporal reverse is OFF by default; when enabled it must flip the
        # frames, the track coordinates' TIME axis, visibility, and the
        # cached frame indices TOGETHER (coordinates are never re-interpolated).
        self.tgl_temporal_reverse = False
        self._track_pack_cache: Dict[str, Dict[str, torch.Tensor]] = {}

        self.mean, self.std = mean, std
        self.num_frames, self.sampling_rate = num_frames, sampling_rate
        self.label_offset = int(label_offset)
        self.num_classes = int(num_classes)

        if random_sample:
            assert num_spatial_views == 1 and num_temporal_views == 1
            self.random_sample = True
            self.mirror = mirror
            self.temporal_flip = temporal_flip
            self.auto_augment = auto_augment
        else:
            assert auto_augment is None and not mirror
            assert not temporal_flip
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


    # ------------------------------------------------------------------ #
    #  TGL trajectory mode (offline caches, deterministic geometry)        #
    # ------------------------------------------------------------------ #
    def _load_track_pack(self, rel_path: str) -> Dict[str, torch.Tensor]:
        """Load (and memoize) the offline track cache of one video.

        Pads/truncates to a fixed ``tgl_max_tracks`` so the default collate
        can stack batches. Padded trajectories have visibility 0 and
        birth_frame == T, i.e. they are masked out everywhere.
        """
        if rel_path in self._track_pack_cache:
            return self._track_pack_cache[rel_path]

        from tgl_tracker_offline import track_cache_path  # lazy: training dir

        cache_file = track_cache_path(self.track_root, rel_path)
        if not os.path.isfile(cache_file):
            raise FileNotFoundError(
                f"TGL track cache not found for {rel_path!r}: {cache_file}. "
                "Run training/run_tgl_pipeline.py --auto-track to generate "
                "the offline trajectories first. Online tracking is "
                "forbidden."
            )
        payload = torch.load(cache_file, map_location="cpu")

        T_cache = int(payload["tracks"].shape[1])
        if T_cache != self.num_frames:
            raise ValueError(
                f"TGL track cache for {rel_path!r} has T={T_cache} but "
                f"training uses num_frames={self.num_frames}. Regenerate the "
                "tracks (run_tgl_pipeline.py --force-regen)."
            )
        frame_indices = payload["frame_indices"].tolist()
        if len(frame_indices) != self.num_frames:
            raise ValueError(
                f"TGL cache frame_indices length {len(frame_indices)} != "
                f"num_frames {self.num_frames} for {rel_path!r}."
            )

        tracks = payload["tracks"].float()
        visibility = payload["visibility"].float()
        birth_frame = payload["birth_frame"].long()
        K = tracks.shape[0]
        K_max = self.tgl_max_tracks
        if K > K_max:
            tracks, visibility, birth_frame = (
                tracks[:K_max], visibility[:K_max], birth_frame[:K_max]
            )
        elif K < K_max:
            pad = K_max - K
            tracks = torch.cat(
                [tracks, torch.zeros(pad, self.num_frames, 2)], dim=0
            )
            visibility = torch.cat(
                [visibility, torch.zeros(pad, self.num_frames)], dim=0
            )
            birth_frame = torch.cat(
                [birth_frame, torch.full((pad,), self.num_frames, dtype=torch.long)]
            )

        pack = {
            "tracks": tracks,               # [K, T, 2]
            "visibility": visibility,       # [K, T]
            "birth_frame": birth_frame,     # [K]
            "frame_indices": torch.tensor(frame_indices, dtype=torch.long),
        }
        self._track_pack_cache[rel_path] = pack
        return pack

    def _getitem_tgl(self, idx):
        rel_path = self.data_list[idx]
        label = self.labels[idx]
        pack = self._load_track_pack(rel_path)
        frame_paths = self._frame_paths(rel_path)

        frame_idx = pack["frame_indices"].tolist()
        if max(frame_idx) >= len(frame_paths) or min(frame_idx) < 0:
            raise IndexError(
                f"TGL cache frame_indices out of range for {rel_path!r}: "
                f"{len(frame_paths)} frames available."
            )
        # Exact cached frames -> pure resize to the tracked image size ->
        # normalize. NO random crop / mirror / temporal flip: they would
        # invalidate the track coordinates.
        frames = torch.stack(
            [self._load_frame(frame_paths[i]) for i in frame_idx]
        )  # [T, 3, H, W]
        if tuple(frames.shape[-2:]) != (self.spatial_size, self.spatial_size):
            frames = torch.nn.functional.interpolate(
                frames,
                size=(self.spatial_size, self.spatial_size),
                mode='bilinear',
                align_corners=False,
            )
        frames = (
            frames - self.mean.view(1, 3, 1, 1)
        ) / self.std.view(1, 3, 1, 1)

        # Horizontal flip with TRACK SYNCHRONIZATION: pixel x' = W - 1 - x
        # (normalized coords would be x' = 1 - x). Visibility/birth are
        # point-wise properties and are unchanged by mirroring; the motion
        # features are recomputed from the flipped tracks downstream, so the
        # geometry stays exactly consistent.
        tracks = pack["tracks"]
        visibility = pack["visibility"]
        birth_frame = pack["birth_frame"]
        if self.mirror and random.random() < 0.5:
            frames = torch.flip(frames, dims=[-1])  # [T, 3, H, W] flip W
            tracks = tracks.clone()
            tracks[..., 0] = (self.spatial_size - 1) - tracks[..., 0]

        # Temporal reverse (config-gated, default OFF): flip frames, the
        # track TIME axis, visibility, and the cached frame indices
        # TOGETHER. Coordinates are never re-interpolated.
        if self.tgl_temporal_reverse:
            frames = torch.flip(frames, dims=[0])  # [T, 3, H, W] flip T
            tracks = torch.flip(tracks, dims=[1])
            visibility = torch.flip(visibility, dims=[1])

        frames = frames.permute(1, 0, 2, 3)  # C, T, H, W

        # track_padding_mask is implicit in `valid` downstream: padded
        # tracks carry visibility == 0 and birth_frame == T, so they are
        # excluded by the visibility/birth mask inside the encoder.
        return (
            frames,
            label,
            tracks,
            visibility,
            birth_frame,
        )

    def __getitem__(self, idx):
        rel_path = self.data_list[idx]
        label = self.labels[idx]
        if self.track_root is not None:
            return self._getitem_tgl(idx)
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

            # Spatial (horizontal) flip: standard action-recognition
            # augmentation. Applied consistently to ALL frames of a clip.
            if self.mirror and random.random() < 0.5:
                frames = frames.flip(-1)  # C, T, H, W -> flip W axis

            # Temporal flip: reverse the frame order with probability 0.5.
            # A cheap, well-known regulariser that improves generalisation
            # for action recognition (motion direction invariance).
            if self.temporal_flip and random.random() < 0.5:
                frames = frames.flip(1)  # C, T, H, W -> flip T axis

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
