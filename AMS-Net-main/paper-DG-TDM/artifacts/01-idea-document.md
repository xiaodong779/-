# DG-TDM Idea Document

## Title
DG-TDM: Dual-Granularity Temporal Dynamics Modeling with Spatial Prior Guidance for Fine-Grained Action Recognition

## Target venue
NeurIPS 2026 (CCF-A), main track, double-blind. 9 content pages.

## Storyline
- **Scientific origin**: Fine-grained action recognition (e.g., Diving48) has classes with similar backgrounds that differ in subtle body movements.
- **Knowledge gap**: Existing video models either inflate to 3D (full spatial cost), use deformable sampling without semantic guidance, or use frame-wise CLIP without cross-frame motion.
- **Core insight**: Spatial localization and motion modeling are complementary and can be composed — a frozen CLIP prior tells the video branch where to look, and a dual-granularity temporal module encodes how things move.
- **Method**: Three branches.
  - Upper: frozen CLIP ViT-B/16 (spatial prior, no gradients).
  - Middle: STG block (bilateral window cross-attention + adaptive gating) injects prior into lower branch.
  - Lower: temporal action encoder (ResNet-50) with DG-TDM at each stage — MoDeA (deformable sampling for local motion) + TSA (bidirectional state scan for channel-to-temporal variation).
- **Evidence**: Diving48 main results + ablation of each branch. Final numbers TBD pending training completion.

## Key references (updated 2022-2025)
- MoDeA mechanism grounded in DVT (Deformable Video Transformer, CVPR 2022), differentiated as dense 3D offset field + continuous trilinear sampling vs patch-level selection.
- TSA mechanism grounded in VideoMamba (ECCV 2024) and Mamba (2023), differentiated as channel-wise bidirectional linear recurrence used as a temporal gate.
- Fine-grained action recognition grounded in Diving48 (2018); CLIP (2021) and EVL (2022) for the frozen spatial prior.

## Confirmed module identities (from code)
- MoDeA = ResNet_DeformAttention in `mmaction/models/common/adsta.py`
- TSA = SimpleMambaGate (bidirectional state scan) in `mmaction/models/common/adsta.py`
- STG = BWSTGBlock in `mmaction/models/common/bw_stg.py`
- Spatial prior = FrozenEVLViTBranch (ViT-B/16, freeze_fp32) in `mmaction/models/common/evl/frozen_evl_vit_branch.py`
- Backbone = ResNetAMS_Pro in `mmaction/models/backbones/resnet_ams_pro.py`
- Head = AMSHead (label-smoothed CE, eps=0.1) in `mmaction/models/heads/ams_head.py`

## Hyperparameters (from config)
- T = 8 frames; reduction r = 4; STG window = 7; STG stages 2/3/4; DG-TDM stages 1/2/3/4
- SGD lr 8e-3, momentum 0.9, wd 5e-4; step [45,55]; 75 epochs; batch 16
- Mamba expand = 1, d_conv = 3; heads/groups per stage = 1/2/4/8

## Experiment status
- Training in progress at last check (56/75 epochs). Final validation/test accuracy TBD.
