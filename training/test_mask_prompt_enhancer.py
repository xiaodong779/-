#!/usr/bin/env python3
"""Minimal verification of the bounded-learnable Mask Prompt alpha/gamma.

Run from the repo root (or training/):
    python training/test_mask_prompt_enhancer.py

Checks:
  1. effective alpha at init ~= 0.40
  2. effective gamma at init ~= 0.20
  3. valid person masks visibly change the input frames
  4. mask_valid == 0 frames are bit-exact identity
  5. loss.backward() produces finite grads on raw_alpha / raw_gamma
  6. optimizer and checkpoint both contain the two learnable tensors
"""

from __future__ import annotations

import argparse
import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

from VitaCLIP_model import LearnableMaskPromptEnhancer, VitaCLIP
from train_OPENAI_ALIGNED import _parameter_family, build_optimizer

BACKBONE = "/hy-tmp/weights/ViT-B-16.pt"
CLASSES = "/hy-tmp/UCF101/annotations/ucfTrainTestlist/classInd.txt"


def main() -> None:
    torch.manual_seed(0)

    # ---- unit-level: enhancer alone ------------------------------------
    enhancer = LearnableMaskPromptEnhancer(
        alpha_init=0.40, alpha_min=0.35, alpha_max=0.80,
        gamma_init=0.20, gamma_min=0.15, gamma_max=0.40,
        learnable_alpha=True, learnable_gamma=True,
    )
    values = enhancer.effective_values()

    assert abs(values["alpha"] - 0.40) < 1e-5, values
    assert abs(values["gamma"] - 0.20) < 1e-5, values
    print(f"1. alpha init = {values['alpha']:.6f} (expect ~0.40)")
    print(f"2. gamma init = {values['gamma']:.6f} (expect ~0.20)")

    B, T, H, W = 2, 8, 224, 224
    rgb = torch.rand(B, 3, T, H, W)
    actor = torch.zeros(B, 1, T, H, W)
    actor[:, :, 0] = 1.0  # person only on frame 0 of every clip
    inter = torch.zeros(B, 1, T, H, W)
    inter[:, :, 1] = 1.0  # interaction region on frame 1
    valid = torch.tensor([[1] * T, [1, 1, 0, 0, 1, 0, 1, 0]], dtype=torch.float32)

    spatial, enhanced = enhancer.compute_prompts_and_enhanced(
        rgb, actor, inter, valid
    )
    # 3. valid person mask visibly changes the input
    person_diff = (enhanced[:, :, 0] - rgb[:, :, 0]).abs().mean().item()
    assert person_diff > 1e-3, person_diff
    print(
        f"3. person-frame mean |delta| = {person_diff:.4f} "
        f"(gain 1+gamma = {values['person_gain']:.4f})"
    )
    # 4. invalid frames are bit-exact identity
    zero_valid = torch.zeros_like(valid)
    _, identity = enhancer.compute_prompts_and_enhanced(
        rgb, actor, inter, zero_valid
    )
    max_diff = (identity - rgb).abs().max().item()
    assert max_diff == 0.0, max_diff
    print(f"4. mask_valid=0 identity max diff = {max_diff}")

    # ---- model-level: registration / forward / backward -----------------
    model = VitaCLIP(
        backbone_path=BACKBONE,
        input_size=(224, 224), num_frames=8, num_classes=101,
        text_prompt_classes_path=CLASSES,
        use_mask_prompt=True,
        mask_prompt_alpha_init=0.40, mask_prompt_alpha_min=0.35,
        mask_prompt_alpha_max=0.80,
        mask_prompt_gamma_init=0.20, mask_prompt_gamma_min=0.15,
        mask_prompt_gamma_max=0.40,
        mask_prompt_learnable_alpha=True, mask_prompt_learnable_gamma=True,
    )

    # registered as an nn.Module submodule with optimizer-managed params
    assert isinstance(model.mask_prompt_enhancer, LearnableMaskPromptEnhancer)
    param_names = [
        name for name, _ in model.named_parameters()
        if "mask_prompt" in name
    ]
    assert param_names == [
        "mask_prompt_enhancer.raw_alpha",
        "mask_prompt_enhancer.raw_gamma",
    ], param_names
    assert model.mask_prompt_enhancer.raw_alpha.requires_grad
    assert model.mask_prompt_enhancer.raw_gamma.requires_grad

    model.eval()  # no dropout in this path; eval only silences training logic
    labels = torch.randint(0, 101, (B,))
    logits = model(
        rgb, labels, bie_step=0,
        mask_prompt_actor=actor,
        mask_prompt_interaction=inter,
        mask_prompt_valid=valid,
    )
    assert logits.shape == (B, 101)
    loss = torch.nn.functional.cross_entropy(logits, labels)
    loss.backward()
    raw_alpha = model.mask_prompt_enhancer.raw_alpha
    raw_gamma = model.mask_prompt_enhancer.raw_gamma
    assert raw_alpha.grad is not None
    assert raw_gamma.grad is not None
    assert torch.isfinite(raw_alpha.grad).all()
    assert torch.isfinite(raw_gamma.grad).all()
    print(
        f"5. grads: raw_alpha={raw_alpha.grad.item():.3e}, "
        f"raw_gamma={raw_gamma.grad.item():.3e}"
    )
    # frozen CLIP check
    assert model.visual.conv1.weight.grad is None
    # bounds hold after backward (parameters unchanged until step)
    values = model.mask_prompt_enhancer.effective_values()
    assert 0.35 <= values["alpha"] <= 0.80
    assert 0.15 <= values["gamma"] <= 0.40

    # ---- optimizer & checkpoint -----------------------------------------
    assert _parameter_family("mask_prompt_enhancer.raw_alpha") == "mask_prompt"
    args = argparse.Namespace(
        lr=1.0e-4, prompt_lr=5e-4, adapter_lr=2e-4, time_embed_lr=1e-4,
        scalar_lr=1e-4, bie_lr=1e-4, weight_decay=5e-4,
        adam_beta1=0.9, adam_beta2=0.999, adam_eps=1e-8,
    )
    model.zero_grad()
    optimizer = build_optimizer(model, args)
    mp_groups = [
        g for g in optimizer.param_groups
        if g["group_name"].startswith("mask_prompt")
    ]
    assert len(mp_groups) == 1
    group = mp_groups[0]
    group_params = [id(p) for p in group["params"]]
    assert id(raw_alpha) in group_params and id(raw_gamma) in group_params
    assert group["lr"] == args.lr and group["weight_decay"] == 0.0
    print(
        f"6a. optimizer group {group['group_name']}: lr={group['lr']}, "
        f"wd={group['weight_decay']}, tensors={len(group['params'])}"
    )

    # checkpoint round-trip
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "mp_test.pth")
        torch.save({"model": model.state_dict()}, path)
        saved = torch.load(path, map_location="cpu")["model"]
        assert "mask_prompt_enhancer.raw_alpha" in saved
        assert "mask_prompt_enhancer.raw_gamma" in saved

        rebuilt = VitaCLIP(
            backbone_path=BACKBONE,
            input_size=(224, 224), num_frames=8, num_classes=101,
            text_prompt_classes_path=CLASSES,
            use_mask_prompt=True,
            mask_prompt_alpha_init=0.40, mask_prompt_alpha_min=0.35,
            mask_prompt_alpha_max=0.80,
            mask_prompt_gamma_init=0.20, mask_prompt_gamma_min=0.15,
            mask_prompt_gamma_max=0.40,
            mask_prompt_learnable_alpha=True, mask_prompt_learnable_gamma=True,
        )
        missing, unexpected = rebuilt.load_state_dict(saved, strict=False)
        assert not unexpected, unexpected
        assert all("mask_prompt" not in k for k in missing), missing
        assert torch.equal(
            rebuilt.mask_prompt_enhancer.raw_alpha.detach(), raw_alpha.detach()
        )
        assert torch.equal(
            rebuilt.mask_prompt_enhancer.raw_gamma.detach(), raw_gamma.detach()
        )
    print("6b. checkpoint save/load round-trip OK")

    # ---- one forward pass with real numbers ------------------------------
    values = model.mask_prompt_enhancer.effective_values()
    print("\n=== one forward pass (effective values) ===")
    print(
        f"alpha = {values['alpha']:.4f}  "
        f"gamma = {values['gamma']:.4f}\n"
        f"person-region gain      1 + gamma          = "
        f"{values['person_gain']:.4f}\n"
        f"interaction-region gain 1 + alpha * gamma  = "
        f"{values['interaction_gain']:.4f}"
    )
    print("\nALL MASK PROMPT ENHANCER TESTS PASSED")


if __name__ == "__main__":
    main()
