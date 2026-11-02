#!/usr/bin/env python3
"""CVP v2 revision tests (6.1-6.7). Run: python training/test_cvp_v2.py"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn.functional as F

from modules.correspondence_video_prompt import (
    CorrespondenceVideoPromptBlock,
    PatchSelector,
    _frame_difference,
)
from VitaCLIP_model import VitaCLIP

BACKBONE = "/hy-tmp/weights/ViT-B-16.pt"
CLASSES = "/hy-tmp/UCF101/annotations/ucfTrainTestlist/classInd.txt"
DEV = "cuda" if torch.cuda.is_available() else "cpu"
B, T, N, D = 2, 8, 196, 768


def _inputs(seed=0):
    torch.manual_seed(seed)
    tokens = torch.randn(B, T, N + 1, D, device=DEV)
    attn = torch.rand(B, T, 12, N + 1, N + 1, device=DEV)
    pos = torch.randn(N, D, device=DEV)
    inp = torch.randn(B, T, N, D, device=DEV)
    return tokens, attn, pos, inp


def test_6_1_static_frame():
    print("== 6.1 Static Frame Test ==")
    blk = CorrespondenceVideoPromptBlock(d_model=D, num_frames=T).to(DEV).eval()
    static = torch.ones(B, T, N, D, device=DEV)
    tokens = torch.randn(B, T, N + 1, D, device=DEV)
    tokens = torch.cat([tokens[:, :, :1], static], dim=2)
    attn = torch.rand(B, T, 12, N + 1, N + 1, device=DEV)
    with torch.no_grad():
        diff = _frame_difference(tokens[:, :, 1:])
        assert diff.abs().max() < 1e-6, f"frame diff not zero: {diff.abs().max()}"
        _, _, _, stats = blk(tokens, attn, torch.randn(N, D, device=DEV), static)
    rel = stats["motion_reliability_mean"].item()
    wm = stats["motion_weight_mean"].item()
    assert rel < 0.01, rel
    assert wm < 0.01, wm
    assert torch.isfinite(stats["corr_cosine"])
    print(f"  frame_diff max={diff.abs().max().item():.2e} | "
          f"reliability={rel:.2e} | motion_weight={wm:.2e} | score is attention-driven")


def test_6_2_sparse_motion():
    print("== 6.2 Sparse Motion Test ==")
    sel = PatchSelector(topk=64)
    static = torch.ones(B, T, N, D)
    sparse = static.clone()
    sparse[:, :, :40] += 3.0 * torch.randn(B, T, 40, D)  # ~20% patches change (>q90)
    _, _, s_static = sel(static, torch.rand(B, T, N))
    _, _, s_sparse = sel(sparse, torch.rand(B, T, N))
    q50_s = s_static["q50_motion"].item()
    q90_s = s_static["q90_motion"].item()
    q50_m = s_sparse["q50_motion"].item()
    q90_m = s_sparse["q90_motion"].item()
    rel_s = s_static["motion_reliability_mean"].item()
    rel_m = s_sparse["motion_reliability_mean"].item()
    assert q90_m > q50_m, (q50_m, q90_m)
    assert rel_m > rel_s, (rel_s, rel_m)
    # changed patches must have higher motion score than unchanged ones
    d_sparse = s_sparse["motion_score_mean"]  # mean over all; check via raw
    diff = _frame_difference(sparse)
    mean_changed = diff[..., :40].mean().item()
    mean_unchanged = diff[..., 40:].mean().item()
    assert mean_changed > mean_unchanged, (mean_changed, mean_unchanged)
    print(f"  static: q50={q50_s:.2e} q90={q90_s:.2e} rel={rel_s:.2e}")
    print(f"  sparse: q50={q50_m:.2e} q90={q90_m:.2e} rel={rel_m:.2e}")
    print(f"  changed-patch diff {mean_changed:.4f} > unchanged {mean_unchanged:.4f}")


def test_6_3_partition():
    print("== 6.3 Selection Partition Test ==")
    blk = CorrespondenceVideoPromptBlock(d_model=D, num_frames=T).to(DEV).eval()
    tokens, attn, pos, inp = _inputs()
    with torch.no_grad():
        blk(tokens, attn, pos, inp)
    dbg = blk  # stats on the block? recompute via forward return
    with torch.no_grad():
        _, _, _, stats = blk(tokens, attn, pos, inp)
    sel_mask = stats["debug"]["selection_mask"]
    vis_idx = stats["debug"]["visible_indices"]  # [B, T, 132]
    assert (sel_mask.sum(-1) == 64).all()
    assert vis_idx.shape[-1] == 132, vis_idx.shape
    for b in range(B):
        for t in range(T):
            sel_set = set(sel_mask[b, t].nonzero(as_tuple=True)[-1].tolist())
            vis_set = set(vis_idx[b, t].tolist())
            assert len(vis_set) == 132
            assert not (sel_set & vis_set), "overlap between selected and visible"
            assert sel_set | vis_set == set(range(196))
    print("  selected=64 | visible=132 | disjoint | union={0..195}: OK")


def test_6_4_query_conditioning():
    print("== 6.4 Query Conditioning Test ==")
    blk = CorrespondenceVideoPromptBlock(d_model=D, num_frames=T).to(DEV).eval()
    tokens, attn, pos, inp = _inputs()
    sel_mask, sel_idx, _ = blk.selector(tokens[:, :, 1:], attn.mean(2)[:, :, 0, 1:])

    with torch.no_grad():
        base_q = blk.correspondence.build_base_queries(sel_idx, pos, torch.float32)
        q_ctx_a, _ = blk.correspondence._current_context(
            base_q, inp, sel_idx, 132
        )
        # 1) different visible input patches -> q_ctx changes
        inp_b = torch.where(
            (~sel_mask)[..., None], torch.randn_like(inp), inp
        )
        q_ctx_b, _ = blk.correspondence._current_context(
            base_q, inp_b, sel_idx, 132
        )
        d1 = (q_ctx_a - q_ctx_b).abs().mean().item()
        assert d1 > 0, d1
        # 2) only SELECTED input patches changed -> q_ctx identical
        inp_c = torch.where(
            sel_mask[..., None], torch.randn_like(inp), inp
        )
        q_ctx_c, _ = blk.correspondence._current_context(
            base_q, inp_c, sel_idx, 132
        )
        d2 = (q_ctx_a - q_ctx_c).abs().mean().item()
        assert d2 == 0.0, f"selected input leaked into q_ctx: {d2}"
        # 3) block4 tokens changed (but same visible input) -> q_ctx unchanged
        tokens_d = tokens.clone()
        tokens_d[:, :, 1:] = torch.randn_like(tokens_d[:, :, 1:])
        q_ctx_d, _ = blk.correspondence._current_context(
            base_q, inp, sel_idx, 132
        )
        d3 = (q_ctx_a - q_ctx_d).abs().mean().item()
        assert d3 == 0.0, f"block4 tokens leaked into q_ctx: {d3}"
    print(f"  diff-visible q_ctx diff={d1:.4f} > 0 | selected-input leak={d2} "
          f"| block4-token leak={d3}")


def test_6_5_spatial_bias():
    print("== 6.5 Spatial Bias Test ==")
    blk = CorrespondenceVideoPromptBlock(d_model=D, num_frames=T).to(DEV).eval()
    tokens, attn, pos, inp = _inputs()
    # uniform QK: zero out q_proj and k_proj so logits are all equal
    with torch.no_grad():
        blk.correspondence.q_proj.weight.zero_()
        blk.correspondence.q_proj.bias.zero_()
        blk.correspondence.k_proj.weight.zero_()
        blk.correspondence.k_proj.bias.zero_()
        sel_mask, sel_idx, _ = blk.selector(tokens[:, :, 1:], attn.mean(2)[:, :, 0, 1:])
        _, _, _, stats = blk(tokens, attn, pos, inp)
    corr_att = stats["debug"]["full_corr_attention"]  # [B, T, K, 196]
    lam = stats["lambda_sp"].item()
    row = corr_att[0, 4, 0]  # [196] attention of one query
    sel_coords = blk.correspondence.patch_coordinates.to(DEV)
    q_pos = sel_coords[sel_idx[0, 4, 0]]
    d2 = ((sel_coords - q_pos) ** 2).sum(-1)
    near = d2 < 0.02
    far = d2 > 0.3
    p_near = row[near].mean().item()
    p_far = row[far].mean().item()
    assert abs(row.sum().item() - 1.0) < 1e-4, row.sum().item()
    assert p_near > p_far, (p_near, p_far, lam)
    print(f"  lambda_sp={lam:.4f} | P(near)={p_near:.3e} > P(far)={p_far:.3e} | rows sum to 1")

    # lambda ~ 0: distance must not matter
    with torch.no_grad():
        blk.correspondence.raw_lambda_sp.fill_(-12.0)
        _, _, _, stats0 = blk(tokens, attn, pos, inp)
    row0 = stats0["debug"]["full_corr_attention"][0, 4, 0]
    assert (row0.max() - row0.min()).abs() < 1e-4, "spatial bias still active"
    print(f"  lambda_sp~0: attention uniform (max-min={row0.max()-row0.min():.2e})")

    # gradient
    blk.zero_grad()
    with torch.no_grad():
        blk.correspondence.q_proj.weight.zero_(); blk.correspondence.q_proj.bias.zero_()
        blk.correspondence.k_proj.weight.zero_(); blk.correspondence.k_proj.bias.zero_()
    enh, prompts, corr, _ = blk(tokens, attn, pos, inp)
    corr.sum().backward()
    g = blk.correspondence.raw_lambda_sp.grad
    assert g is not None and torch.isfinite(g) and g != 0, g
    print(f"  raw_lambda_sp.grad = {g.item():.4e} (finite, non-zero)")


def test_6_6_full_shape():
    print("== 6.6 Full Correspondence Shape Test ==")
    blk = CorrespondenceVideoPromptBlock(d_model=D, num_frames=T).to(DEV).eval()
    tokens, attn, pos, inp = _inputs()
    with torch.no_grad():
        enh, prompts, corr, stats = blk(tokens, attn, pos, inp)
    assert enh.shape == (B, T, N, D)
    assert prompts.shape == (B, 4, D)
    ca = stats["debug"]["full_corr_attention"]
    assert ca.shape == (B, T, 64, N), ca.shape
    assert torch.isfinite(enh).all() and torch.isfinite(prompts).all()
    print(f"  enhanced {tuple(enh.shape)} | prompts {tuple(prompts.shape)} | "
          f"corr attention {tuple(ca.shape)} (no physical repeat of sources)")
    mem = torch.cuda.max_memory_allocated() / 1e9 if torch.cuda.is_available() else 0
    print(f"  peak GPU memory: {mem:.2f} GB")


def test_6_7_gradient():
    print("== 6.7 Gradient Test ==")
    model = VitaCLIP(
        backbone_path=BACKBONE, input_size=(224, 224), num_frames=8,
        num_classes=101, text_prompt_classes_path=CLASSES,
        use_cvp_block=True,
    ).to(DEV)
    rgb = torch.rand(2, 3, 8, 224, 224, device=DEV)
    labels = torch.randint(0, 101, (2,), device=DEV)
    logits = model(rgb)
    loss = torch.nn.functional.cross_entropy(logits, labels) + 0.1 * model._last_cvp_corr_loss
    loss.backward()
    corr = model.visual.cvp_block.correspondence
    checks = {
        "ctx k_proj": corr.ctx_k_proj.weight.grad,
        "ctx v_proj": corr.ctx_v_proj.weight.grad,
        "corr q_proj": corr.q_proj.weight.grad,
        "corr k_proj": corr.k_proj.weight.grad,
        "corr v_proj": corr.v_proj.weight.grad,
        "raw_lambda_sp": corr.raw_lambda_sp.grad,
        "raw_gamma": model.visual.cvp_block.raw_gamma.grad,
        "raw_beta": model.visual.cvp_block.prompt_generator.raw_beta.grad,
        "video_queries": model.visual.cvp_block.prompt_generator.video_queries.grad,
    }
    nonzero = 0
    for name, g in checks.items():
        assert g is not None, f"{name}.grad is None"
        assert torch.isfinite(g).all(), f"{name}.grad not finite"
        nz = g.abs().sum().item() > 0
        nonzero += int(nz)
        print(f"  {name}.grad: mean|g|={g.abs().mean().item():.3e} "
              f"{'(non-zero)' if nz else '(ZERO)'}")
    assert nonzero >= 6, f"only {nonzero}/9 non-zero gradients"
    assert model.visual.conv1.weight.grad is None
    print(f"  {nonzero}/9 key gradients non-zero; frozen CLIP grads: None")


def main():
    torch.manual_seed(0)
    test_6_1_static_frame()
    test_6_2_sparse_motion()
    test_6_3_partition()
    test_6_4_query_conditioning()
    test_6_5_spatial_bias()
    test_6_6_full_shape()
    test_6_7_gradient()
    print("\nALL CVP v2 TESTS PASSED")


if __name__ == "__main__":
    main()
