#!/usr/bin/env python3
"""TGL-Adapter test suite. Run: python training/test_tgl_adapter.py

Covers the mandatory contract:
  1/2. I/O shapes ([2,8,196,768] in -> [2,8,196,768] out, tracks [2,64,8,2])
  3.   gamma == 0  -> enhanced_tokens == patch_tokens (identity start)
  4.   visibility all 0 -> enhanced_tokens == patch_tokens
  5.   nearest & bilinear modes produce no NaN
  6.   multiple trajectories on one patch scatter as a WEIGHTED AVERAGE
  7.   pre-birth frames are masked out of attention (exact invariance)
plus offline preprocessing logic (dedup / samplers / query points).
"""

from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

from TGL_adapter import TGLAdapter
from tgl_tracker_offline import (
    dedup_trajectories,
    make_query_points,
    motion_scores,
    select_trajectories,
    visibility_scores,
)

DEV = "cuda" if torch.cuda.is_available() else "cpu"
B, T, N, C, K = 2, 8, 196, 768, 64
PATCH_HW = (14, 14)
IMAGE_SIZE = (224, 224)


def _gen(seed: int) -> torch.Generator:
    return torch.Generator(device=DEV).manual_seed(seed)


def _make_inputs(seed: int = 0, vis_fill: float = None) -> dict:
    g = _gen(seed)
    patch_tokens = torch.randn(B, T, N, C, generator=g, device=DEV)
    tracks = torch.rand(B, K, T, 2, generator=g, device=DEV) * 224.0
    visibility = torch.rand(B, K, T, generator=g, device=DEV)
    if vis_fill is not None:
        visibility = torch.full_like(visibility, vis_fill)
    birth_frame = torch.randint(0, 4, (B, K), generator=g, device=DEV)
    return dict(patch_tokens=patch_tokens, tracks=tracks, visibility=visibility, birth_frame=birth_frame)


def _build(**overrides) -> TGLAdapter:
    kwargs = dict(dim=C, num_layers=2, num_heads=4, dropout=0.1, gather_mode="nearest",
                  local_tube_size=3, max_scale=0.1)
    kwargs.update(overrides)
    m = TGLAdapter(**kwargs).to(DEV)
    m.eval()  # deterministic (dropout off) for all assertions
    return m


def test_1_2_shapes():
    print("== TGL 1/2. Shape Test ==")
    m = _build()
    inp = _make_inputs()
    with torch.no_grad():
        enhanced, aux = m(inp["patch_tokens"], inp["tracks"], inp["visibility"],
                          inp["birth_frame"], PATCH_HW, IMAGE_SIZE)
    assert enhanced.shape == (B, T, N, C), enhanced.shape
    assert aux["trajectory_tokens"].shape == (B, K, T, C // 4), aux["trajectory_tokens"].shape
    assert aux["residual_tokens"].shape == (B, K, T, C), aux["residual_tokens"].shape
    assert aux["scatter_weight"].shape == (B, T, N), aux["scatter_weight"].shape
    assert aux["adapter_scale"].numel() == 1 and float(aux["adapter_scale"]) == 0.0
    assert aux["valid_mask"].dtype == torch.bool and aux["valid_mask"].shape == (B, K, T)
    assert torch.isfinite(enhanced).all()
    print(f"  enhanced: {tuple(enhanced.shape)} | traj_tokens: {tuple(aux['trajectory_tokens'].shape)}"
          f" | residual: {tuple(aux['residual_tokens'].shape)} | scale={float(aux['adapter_scale'])}")


def test_3_gamma_zero_identity():
    print("== TGL 3. gamma=0 -> identity ==")
    m = _build()
    assert float(m.gamma) == 0.0, "gamma must be zero-init"
    assert float(m.up_proj.weight.abs().max()) < 1e-2, \
        "up_proj must be (at most) tiny-init; exact identity comes from gamma=0"
    inp = _make_inputs(1)
    with torch.no_grad():
        enhanced, _ = m(inp["patch_tokens"], inp["tracks"], inp["visibility"],
                        inp["birth_frame"], PATCH_HW, IMAGE_SIZE)
    assert torch.allclose(enhanced, inp["patch_tokens"], atol=1e-6), \
        "gamma=0 must leave patch tokens untouched"
    print(f"  max |enhanced - patch| = {(enhanced - inp['patch_tokens']).abs().max().item():.3e}")


def test_4_zero_visibility_identity():
    print("== TGL 4. visibility all 0 -> identity ==")
    m = _build()
    inp = _make_inputs(2, vis_fill=0.0)
    with torch.no_grad():
        enhanced, aux = m(inp["patch_tokens"], inp["tracks"], inp["visibility"],
                          inp["birth_frame"], PATCH_HW, IMAGE_SIZE)
    assert torch.allclose(enhanced, inp["patch_tokens"], atol=1e-6), \
        "zero visibility must produce a zero residual"
    assert float(aux["scatter_weight"].abs().max()) == 0.0, "no trajectory may receive weight"
    assert bool(aux["valid_mask"].any()) is False
    print(f"  max |enhanced - patch| = {(enhanced - inp['patch_tokens']).abs().max().item():.3e}"
          f" | max scatter_weight = {float(aux['scatter_weight'].abs().max())}")


def test_5_no_nan_both_modes():
    print("== TGL 5. nearest/bilinear: no NaN ==")
    for mode in ("nearest", "bilinear"):
        for tube in (1, 3):
            m = _build(gather_mode=mode, local_tube_size=tube)
            inp = _make_inputs(3)
            # stress: out-of-range coords + zero-visibility columns + late births
            inp["tracks"][..., 0] = inp["tracks"][..., 0] * 1.2 - 15.0
            inp["tracks"][..., 1] = inp["tracks"][..., 1] * 1.2 - 15.0
            inp["visibility"][:, ::3] = 0.0
            inp["birth_frame"][:, ::2] = 6
            with torch.no_grad():
                enhanced, aux = m(inp["patch_tokens"], inp["tracks"], inp["visibility"],
                                  inp["birth_frame"], PATCH_HW, IMAGE_SIZE)
            assert torch.isfinite(enhanced).all(), f"{mode}/tube{tube}: NaN in enhanced"
            assert torch.isfinite(aux["residual_tokens"]).all(), f"{mode}/tube{tube}: NaN in residual"
            print(f"  mode={mode:8s} tube={tube}: finite ok (scale={float(aux['adapter_scale']):.4f})")


def test_6_scatter_weighted_average():
    print("== TGL 6. same-patch scatter is a weighted average ==")
    m = _build(dropout=0.0)
    # Non-zero residual path: wake up up_proj and gamma (tests identity start
    # separately in test 3).
    torch.nn.init.normal_(m.up_proj.weight, std=0.02)
    with torch.no_grad():
        m.gamma.fill_(0.8)
    scale = m.max_scale * float(torch.tanh(m.gamma))

    cell = IMAGE_SIZE[1] / PATCH_HW[1]
    x, y = (5 + 0.5) * cell, (3 + 0.5) * cell  # patch center (row 3, col 5) -> idx 47
    patch_idx = 3 * PATCH_HW[1] + 5

    patch_tokens = torch.randn(1, T, N, C, device=DEV)
    tracks = torch.tensor([x, y], device=DEV).view(1, 1, 1, 2).repeat(1, 2, T, 1)
    visibility = torch.tensor([0.9, 0.7], device=DEV).view(1, 2, 1).repeat(1, 1, T)
    birth = torch.zeros(1, 2, dtype=torch.long, device=DEV)
    with torch.no_grad():
        enhanced, aux = m(patch_tokens, tracks, visibility, birth, PATCH_HW, IMAGE_SIZE)

    r_hat = (enhanced[0, :, patch_idx] - patch_tokens[0, :, patch_idx]) / scale  # [T, C]
    r0, r1 = aux["residual_tokens"][0, 0], aux["residual_tokens"][0, 1]  # [T, C]
    assert not torch.allclose(r0, r1, atol=1e-4), "two trajectories must yield different residuals"
    expected = (visibility[0, 0].unsqueeze(-1) * r0 + visibility[0, 1].unsqueeze(-1) * r1) \
        / visibility[0].sum(dim=0).unsqueeze(-1)  # per-frame K-sum = 0.9 + 0.7
    # Norm-conserving write-back: direction == weighted scatter result,
    # magnitude == ||patch token|| per frame.
    cos = torch.nn.functional.cosine_similarity(r_hat, expected, dim=-1)
    assert float(cos.min()) > 0.9999, \
        f"scatter direction must be sum(w*R)/sum(w), min cos = {float(cos.min()):.4f}"
    pnorm = patch_tokens[0, :, patch_idx].norm(dim=-1)
    assert torch.allclose(r_hat.norm(dim=-1), pnorm, rtol=1e-4), \
        "norm-conserving write-back: ||delta|| must equal ||patch||"
    assert float(aux["scatter_weight"][0, :, patch_idx].min()) > 0.0
    print(f"  weighted-average direction match: min cos = {float(cos.min()):.5f}"
          f" | ||delta||/||patch|| = {float((r_hat.norm(dim=-1) / pnorm).mean()):.4f}"
          f" (== scale, norm-conserving)")


def test_7_birth_frame_masking():
    print("== TGL 7. pre-birth frames masked in attention ==")
    m = _build(dropout=0.0)
    torch.nn.init.normal_(m.up_proj.weight, std=0.02)
    with torch.no_grad():
        m.gamma.fill_(1.0)

    g = _gen(7)
    patch_tokens = torch.randn(1, T, N, C, generator=g, device=DEV)
    tracks = torch.rand(1, 4, T, 2, generator=g, device=DEV) * 224.0
    visibility = torch.ones(1, 4, T, device=DEV)
    birth = torch.full((1, 4), 4, dtype=torch.long, device=DEV)

    with torch.no_grad():
        base, aux_base = m(patch_tokens, tracks, visibility, birth, PATCH_HW, IMAGE_SIZE)
        pert = patch_tokens.clone()
        pert[:, :4] += 10.0 * torch.randn(1, 4, N, C, generator=g, device=DEV)
        mod, aux_mod = m(pert, tracks, visibility, birth, PATCH_HW, IMAGE_SIZE)

    assert torch.allclose(base[:, 4:], mod[:, 4:], atol=1e-5), \
        "changing pre-birth patch tokens must not affect post-birth outputs"
    assert torch.allclose(aux_base["trajectory_tokens"][:, :, :4],
                          torch.zeros_like(aux_base["trajectory_tokens"][:, :, :4]), atol=1e-7), \
        "pre-birth trajectory tokens must be exactly zero"
    assert float(aux_base["scatter_weight"][:, :4].abs().max()) == 0.0, \
        "pre-birth frames must receive zero scatter weight"
    assert not torch.allclose(base[:, :4], mod[:, :4]), "sanity: pre-birth tokens did change"
    print(f"  post-birth invariance: max diff = {(base[:, 4:] - mod[:, 4:]).abs().max().item():.3e}"
          f" | pre-birth traj tokens = 0 | pre-birth scatter weight = 0")


def test_8_offline_logic():
    print("== TGL 8. offline: query points / dedup / samplers ==")
    q = make_query_points(num_frames=8, patch_grid=14, image_size=(224, 224), query_frame_stride=2)
    assert q.shape == (784, 3), q.shape
    births = sorted(set(int(f) for f in q[:, 0]))
    assert births == [0, 2, 4, 6], f"TAT-style births expected 0/2/4/6, got {births}"
    assert q[:, 1:].min() > 0 and q[:, 1:].max() < 224

    # 8 synthetic trajectories on T=8: {base0, base1, base2, static} x 2
    # (the second copy of each base path is shifted 2px -> redundant).
    t = torch.arange(8, dtype=torch.float32).view(1, 8, 1)
    base_paths = torch.tensor([[20.0, 20.0], [100.0, 50.0], [180.0, 180.0]], dtype=torch.float32)
    dirs = torch.tensor([[1.0, 0.5], [-0.5, 1.0], [0.0, -1.0]], dtype=torch.float32) * 8.0
    paths = base_paths.unsqueeze(1) + dirs.unsqueeze(1) * t  # [3, 8, 2]
    static = torch.tensor([[10.0, 200.0]]) + torch.tensor([[3.0, -2.0]]) * t  # [1, 8, 2]
    tracks = torch.cat([paths, paths + torch.tensor([2.0, 0.0]), static, static], dim=0)  # [8,8,2]
    vis = torch.ones(8, 8)
    birth = torch.tensor([0, 2, 0, 4, 0, 0, 0, 0])
    keep = dedup_trajectories(tracks, vis, birth, delta=8.0, dir_thresh=0.9)
    # Duplicate pairs are (0,3), (1,4), (2,5), (6,7); the earlier-born (or
    # equal-birth-first) copy must survive: expect exactly {0, 2, 4, 6}.
    assert keep.numel() == 4, f"expected 4 survivors after dedup, got {keep.tolist()}"
    assert set(keep.tolist()) == {0, 2, 4, 6}, \
        f"earlier-born duplicates must survive, got {keep.tolist()}"

    sel = select_trajectories(tracks[keep], vis[keep], birth[keep], max_tracks=2,
                              sampler="tat_random", seed=0)
    assert sel.numel() == 2 and sel.unique().numel() == 2
    sel_b = select_trajectories(tracks[keep], vis[keep], birth[keep], max_tracks=2,
                                sampler="tat_random", seed=0)
    assert torch.equal(sel, sel_b), "tat_random must be seeded/deterministic"
    sel_all = select_trajectories(tracks[keep], vis[keep], birth[keep], max_tracks=64)
    assert sel_all.numel() == 4, "no sampling needed when S <= max_tracks"

    ms = motion_scores(tracks[keep], vis[keep], (224, 224))
    vs = visibility_scores(vis[keep])
    assert ms.shape == (4,) and vs.shape == (4,)
    assert torch.isfinite(ms).all() and float(ms.min()) >= 0.0 and float(ms.max()) <= 1.0
    print(f"  births={births} | dedup 8->4 (keep {keep.tolist()}) | "
          f"tat_random K=2 deterministic | scores in [0,1]")


def test_9_gradient_flow():
    print("== TGL 9. wake-up sequence: gamma first, then the full path ==")
    m = _build()
    m.train()
    inp = _make_inputs(11)

    def _grad_sum(mod, path):
        cur = m
        for part in path.split("."):
            cur = getattr(cur, part)
        if isinstance(cur, torch.nn.Module):
            return sum(float(p.grad.abs().sum()) for p in cur.parameters() if p.grad is not None)
        g = cur.grad
        return 0.0 if g is None else float(g.abs().sum())

    # --- step 0: scale == 0, so only gamma is awake (tiny up_proj makes its
    # gradient non-zero); the gated path is expected to have zero grad yet.
    enhanced, _ = m(inp["patch_tokens"], inp["tracks"], inp["visibility"],
                    inp["birth_frame"], PATCH_HW, IMAGE_SIZE)
    enhanced.square().mean().backward()
    assert _grad_sum(m, "gamma") > 0.0, "gamma must receive gradient at step 0"
    for path in ("up_proj", "down_proj", "blocks.0.attn_t.qkv", "motion_mlp.0",
                 "tube_pool"):
        assert _grad_sum(m, path) == 0.0, f"{path} grad must be 0 while scale == 0"
    m.zero_grad(set_to_none=True)

    # --- step 2 (after gamma moves): the full path must receive gradients.
    with torch.no_grad():
        m.gamma += 0.5
    enhanced, _ = m(inp["patch_tokens"], inp["tracks"], inp["visibility"],
                    inp["birth_frame"], PATCH_HW, IMAGE_SIZE)
    enhanced.square().mean().backward()
    for path in ("gamma", "up_proj", "down_proj", "blocks.0.attn_t.qkv", "motion_mlp.0",
                 "tube_pool"):
        assert _grad_sum(m, path) > 0.0, f"no grad on {path} once scale != 0"

    # identity start still holds while gamma is 0 (fresh module, weights trained)
    m2 = _build()
    m2.load_state_dict(m.state_dict(), strict=True)
    with torch.no_grad():
        m2.gamma.zero_()
        enhanced0, _ = m2(inp["patch_tokens"], inp["tracks"], inp["visibility"],
                          inp["birth_frame"], PATCH_HW, IMAGE_SIZE)
    assert torch.allclose(enhanced0, inp["patch_tokens"], atol=1e-6)
    print("  step0: grad(gamma)>0, gated path 0 | step2: full path awake | identity at gamma=0")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true", help="skip gradient test")
    args = ap.parse_args()
    tests = [test_1_2_shapes, test_3_gamma_zero_identity, test_4_zero_visibility_identity,
             test_5_no_nan_both_modes, test_6_scatter_weighted_average, test_7_birth_frame_masking,
             test_8_offline_logic]
    if not args.quick:
        tests.append(test_9_gradient_flow)
    for t in tests:
        t()
    print(f"\nALL {len(tests)} TGL-ADAPTER TESTS PASSED")


if __name__ == "__main__":
    main()
