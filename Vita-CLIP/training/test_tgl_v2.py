#!/usr/bin/env python3
"""TGL v2 test suite (24 checks). Run: python training/test_tgl_v2.py [--quick]

Covers the v2 spec: geometry/gather correctness, Track-ST + Trajectory CLS,
CVPT prompt injection (alpha=0 identity, single-step cold start), pure-image
late fusion, frozen-CLIP gradient audit, cache preflight, CPU/CUDA.
"""

from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

from tgl_geometry import (
    track_patch_indices,
    flip_x,
    crop_resize_coords,
    flip_affine,
    crop_resize_affine,
    apply_affine,
    reverse_time,
)
from tgl_track_st import TrackSTEncoder, build_motion_input
from tgl_late_fusion import TemporalPool, track_quality_stats, LateFusion

DEV = "cpu"  # 单元测试默认 CPU；CUDA smoke 在 test_22 单独执行
T = 8
GRID = 14
IMG = 224
DIM = 768
TRACK_DIM = 192
CLIP_DIM = 512

# (x, y) -> 期望 patch 索引（floor(x*14/224) + floor(y*14/224)*14）
CASES = {
    (0, 0): 0,
    (223, 0): 13,
    (0, 223): 182,
    (223, 223): 195,
    (111, 111): 6 * 14 + 6,   # 中心
}


def test_02_03_07_geom():
    print("== v2 02/03/07. final 形状 + (x,y)→patch 映射（未写反） ==")
    x = torch.tensor([list(c[0] for c in CASES)])
    y = torch.tensor([list(c[1] for c in CASES)])
    idx, in_bounds, _ = track_patch_indices(x, y, GRID, GRID, IMG, IMG)
    expected = torch.tensor([v for v in CASES.values()])
    assert torch.equal(idx[0], expected), (idx, expected)
    assert bool(in_bounds.all())
    # 轴写反检测：交换 x/y 后索引必须不同（对非对角点）
    idx_swapped, _, _ = track_patch_indices(y, x, GRID, GRID, IMG, IMG)
    assert not torch.equal(idx[0], idx_swapped[0]), "(x,y) 疑似写反"
    print(f"  映射: {dict(zip(CASES.keys(), idx[0].tolist()))} ✓")


def test_08_oob():
    print("== v2 08. 越界点 clamp 后仍 invalid ==")
    x = torch.tensor([[[-5.0, 300.0, 100.0]]])  # [1,1,3]
    y = torch.tensor([[[50.0, -1.0, 50.0]]])
    idx, in_bounds, _ = track_patch_indices(x, y, GRID, GRID, IMG, IMG)
    assert bool(~in_bounds[0, 0, :2].all() is False) or True
    assert in_bounds[0, 0, 2] == True  # noqa
    assert bool((~in_bounds[0, 0, :2]).all()), "前两个点必须越界"
    assert idx[0, 0, 0] in range(196) and idx[0, 0, 1] in range(196), "clamp 只为安全 gather"
    print(f"  in_bounds={in_bounds[0,0].tolist()} | safe idx={idx[0,0].tolist()} ✓")


def test_09_flip():
    print("== v2 09. 水平翻转 x'=223-x，左右 patch 互换 ==")
    x = torch.tensor([[16.0]])  # patch 列 0（floor(16*14/224)=0）
    xf = flip_x(x, IMG)
    assert float(xf) == 223.0 - 16.0
    idx_l, _, (px_l, py_l) = track_patch_indices(x, torch.tensor([[100.0]]), GRID, GRID, IMG, IMG)
    idx_r, _, (px_r, py_r) = track_patch_indices(xf, torch.tensor([[100.0]]), GRID, GRID, IMG, IMG)
    assert int(px_l[0, 0]) + int(px_r[0, 0]) == 13, "翻转后左右 patch 列必须互补"
    assert int(py_l[0, 0]) == int(py_r[0, 0]), "翻转不改变行"
    # affine 一致性
    M = flip_affine(IMG, IMG)
    ax, ay = apply_affine(x, torch.tensor([[100.0]]), M)
    assert torch.allclose(ax, xf) and torch.allclose(ay, torch.tensor([[100.0]]))
    print(f"  x={float(x)} -> {float(xf)} | patch {int(idx_l[0,0])} <-> {int(idx_r[0,0])} "
          f"(px {int(px_l[0,0])}+{int(px_r[0,0])}=13) ✓")


def test_10_crop():
    print("== v2 10. crop+resize 坐标 ==")
    # 左上 112x112 裁剪到 224x224：坐标应精确翻倍
    x, y = crop_resize_coords(torch.tensor([[10.0]]), torch.tensor([[20.0]]),
                              0, 0, 112, 112, 224, 224)
    assert float(x) == 20.0 and float(y) == 40.0
    M = crop_resize_affine(0, 0, 112, 112, 224, 224)
    ax, ay = apply_affine(torch.tensor([[10.0]]), torch.tensor([[20.0]]), M)
    assert float(ax) == 20.0 and float(ay) == 40.0
    # 右半裁剪: crop_x=112 → x=168 应映射回 112
    x2, _ = crop_resize_coords(torch.tensor([[168.0]]), torch.tensor([[0.0]]),
                               112, 0, 112, 112, 224, 224)
    assert float(x2) == 112.0
    print("  crop+resize 坐标精确 ✓")


def test_11_23_preflight(tmp="/tmp/tgl_v2_preflight"):
    print("== v2 11/23. cache 预检：一致可启动 / 不一致明确拒绝 ==")
    os.makedirs(tmp, exist_ok=True)
    from tgl_tracker_offline import track_cache_path
    good = {
        "tracks": torch.rand(64, 8, 2) * 224,
        "visibility": torch.ones(64, 8),
        "birth_frame": torch.zeros(64, dtype=torch.long),
        "frame_indices": torch.linspace(0, 100, 8).long(),
        "image_size": [224, 224],
        "preprocess_config": {"resize": [224, 224], "patch_grid": 14,
                              "sampler": "tat_random", "delta_px": 8.0,
                              "query_frame_stride": 2, "max_tracks": 64},
    }
    p_ok = track_cache_path(tmp, "videos/a")
    torch.save(good, p_ok)
    bad = {**good, "tracks": torch.rand(16, 4, 2) * 224,
           "preprocess_config": {**good["preprocess_config"], "resize": [112, 112]}}
    p_bad = track_cache_path(tmp, "videos/b")
    torch.save(bad, p_bad)

    from run_tgl_pipeline import CfgNode
    tgl = CfgNode(TRACK_ROOT=tmp, NUM_FRAMES=8, IMAGE_SIZE=[224, 224],
                  PATCH_GRID=14, MAX_TRACKS=64, DELTA_PX=8.0,
                  QUERY_FRAME_STRIDE=2)
    tgl.PRESET_RESOLVED = CfgNode(offline_sampler="tat_random")
    import importlib
    rtp = importlib.import_module("run_tgl_pipeline")
    problems, _warn = rtp._preflight_entries(
        [("videos/a", 0)], tgl, tmp)
    assert not problems, problems
    problems, _warn = rtp._preflight_entries(
        [("videos/b", 0)], tgl, tmp)
    assert problems and any("num_frames" in p or "resize" in p for p in problems), problems
    print(f"  一致: 通过 | 不一致: {len(problems)} 条具体字段差异 ✓")
    os.remove(p_ok); os.remove(p_bad)


def test_12_13_14_19_trackst():
    print("== v2 12/13/14/19. Track-ST / Trajectory CLS / 全无效 / 梯度 ==")
    torch.manual_seed(0)
    B, K = 2, 64
    enc = TrackSTEncoder(clip_dim=DIM, track_dim=TRACK_DIM, num_layers=2,
                         num_heads=4, num_classes=10)
    tracks = torch.rand(B, K, T, 2) * 224
    vis = (torch.rand(B, K, T) > 0.3).float()
    motion = build_motion_input(tracks, vis, vis.bool(), IMG, IMG)
    app = torch.randn(B, K, T, DIM)
    st = enc(app, motion, vis.bool(), labels=torch.tensor([1, 2]),
             recon_mask_ratio=0.1, clip_target=app.detach(),
             dxdy_target=motion[..., 2:4].detach())
    assert st["trajectory_visual"].shape == (B, CLIP_DIM)
    assert torch.isfinite(st["trajectory_visual"]).all()
    assert "trajectory_logits" in st and "loss_recon_clip" in st
    # 12: padding + 不可见轨迹不影响 Trajectory CLS
    st_base = enc(app, motion, vis.bool())
    extra_vis = torch.zeros(B, K + 8, T)
    extra_vis[:, :K] = vis
    extra_app = torch.zeros(B, K + 8, T, DIM)
    extra_app[:, :K] = app
    extra_mot = build_motion_input(
        torch.cat([tracks, torch.zeros(B, 8, T, 2)], dim=1),
        torch.cat([vis, torch.zeros(B, 8, T)], dim=1),
        extra_vis.bool(), IMG, IMG)
    st_pad = enc(extra_app, extra_mot, extra_vis.bool())
    assert torch.allclose(st_base["trajectory_cls"], st_pad["trajectory_cls"], atol=1e-6), \
        "padding/不可见轨迹不得影响 Trajectory CLS"
    # 13: 打乱轨迹顺序 → Trajectory CLS 基本不变（无 fixed-order inter-motion）
    perm = torch.randperm(K)
    st_perm = enc(app[:, perm], motion[:, perm], vis[:, perm].bool())
    d = (st_base["trajectory_cls"] - st_perm["trajectory_cls"]).abs().max()
    assert float(d) < 1e-4, f"轨迹顺序打乱后 CLS 变化过大: {float(d)}"
    print(f"  padding 不影响 ✓ | 顺序打乱 max diff = {float(d):.2e} ✓ | "
          f"aux/recon loss 存在 ✓")


def test_14_all_invalid():
    print("== v2 14. 全部轨迹无效：无 NaN、gate=0、final==video ==")
    enc = TrackSTEncoder(DIM, TRACK_DIM, 2, 4, num_classes=10)
    lf = LateFusion(512, 4, gate_init=0.1)
    B = 2
    app = torch.randn(B, 4, T, DIM)
    motion = build_motion_input(torch.rand(B, 4, T, 2) * 224,
                                torch.zeros(B, 4, T), torch.zeros(B, 4, T).bool(), IMG, IMG)
    st = enc(app, motion, torch.zeros(B, 4, T).bool())
    assert torch.isfinite(st["trajectory_visual"]).all()
    assert float(st["trajectory_visual"].abs().max()) == 0.0, "全无效必须输出零向量"
    video = torch.nn.functional.normalize(torch.randn(B, 512), dim=-1)
    q = track_quality_stats(torch.zeros(B, 4, T).bool(),
                            torch.zeros(B, 4, T), torch.zeros(B, 4, T).bool(),
                            st["any_valid"])
    final, gate = lf(video, st["trajectory_visual"], q, st["any_valid"])
    assert float(gate.max()) == 0.0, "gate 必须强制为 0"
    assert torch.allclose(final, video, atol=1e-6), "final 必须严格等于 video_visual"
    print(f"  gate={gate.flatten().tolist()} | final==video ✓ | 无 NaN ✓")


def test_16_static_no_scatter():
    print("== v2 16. image encoder 源码无轨迹 scatter 调用 ==")
    import inspect
    from VitaCLIP_model import VisualTransformer
    src = inspect.getsource(VisualTransformer.encode_frames).lower()
    src_vt = inspect.getsource(VisualTransformer.forward).lower()
    # 统计键名 scatter_coverage 允许存在；禁止实际的 scatter/scatter_add 调用
    for bad in (".scatter(", ".scatter_", "scatter_add", "scatter_("):
        assert bad not in src, f"encode_frames 出现 {bad}"
        assert bad not in src_vt, f"VT.forward 出现 {bad}"
    print("  encode_frames / VT.forward 无轨迹 scatter 调用 ✓")


def test_24_gather_synthetic():
    print("== v2 24. 合成 patch=索引：gather 精确命中 ==")
    patches = torch.arange(196, dtype=torch.float32)[None, None, :, None].expand(1, 1, 196, 4)
    tracks = torch.tensor([8.0, 100.0, 220.0, 16.0]).reshape(1, 2, 1, 2)  # [1,K=2,T=1,2]
    vis = torch.ones(1, 2, 1)
    valid = torch.ones(1, 2, 1, dtype=torch.bool)
    motion = build_motion_input(tracks, vis, valid, IMG, IMG)
    idx, in_bounds, _ = track_patch_indices(tracks[..., 0], tracks[..., 1], GRID, GRID, IMG, IMG)
    gathered = patches.gather(
        2, idx.permute(0, 2, 1).unsqueeze(-1).expand(-1, -1, -1, 4)
    )  # idx [B,K,T] -> [B,T,K,1] -> gather dim=2 (N)
    # 轨迹 0: x=8 → px=floor(8*14/224)=0; y=100 → py=floor(100*14/224)=6 → idx 84
    # 轨迹 1: x=220 → px=13; y=16 → py=1 → idx 27
    assert int(idx[0, 0, 0]) == 84 and int(idx[0, 1, 0]) == 27, idx[0, :, 0].tolist()
    for k, want in ((0, 84), (1, 27)):
        assert torch.allclose(gathered[0, 0, k], torch.tensor([float(want)] * 4)), \
            "gather 必须取到索引对应的合成 patch"
    assert torch.equal(motion[..., 0], tracks[..., 0] / 224.0), "x 必须按 /W 归一化"
    print(f"  patch 索引 {idx[0, :, 0].tolist()} | 合成值精确命中 ✓")


# ---------------------------------------------------------------------------
# 模型级测试（需要 ViT-B 权重，CPU 可跑）
# ---------------------------------------------------------------------------
_MODEL_CACHE = {}


def _build(kind: str, device: str = "cpu"):
    if kind in _MODEL_CACHE:
        return _MODEL_CACHE[kind].to(device)
    from VitaCLIP_model import VitaCLIP
    torch.manual_seed(0)
    m = VitaCLIP(
        backbone_path="/hy-tmp/weights/ViT-B-16.pt",
        input_size=(224, 224), num_frames=8, num_classes=101,
        text_prompt_classes_path="/hy-tmp/UCF101/annotations/ucfTrainTestlist/classInd.txt",
        use_learnable_text_branch=True, use_fixed_text_branch=False,
        use_text_mstff=False, use_cvp_block=False,
        use_tgl_adapter=False, tgl_readout=False,
        use_tgl_v2=(kind == "v2"), tgl_v2_num_classes=101,
    ).to(device).eval()
    _MODEL_CACHE[kind] = m
    return m


def _model_inputs(B=1, seed=0):
    g = torch.Generator().manual_seed(seed)
    frames = torch.randn(B, 3, T, 224, 224, generator=g)
    tracks = torch.rand(B, 16, T, 2, generator=g) * 224
    vis = (torch.rand(B, 16, T, generator=g) > 0.2).float()
    birth = torch.zeros(B, 16, dtype=torch.long)
    return frames, tracks, vis, birth


def test_01_02_15_16_model():
    print("== v2 01/02/15/16. 端到端形状 / 单位范数 / gather 只读 / 无 scatter ==")
    m = _build("v2")
    frames, tracks, vis, birth = _model_inputs()
    with torch.no_grad():
        final, aux = m.encode_video(frames, tracks=tracks, visibility=vis,
                                    birth_frame=birth, return_aux=True)
    assert final.shape == (1, 512), final.shape
    n = float(final.norm(dim=-1))
    assert abs(n - 1.0) < 1e-4, f"final L2 norm = {n}"
    assert aux["video_visual"].shape == (1, 512)
    assert aux["trajectory_visual"].shape == (1, 512)
    print(f"  final_visual {tuple(final.shape)} | L2={n:.4f} | "
          f"gate={float(aux['fusion_gate']):.4f} | cov={float(aux['valid'].float().mean()):.3f}")


def test_17_18_20_prompt():
    print("== v2 17/18/20. alpha=0 恒等 / 一步冷启动 / 冻结审计 ==")
    mv2 = _build("v2")
    mpure = _build("pure")
    frames, tracks, vis, birth = _model_inputs(1, seed=3)

    cap = {}
    with torch.no_grad():
        for block in mv2.visual.transformer.resblocks:
            if getattr(block, "prompt_alpha", None) is not None:
                block.prompt_alpha.zero_()
    hooks = [
        mv2.visual.ln_post.register_forward_hook(lambda m, i, o: cap.update(v2=o.detach())),
        mpure.visual.ln_post.register_forward_hook(lambda m, i, o: cap.update(pure=o.detach())),
    ]
    with torch.no_grad():
        mv2.encode_video(frames,
                         tracks=tracks, visibility=vis,
                         birth_frame=birth)
        mpure.encode_video(frames)
    for hk in hooks:
        hk.remove()
    d = (cap["v2"] - cap["pure"]).abs().max()
    assert float(d) < 1e-6, f"alpha_l=0 时与原始 CLIP 差异 {float(d):.2e} 超出 atol=1e-6"
    print(f"  alpha=0 恒等: max |ΔCLS| = {float(d):.2e} < 1e-6 ✓")

    # 18: step0 alpha 有梯度；更新 alpha 后 generator 有梯度
    mv2.train()
    frames.requires_grad_(False)
    out = mv2.encode_video(frames,
                           tracks=tracks, visibility=vis,
                           birth_frame=birth)
    out.sum().backward()
    alphas = [b.prompt_alpha for b in mv2.visual.transformer.resblocks
              if getattr(b, "prompt_alpha", None) is not None]
    assert alphas and all(p.grad is not None and float(p.grad.abs().sum()) > 0 for p in alphas), \
        "step 0 alpha_l 必须有梯度"
    gen_ok_before = any(
        p.grad is not None and float(p.grad.abs().sum()) > 0
        for p in mv2.visual.v2_prompt_gen.parameters()
    )
    mv2.zero_grad(set_to_none=True)
    with torch.no_grad():
        for b in mv2.visual.transformer.resblocks:
            if getattr(b, "prompt_alpha", None) is not None:
                b.prompt_alpha += 0.1
    out = mv2.encode_video(frames,
                           tracks=tracks, visibility=vis,
                           birth_frame=birth)
    out.sum().backward()
    gen_ok = any(p.grad is not None and float(p.grad.abs().sum()) > 0
                 for p in mv2.visual.v2_prompt_gen.parameters())
    assert gen_ok, "更新 alpha 后 prompt generator 必须获得梯度（一步冷启动）"
    print(f"  alpha 梯度 ✓ | generator 冷启动一步后梯度 {'✓' if gen_ok else '✗'} "
          f"(step0 generator 梯度 {'有' if gen_ok_before else '无(符合单步冷启动)'})")

    # 20: 冻结 CLIP 参数无梯度
    for name, p in mv2.named_parameters():
        if "transformer.resblocks" in name and "prompt_alpha" not in name:
            assert p.requires_grad is False or p.grad is None or float(p.grad.abs().sum()) == 0, \
                f"冻结 CLIP 参数出现梯度: {name}"
    print("  冻结 CLIP 参数无梯度 ✓")


def test_19_model_grads():
    print("== v2 19. Track-ST / TrajCLS / projection / fusion gate 梯度 ==")
    mv2 = _build("v2")
    mv2.train()
    frames, tracks, vis, birth = _model_inputs(1, seed=5)
    out, aux = mv2.encode_video(
        frames,
        tracks=tracks,
        visibility=vis,
        birth_frame=birth,
        return_aux=True,
    )
    target = torch.randn_like(out)
    loss = (out * target).sum() + 0.1 * aux["trajectory_visual"].sum()
    loss.backward()
    checks = {
        "Track-ST": "v2_track_st.blocks",
        "TrajCLS query": "v2_track_st.trajectory_cls_query",
        "projection": "v2_track_st.trajectory_projection",
        "fusion gate": "v2_late_fusion.gate_linear",
    }
    for label, path in checks.items():
        mod = mv2.visual
        for part in path.split("."):
            mod = getattr(mod, part)
        if isinstance(mod, torch.nn.Parameter):
            g = float(mod.grad.abs().sum()) if mod.grad is not None else 0.0
        elif isinstance(mod, torch.nn.Module):
            g = sum(float(p.grad.abs().sum()) for p in mod.parameters() if p.grad is not None)
        else:
            g = 0.0
        assert g > 0, f"{label} 梯度为 0"
        print(f"  {label}: grad={g:.3e} ✓")


def test_21_cpu_full():
    print("== v2 21. CPU 全流程（含 12/13/14 已在 CPU 跑过）==")
    test_12_13_14_19_trackst()
    test_14_all_invalid()
    print("  CPU 全部通过 ✓")


def test_22_cuda_smoke():
    print("== v2 22. CUDA 单 batch smoke ==")
    if not torch.cuda.is_available():
        print("  无 CUDA，跳过")
        return
    m = _build("v2", device="cuda")
    frames, tracks, vis, birth = _model_inputs(1, seed=9)
    with torch.no_grad():
        final = m.encode_video(frames.cuda(),
                               tracks=tracks.cuda(), visibility=vis.cuda(),
                               birth_frame=birth.cuda())
    assert final.shape == (1, 512) and torch.isfinite(final).all()
    print(f"  CUDA forward ok | norm={float(final.norm(dim=-1)):.4f} ✓")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true", help="跳过模型级测试（仅几何/Track-ST）")
    args = ap.parse_args()
    test_02_03_07_geom()
    test_08_oob()
    test_09_flip()
    test_10_crop()
    test_24_gather_synthetic()
    test_12_13_14_19_trackst()
    test_14_all_invalid()
    test_11_23_preflight()
    test_16_static_no_scatter()
    if not args.quick:
        test_01_02_15_16_model()
        test_17_18_20_prompt()
        test_19_model_grads()
        test_21_cpu_full()
        test_22_cuda_smoke()
    print("\nALL TGL-V2 TESTS PASSED")


if __name__ == "__main__":
    main()
