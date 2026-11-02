"""Smoke tests for the RAP true cross-frame ASA branch (2-D offsets + S-window).

Run: cd /hy-tmp/Vita-CLIP/training && python _smoke_rap.py
"""

from __future__ import annotations

import sys

import torch
from torch import Tensor, nn

sys.path.insert(0, __file__.rsplit("/", 1)[0])

from VitaCLIP_discriminative_rap import _ASABranch, VitaCLIPDiscriminativeRAP  # noqa: E402

D = 8
N_PATCH = 16  # 4 x 4 grid


def _branch(window_size=3, heads=2, space_offset_max=4.0, dim=D):
    return _ASABranch(
        dim=dim,
        num_spatial_patches=N_PATCH,
        heads=heads,
        time_offset_max=1.0,
        space_offset_max=space_offset_max,
        window_size=window_size,
        eps=1e-6,
    )


def _block():
    return VitaCLIPDiscriminativeRAP(
        dim=D, bottleneck=D, num_spatial_patches=N_PATCH, top_k=4, alpha=0.4,
        use_conv3d=True, use_asa=True, asa_heads=2,
        asa_time_offset_max=1.0, asa_space_offset_max=1.0, asa_window_size=3,
        residual_scale_init=0.01, seed_scale_init=0.05, stability_clip=100.0,
        min_frames=4, eps=1e-6,
    )


def _finite(t):
    return bool(torch.isfinite(t).all().item())


def test_zero_init_offsets():
    br = _branch(window_size=3)
    x = torch.randn(2, 4, N_PATCH, D)
    top = torch.randint(0, N_PATCH, (2, 4, 3))
    _, off_t, off_h, off_w = br(x, top)
    assert torch.all(off_t == 0), f"off_t not zero: {off_t.abs().max()}"
    assert torch.all(off_h == 0), f"off_h not zero: {off_h.abs().max()}"
    assert torch.all(off_w == 0), f"off_w not zero: {off_w.abs().max()}"
    assert _finite(off_t) and _finite(off_h) and _finite(off_w)


def test_2d_row_offset():
    # Force off_h ~ +0.46 (off_t = off_w = 0): last offset_net layer bias.
    br = _branch(window_size=3, space_offset_max=1.0)
    with torch.no_grad():
        last = br.offset_net[-1]
        last.weight.zero_()
        last.bias.zero_()
        last.bias.copy_(torch.tensor([0.0, 0.5, 0.0]))

    x = torch.zeros(1, 2, N_PATCH, D)
    x[0, :, 9] = 1.0  # only (h=2, w=1) idx 9 is non-zero
    top = torch.full((1, 2, 1), 5, dtype=torch.long)  # select (h=1, w=1)

    delta, off_t, off_h, off_w = br(x, top)
    assert torch.all(off_t == 0)
    assert abs(float(off_h[0, 0, 0]) - 0.4621) < 1e-2, f"off_h={off_h}"
    assert torch.all(off_w == 0)

    # 2-D impl shifts sampling toward row h=2 (idx 9) -> delta@5 picks x[9].
    # A broken flat-axis impl would shift toward idx 6 (= 0) -> zero delta.
    sel_delta = delta[0, 0, 5].abs().sum().item()
    assert sel_delta > 1e-4, (
        f"2-D row offset did not reach idx 9; delta@5 = {sel_delta}"
    )


def test_candidate_window():
    br1 = _branch(window_size=1)  # S = 1
    br5 = _branch(window_size=5)  # S = 25
    x = torch.randn(1, 2, N_PATCH, D)
    top = torch.full((1, 2, 1), 5, dtype=torch.long)
    d1, _, _, _ = br1(x, top)
    d5, _, _, _ = br5(x, top)
    v1, v5 = d1[0, 0, 5], d5[0, 0, 5]
    assert not torch.allclose(v1, v5, atol=1e-5), (
        "window=1 and window=5 identical: candidate window inactive"
    )
    assert v1.abs().sum().item() > 0 and v5.abs().sum().item() > 0
    assert d1.shape == (1, 2, N_PATCH, D)
    assert d5.shape == (1, 2, N_PATCH, D)


def test_sparsity():
    br = _branch(window_size=3)
    x = torch.randn(1, 3, N_PATCH, D)
    top = torch.tensor([[[2, 5, 8]] * 3], dtype=torch.long)
    delta, _, _, _ = br(x, top)
    mask = torch.zeros(1, 3, N_PATCH, dtype=torch.bool)
    mask.scatter_(-1, top, True)
    unsel = delta[~mask.unsqueeze(-1).expand_as(delta)]
    assert torch.all(unsel == 0), "unselected positions are not zero"


def test_offset_gradients():
    br = _branch(window_size=3)
    x = torch.randn(2, 4, N_PATCH, D)
    top = torch.randint(0, N_PATCH, (2, 4, 3))
    delta, _, _, _ = br(x, top)
    delta.float().pow(2).sum().backward()
    any_grad = False
    for name, p in br.offset_net.named_parameters():
        assert p.grad is not None, f"offset_net.{name} has no gradient"
        any_grad = any_grad or bool(p.grad.abs().sum().item() > 0)
    assert any_grad, "offset_net has no non-zero gradient"


def test_sanitize_real():
    blk = _block()
    y = torch.tensor(
        [[[-5.0, 0.1, -1e38, 1e38, float("nan"), float("inf"), -3.2, 2.0]]]
    )
    out = blk._sanitize_real(y)
    assert _finite(out), "sanitize left NaN/Inf behind"
    assert out[0, 0, 0] < 0, "negative value not preserved"
    assert out[0, 0, 6] < 0, "negative outlier not preserved"
    assert out[0, 0, 1] > 0, "small positive value flipped"
    assert float(out[0, 0, 3].abs()) <= 100.0, "positive outlier not clamped"
    assert float(out[0, 0, 4]) == 0.0, "NaN not zeroed"
    assert float(out[0, 0, 5]) == 100.0, "Inf not mapped to +limit"
    assert float(out[0, 0, 2]) == -100.0, "-1e38 not clamped to -limit"


def test_input_not_mutated():
    blk = _block()
    b, t, n_tok = 2, 4, 1 + N_PATCH
    x = torch.randn(b * t, n_tok, D)
    attn = torch.rand(b * t, N_PATCH)
    x_copy = x.clone()
    with torch.no_grad():
        blk(x, attn, num_frames=t, patch_start=1)
    assert torch.equal(x, x_copy), "module mutated its input in place"


def test_skip_identity():
    blk = _block()
    x = torch.randn(2, 1 + N_PATCH, D)
    attn = torch.rand(2, N_PATCH)
    with torch.no_grad():
        out = blk(x, attn, num_frames=2, patch_start=1)  # 2 < min_frames=4
    assert torch.equal(out, x), "skip path is not identity"
    assert isinstance(out, Tensor)


def test_fp16_dtype():
    br = _branch(window_size=3)
    x = torch.randn(1, 2, N_PATCH, D).half()
    top = torch.randint(0, N_PATCH, (1, 2, 2))
    delta, _, _, _ = br(x, top)
    assert delta.dtype == torch.float16, f"delta dtype {delta.dtype} != fp16"
    assert _finite(delta)


def test_info_keys():
    blk = _block()
    b, t, n_tok = 2, 4, 1 + N_PATCH
    x = torch.randn(b * t, n_tok, D)
    attn = torch.rand(b * t, N_PATCH)
    with torch.no_grad():
        out, info = blk(x, attn, num_frames=t, patch_start=1,
                        return_information=True)
    assert isinstance(out, Tensor) and isinstance(info, dict)
    required = {
        "asa/off_time_abs", "asa/off_time_max",
        "asa/off_h_abs", "asa/off_h_max",
        "asa/off_w_abs", "asa/off_w_max",
        "cls/delta_abs_mean", "info",
    }
    missing = required - set(info.keys())
    assert not missing, f"missing info keys: {sorted(missing)}"


def main():
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    failed = []
    for fn in tests:
        try:
            fn()
            print(f"[PASS] {fn.__name__}")
        except Exception as exc:  # noqa: BLE001
            failed.append(fn.__name__)
            print(f"[FAIL] {fn.__name__}: {exc}")
    print(f"\n{len(tests) - len(failed)}/{len(tests)} passed")
    if failed:
        print("failed:", ", ".join(failed))
        sys.exit(1)


if __name__ == "__main__":
    main()
