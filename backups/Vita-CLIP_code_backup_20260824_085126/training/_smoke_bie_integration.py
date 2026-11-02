#!/usr/bin/env python
"""Model-level integration smoke for BIE (Bidirectional Interpolation Enhancement).

Verifies the VitaCLIP wiring for the BIE module:
  1. BIE instantiation with the BIE hyper-parameters
  2. forward(bie_step=...) from the training-loop signature
  3. runtime monitors (_last_gate_t / _last_gate_v / _last_gate_scale)
  4. identity start: step 0 (warmup not started) -> gate_scale == 0
  5. backward through the CE loss only (BIE has no auxiliary loss)
  6. gate/transform parameters receive gradients once the gate wakes up
  7. eval forward without bie_step (validation path) still works

Runs on CPU to avoid touching the busy GPU. One forward/backward is enough.
"""

from __future__ import annotations

import sys

import torch

sys.path.insert(0, "/hy-tmp/Vita-CLIP/training")
sys.path.insert(0, "/hy-tmp/Vita-CLIP")

from VitaCLIP_model import VitaCLIP  # noqa: E402

CLASSES_PATH = "/hy-tmp/Vita-CLIP/classes/ucf101_classes.txt"


def build_model(**overrides) -> VitaCLIP:
    kw = dict(
        input_size=(224, 224),
        num_frames=8,
        feature_dim=768,
        patch_size=(16, 16),
        num_heads=12,
        num_layers=12,
        mlp_factor=4.0,
        embed_dim=512,
        use_original_visual_prompts=False,
        use_summary_token=False,
        use_local_prompts=False,
        use_global_prompts=False,
        use_text_prompt_learning=True,
        text_context_length=77,
        text_vocab_size=49408,
        text_transformer_width=512,
        text_transformer_heads=8,
        text_transformer_layers=12,
        text_num_prompts=8,
        text_prompt_pos="end",
        text_prompt_init="",
        text_prompt_CSC=False,
        text_prompt_classes_path=CLASSES_PATH,
        use_fixed_text_branch=True,
        use_learnable_text_branch=True,
        use_text_mstff=True,
        fixed_text_template="a video of {}",
        zeroshot_evaluation=False,
        zeroshot_text_features_path="",
        # BIE
        use_bie=True,
        bie_p_temperature=10.0,
        bie_max_gate=0.5,
        bie_gate_warmup_steps=500,
        bie_use_transform=True,
        bie_transform_bottleneck=128,
        bie_gate_bottleneck=128,
    )
    kw.update(overrides)
    return VitaCLIP(**kw)


def check_finite(tag: str, t: torch.Tensor) -> None:
    assert torch.isfinite(t).all(), f"{tag} has NaN/Inf"
    print(f"  [{tag}] finite, shape={tuple(t.shape)}")


def main() -> None:
    device = "cpu"
    print(f"device={device}")

    print("\n=== 1. build model with BIE ===")
    model = build_model().to(device)
    assert model.bie is not None, "bie module not created"
    assert model.bie.mlp1 is not None and model.bie.mlp2 is not None, (
        "transform MLPs missing"
    )
    print(f"  bie: p_temperature={model.bie.p_temperature} "
          f"max_gate={model.bie.max_gate} "
          f"warmup_steps={model.bie.gate_warmup_steps} "
          f"transform={model.bie.use_transform}")

    print("\n=== 2. training forward bie_step=0 (identity start) ===")
    model.train()
    B, T = 2, 8
    x = torch.randn(B, 3, T, 224, 224, device=device)
    labels = torch.tensor([0, 1], dtype=torch.long, device=device)
    logits = model(x, labels, bie_step=0)
    check_finite("logits", logits)
    assert logits.shape[0] == B and logits.shape[1] == 101, logits.shape
    gs = getattr(model, "_last_gate_scale", None)
    assert gs is not None, "gate_scale monitor missing"
    # Warmup not started -> no injection -> output must be exactly the baseline
    # (identity start contract).
    assert float(gs) == 0.0, f"gate_scale at step 0 should be 0, got {gs}"
    print(f"  gate_scale={float(gs):.3f} (identity start OK)")

    print("\n=== 3. training forward bie_step=1000 (warmup over) ===")
    logits2 = model(x, labels, bie_step=1000)
    check_finite("logits2", logits2)
    gs2 = getattr(model, "_last_gate_scale", None)
    assert float(gs2) == 1.0, f"gate_scale should be 1 after warmup, got {gs2}"
    gt = getattr(model, "_last_gate_t", None)
    gv = getattr(model, "_last_gate_v", None)
    assert gt is not None and torch.isfinite(gt).all(), f"gate_t bad: {gt}"
    assert gv is not None and torch.isfinite(gv).all(), f"gate_v bad: {gv}"
    print(f"  gate_t={float(gt):.4f}  gate_v={float(gv.mean()):.4f}  "
          f"gate_scale={float(gs2):.3f}")

    print("\n=== 4. backward through CE only (BIE has no auxiliary loss) ===")
    loss = torch.nn.functional.cross_entropy(logits2.float(), labels)
    loss.backward()
    n_bie_grad = sum(
        1 for n, p in model.named_parameters()
        if "bie." in n and p.grad is not None and float(p.grad.abs().sum()) > 0
    )
    print(f"  bie params with non-zero grad: {n_bie_grad}")
    assert n_bie_grad >= 1, "no BIE parameter received gradient"
    gate_grad = any(
        ("gate_t" in n or "gate_v" in n)
        for n, p in model.named_parameters()
        if p.grad is not None and float(p.grad.abs().sum()) > 0
    )
    assert gate_grad, "no gate parameter has gradient"

    # With the gates awake (and warmup over) the transform MLPs must also
    # receive gradients through the residual bypass.
    print("\n=== 4b. gates awake -> transform MLP gradients ===")
    model.zero_grad()
    with torch.no_grad():
        model.bie.gate_t.logit.fill_(2.0)
    logits3 = model(x, labels, bie_step=2000)
    loss3 = torch.nn.functional.cross_entropy(logits3.float(), labels)
    loss3.backward()
    n_mlp_grad = sum(
        1 for n, p in model.named_parameters()
        if ("bie.mlp1" in n or "bie.mlp2" in n)
        and p.grad is not None
        and float(p.grad.abs().sum()) > 0
    )
    print(f"  transform MLP params with non-zero grad: {n_mlp_grad}")
    assert n_mlp_grad >= 2, "transform MLPs have no gradient once gates awake"

    print("\n=== 5. eval forward without bie_step (validation path) ===")
    model.eval()
    with torch.no_grad():
        logits_eval = model(x)
    check_finite("logits_eval", logits_eval)
    assert logits_eval.shape == logits.shape
    gs_eval = getattr(model, "_last_gate_scale", None)
    assert float(gs_eval) == 1.0, "eval gate_scale should be 1"
    print(f"  gate_scale={float(gs_eval):.3f} (full injection at eval)")

    print("\n[PASS] BIE integration smoke OK")


if __name__ == "__main__":
    main()
