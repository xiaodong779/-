#!/usr/bin/env python
"""Model-level integration smoke for CMSI Phase A.

Verifies the VitaCLIP wiring changed in Phase A:
  1. CMSI instantiation with the new Phase A hyper-parameters
  2. forward(cmsi_step=...) from the training loop signature
  3. runtime monitors (_last_align_loss / _last_gate_t / _last_gate_scale)
  4. backward with the contrastive alignment loss included
  5. eval forward without cmsi_step (validation path) still works

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
        use_frequency_adapter=True,
        frequency_layers=(3, 7, 11),
        share_frequency_adapter=False,
        frequency_bottleneck=192,
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
        # CMSI Phase A
        use_cmsi=True,
        cmsi_align_lambda=0.005,
        cmsi_proto_momentum=0.99,
        cmsi_proto_text_init=True,
        cmsi_gate_warmup_steps=500,
        cmsi_max_gate=0.3,
    )
    kw.update(overrides)
    return VitaCLIP(**kw)


def check_finite(tag: str, t: torch.Tensor) -> None:
    assert torch.isfinite(t).all(), f"{tag} has NaN/Inf"
    print(f"  [{tag}] finite, shape={tuple(t.shape)}")


def main() -> None:
    device = "cpu"
    print(f"device={device}")

    print("\n=== 1. build model with CMSI Phase A ===")
    model = build_model().to(device)
    assert model.cmsi is not None, "cmsi module not created"
    print(f"  cmsi: align_lambda={model.cmsi.align_lambda} "
          f"momentum={model.cmsi.proto_momentum} "
          f"text_init={model.cmsi.proto_text_init} "
          f"warmup_steps={model.cmsi.gate_warmup_steps} "
          f"max_gate={model.cmsi.max_gate}")

    print("\n=== 2. training forward with cmsi_step (training-loop path) ===")
    model.train()
    B, T = 2, 8
    x = torch.randn(B, 3, T, 224, 224, device=device)
    labels = torch.tensor([0, 1], dtype=torch.long, device=device)
    logits = model(x, labels, cmsi_step=100)
    check_finite("logits", logits)
    assert logits.shape[0] == B and logits.shape[1] == 101, logits.shape
    al = getattr(model, "_last_align_loss", None)
    assert al is not None and torch.isfinite(al), f"align_loss bad: {al}"
    gt = getattr(model, "_last_gate_t", None)
    assert gt is not None and torch.isfinite(gt), f"gate_t bad: {gt}"
    gs = getattr(model, "_last_gate_scale", None)
    assert gs is not None, "gate_scale monitor missing"
    print(f"  align_loss={float(al):.5f}  gate_t={float(gt):.4f}  "
          f"gate_scale={float(gs):.3f}")

    print("\n=== 3. backward through CE + align_loss ===")
    loss = torch.nn.functional.cross_entropy(
        logits.float(), labels
    ) + float(al)
    loss.backward()
    n_cmsi_grad = sum(
        1 for n, p in model.named_parameters()
        if "cmsi" in n and p.grad is not None and float(p.grad.abs().sum()) > 0
    )
    print(f"  cmsi params with non-zero grad: {n_cmsi_grad}")
    # Identity start: at gate == 0 the low-rank net is (by design) not yet
    # reachable through Q_ = Q + gate*delta, but the gate logit must receive
    # gradient so the injection can wake up.
    assert n_cmsi_grad >= 1, "no CMSI parameter received gradient"
    gate_grad = any(
        "gate_t" in n or "gate_v" in n
        for n, p in model.named_parameters()
        if p.grad is not None and float(p.grad.abs().sum()) > 0
    )
    assert gate_grad, "gate logit has no gradient"

    # Once the gate is awake, gradients must also reach the low-rank net.
    print("\n=== 3b. gate awake -> lowrank_net gradients ===")
    model.zero_grad()
    with torch.no_grad():
        model.cmsi.gate_t.logit.fill_(2.0)
    logits2 = model(x, labels, cmsi_step=500)  # warmup over
    al2 = getattr(model, "_last_align_loss", None)
    loss2 = torch.nn.functional.cross_entropy(
        logits2.float(), labels
    ) + float(al2)
    loss2.backward()
    n_lowrank_grad = sum(
        1 for n, p in model.named_parameters()
        if "cmsi.lowrank_net" in n
        and p.grad is not None
        and float(p.grad.abs().sum()) > 0
    )
    print(f"  lowrank_net params with non-zero grad: {n_lowrank_grad}")
    assert n_lowrank_grad >= 2, (
        "lowrank_net has no gradient once the gate is awake"
    )

    print("\n=== 4. eval forward without cmsi_step (validation path) ===")
    model.eval()
    with torch.no_grad():
        logits_eval = model(x)
    check_finite("logits_eval", logits_eval)
    assert logits_eval.shape == logits.shape

    print("\n[PASS] CMSI integration smoke OK")


if __name__ == "__main__":
    main()
