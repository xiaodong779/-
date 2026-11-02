"""Smoke test for _TemporalAggregationMixer (mean-before temporal Conv1d).

Checks:
1. Identity start: with scale=0 (init), forward(x) == x exactly.
2. Gradient flow: after scale is set non-zero, both conv1d.weight and
   scale receive gradients in one backward.
3. Model integration: parameters registered, trainable after
   _configure_trainable_parameters, and missing-key whitelist accepts them.
"""
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from VitaCLIP_model import _TemporalAggregationMixer, VitaCLIP  # noqa: E402

DEVICE = "cpu"
B, T, D = 2, 8, 512


def test_identity_start() -> None:
    mixer = _TemporalAggregationMixer(dim=D, kernel_size=3)
    mixer.eval()
    x = torch.randn(B, T, D, dtype=torch.float32)
    with torch.no_grad():
        out = mixer(x)
    assert torch.equal(out, x), (
        f"identity start broken: max|out - x| = "
        f"{(out - x).abs().max().item():.3e}"
    )
    print("[PASS] identity start: mixer(x) == x at init")


def test_gradient_flow() -> None:
    mixer = _TemporalAggregationMixer(dim=D, kernel_size=3)
    x = torch.randn(B, T, D, dtype=torch.float32, requires_grad=True)
    # Wake the gate so the conv path is active.
    mixer.scale.data.fill_(0.7)
    loss = mixer(x).pow(2).mean()
    loss.backward()
    assert mixer.conv1d.weight.grad is not None, "conv1d.weight has no grad"
    assert mixer.scale.grad is not None, "scale has no grad"
    assert mixer.conv1d.weight.grad.abs().sum().item() > 0
    print("[PASS] gradient flow: conv1d.weight and scale both get grad")


def test_model_integration() -> None:
    model = VitaCLIP(
        backbone_path="",
        num_frames=8,
        num_classes=101,
        use_discriminative_module=False,
        use_bie=False,
        use_aggregation_mixer=True,
        zeroshot_evaluation=False,
        text_prompt_classes_path=(
            "/hy-tmp/UCF101/annotations/"
            "ucfTrainTestlist/classInd.txt"
        ),
    )
    model._configure_trainable_parameters()
    names = set(model.trainable_parameter_names())
    mixer_params = [
        (name, param)
        for name, param in model.named_parameters()
        if "aggregation_mixer" in name
    ]
    assert mixer_params, "aggregation_mixer parameters not registered"
    for name, param in mixer_params:
        assert param.requires_grad, f"{name} not trainable"
        assert name in names, f"{name} missing from trainable names"
    # Missing-key whitelist accepts the new keys.
    for name, _ in mixer_params:
        assert model._allowed_backbone_missing_key(name), (
            f"{name} not whitelisted"
        )
    print(f"[PASS] model integration: {len(mixer_params)} mixer params "
          f"trainable + whitelisted")


if __name__ == "__main__":
    test_identity_start()
    test_gradient_flow()
    test_model_integration()
    print("ALL SMOKE TESTS PASSED")
