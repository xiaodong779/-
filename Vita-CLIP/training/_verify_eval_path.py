"""One-off eval-path verification for BIE + aggregation mixer (CPU).

Mirrors run_ucf101_rap_bie_mixer.sh exactly; runs the eval code path
(model(data) with no bie_step) and asserts BIE gate_scale == 1.0, monitors
populated, and logits shape correct. Deleted after use.
"""
import sys
import time

sys.path.insert(0, ".")

import torch

from VitaCLIP_model import VitaCLIP


def build_model():
    model = VitaCLIP(
        backbone_path="/hy-tmp/weights/ViT-B-16.pt",
        input_size=(224, 224),
        num_frames=8,
        num_classes=101,
        feature_dim=768,
        embed_dim=512,
        patch_size=(16, 16),
        num_heads=12,
        num_layers=12,
        mlp_factor=4,
        use_discriminative_module=True,
        dr_layers=(3, 5, 7, 9),
        dr_arch="rap",
        dr_top_k=12,
        dr_asa_heads=8,
        dr_asa_time_offset_max=1.0,
        dr_asa_space_offset_max=4.0,
        dr_asa_window_size=5,
        use_bie=True,
        bie_p_temperature=10.0,
        bie_max_gate=0.5,
        bie_gate_warmup_steps=100,
        bie_transform_bottleneck=128,
        bie_gate_bottleneck=128,
        use_aggregation_mixer=True,
        aggregation_mixer_kernel=3,
        use_text_prompt_learning=True,
        text_prompt_CSC=False,
        text_prompt_classes_path=(
            "/hy-tmp/UCF101/annotations/ucfTrainTestlist/classInd.txt"
        ),
    )
    return model


def main():
    torch.manual_seed(0)
    model = build_model()
    model.eval()  # <-- exactly what evaluate() does before iterating

    # Eval loader feeds [B*V, 3, T, H, W]; use a small B to keep CPU fast.
    video = torch.randn(2, 3, 8, 224, 224)

    with torch.no_grad():
        t0 = time.time()
        logits = model(video)  # no labels, no bie_step -> the eval path
        elapsed = time.time() - t0

    assert logits.shape == (2, 101), logits.shape
    gs = model._last_gate_scale
    assert gs is not None, "gate_scale monitor missing"
    assert float(gs) == 1.0, f"eval gate_scale must be 1.0, got {float(gs)}"
    assert model._last_gate_t is not None and model._last_gate_v is not None
    assert torch.isfinite(logits).all(), "non-finite logits in eval"

    # train mode without step must also not crash (gate_scale -> 0 path)
    model.train()
    with torch.no_grad():
        logits2 = model(video)
    assert logits2.shape == (2, 101)

    print(f"EVAL-PATH OK | logits {tuple(logits.shape)} | "
          f"gate_scale={float(gs):.4f} | gate_t={float(model._last_gate_t):.4f} | "
          f"gate_v(mean)={float(model._last_gate_v.detach().mean()):.4f} | "
          f"eval forward {elapsed:.1f}s | train-mode-no-step OK")


if __name__ == "__main__":
    main()
