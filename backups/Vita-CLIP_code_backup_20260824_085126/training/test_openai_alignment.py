#!/usr/bin/env python3
"""Stage-1/2 validation for the OpenAI-aligned VitaCLIP_model.py.

Run:
    cd /hy-tmp/Vita-CLIP/training
    CUDA_VISIBLE_DEVICES=0 python test_openai_alignment.py \
        --backbone /hy-tmp/weights/ViT-B-16.pt \
        --classes /hy-tmp/UCF101/annotations/ucfTrainTestlist/classInd.txt

The script checks:
  1. Correct module path and constructor signature.
  2. Official ViT-B/16 SHA-256.
  3. 12/12 visual and 12/12 text blocks loaded.
  4. Key parameter tensors exactly match the OpenAI checkpoint.
  5. Image, text, and 8-frame mean features align with OpenAI CLIP.
  6. Frequency adapters exist at blocks 3, 7, and 11.
  7. One real-shaped video forward/backward succeeds.
  8. New modules get gradients while the frozen CLIP backbone does not.
"""

from __future__ import annotations

import argparse
import hashlib
import inspect
from pathlib import Path
from typing import Iterable

import torch
import torch.nn.functional as F

import VitaCLIP_model
from VitaCLIP_model import VitaCLIP, tokenize, convert_weights


EXPECTED_SHA256 = (
    "5806e77cd80f8b59890b7e101eabd078d"
    "9fb84e6937f9e85e4ecb61988df416f"
)


def sha256sum(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def cosine_and_max_diff(
    left: torch.Tensor,
    right: torch.Tensor,
) -> tuple[float, float]:
    left = left.detach().float()
    right = right.detach().float()
    cosine = F.cosine_similarity(left, right, dim=-1).mean().item()
    max_diff = (left - right).abs().max().item()
    return cosine, max_diff


def assert_alignment(
    name: str,
    left: torch.Tensor,
    right: torch.Tensor,
    *,
    min_cosine: float = 0.99999,
    max_abs_diff: float = 1e-2,
) -> None:
    cosine, max_diff = cosine_and_max_diff(left, right)
    print(
        f"[ALIGN] {name:<12} cosine={cosine:.8f} "
        f"max_abs_diff={max_diff:.6e}"
    )
    if cosine < min_cosine or max_diff > max_abs_diff:
        raise AssertionError(
            f"{name} alignment failed: cosine={cosine}, "
            f"max_abs_diff={max_diff}."
        )


def assert_finite(name: str, tensor: torch.Tensor) -> None:
    if not bool(torch.isfinite(tensor).all()):
        raise AssertionError(f"{name} contains NaN or Inf.")


def has_nonzero_finite_gradient(
    named_parameters: Iterable[tuple[str, torch.nn.Parameter]],
) -> tuple[int, list[str]]:
    count = 0
    names: list[str] = []
    for name, parameter in named_parameters:
        gradient = parameter.grad
        if gradient is None:
            continue
        if not bool(torch.isfinite(gradient).all()):
            raise AssertionError(f"Gradient has NaN/Inf: {name}")
        if gradient.abs().max().item() > 0:
            count += 1
            names.append(name)
    return count, names


def make_model(
    *,
    backbone: str,
    classes: str,
    use_frequency_adapter: bool,
) -> VitaCLIP:
    return VitaCLIP(
        backbone_path=backbone,
        input_size=(224, 224),
        num_frames=8,
        num_classes=101,
        feature_dim=768,
        patch_size=(16, 16),
        num_heads=12,
        num_layers=12,
        embed_dim=512,
        use_original_visual_prompts=False,
        use_summary_token=False,
        use_local_prompts=False,
        use_global_prompts=False,
        use_frequency_adapter=use_frequency_adapter,
        frequency_layers=(3, 7, 11),
        share_frequency_adapter=False,
        frequency_bottleneck=192,
        frequency_residual_scale_init=0.01,
        # Stage-1 uses only the fixed handcrafted text branch.
        use_text_prompt_learning=False,
        use_fixed_text_branch=True,
        use_learnable_text_branch=False,
        use_text_mstff=False,
        fixed_text_template="a video of a person {}",
        text_prompt_classes_path=classes,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--backbone", required=True)
    parser.add_argument("--classes", required=True)
    args = parser.parse_args()

    backbone = Path(args.backbone).resolve()
    classes = Path(args.classes).resolve()

    print("===== Environment =====")
    print("torch:", torch.__version__)
    print("cuda available:", torch.cuda.is_available())
    print("imported model file:", Path(VitaCLIP_model.__file__).resolve())
    print("constructor:", inspect.signature(VitaCLIP.__init__))

    if not backbone.is_file():
        raise FileNotFoundError(backbone)
    if not classes.is_file():
        raise FileNotFoundError(classes)

    digest = sha256sum(backbone)
    print("backbone sha256:", digest)
    if digest != EXPECTED_SHA256:
        raise AssertionError(
            "The backbone SHA-256 is not the official OpenAI ViT-B/16 hash."
        )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(7)

    # ------------------------------------------------------------------
    # Stage 1: exact OpenAI backbone loading and numerical alignment.
    # ------------------------------------------------------------------
    print("\n===== Stage 1: OpenAI CLIP alignment =====")
    model = make_model(
        backbone=str(backbone),
        classes=str(classes),
        use_frequency_adapter=False,
    ).eval()

    print("missing keys:", model.backbone_missing_keys)
    print("unexpected keys:", model.backbone_unexpected_keys)

    if model.backbone_unexpected_keys:
        raise AssertionError(
            f"Unexpected keys: {model.backbone_unexpected_keys}"
        )
    if tuple(model.backbone_missing_keys) != ("visual.time_embed",):
        raise AssertionError(
            "With frequency adapters disabled, the only allowed missing "
            f"key is visual.time_embed; got {model.backbone_missing_keys}."
        )

    official_cpu = torch.jit.load(
        str(backbone),
        map_location="cpu",
    ).eval()
    official_state = official_cpu.state_dict()
    custom_state = model.state_dict()

    key_parameters = [
        "visual.conv1.weight",
        "visual.class_embedding",
        "visual.positional_embedding",
        "visual.transformer.resblocks.0.attn.in_proj_weight",
        "visual.transformer.resblocks.11.mlp.c_proj.weight",
        "visual.ln_post.weight",
        "visual.proj",
        "token_embedding.weight",
        "positional_embedding",
        "transformer.resblocks.0.attn.in_proj_weight",
        "transformer.resblocks.11.mlp.c_proj.weight",
        "ln_final.weight",
        "text_projection",
        "logit_scale",
    ]

    print("\n===== Parameter equality =====")
    for key in key_parameters:
        if key not in custom_state or key not in official_state:
            raise KeyError(f"Missing comparison key: {key}")
        max_diff = (
            custom_state[key].detach().float()
            - official_state[key].detach().float()
        ).abs().max().item()
        print(f"[PARAM] {key:<70} max_abs_diff={max_diff:.3e}")
        if max_diff != 0.0:
            raise AssertionError(f"Parameter mismatch: {key}")

    # The official OpenAI TorchScript graph internally casts image
    # inputs to FP16. Keep both models in FP16 for numerical alignment.
    # FollowAttention/Projection -> FP16,
    # while LayerNorm parameters remain FP32.
    model = model.to(device).eval()
    convert_weights(model)

    official = torch.jit.load(
        str(backbone),
        map_location=device,
    ).eval()

    images = torch.randn(
        2, 3, 224, 224,
        device=device,
        dtype=torch.float32,
    )
    tokens = tokenize(
        [
            "a video of a person playing guitar.",
            "a video of a person applying eye makeup.",
        ],
        context_length=77,
    ).to(device)

    with torch.no_grad():
        official_image = official.encode_image(images)
        custom_image = model.encode_image(images)
        official_text = official.encode_text(tokens)
        custom_text = model.encode_text(tokens, apply_mstff=False)

    assert_alignment("image", custom_image, official_image)
    assert_alignment("text", custom_text, official_text)

    video = torch.randn(
        1, 3, 8, 224, 224,
        device=device,
        dtype=torch.float32,
    )
    flat_frames = video.permute(
        0, 2, 1, 3, 4
    ).reshape(8, 3, 224, 224)

    with torch.no_grad():
        official_video = official.encode_image(
            flat_frames
        ).reshape(1, 8, 512).mean(dim=1)
        custom_video_frames = flat_frames.to(
            device=device,
            dtype=model.visual.conv1.weight.dtype,
        )
        custom_video = model.visual(
            custom_video_frames,
            num_frames=8,
            apply_temporal_modules=False,
        )

    assert_alignment("video mean", custom_video, official_video)
    print("[PASS] Stage 1 OpenAI alignment passed.")

    # Free GPU memory before the trainability test.
    del model, official, official_cpu
    if device.type == "cuda":
        torch.cuda.empty_cache()

    # ------------------------------------------------------------------
    # Stage 2: Frequency Adapter integration and gradient isolation.
    # ------------------------------------------------------------------
    print("\n===== Stage 2: Frequency Adapter integration =====")
    model = make_model(
        backbone=str(backbone),
        classes=str(classes),
        use_frequency_adapter=True,
    ).float().to(device)
    model.train()

    adapter_keys = list(model.visual.frequency_adapters.keys())
    print("frequency adapter keys:", adapter_keys)
    if adapter_keys != ["3", "7", "11"]:
        raise AssertionError(
            f"Expected ['3', '7', '11'], got {adapter_keys}."
        )

    trainable_names = model.trainable_parameter_names()
    print("trainable tensor count:", len(trainable_names))
    for name in trainable_names:
        print("  TRAIN:", name)

    batch_video = torch.randn(
        2, 3, 8, 224, 224,
        device=device,
        dtype=torch.float32,
    )
    labels = torch.tensor([0, 1], device=device)

    logits = model(batch_video)
    print("logits shape:", tuple(logits.shape))
    if logits.shape != (2, 101):
        raise AssertionError(
            f"Expected logits [2, 101], got {tuple(logits.shape)}."
        )
    assert_finite("logits", logits)

    loss = F.cross_entropy(logits, labels)
    print("loss:", float(loss.detach().cpu()))
    assert_finite("loss", loss)

    model.zero_grad(set_to_none=True)
    loss.backward()

    adapter_grad_count, adapter_grad_names = (
        has_nonzero_finite_gradient(
            (
                (name, parameter)
                for name, parameter in model.named_parameters()
                if name.startswith(
                    "visual.frequency_adapters."
                )
            )
        )
    )
    print(
        "frequency adapter tensors with non-zero gradient:",
        adapter_grad_count,
    )
    for name in adapter_grad_names[:20]:
        print("  GRAD:", name)

    if adapter_grad_count == 0:
        raise AssertionError(
            "No Frequency Adapter parameter received a non-zero gradient."
        )

    time_gradient = model.visual.time_embed.grad
    if time_gradient is None:
        raise AssertionError("time_embed received no gradient.")
    assert_finite("time_embed.grad", time_gradient)
    print(
        "time_embed max_abs_grad:",
        float(time_gradient.abs().max().item()),
    )

    forbidden_backbone_gradients = []
    for name, parameter in model.named_parameters():
        if (
            name.startswith("visual.")
            and not name.startswith("visual.frequency_adapters.")
            and name != "visual.time_embed"
            and parameter.grad is not None
            and parameter.grad.abs().max().item() > 0
        ):
            forbidden_backbone_gradients.append(name)

        if (
            (
                name.startswith("transformer.")
                or name.startswith("token_embedding.")
                or name == "positional_embedding"
                or name.startswith("ln_final.")
                or name == "text_projection"
            )
            and parameter.grad is not None
            and parameter.grad.abs().max().item() > 0
        ):
            forbidden_backbone_gradients.append(name)

    if forbidden_backbone_gradients:
        raise AssertionError(
            "Frozen CLIP backbone received gradients:\n  "
            + "\n  ".join(forbidden_backbone_gradients)
        )

    print("[PASS] Stage 2 Frequency Adapter integration passed.")
    print("\n[PASS] All alignment and integration checks completed.")


if __name__ == "__main__":
    main()