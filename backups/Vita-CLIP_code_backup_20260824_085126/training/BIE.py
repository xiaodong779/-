"""BIE: Bidirectional Interpolation Enhancement for supervised video action recognition.

Placement
---------
BIE runs after the frozen CLIP backbone heads, exactly between feature
normalization and the final cosine matmul:

    V (video, [B, D])  : time_embed + frequency adapter output, L2-normalized
    Q (text,  [C, D])  : prompt learning + TextMSTFF class embeddings, L2-normalized,
                         used as classifier weights

Both modalities enhance each other *before* the logit_scale * cosine match.
Unlike MIP-CLIP's DFR (which *hard-adds* text reconstructions to the video:
``V + sum_c p_c * Q_c``), BIE uses *soft interpolation* between the two
modalities, so every enhanced feature stays on the segment between the
original features and can never be dragged outside the shared embedding
ball, and every injection path starts from zero (identity start) so the
already-converged adapters are not disturbed.

Forward contract
----------------
    logits_text = exp(logit_scale) * einsum("bd,bcd->bc", V_, Q_)
    p  = softmax(p_temperature * (V @ Q.T))                 # (B, C) classification result
    Q_ = normalize(Q + g_t * p.detach() * (V - Q))          # (B, C, D) video -> text
    V_ = normalize(V + g_v * (p.detach() @ Q - V))          # (B, D)   text -> video
    g_t : zero-init scalar gate (max_gate * tanh(theta), theta init 0)
    g_v : zero-init dynamic gate driven by p (max_gate * tanh(MLP(p)))

Design rationale (vs MIP-CLIP's DFR)
------------------------------------
* Reconstruction material: MIP-CLIP reconstructs the video from *text*
  (``A @ Q``), so the result is trapped in the subspace spanned by the frozen
  class texts and inherits the text-domain bias. BIE's video enhancement
  ``V + g_v * (p @ Q - V)`` is an *interpolation* between V and the predicted
  text centroid: the enhanced video always lies on the segment [V, p @ Q],
  and g_v (bounded, zero-init, driven by p) decides how far to move. The
  text enhancement ``Q + g_t * p * (V - Q)`` only moves the classes the model
  actually considers likely (p_c large); unlikely classes keep their text.
* p is detached in *both* directions (MIP-CLIP detaches ``visual_logits``
  too): the loop V -> p -> Qhat -> V_ is broken, and V's gradients come from
  the structural interpolation terms only -- stable and faithful to the
  "detach" contract of the supervised design.
* Identity start: g_t == 0 and g_v == 0 at init, so step 0 is exactly the
  baseline; with ``gate_warmup_steps > 0`` the injection ramps in linearly.
* Optional per-direction transform MLPs (``use_transform``): a residual
  ``D -> bottleneck -> D`` block with a zero-init output layer, i.e. identity
  at start, giving the module extra expressiveness once the gates wake up.

Loss contract
-------------
    loss = CE(label_smoothing=0.1)   (no auxiliary losses)

All math runs in float32 (project `_fp32_linear` convention), safe under AMP.
"""

from __future__ import annotations

import math
from typing import Dict, Optional

import torch
from torch import Tensor, nn
import torch.nn.functional as F


def _fp32_linear(module: nn.Linear, x: Tensor) -> Tensor:
    """Execute Linear in float32 while preserving gradient flow."""
    weight = module.weight.float()
    bias = None if module.bias is None else module.bias.float()
    return F.linear(x.float(), weight, bias)


def _l2_normalize(x: Tensor, eps: float = 1e-6) -> Tensor:
    return F.normalize(x.float(), dim=-1, eps=eps)


class _ZeroInitLinear(nn.Linear):
    """Linear whose output is exactly zero at initialization (zero weight+bias)."""

    def __init__(self, in_features: int, out_features: int, bias: bool = True) -> None:
        super().__init__(in_features, out_features, bias=bias)
        nn.init.zeros_(self.weight)
        if self.bias is not None:
            nn.init.zeros_(self.bias)


class _ScalarGate(nn.Module):
    """g = max_gate * tanh(learnable_scalar); scalar init 0 -> g == 0 at start.

    ``max_gate`` caps the injection strength so the frozen CLIP feature keeps
    the dominant weight in V_ / Q_ (the interpolation never becomes a full
    replacement).
    """

    def __init__(self, max_gate: float = 0.5) -> None:
        super().__init__()
        self.max_gate = float(max_gate)
        self.logit = nn.Parameter(torch.zeros(1))

    def forward(self, x: Tensor) -> Tensor:
        # x unused; returns shape (1,), broadcastable against (B, C, D).
        return torch.tanh(self.logit.float()) * self.max_gate


class _DynamicGate(nn.Module):
    """g = max_gate * tanh(MLP(x)); MLP output layer zero-init -> g == 0 at start.

    IMPORTANT: no relu here. ``relu(tanh(z))`` has a *zero derivative at
    z == 0*, and the output layer is zero-inited, so z stays exactly 0 and the
    gate would never receive gradients (permanent lock-up of the whole video
    enhancement path). ``tanh`` has tanh'(0) = 1, so the gate wakes up
    immediately, stays bounded in [-max_gate, max_gate] (a small negative
    value means "move slightly *away* from the predicted text centroid",
    which the loss can exploit or discard; it cannot diverge).
    """

    def __init__(self, in_dim: int, bottleneck: int, max_gate: float = 0.5) -> None:
        super().__init__()
        self.max_gate = float(max_gate)
        self.fc1 = nn.Linear(in_dim, bottleneck)
        self.fc2 = _ZeroInitLinear(bottleneck, 1)

    def forward(self, x: Tensor) -> Tensor:
        h = F.gelu(_fp32_linear(self.fc1, x))
        return torch.tanh(_fp32_linear(self.fc2, h)) * self.max_gate  # (..., 1)


class _TransformNet(nn.Module):
    """Residual ``D -> bottleneck -> D`` transform with identity start.

    The output layer is zero-inited, so ``out = x + MLP(x)`` is exactly ``x``
    at init. Gradients reach fc2 through the residual bypass
    (dL/dx flows even when MLP(x) == 0), so the block wakes up once the
    gates do; fc1 only starts moving after fc2, which is fine -- it is the
    standard zero-init residual adapter behavior (no permanent lock-up:
    dL/dW2 = dL/dout . h.T != 0 because dL/dout flows from the bypass).
    """

    def __init__(self, dim: int, bottleneck: int) -> None:
        super().__init__()
        self.fc1 = nn.Linear(dim, bottleneck)
        self.fc2 = _ZeroInitLinear(bottleneck, dim)

    def forward(self, x: Tensor) -> Tensor:
        return x + _fp32_linear(self.fc2, F.gelu(_fp32_linear(self.fc1, x)))


class BIE(nn.Module):
    """Bidirectional Interpolation Enhancement.

    Args:
        num_classes: number of classes C.
        dim: feature dimension D (512 for Vita-CLIP).
        p_temperature: fixed temperature of the classification softmax
            ``softmax(p_temperature * V @ Q.T)`` that produces the soft class
            weights p. The shared logit_scale (~100) saturates the softmax
            into one-hot and would make the interpolation either 0 or 1 for
            every class (plus starve the p path of gradients); 10.0 keeps p
            soft while preserving class separability (same value CMSI uses
            for its attention temperature).
        max_gate: upper bound of both injection gates (identity start stays 0).
        gate_warmup_steps: number of steps over which the gates ramp from 0 to
            1 (linear warmup). 0 disables warmup. At eval the scale is always 1.
        use_transform: enable the per-direction residual transform MLPs
            (identity at start). When False the module is pure interpolation.
        transform_bottleneck: hidden dim of the transform MLPs.
        gate_bottleneck: hidden dim of the dynamic video gate.
        external_logit_scale: optional shared logit_scale parameter (reused
            from the parent CLIP model so a single temperature is trained).
        eps: L2-normalization epsilon.
    """

    def __init__(
        self,
        num_classes: int,
        dim: int,
        p_temperature: float = 10.0,
        max_gate: float = 0.5,
        gate_warmup_steps: int = 0,
        use_transform: bool = True,
        transform_bottleneck: int = 128,
        gate_bottleneck: int = 128,
        external_logit_scale: Optional[nn.Parameter] = None,
        eps: float = 1e-6,
    ) -> None:
        super().__init__()

        if num_classes <= 0:
            raise ValueError("num_classes must be positive.")
        if dim <= 0:
            raise ValueError("dim must be positive.")
        if p_temperature is None or p_temperature <= 0:
            raise ValueError("p_temperature must be a positive float.")

        self.num_classes = int(num_classes)
        self.dim = int(dim)
        self.p_temperature = float(p_temperature)
        self.max_gate = float(max_gate)
        self.gate_warmup_steps = int(gate_warmup_steps)
        self.use_transform = bool(use_transform)
        self.eps = float(eps)

        # --- Gates (created unconditionally: both directions are always on) ---
        # Text enhancement gate: zero-init scalar -> identity start, then the
        # per-class interpolation strength p_c does the fine-grained work.
        self.gate_t = _ScalarGate(max_gate=self.max_gate)
        # Video enhancement gate: driven by the classification result p
        # (confidence-conditioned), zero-init -> identity start.
        self.gate_v = _DynamicGate(
            self.num_classes, gate_bottleneck, max_gate=self.max_gate
        )

        # --- Per-direction residual transforms (optional, identity start) ---
        self.mlp1: Optional[nn.Module] = None  # text side
        self.mlp2: Optional[nn.Module] = None  # video side
        if self.use_transform:
            self.mlp1 = _TransformNet(self.dim, transform_bottleneck)
            self.mlp2 = _TransformNet(self.dim, transform_bottleneck)

        # --- Classification ---
        if external_logit_scale is not None:
            # Reuse the parent CLIP model's logit_scale so that a single
            # shared, trainable temperature scalar is used (and still subject
            # to the trainer's clamp_logit_scale). object.__setattr__ avoids
            # registering it again as a child parameter of this module.
            object.__setattr__(self, "logit_scale", external_logit_scale)
        else:
            self.logit_scale = nn.Parameter(
                torch.tensor(math.log(100.0), dtype=torch.float32)
            )

    # ------------------------------------------------------------------ #
    #  Forward                                                            #
    # ------------------------------------------------------------------ #
    def forward(
        self,
        V: Tensor,
        Q: Tensor,
        y: Optional[Tensor] = None,
        step: Optional[int] = None,
    ) -> Dict[str, object]:
        """V: (B, D), Q: (C, D), y: (B,) optional labels (unused), step: global step.

        ``step`` drives the gate warmup scale during training
        (``min(1, step / gate_warmup_steps)``); None + training means warmup
        is treated as "not started yet" (scale 0), None + eval means scale 1.

        Returns: {"logits", "V_", "Q_", "gate_v", "gate_t", "gate_scale"}
        """
        if V.dim() != 2 or Q.dim() != 2:
            raise ValueError(f"V/Q must be 2-D, got {tuple(V.shape)} / {tuple(Q.shape)}.")
        if V.shape[1] != self.dim or Q.shape[1] != self.dim:
            raise ValueError(
                f"Feature dim mismatch: V={V.shape[1]}, Q={Q.shape[1]}, expected {self.dim}."
            )
        if Q.shape[0] != self.num_classes:
            raise ValueError(
                f"Q rows {Q.shape[0]} != num_classes {self.num_classes}."
            )

        video = _l2_normalize(V, self.eps)  # (B, D)
        text = _l2_normalize(Q, self.eps)  # (C, D)
        B = video.shape[0]

        # --- Gate warmup: ramp injection strength from 0 during early training ---
        # Eval always uses the full injection; during training the scale is
        # min(1, step / gate_warmup_steps) so the first steps are exactly the
        # baseline behavior.
        gate_scale = 1.0
        if self.training and self.gate_warmup_steps > 0:
            if step is None:
                gate_scale = 0.0
            else:
                gate_scale = min(1.0, float(step) / float(self.gate_warmup_steps))

        # --- Classification result p (the driver of both directions) ---
        # Fixed dedicated temperature: logit_scale (~100) would saturate the
        # softmax into one-hot and starve the p path of gradients (same
        # rationale as CMSI's v2t_attn_temperature). p is detached in BOTH
        # directions, mirroring MIP-CLIP's ``visual_logits.detach()``: this
        # breaks the video -> p -> (qhat, g_v) -> video_enh self-loop, so the
        # video's gradients come only from the structural interpolation
        # terms (stable).
        p = torch.softmax(
            self.p_temperature * (video @ text.t()), dim=-1
        )  # (B, C)
        p_d = p.detach()

        # --- Video -> text: soft interpolation toward the video, only for
        #     likely classes ---
        # text_enh[b, c] = text[c] + g_t * p[b, c] * (video[b] - text[c])
        #                = (1 - g_t * p[b, c]) * text[c] + g_t * p[b, c] * video[b]
        # Every row is a convex combination of its original text and the video
        # (g_t capped by max_gate and bounded by 1), so the enhanced text can
        # never leave the segment [text[c], video[b]].
        gate_t = self.gate_t(video) * gate_scale  # (1,)
        diff = video.unsqueeze(1) - text.unsqueeze(0)  # (B, C, D)
        text_enh = text.unsqueeze(0) + gate_t * p_d.unsqueeze(-1) * diff  # (B, C, D)
        text_enh = _l2_normalize(text_enh, self.eps)
        if self.mlp1 is not None:
            text_enh = _l2_normalize(
                self.mlp1(
                    text_enh.reshape(B * self.num_classes, self.dim)
                ).reshape(B, self.num_classes, self.dim),
                self.eps,
            )

        # --- Text -> video: interpolation toward the predicted text centroid ---
        # video_enh = video + g_v * (p @ text - video)
        #           = (1 - g_v) * video + g_v * (p @ text)
        # g_v is driven by p (confidence-conditioned) and zero-init, so the
        # video starts at the baseline and the network learns how far to move
        # toward the text centroid of its own prediction.
        qhat = p_d @ text  # (B, D) predicted text centroid (no grad through p)
        gate_v = self.gate_v(p_d) * gate_scale  # (B, 1)
        video_enh = video + gate_v * (qhat - video)  # (B, D)
        video_enh = _l2_normalize(video_enh, self.eps)
        if self.mlp2 is not None:
            video_enh = _l2_normalize(self.mlp2(video_enh), self.eps)

        # --- Classification ---
        scale = torch.exp(self.logit_scale.float())
        # text_enh is sample-dependent (B, C, D):
        # logits[b, c] = scale * <video_enh[b], text_enh[b, c]>
        logits = scale * torch.einsum("bd,bcd->bc", video_enh, text_enh)  # (B, C)

        return {
            "logits": logits,
            "V_": video_enh,
            "Q_": text_enh,
            "gate_v": gate_v,
            "gate_t": gate_t,
            "gate_scale": gate_scale,
        }
