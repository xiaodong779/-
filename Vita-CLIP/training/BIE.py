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
    V_ = normalize(V + g_v * CrossAttn(V, Q))               # (B, D)   text -> video
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
import warnings
from typing import Dict, Optional, Tuple

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


class _SigmoidGate(nn.Module):
    """Non-negative video-side gate: g = max_gate * sigmoid(MLP(x) + bias).

    Unlike ``_DynamicGate`` (tanh -> [-max_gate, max_gate]), this gate outputs
    only [0, max_gate]: the video enhancement *adds* collected text context and
    never subtracts it. This is intentional -- the empirical BIE runs showed the
    video direction kept learning a *negative* gate (model preferred to move
    *away* from the text context), meaning the "pull toward text" direction was
    rejected. With a non-negative gate, the video can only *absorb* the
    collected text semantics (matching MIP-CLIP's DFR ``V + alpha*text`` add),
    so it cannot cancel the enhancement by flipping sign.

    The logit_bias starts negative (sigmoid ~ 0) so the gate starts at ~0
    (identity start); it is learnable so the model can increase the injection
    as it learns the collected context is useful. fc2 is zero-inited so the MLP
    output is 0 at start and gate == sigmoid(init_logit)*max_gate (small).
    """

    def __init__(
        self,
        in_dim: int,
        bottleneck: int,
        max_gate: float = 0.5,
        init_logit: float = -5.0,
    ) -> None:
        super().__init__()
        self.max_gate = float(max_gate)
        self.fc1 = nn.Linear(in_dim, bottleneck)
        self.fc2 = _ZeroInitLinear(bottleneck, 1)
        self.logit_bias = nn.Parameter(torch.tensor([float(init_logit)]))

    def forward(self, x: Tensor) -> Tensor:
        h = F.gelu(_fp32_linear(self.fc1, x))
        z = _fp32_linear(self.fc2, h) + self.logit_bias
        return torch.sigmoid(z) * self.max_gate  # (..., 1) in [0, max_gate]


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


class _CrossAttentionTextCollect(nn.Module):
    """Video-side text collection via learnable-projection cross-attention.

    Replaces the old ``qhat = p @ text`` (classification-probability centroid),
    which the model learned to *reject* (gate_v ended at -0.161 in the first
    BIE run: the centroid ~= the predicted class text carries little new
    information). Here the video acts as a query that *selectively* collects
    discriminative context from all class texts through learned projections,
    so the video can pull in richer, more targeted text information.

        q = video @ Wq                       # (B, d)
        k = text @ Wk ; v = text @ Wv        # (C, d)
        attn = softmax( (q @ k.T) / sqrt(d) )  # (B, C)
        context = attn @ v                   # (B, d)
        residual = Wout(context - video)     # (B, D)

    ``Wout`` is given a small (std=1e-3) non-zero init rather than zero: a
    zero Wout would zero the residual and cut gradients to both gate_v and the
    collector, locking the branch dead. With std=1e-3 the initial delta is
    O(1e-3), keeping the start effectively at baseline while letting gate_v
    wake up immediately. The text is detached on the K/V path so the gradient
    loop ``video -> attn -> context -> video_enh -> loss -> video`` is broken
    while the *learned* projections Wk/Wv still receive gradients.
    Computations run in fp32 (project ``_fp32_linear`` convention).
    """

    def __init__(
        self,
        dim: int,
        num_classes: int,
        proj_dim: int = 128,
        temperature: float = 10.0,
        attn_dropout: float = 0.1,
        proj_dropout: float = 0.1,
        eps: float = 1e-6,
    ) -> None:
        super().__init__()
        self.dim = int(dim)
        self.num_classes = int(num_classes)
        self.proj_dim = int(proj_dim)
        self.temperature = float(temperature)
        self.eps = float(eps)

        # Dropout to regularize the cross-attention on small datasets: applied
        # to the attention weights (prevents over-reliance on a few classes)
        # and to the projected context before wout (prevents the collection
        # branch from memorizing).
        self.attn_drop = nn.Dropout(p=float(attn_dropout))
        self.proj_drop = nn.Dropout(p=float(proj_dropout))

        # Learnable cross-modal projections.
        self.wq = nn.Linear(dim, proj_dim, bias=False)
        self.wk = nn.Linear(dim, proj_dim, bias=False)
        self.wv = nn.Linear(dim, proj_dim, bias=False)
        # Project collected-text context back to the video feature dim. wout is
        # NOT zero-initialized (a zero wout would zero delta and cut gradients
        # to both gate_v and the collector). The input to wout is an
        # L2-normalized context (norm 1), so a wout std of 0.02 yields delta
        # norm ~ 0.02 * sqrt(dim) ~ 0.45: a *visible but bounded* additive
        # injection that the non-negative sigmoid gate can control (full gate
        # moves the unit-sphere video by ~0.2-0.3, keeping the consistency
        # regularizer meaningful without overwhelming the baseline). Earlier
        # attempts used std=1e-3 (delta ~1e-3, branch a no-op) and std=0.1
        # (delta ~2.2, over-injection).
        self.wout = nn.Linear(proj_dim, dim, bias=False)
        nn.init.normal_(self.wout.weight, std=0.02)

    def forward(
        self, video: Tensor, text: Tensor
    ) -> Tuple[Tensor, Tensor, Tensor]:
        """Return (residual [B, D], attn [B, C], context [B, proj_dim]).

        ``text`` must already be L2-normalized by the caller. The returned
        residual is the *projected* collected-text delta to add to video.
        """
        # Detach text on the K/V path to break the video->text->video loop,
        # while still letting the learned Wk/Wv projections be trained.
        t_detached = text.detach()

        q = _fp32_linear(self.wq, video)          # (B, d)
        k = _fp32_linear(self.wk, t_detached)     # (C, d)
        v = _fp32_linear(self.wv, t_detached)     # (C, d)

        # Scaled dot-product attention with a dedicated (soft) temperature.
        scale = self.temperature / math.sqrt(max(1, self.proj_dim))
        attn = torch.softmax(scale * (q @ k.t()), dim=-1)  # (B, C)
        attn = self.attn_drop(attn)               # (B, C) drop for regularization
        context = attn @ v                        # (B, d)
        context = self.proj_drop(context)         # (B, d) drop before wout

        # Project the collected-text context back to the video dimension.
        # NOTE: additive (delta = Wout(context)), NOT interpolative
        # (Wout(context - q)). The video enhancement becomes
        # ``video + gate_v * delta`` (MIP-CLIP DFR's ``V + alpha*text`` add),
        # which the empirical runs showed the model actually wants: the
        # interpolative ``+ gate*(context - video)`` version kept learning a
        # negative gate (rejecting the pull), whereas adding the collected
        # context is a pure information supplement the model cannot cancel.
        #
        # context is L2-normalized BEFORE wout: the raw context norm is tiny
        # (~0.03, since v = Wv(text) with kaiming-init Wv), so an unnormalized
        # Wout(context) made delta ~0.03 and the video side barely moved even at
        # full gate -- the consistency loss never engaged and the branch was a
        # no-op. Normalizing first lets wout's std (0.1) set a real, controllable
        # injection scale independent of the (arbitrary) projection norm.
        delta = _fp32_linear(self.wout, _l2_normalize(context, self.eps))  # (B, D)
        return delta, attn, context


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
        attn_temperature: temperature of the video->text cross-attention
            (default 10.0). Kept independent of logit_scale (~100) so the
            attention stays soft and gradients reach the attention weights.
        attn_proj_dim: hidden dim of the cross-attention Q/K/V projections.
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
        attn_temperature: float = 10.0,
        attn_proj_dim: int = 128,
        attn_dropout: float = 0.1,
        proj_dropout: float = 0.1,
        max_dist: float = 0.05,
        max_dist_t: float = 0.05,
        consistency_lambda: float = 0.1,
        enh_thr: float = 0.0,
        alpha_aux_lambda: float = 0.1,
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
        if max_dist < 0.0:
            raise ValueError("max_dist must be non-negative.")
        if max_dist_t < 0.0:
            raise ValueError("max_dist_t must be non-negative.")
        if consistency_lambda < 0.0:
            raise ValueError("consistency_lambda must be non-negative.")
        if not 0.0 <= attn_dropout < 1.0 or not 0.0 <= proj_dropout < 1.0:
            raise ValueError("dropout must be in [0, 1).")
        if enh_thr < 0.0:
            raise ValueError("enh_thr must be non-negative.")

        self.num_classes = int(num_classes)
        self.dim = int(dim)
        self.p_temperature = float(p_temperature)
        self.max_gate = float(max_gate)
        self.gate_warmup_steps = int(gate_warmup_steps)
        self.use_transform = bool(use_transform)
        self.attn_temperature = float(attn_temperature)
        self.attn_proj_dim = int(attn_proj_dim)
        self.attn_dropout = float(attn_dropout)
        self.proj_dropout = float(proj_dropout)
        self.max_dist = float(max_dist)
        self.max_dist_t = float(max_dist_t)
        self.consistency_lambda = float(consistency_lambda)
        self.enh_thr = float(enh_thr)
        self.alpha_aux_lambda = float(alpha_aux_lambda)
        self.eps = float(eps)

        # --- Independent visual classification head α (MIP-CLIP DFR style) ---
        # α = FC(LN(video)): a *separate, learnable* classifier on the video
        # feature, NOT the CLIP contrastive similarity `video @ text.T`. It
        # provides an independent "which class does this video belong to" signal
        # (raw confidence scores) that drives the text-side enhancement
        # weighting, decoupled from the final logits path. It is trained by the
        # auxiliary classification loss (alpha_aux_lambda * CE(alpha_logits, y)).
        self.alpha_head = nn.Sequential(
            nn.LayerNorm(self.dim),
            nn.Linear(self.dim, self.num_classes, bias=True),
        )

        # --- Gates (created unconditionally: both directions are always on) ---
        # Text enhancement gate: zero-init scalar -> identity start, then the
        # per-class interpolation strength p_c does the fine-grained work.
        self.gate_t = _ScalarGate(max_gate=self.max_gate)
        # Video enhancement gate: NON-NEGATIVE sigmoid gate driven by α.
        # Range [0, max_gate] so the video only *adds* collected text context
        # and can never learn a negative (rejecting) direction. sigmoid(MLP(α))
        # starts ~0 (identity start) via the negative logit_bias.
        self.gate_v = _SigmoidGate(
            self.num_classes, gate_bottleneck, max_gate=self.max_gate
        )
        # Video-side text collection: cross-attention with learnable
        # projections (replaces the old p-weighted centroid; identity start).
        self.attn_collect = _CrossAttentionTextCollect(
            dim=self.dim,
            num_classes=self.num_classes,
            proj_dim=self.attn_proj_dim,
            temperature=self.attn_temperature,
            attn_dropout=self.attn_dropout,
            proj_dropout=self.proj_dropout,
            eps=self.eps,
        )
        self._last_attn: Optional[Tensor] = None
        self._last_attn_max: float = 0.0
        self._last_consistency_loss: Optional[Tensor] = None
        self._last_alpha_aux_loss: Optional[Tensor] = None

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
            # Defensive: BIE owns a private temperature ONLY when used in
            # isolation (e.g. unit tests). In the real Vita-CLIP model the
            # parent always passes external_logit_scale, so exactly one
            # temperature is trained. Creating a private one here while a
            # parent also has one would silently train two temperatures, so we
            # warn loudly to catch accidental standalone usage.
            warnings.warn(
                "BIE created WITHOUT external_logit_scale: it owns a private "
                "logit_scale=log(100). When embedded in a CLIP model this "
                "should be passed the parent's logit_scale to keep a single "
                "shared temperature.",
                stacklevel=2,
            )
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

        # --- Image-side classification confidence α (MIP-CLIP DFR style) ---
        # α = alpha_head(video) is an independent, learnable classifier on the
        # video feature (FC(LN(video))), NOT the CLIP contrastive similarity
        # `video @ text.T`. It is the raw confidence score (MIP-CLIP DFR's
        # α = FC(LN(ev))), NOT a softmax distribution: the user explicitly wants
        # the confidence score, not a probability. For the text-side weighting,
        # low/negative scores naturally fall below enh_thr and keep their
        # prototype text, so only genuinely confident classes get enhanced.
        #
        # α is detached on the enhancement paths (mirroring MIP-CLIP's
        # ``visual_logits.detach()``) to break the video->α->enhancement loop.
        # To keep alpha_head meaningful it is trained by an *auxiliary
        # classification loss* on alpha_logits vs. the labels y when y is
        # provided (supervised training); in eval y is None so the head is
        # frozen-but-used.
        alpha_logits = self.alpha_head(video)  # (B, C) raw confidence scores
        α = alpha_logits  # (B, C) raw confidence scores (can be negative)
        α_d = α.detach()

        # --- Video -> text: soft interpolation toward the video, only for
        #     likely classes ---
        # text_enh[b, c] = text[c] + g_t * α[b, c] * (video[b] - text[c])
        #                = (1 - g_t * α[b, c]) * text[c] + g_t * α[b, c] * video[b]
        # The weighting uses the *independent* image-side confidence α (not the
        # CLIP contrastive similarity), so the text is pulled toward the video
        # only for the classes the independent classifier believes are present.
        # Every row is a convex combination of its original text and the video
        # (g_t capped by max_gate and bounded by 1), so the enhanced text can
        # never leave the segment [text[c], video[b]].
        gate_t = self.gate_t(video) * gate_scale  # (1,)
        diff = video.unsqueeze(1) - text.unsqueeze(0)  # (B, C, D)
        # α is the raw confidence score (can be negative). The text-side
        # weighting uses the *non-negative* part: classes with positive
        # confidence get their text pulled toward the video proportionally to
        # their confidence; classes with negative/low confidence (below
        # enh_thr) keep their prototype text untouched. This prevents both
        # reverse-enhancement (negative α) and irrelevant-class distortion.
        # ReLU on the detached α gates the *amplitude* without breaking the
        # gradient path of the surviving classes.
        α_w = torch.clamp(α_d, min=0.0)  # (B, C) non-negative confidence
        if self.enh_thr > 0.0:
            keep = (α_w > self.enh_thr).float().unsqueeze(-1)  # (B, C, 1)
            text_enh = (
                text.unsqueeze(0)
                + gate_t * α_w.unsqueeze(-1) * keep * diff
            )  # (B, C, D)
        else:
            text_enh = text.unsqueeze(0) + gate_t * α_w.unsqueeze(-1) * diff
        text_enh = _l2_normalize(text_enh, self.eps)
        if self.mlp1 is not None:
            text_enh = _l2_normalize(
                self.mlp1(
                    text_enh.reshape(B * self.num_classes, self.dim)
                ).reshape(B, self.num_classes, self.dim),
                self.eps,
            )

        # --- Text -> video: collect text context via cross-attention, then
        #     ADD it with a non-negative gate (MIP-CLIP DFR's V + α*text). ---
        # Replaces the old ``video + g_v*(p @ text - video)`` (interpolative pull
        # to the p-weighted centroid), which the model learned to reject
        # (gate_v went negative). Here the video collects text context through
        # learnable projections (delta = Wout(context)) and ADD it to the video:
        # ``video + gate_v * delta`` with gate_v in [0, max_gate] (sigmoid), so
        # the video can only *absorb* the collected semantics and can never flip
        # sign to cancel it. This matches DFR's additive reconstruction and lets
        # the gate stay positive (it controls how much context to absorb).
        delta, attn, _ = self.attn_collect(video, text)  # delta: (B, D)
        gate_v = self.gate_v(α_d) * gate_scale  # (B, 1) in [0, max_gate]
        video_enh = video + gate_v * delta  # (B, D)
        video_enh = _l2_normalize(video_enh, self.eps)
        if self.mlp2 is not None:
            video_enh = _l2_normalize(self.mlp2(video_enh), self.eps)

        # --- Consistency losses: keep enhanced features close to the originals ---
        # We penalize the *actual* movement (L2 distance on the unit sphere)
        # rather than the cosine: enhanced and original are both L2-normalized
        # and the collected context is directionally close to the original, so
        # the cosine stays ~0.999 even when the injection is large -- a cosine
        # threshold would be a no-op. The L2 distance (max 2 on the sphere) is a
        # faithful measure of how far the enhancement moved the feature; if it
        # exceeds the per-modality threshold we penalize linearly. This stops
        # the model exploiting gate freedom to arbitrarily distort features to
        # fit labels instead of learning real cross-modal alignment.
        # Both losses are clamped per-sample BEFORE averaging: clamping the
        # batch mean first would let a few badly-distorted samples be diluted by
        # the many well-behaved ones (the mean stays under the threshold and the
        # penalty is silently 0), so the regularizer would fail to constrain
        # outlier enhancements. consistency_lambda is folded in so the trainer
        # just adds the returned scalar.
        consistency_loss = torch.zeros((), device=video.device)
        if self.consistency_lambda > 0.0:
            # Video side: ||video_enh - video|| per sample.
            dist_v = (video_enh - video).norm(dim=-1)  # (B,)
            consistency_loss = consistency_loss + self.consistency_lambda * (
                torch.clamp(dist_v - self.max_dist, min=0.0).mean()
            )
            # Text side: ||text_enh[b,c] - text[c]|| per (sample, class) pair.
            # text and text_enh are both L2-normalized, so this is on the sphere.
            dist_t = (text_enh - text.unsqueeze(0)).norm(dim=-1)  # (B, C)
            consistency_loss = consistency_loss + self.consistency_lambda * (
                torch.clamp(dist_t - self.max_dist_t, min=0.0).mean()
            )
        self._last_consistency_loss = consistency_loss.detach()

        # --- Classification ---
        scale = torch.exp(self.logit_scale.float())
        # text_enh is sample-dependent (B, C, D):
        # logits[b, c] = scale * <video_enh[b], text_enh[b, c]>
        logits = scale * torch.einsum("bd,bcd->bc", video_enh, text_enh)  # (B, C)

        # Monitor the attention (detached): how soft/peak is the video's text
        # attention, and how much actual text context got collected.
        self._last_attn = attn.detach()
        self._last_attn_max = float(attn.detach().max(dim=-1).values.mean())

        # --- Auxiliary loss to train the independent α classifier ---
        # α is detached on the enhancement paths, so the only supervision for
        # alpha_head is this auxiliary classification loss on alpha_logits vs y
        # (when y is provided, i.e. supervised training). In eval y is None and
        # alpha_head is used as a frozen classifier. Weight is folded in here.
        alpha_aux_loss = torch.zeros((), device=video.device)
        if y is not None and self.alpha_aux_lambda > 0.0:
            alpha_aux_loss = self.alpha_aux_lambda * F.cross_entropy(
                alpha_logits, y
            )
        self._last_alpha_aux_loss = alpha_aux_loss.detach()

        return {
            "logits": logits,
            "V_": video_enh,
            "Q_": text_enh,
            "gate_v": gate_v,
            "gate_t": gate_t,
            "gate_scale": gate_scale,
            "attn_max": self._last_attn_max,
            "consistency_loss": consistency_loss,
            "alpha_aux_loss": alpha_aux_loss,
        }
