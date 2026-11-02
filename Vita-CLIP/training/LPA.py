"""LPA: Learnable Projection Alignment for supervised video action recognition.

Placement
---------
LPA runs after the frozen CLIP backbone heads, exactly between feature
normalization and the final cosine matmul (same contract as BIE / CMSI):

    V (video, [B, D])  : time_embed + frequency adapter output, L2-normalized
    Q (text,  [C, D])  : prompt learning + TextMSTFF class embeddings, L2-normalized,
                         used as classifier weights

Unlike BIE (bidirectional *interpolation* between the raw V and Q) and CMSI
(EMA-prototype semantic injection), LPA views the residual gap between the two
modalities as a *learnable-projection* problem: the frozen CLIP backbone embeds
both images and text in a *shared* space (via ``visual_projection`` /
``text_projection``), but the supervised video stream distorts V with
time_embed + adapters, so V drifts away from the text-manifold that Q still
lives on. LPA therefore learns explicit cross-modal projection layers that
re-align the two streams *before* classification, plus an EMA class-prototype
contrastive objective that keeps the projected text classifier aligned to the
true video class centroids.

Forward contract
----------------
    logits = exp(logit_scale) * <V_, Q_>
    Q_ = normalize(Proj_T2V(Q))                  # learnable text->video projector
    V_ = normalize(Proj_V2S(V))                  # learnable video->shared projector
    Both projectors start as identity (zero-init residual gate) so step 0 is
    exactly the baseline, and they are regularized to stay near identity.

Design rationale
----------------
* Learnable projection (CLIP's own structure): CLIP already aligns image and
  text by projecting both into one shared space. LPA makes this alignment
  *task-adaptive*: the text->video projector Proj_T2V pulls each class text into
  the (task) video manifold so the classifier weights are co-located with the
  video features; the video->shared projector Proj_V2S removes the domain shift
  introduced by the video-specific adapters. This is a direct, interpretable
  answer to "how do I align my text output vectors with my image output
  vectors".
* EMA class prototypes (CMSI / cross-modality prototype collaboration): the
  contrastive alignment loss pulls the projected text direction Q'_i toward the
  EMA class-video centroid P_i (positive) and repels it from the other class
  text directions Q_j (negatives, j != i). Unlike plain MSE -- which collapses
  the classifier inter-class margin -- InfoNCE keeps Q' close to the video class
  center *while* preserving discriminating separation from the other classes.
* Identity start (ReZero-style, BIE/CMSI convention): every projection delta is
  scaled by a zero-init gate, so at init V_ == V and Q_ == Q exactly; the
  already-converged adapters/prompts are not disturbed, and any regression is
  attributable to the projection.
* float32 math throughout (project ``_fp32_linear`` convention), safe under AMP.

References (all real, cited for design provenance)
--------------------------------------------------
* CLIP [Radford et al. 2021]  -- the shared embedding space + projection design
  that LPA re-uses for its projection layers.
* Vita-CLIP [Wasim et al., CVPR 2023] -- the host backbone (this repo).
* ActionCLIP [Wang et al., NeurIPS 2021] -- language-image pretrained models for
  video action recognition.
* MIP-CLIP [Gao et al., IEEE TMM 2025] -- cross-modal semantic injection /
  feature reconstruction that this repo's BIE/CMSI already build on.
* VLPA-CLIP [Pattern Recognition 2025] -- video language prompting + adapting.
* X-CLIP [Ni et al., AAAI 2022] -- cross-frame attention / text-video interplay.
* Cross-modality Prototype Collaboration (CPC, X-ReID [AAAI 2026]) -- the
  prototype-collaboration idea behind the EMA contrastive alignment below.

Loss contract
-------------
    total = CE(label_smoothing=0.1)
          + align_lambda * InfoNCE_align(Proj_T2V(Q), P)      # text->video
          + idt_lambda * identity_regularizer(V_, Q_)          # stay near CLIP
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


class _IdentityGate(nn.Module):
    """g = tanh(learnable_scalar); scalar init 0 -> g == 0 -> identity start.

    ``max_gate`` caps the projection delta so the frozen CLIP feature keeps the
    dominant weight in V_ / Q_ (the projection is a bounded residual, never a
    full replacement).
    """

    def __init__(self, max_gate: float = 0.5) -> None:
        super().__init__()
        self.max_gate = float(max_gate)
        self.logit = nn.Parameter(torch.zeros(1))

    def forward(self, x: Tensor) -> Tensor:
        # x unused; returns shape (1,), broadcastable against (B, D) / (C, D).
        return torch.tanh(self.logit.float()) * self.max_gate


class _LowRankProjector(nn.Module):
    """Residual ``D -> bottleneck -> D`` cross-modal projector (identity start).

    ``out = x + g * MLP(x)``, where g is a zero-init gate. At init the MLP delta
    is zeroed by the gate, so the projector is exactly the identity map. The
    bottleneck MLP gives the projector extra expressiveness once the gate wakes
    up. The output layer uses a *small random* init (not zero) so that the
    projector can learn even when used without a gate (defensive), while the
    gate alone already guarantees the identity start.
    """

    def __init__(
        self,
        dim: int,
        bottleneck: int,
        max_gate: float = 0.5,
        delta_std: float = 0.05,
        use_gate: bool = True,
    ) -> None:
        super().__init__()
        if dim <= 0:
            raise ValueError("dim must be positive.")
        if bottleneck <= 0:
            raise ValueError("bottleneck must be positive.")
        self.dim = int(dim)
        self.bottleneck = int(bottleneck)
        self.max_gate = float(max_gate)
        self.delta_std = float(delta_std)
        self.use_gate = bool(use_gate)

        self.fc1 = nn.Linear(dim, bottleneck)
        self.fc2 = nn.Linear(bottleneck, dim)
        # Small random output init: delta magnitude controls how fast the gate
        # wakes up (dL/dg is proportional to delta). Identity start is
        # guaranteed by the gate, so a small non-zero delta is safe and keeps
        # the projector trainable from step 1.
        nn.init.normal_(self.fc2.weight, std=float(delta_std))
        nn.init.zeros_(self.fc2.bias)

        if self.use_gate:
            self.gate = _IdentityGate(max_gate=self.max_gate)
        else:
            self.gate = None

    def forward(self, x: Tensor, gate_scale: float = 1.0) -> Tensor:
        h = F.gelu(_fp32_linear(self.fc1, x))
        delta = _fp32_linear(self.fc2, h)  # (..., D)
        if self.gate is not None:
            # identity start via zero-init gate; gate_scale ramps the injection
            # in during warmup (early steps are exactly the baseline).
            delta = delta * self.gate(x) * gate_scale
        return x + delta


class _CrossModalAttention(nn.Module):
    """Video->text cross-attention to collect discriminative class context.

    The video acts as a query that *selectively* attends to the projected class
    text directions, collecting the most relevant semantic context for the
    video's actual content. Used to enrich the video feature before it is
    projected into the shared space. This is a X-CLIP / ActionCLIP style
    text-video interplay.
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

        self.attn_drop = nn.Dropout(p=float(attn_dropout))
        self.proj_drop = nn.Dropout(p=float(proj_dropout))

        self.wq = nn.Linear(dim, proj_dim, bias=False)
        self.wk = nn.Linear(dim, proj_dim, bias=False)
        self.wv = nn.Linear(dim, proj_dim, bias=False)
        self.wout = nn.Linear(proj_dim, dim, bias=False)
        nn.init.normal_(self.wout.weight, std=0.02)

    def forward(self, video: Tensor, text: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        """Return (residual [B, D], attn [B, C], context [B, proj_dim]).

        ``text`` must already be L2-normalized by the caller.
        """
        # Detach text on the K/V path to break the video->text->video loop.
        t_detached = text.detach()

        q = _fp32_linear(self.wq, video)          # (B, d)
        k = _fp32_linear(self.wk, t_detached)     # (C, d)
        v = _fp32_linear(self.wv, t_detached)     # (C, d)

        scale = self.temperature / math.sqrt(max(1, self.proj_dim))
        attn = torch.softmax(scale * (q @ k.t()), dim=-1)  # (B, C)
        attn = self.attn_drop(attn)
        context = attn @ v                        # (B, d)
        context = self.proj_drop(context)

        delta = _fp32_linear(self.wout, _l2_normalize(context, self.eps))  # (B, D)
        return delta, attn, context


class LPA(nn.Module):
    """Learnable Projection Alignment.

    Args:
        num_classes: number of classes C.
        dim: feature dimension D (512 for Vita-CLIP).
        max_gate: upper bound of the projection gates (identity start stays 0).
        proj_bottleneck: hidden dim of the cross-modal projector MLPs.
        gate_warmup_steps: number of steps over which the projection gates ramp
            from 0 to 1 (linear warmup). 0 disables warmup. At eval always 1.
        use_text_proj: enable the learnable text->video projector.
        use_video_proj: enable the learnable video->shared projector.
        use_video_attn: enable the cross-attention text context collection on
            the video side (X-CLIP / ActionCLIP style).
        align_lambda: weight of the InfoNCE text->prototype alignment loss.
        idt_lambda: weight of the identity regularizer (keeps V_/Q_ near CLIP).
        proto_momentum: EMA momentum for class-video prototypes.
        align_temperature: temperature of the InfoNCE alignment loss.
        align_neg_num: number of negative classes in the alignment loss
            (0 or >= num_classes-1 uses ALL other classes).
        external_logit_scale: optional shared logit_scale parameter (reused
            from the parent CLIP model so a single temperature is trained).
        eps: L2-normalization epsilon.
    """

    def __init__(
        self,
        num_classes: int,
        dim: int,
        max_gate: float = 0.5,
        proj_bottleneck: int = 128,
        gate_warmup_steps: int = 0,
        use_text_proj: bool = True,
        use_video_proj: bool = True,
        use_video_attn: bool = True,
        align_lambda: float = 0.005,
        idt_lambda: float = 0.0,
        proto_momentum: float = 0.99,
        align_temperature: float = 10.0,
        align_neg_num: int = 0,
        attn_proj_dim: int = 128,
        attn_dropout: float = 0.1,
        proj_dropout: float = 0.1,
        external_logit_scale: Optional[nn.Parameter] = None,
        eps: float = 1e-6,
    ) -> None:
        super().__init__()

        if num_classes <= 0:
            raise ValueError("num_classes must be positive.")
        if dim <= 0:
            raise ValueError("dim must be positive.")
        if max_gate < 0.0:
            raise ValueError("max_gate must be non-negative.")
        if align_lambda < 0.0:
            raise ValueError("align_lambda must be non-negative.")
        if idt_lambda < 0.0:
            raise ValueError("idt_lambda must be non-negative.")
        if not 0.0 < proto_momentum < 1.0:
            raise ValueError("proto_momentum must be in (0, 1).")
        if align_temperature <= 0.0:
            raise ValueError("align_temperature must be positive.")
        if not 0.0 <= attn_dropout < 1.0 or not 0.0 <= proj_dropout < 1.0:
            raise ValueError("dropout must be in [0, 1).")

        self.num_classes = int(num_classes)
        self.dim = int(dim)
        self.max_gate = float(max_gate)
        self.proj_bottleneck = int(proj_bottleneck)
        self.gate_warmup_steps = int(gate_warmup_steps)
        self.use_text_proj = bool(use_text_proj)
        self.use_video_proj = bool(use_video_proj)
        self.use_video_attn = bool(use_video_attn)
        self.align_lambda = float(align_lambda)
        self.idt_lambda = float(idt_lambda)
        self.proto_momentum = float(proto_momentum)
        self.align_temperature = float(align_temperature)
        self.align_neg_num = int(align_neg_num)
        self.attn_proj_dim = int(attn_proj_dim)
        self.attn_dropout = float(attn_dropout)
        self.proj_dropout = float(proj_dropout)
        self.eps = float(eps)

        # --- Learnable cross-modal projectors (identity start) ---
        # Text->video projector: pulls each class text into the task video
        # manifold so the classifier weights co-locate with the video features.
        self.proj_t2v = _LowRankProjector(
            self.dim, self.proj_bottleneck, max_gate=self.max_gate
        )
        # Video->shared projector: removes the domain shift introduced by the
        # video-specific adapters, re-aligning V with the text manifold.
        self.proj_v2s = _LowRankProjector(
            self.dim, self.proj_bottleneck, max_gate=self.max_gate
        )

        # --- Cross-attention text context collection on the video side ---
        self.attn_collect = _CrossModalAttention(
            dim=self.dim,
            num_classes=self.num_classes,
            proj_dim=self.attn_proj_dim,
            temperature=10.0,
            attn_dropout=self.attn_dropout,
            proj_dropout=self.proj_dropout,
            eps=self.eps,
        )

        # --- EMA class-video prototypes (buffers, no grad) ---
        # NOTE (audit 2026-08): updated rank-locally, NOT DDP-synced. Under
        # nproc_per_node=1 (current setup) this is fine; on multi-GPU the EMA
        # diverges per rank. Sync before/after training if multi-GPU is enabled.
        self.register_buffer("proto", torch.zeros(self.num_classes, self.dim))
        self.register_buffer(
            "proto_count", torch.zeros(self.num_classes, dtype=torch.long)
        )

        # --- Classification ---
        if external_logit_scale is not None:
            object.__setattr__(self, "logit_scale", external_logit_scale)
        else:
            warnings.warn(
                "LPA created WITHOUT external_logit_scale: it owns a private "
                "logit_scale=log(100). When embedded in a CLIP model this "
                "should be passed the parent's logit_scale to keep a single "
                "shared temperature.",
                stacklevel=2,
            )
            self.logit_scale = nn.Parameter(
                torch.tensor(math.log(100.0), dtype=torch.float32)
            )

        self._last_attn_max: float = 0.0
        self._last_align_loss: Optional[Tensor] = None
        self._last_idt_loss: Optional[Tensor] = None

    # ------------------------------------------------------------------ #
    #  EMA prototypes (train only, no grad)                              #
    # ------------------------------------------------------------------ #
    @torch.no_grad()
    def _update_proto(self, V: Tensor, y: Tensor) -> None:
        """Update P[c] with EMA of the normalized mean of class-c videos."""
        onehot = F.one_hot(y, num_classes=self.num_classes).float()  # (B, C)
        sums = onehot.t() @ V.float()  # (C, D)
        counts = onehot.sum(0)  # (C,)
        present = counts > 0
        if not present.any():
            return

        new_mean = torch.zeros_like(self.proto)
        new_mean[present] = F.normalize(sums[present], dim=-1, eps=self.eps)

        first = present & (self.proto_count == 0)
        self.proto[first] = new_mean[first]
        self.proto_count[first] += 1

        ema = present & (~first)
        self.proto[ema] = F.normalize(
            self.proto[ema] * self.proto_momentum
            + (1.0 - self.proto_momentum) * new_mean[ema],
            dim=-1,
            eps=self.eps,
        )
        self.proto_count[ema] += 1

    def _proto_initialized(self) -> bool:
        return bool((self.proto_count > 0).any())

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
        """V: (B, D), Q: (C, D), y: (B,) optional labels, step: global step.

        ``step`` drives the projection gate warmup during training
        (``min(1, step / gate_warmup_steps)``); None + training means warmup is
        treated as "not started yet" (scale 0), None + eval means scale 1.

        Returns: {"logits", "V_", "Q_", "gate_scale", "align_loss",
                  "idt_loss", "attn_max"}
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
        device = video.device

        # --- Gate warmup: ramp projection strength from 0 during early training ---
        gate_scale = 1.0
        if self.training and self.gate_warmup_steps > 0:
            if step is None:
                gate_scale = 0.0
            else:
                gate_scale = min(1.0, float(step) / float(self.gate_warmup_steps))

        # --- Update EMA prototypes (train only, no grad) ---
        if self.training and y is not None:
            self._update_proto(video, y)

        # --- Video->shared projection + cross-attention text context ---
        if self.use_video_attn:
            delta, attn, _ = self.attn_collect(video, text)  # delta: (B, D)
            # Scale the cross-attn delta by gate_scale so the early warmup
            # steps (gate_scale == 0) are exactly the baseline: no context
            # injection until the projection ramps in.
            video_collected = _l2_normalize(
                video + gate_scale * delta, self.eps
            )
            self._last_attn_max = float(attn.detach().max(dim=-1).values.mean())
        else:
            video_collected = video
            self._last_attn_max = 0.0

        if self.use_video_proj:
            # gate_scale ramps the *residual* delta, preserving identity start.
            proj_v = self.proj_v2s(video_collected, gate_scale=gate_scale)
            V_ = _l2_normalize(proj_v, self.eps)
        else:
            V_ = video_collected

        # --- Text->video projection (learnable alignment of class weights) ---
        if self.use_text_proj:
            proj_q = self.proj_t2v(text, gate_scale=gate_scale)  # identity at init
            Q_ = _l2_normalize(proj_q, self.eps)
        else:
            Q_ = text

        # --- Identity regularizer: keep enhanced features near the CLIP originals ---
        idt_loss = torch.zeros((), device=device)
        if self.idt_lambda > 0.0:
            dist_v = (V_ - video).norm(dim=-1)  # (B,)
            dist_t = (Q_ - text).norm(dim=-1)  # (C,)
            idt_loss = self.idt_lambda * (dist_v.mean() + dist_t.mean())
        self._last_idt_loss = idt_loss.detach()

        # --- Classification ---
        scale = torch.exp(self.logit_scale.float())
        logits = scale * (V_ @ Q_.t())  # (B, C)

        # --- InfoNCE text->prototype alignment loss (train only) ---
        align_loss = torch.zeros((), device=device)
        if self.training and y is not None and self.align_lambda > 0.0:
            with torch.no_grad():
                onehot = F.one_hot(y, num_classes=self.num_classes).float()
                present = onehot.sum(0) > 0
            if present.any() and self._proto_initialized():
                align_loss = self.align_lambda * self._contrastive_align_loss(
                    Q_, self.proto, present, text
                )
        self._last_align_loss = align_loss.detach()

        return {
            "logits": logits,
            "V_": V_,
            "Q_": Q_,
            "gate_scale": gate_scale,
            "align_loss": align_loss,
            "idt_loss": idt_loss,
            "attn_max": self._last_attn_max,
        }

    # ------------------------------------------------------------------ #
    #  Losses                                                             #
    # ------------------------------------------------------------------ #
    def _contrastive_align_loss(
        self, Q_: Tensor, P: Tensor, present: Tensor, Q: Tensor
    ) -> Tensor:
        """InfoNCE-style alignment that preserves inter-class margin.

        For each class i present in the batch:
            anchor    = Q'_i   (projected text direction, has grad)
            positive  = P_i    (EMA video prototype, detached)
            negatives = Q_j    (j != i, *original* text directions)
        """
        idx = torch.nonzero(present, as_tuple=False).squeeze(1)  # (K,)
        K = idx.numel()
        if K == 0:
            return torch.zeros((), device=Q_.device)

        scale = float(self.align_temperature)
        Q_p = Q_[idx]                          # (K, D) anchors
        P_p = P[idx].detach()                  # (K, D) positives (no grad)
        pos = scale * (Q_p * P_p).sum(-1)      # (K,)
        neg = scale * (Q_p @ Q.t())            # (K, C) original text dirs
        neg = neg.scatter(1, idx.unsqueeze(1), float("-inf"))

        if 0 < self.align_neg_num < self.num_classes - 1:
            rand = torch.randint(
                0,
                self.num_classes,
                (K, self.align_neg_num),
                device=neg.device,
                dtype=torch.long,
            )
            neg = neg.gather(1, rand)
            neg = neg.masked_fill(rand == idx.unsqueeze(1), float("-inf"))

        logits = torch.cat([pos.unsqueeze(1), neg], dim=1)  # (K, 1 + N_neg)
        targets = torch.zeros(K, dtype=torch.long, device=logits.device)
        return F.cross_entropy(logits, targets)

    def compute_loss(
        self,
        logits: Tensor,
        y: Tensor,
        align_loss: Optional[Tensor] = None,
        idt_loss: Optional[Tensor] = None,
        label_smoothing: float = 0.1,
    ) -> Tensor:
        """Main loss: CE(label_smoothing) + align_loss + idt_loss."""
        ce = F.cross_entropy(logits.float(), y, label_smoothing=label_smoothing)
        total = ce
        if align_loss is not None:
            total = total + align_loss
        if idt_loss is not None:
            total = total + idt_loss
        return total
