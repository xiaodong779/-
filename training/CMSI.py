"""CMSI: Cross-Modal Semantic Injection for supervised video action recognition.

Placement
---------
CMSI runs after the frozen CLIP backbone heads, exactly between feature
normalization and the final cosine matmul:

    V (video, [B, D])  : time_embed + frequency adapter output, L2-normalized
    Q (text,  [C, D])  : prompt learning + TextMSTFF class embeddings, L2-normalized,
                         used as classifier weights

The two modalities correct each other *before* the logit_scale * cosine match,
while every injection path starts from zero so the already-converged adapters
are not disturbed (identity start).

Forward contract
----------------
    logits_text = exp(logit_scale) * (V_ @ Q_.T)
    V_ = normalize(V + g_v * (softmax(V @ Q_.T * s) @ Q_ - V))   # V->T residual
    Q_ = normalize(Q + g_t * LowRankNet(normalize(P - Q)))        # T->V proto fix
    P  : EMA class-video prototypes, updated under torch.no_grad()

Loss contract
-------------
    loss = CE(label_smoothing=0.1) + align_lambda * contrastive_align(P, Q_)
           + hard_neg_weight * hard_negative_loss

contrastive_align: InfoNCE-style alignment loss. For each class i present in
the current batch, Q'_i is pulled toward P_i (positive, detached prototype)
and repelled from the other class text directions Q_j (negatives, j != i).
Unlike plain MSE -- which isotropically collapses the inter-class margins of
the classifier -- this keeps Q' close to the video class center *while*
preserving discriminating separation from the other text class directions.

Injection warmup
----------------
When ``gate_warmup_steps > 0``, the gates are scaled by
``min(1, step / gate_warmup_steps)`` during training: early steps behave
exactly like the baseline (identity injection), then the correction ramps in.
At evaluation time the scale is always 1 (full injection).

All math runs in float32 (project `_fp32_linear` convention), safe under AMP.

Design notes
------------
* Zero-init gates (scalar: tanh(param), param=0; dynamic: MLP output layer
  zero-init) guarantee g == 0 at start -> identity injection.
* EMA prototypes are buffers; updates never carry gradients.
* ``Q`` should be frozen or trained with a very low LR; CMSI params train
  normally.
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


class _LowRankNet(nn.Module):
    """D -> lowrank -> D for the T->V prototype correction.

    Identity start is guaranteed by the zero-init gates (g_t == 0 makes
    Q_ == Q exactly, regardless of delta). The output layer therefore uses
    a *small random* init by default: a strictly zero output layer would
    starve the whole T->V path of gradients (with g_t == 0 AND delta == 0
    both dQ_/dg_t == delta and dQ_/ddelta == g_t vanish -> permanent
    lock-up, nothing ever learns). Set ``zero_init=True`` to strictly
    follow an "initial Delta == 0" contract; T->V then stays frozen unless
    another gradient source appears.
    """

    def __init__(
        self,
        dim: int,
        lowrank_bottleneck: int,
        zero_init: bool = False,
        delta_std: float = 0.05,
    ) -> None:
        super().__init__()
        self.fc1 = nn.Linear(dim, lowrank_bottleneck)
        self.fc2 = nn.Linear(lowrank_bottleneck, dim)
        if zero_init:
            nn.init.zeros_(self.fc2.weight)
            nn.init.zeros_(self.fc2.bias)
        else:
            # Small random init. delta magnitude controls how fast g_t wakes up
            # (dL/dg_t is proportional to delta); identity start is guaranteed
            # by the zero-init gate regardless of this value. 0.05 -> |delta|~0.3
            # which wakes g_t within ~100 steps while staying gate-bounded.
            nn.init.normal_(self.fc2.weight, std=float(delta_std))
            nn.init.zeros_(self.fc2.bias)

    def forward(self, x: Tensor) -> Tensor:
        h = F.gelu(_fp32_linear(self.fc1, x))
        return _fp32_linear(self.fc2, h)


class _ScalarGate(nn.Module):
    """g = max_gate * tanh(learnable_scalar); scalar init 0 -> g == 0 at start.

    ``max_gate`` caps the injection strength so the frozen CLIP feature keeps
    the dominant weight in V_ / Q_ (Q_ is an interpolation between Q and the
    prototype-corrected direction, never a full replacement).
    """

    def __init__(self, max_gate: float = 0.5) -> None:
        super().__init__()
        self.max_gate = float(max_gate)
        self.logit = nn.Parameter(torch.zeros(1))

    def forward(self, x: Tensor) -> Tensor:
        # x unused; returns shape (1,), broadcastable against (B, D) / (C, D).
        return torch.tanh(self.logit.float()) * self.max_gate


class _DynamicGate(nn.Module):
    """g = max_gate * tanh(MLP(x)); MLP output layer zero-init -> tanh(0) == 0 at start."""

    def __init__(self, in_dim: int, bottleneck: int, max_gate: float = 0.5) -> None:
        super().__init__()
        self.max_gate = float(max_gate)
        self.fc1 = nn.Linear(in_dim, bottleneck)
        self.fc2 = _ZeroInitLinear(bottleneck, 1)

    def forward(self, x: Tensor) -> Tensor:
        h = F.gelu(_fp32_linear(self.fc1, x))
        # Non-negative clamp, same rationale as _ScalarGate (see above).
        return torch.relu(torch.tanh(_fp32_linear(self.fc2, h))) * self.max_gate  # (..., 1)


class CMSI(nn.Module):
    """Cross-Modal Semantic Injection.

    Args:
        num_classes: number of classes C.
        dim: feature dimension D (512 for Vita-CLIP).
        use_v2t: enable V->T semantic reconstruction residual injection.
        use_t2v: enable T->V class-prototype correction (also drives proto EMA).
        use_gate: when False, injections are ungated (equivalent to g = 1).
        gate_type: "scalar" | "dynamic".
        use_dual_head: fuse logits_text with a zero-init linear head via sigmoid(alpha).
        align_lambda: weight of the prototype-alignment loss.
        proto_momentum: EMA momentum for class-video prototypes.
        proto_text_init: defensive zero-prototype guard. On the first training
            step the prototypes are copied from the (normalized) text
            directions Q *before* the video EMA of that same forward pass
            runs. Because _update_proto then immediately overwrites every
            class that appears in the batch (count == 0 -> direct assignment),
            the text seed only survives for classes absent from the batch --
            and those have seen_mask == 0, so they never participate in
            injection. Net effect: prevents the normalized (0 - Q) degenerate
            direction for never-seen classes; it is NOT a real semantic seed.
        v2t_attn_temperature: temperature used by the V->T semantic
            reconstruction attention ``softmax((V @ Q_.T) * t)``. The shared
            logit_scale (~100) saturates the softmax into one-hot and kills
            gradients (dA/d(V@Q_.T) ~ 0), so the V->T path degenerates into
            "gate-only" learning. Defaults to 10.0, which keeps the attention
            soft while preserving class separability; None restores the legacy
            behavior of reusing logit_scale.
        v2t_detach_text: when True (default), the V->T reconstruction
            attention uses a detached Q_ (``yhat = softmax(...) @ Q_.detach()``),
            so V_ is only pulled by the current text directions and no gradient
            flows back through the V->P->Q_->V_->V loop. Set to False to let
            gradients flow back through the loop (experimental).
        v2t_dict: attention dictionary of the V->T injection. "proto" (default)
            pulls V toward its own class-video EMA centroid, masking unseen
            classes: it directly serves the classification margin and is
            identical in train/eval (no label needed). "text" reconstructs
            from the (detached) text directions Q_ (v3 behavior).
        align_temperature: temperature of the InfoNCE prototype-alignment
            loss (default 10.0). The shared logit_scale (~100) saturates the
            InfoNCE logits and starves the Q_ alignment gradient; a lower
            fixed temperature keeps the pull signal alive.
        align_lambda_decay: final/initial ratio of the prototype-alignment loss
            after align_lambda_decay_steps (linear decay). 1.0 disables decay.
        align_lambda_decay_steps: number of steps over which the alignment loss
            decays linearly from align_lambda to align_lambda * align_lambda_decay
            (0 disables decay).
        gate_warmup_steps: number of steps over which the injection gates ramp
            from 0 to 1 (linear warmup). 0 disables warmup (full gate from
            step 0). At eval the scale is always 1.
        align_neg_num: number of negative classes in the contrastive alignment
            loss. 0 or >= num_classes-1 uses ALL other classes as negatives.
        t2v_direction_only: when True (default), the T->V correction is a fixed
            direction toward the class-video prototype (delta = normalize(P-Q))
            scaled by the gate only -- interpretable, and it cannot wander into
            a semantically wrong direction the way a free MLP can (v3 observed
            this: gates went negative). When False, the LowRankNet MLP is used.
        max_gate: upper bound of the injection gates (identity start stays 0).
        lowrank_bottleneck: hidden dim of the T->V LowRankNet.
        init_logit_scale: initial exp(logit_scale) value (100.0 == CLIP default).
        gate_bottleneck: hidden dim of dynamic gates.
        hard_neg_weight: weight of the hard-negative hinge loss (0 disables).
        hard_neg_margin: hinge margin.
        hard_neg_top_k: number of hardest fake classes per sample.
        eps: L2-normalization epsilon.
    """

    def __init__(
        self,
        num_classes: int,
        dim: int,
        use_v2t: bool = True,
        use_t2v: bool = True,
        use_gate: bool = True,
        gate_type: str = "scalar",  # "scalar" | "dynamic"
        use_dual_head: bool = True,
        align_lambda: float = 0.005,
        proto_momentum: float = 0.99,
        proto_text_init: bool = True,
        v2t_attn_temperature: Optional[float] = 10.0,
        v2t_detach_text: bool = True,
        v2t_dict: str = "proto",  # "proto" | "text"
        align_temperature: float = 10.0,
        align_lambda_decay: float = 1.0,  # final/initial ratio, 1.0 disables
        align_lambda_decay_steps: int = 0,
        gate_warmup_steps: int = 0,
        align_neg_num: int = 0,
        t2v_direction_only: bool = True,
        max_gate: float = 0.3,
        lowrank_bottleneck: int = 128,
        init_logit_scale: float = 100.0,
        gate_bottleneck: int = 128,
        hard_neg_weight: float = 0.0,
        hard_neg_margin: float = 0.1,
        hard_neg_top_k: int = 3,
        lowrank_zero_init: bool = False,
        lowrank_delta_std: float = 0.05,
        eps: float = 1e-6,
        external_logit_scale: Optional[nn.Parameter] = None,
    ) -> None:
        super().__init__()

        if num_classes <= 0:
            raise ValueError("num_classes must be positive.")
        if dim <= 0:
            raise ValueError("dim must be positive.")
        if gate_type not in ("scalar", "dynamic"):
            raise ValueError(f"gate_type must be 'scalar' or 'dynamic', got {gate_type!r}.")

        self.num_classes = int(num_classes)
        self.dim = int(dim)
        self.use_v2t = bool(use_v2t)
        self.use_t2v = bool(use_t2v)
        self.use_gate = bool(use_gate)
        self.use_dual_head = bool(use_dual_head)
        self.align_lambda = float(align_lambda)
        self.proto_momentum = float(proto_momentum)
        self.proto_text_init = bool(proto_text_init)
        self.v2t_attn_temperature = (
            None if v2t_attn_temperature is None else float(v2t_attn_temperature)
        )
        self.v2t_detach_text = bool(v2t_detach_text)
        if v2t_dict not in ("proto", "text"):
            raise ValueError(
                f"v2t_dict must be 'proto' or 'text', got {v2t_dict!r}."
            )
        self.v2t_dict = str(v2t_dict)
        self.align_temperature = (
            10.0 if align_temperature is None else float(align_temperature)
        )
        self.align_lambda_decay = float(align_lambda_decay)
        self.align_lambda_decay_steps = int(align_lambda_decay_steps)
        self.gate_warmup_steps = int(gate_warmup_steps)
        self.align_neg_num = int(align_neg_num)
        self.t2v_direction_only = bool(t2v_direction_only)
        self.max_gate = float(max_gate)
        self.hard_neg_weight = float(hard_neg_weight)
        self.hard_neg_margin = float(hard_neg_margin)
        self.hard_neg_top_k = int(hard_neg_top_k)
        self.lowrank_zero_init = bool(lowrank_zero_init)
        self.lowrank_delta_std = float(lowrank_delta_std)
        self.eps = float(eps)

        # --- Component 1: zero-init gates (one per injection direction) ---
        # Gates are created only for the injection directions that are enabled:
        # a created-but-unused gate would trip DDP (find_unused_parameters=False).
        self.gate_v: Optional[nn.Module]
        self.gate_t: Optional[nn.Module]
        if self.use_v2t:
            if self.use_gate and gate_type == "dynamic":
                self.gate_v = _DynamicGate(2 * self.dim, gate_bottleneck, max_gate=self.max_gate)
            else:
                # scalar gate (or unused placeholder when use_gate=False)
                self.gate_v = _ScalarGate(max_gate=self.max_gate)
        else:
            self.gate_v = None
        if self.use_t2v:
            if self.use_gate and gate_type == "dynamic":
                self.gate_t = _DynamicGate(2 * self.dim, gate_bottleneck, max_gate=self.max_gate)
            else:
                self.gate_t = _ScalarGate(max_gate=self.max_gate)
        else:
            self.gate_t = None

        # --- Component 4: T->V low-rank prototype correction ---
        self.lowrank_net = _LowRankNet(
            self.dim,
            lowrank_bottleneck,
            zero_init=self.lowrank_zero_init,
            delta_std=self.lowrank_delta_std,
        )

        # --- Component 2: EMA class-video prototypes (buffers, no grad) ---
        # NOTE (audit 2026-08): these buffers are updated rank-locally by
        # _update_proto and are NOT DDP-synced. Under nproc_per_node=1 (the
        # current setup) this is fine; on multi-GPU the EMA diverges per rank
        # and only rank 0's values are saved, so a rank that loads the
        # checkpoint would inject with different prototype directions than the
        # ranks it was trained with. If multi-GPU is ever enabled, sync proto
        # (e.g. broadcast from rank 0 or all-reduce) before/after training and
        # before checkpointing.
        self.register_buffer("proto", torch.zeros(self.num_classes, self.dim))
        self.register_buffer(
            "proto_count", torch.zeros(self.num_classes, dtype=torch.long)
        )

        # --- Classification ---
        if external_logit_scale is not None:
            # Reuse the parent CLIP model's logit_scale so that a single
            # shared, trainable temperature scalar is used (and still subject
            # to the trainer's clamp_logit_scale). object.__setattr__ avoids
            # registering it again as a child parameter of this module.
            object.__setattr__(self, "logit_scale", external_logit_scale)
        else:
            self.logit_scale = nn.Parameter(
                torch.tensor(math.log(float(init_logit_scale)), dtype=torch.float32)
            )
        if self.use_dual_head:
            # Identity start for the fused logits: linear_head is zero-inited
            # (outputs exactly 0), so logits = sigmoid(alpha) * logits_text +
            # (1 - sigmoid(alpha)) * 0. sigmoid(alpha) ~= 1 therefore keeps the
            # initial logits ~= the baseline cosine logits. alpha = +4.0 gives
            # sigmoid(4.0) ~= 0.982 (alpha = 0 -> sigmoid 0.5 would HALVE the
            # initial logits and break the identity-start contract).
            self.alpha = nn.Parameter(torch.full((1,), 4.0))
            self.linear_head = _ZeroInitLinear(self.dim, self.num_classes)
        else:
            self.register_parameter("alpha", None)
            self.linear_head = None

    # ------------------------------------------------------------------ #
    #  EMA prototypes (train only, t2v only, torch.no_grad)              #
    # ------------------------------------------------------------------ #
    @torch.no_grad()
    def _update_proto(self, V: Tensor, y: Tensor) -> None:
        """Update P[c] with EMA of the normalized mean of class-c videos.

        count[c] == 0 -> direct assignment; otherwise EMA. Never carries grads.
        """
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

        ``step`` drives the gate warmup scale during training
        (``min(1, step / gate_warmup_steps)``); None + training means warmup
        is treated as "not started yet" (scale 0), None + eval means scale 1.

        Returns: {"logits", "logits_text", "align_loss", "V_", "Q_",
                  "gate_v", "gate_t", "gate_scale", "proto"}
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

        device = V.device
        V = _l2_normalize(V, self.eps)  # (B, D)
        Q = _l2_normalize(Q, self.eps)  # (C, D)

        # --- Gate warmup: ramp injection strength from 0 during early training ---
        # Eval always uses the full injection; during training the scale is
        # min(1, step / gate_warmup_steps) so the first epochs are exactly the
        # baseline behavior, and any regression is attributable to the injection.
        gate_scale = 1.0
        if self.training and self.gate_warmup_steps > 0:
            if step is None:
                gate_scale = 0.0
            else:
                gate_scale = min(1.0, float(step) / float(self.gate_warmup_steps))

        # --- Component 2: EMA prototypes (no grad, train only, t2v only) ---
        if self.training and self.use_t2v and y is not None:
            if (
                self.proto_text_init
                and self.proto_count.sum().item() == 0
            ):
                # NOTE (audit 2026-08): this text "seed" is only a defensive
                # zero-prototype guard, NOT a real semantic seed. The copy
                # below lives only between this line and _update_proto below:
                # classes present in this batch immediately get overwritten by
                # the video EMA (count == 0 -> direct assignment), and classes
                # absent from the batch keep the text direction but their
                # seen_mask stays 0 so they never participate in injection.
                with torch.no_grad():
                    self.proto.copy_(Q.float())
            self._update_proto(V, y)

        # --- Component 4: T->V class-prototype correction (compute Q_ first) ---
        Q_ = Q
        gate_t = torch.ones(1, device=device)
        if self.use_t2v and self._proto_initialized():
            P = self.proto.float()  # (C, D), buffer, no grad
            # Classes never seen so far (proto_count == 0) have P == 0; a
            # normalized (0 - Q) is a *directional* correction with unit norm,
            # and LowRankNet(0) != 0 either (fc1 bias). Mask delta for those
            # classes so Q_ stays exactly Q until their prototype exists.
            seen_mask = (self.proto_count > 0).float().unsqueeze(-1)  # (C, 1)
            diff = _l2_normalize(P - Q, self.eps)  # unit-norm difference vector
            if self.t2v_direction_only:
                # Fixed direction (toward the class-video prototype), learned
                # magnitude via the gate only: interpretable and cannot wander
                # into a semantically wrong direction the way a free MLP can
                # (v3 observed gates going negative = the MLP correction was
                # anti-aligned with classification).
                delta = diff * seen_mask
            else:
                delta = self.lowrank_net(diff) * seen_mask
            if self.use_gate:
                gate_t = self.gate_t(torch.cat([Q, P], dim=-1))  # (C, 1) or (1,)
            gate_t = gate_t * gate_scale  # warmup scaling
            Q_ = Q + gate_t * delta
            Q_ = _l2_normalize(Q_, self.eps)

        # --- Component 3: V->T semantic reconstruction residual injection ---
        V_ = V
        gate_v = torch.ones(1, device=device)
        if self.use_v2t:
            # Use a dedicated, lower attention temperature for the V->T
            # reconstruction. Reusing the shared logit_scale (~100) saturates
            # softmax((V @ Q_.T) * 100) into one-hot and kills the gradient of
            # the V/Q_ reconstruction path (dA/d(V @ Q_.T) ~ 0), leaving only
            # gate_v trainable. A temperature in [5, 10] keeps the attention
            # soft while preserving class separability.
            if self.v2t_attn_temperature is not None:
                attn_t = torch.tensor(
                    float(self.v2t_attn_temperature), device=device, dtype=torch.float32
                )
            else:
                attn_t = torch.exp(self.logit_scale.float())
            if self.v2t_detach_text:
                # Break the V->P->Q_->V_->V loop: V_ is pulled only by the
                # current text directions and no gradient flows back to Q_.
                q_attn = Q_.detach()
            else:
                q_attn = Q_
            A = torch.softmax((V @ q_attn.t()) * attn_t, dim=-1)  # (B, C)
            yhat = A @ q_attn  # (B, D) semantic reconstruction
            residual = yhat - V
            if self.use_gate:
                gate_v = self.gate_v(torch.cat([V, yhat], dim=-1))  # (B, 1) or (1,)
            gate_v = gate_v * gate_scale  # warmup scaling
            V_ = V + gate_v * residual
            V_ = _l2_normalize(V_, self.eps)

        # --- Classification ---
        scale = torch.exp(self.logit_scale.float())
        logits_text = scale * (V_ @ Q_.t())  # (B, C)
        if self.use_dual_head:
            w = torch.sigmoid(self.alpha.float())  # scalar, init ~0.982 (identity start)
            logits = w * logits_text + (1.0 - w) * self.linear_head(V_)
        else:
            logits = logits_text

        # --- Prototype alignment loss (train only, t2v only, grads -> Q_) ---
        # Contrastive (InfoNCE-style) form: pulls Q'_i toward P_i while keeping
        # it separated from the other class text directions, so the classifier
        # inter-class margin is preserved (MSE isotropically collapses it).
        align_loss = torch.zeros((), device=device)
        if self.training and self.use_t2v and y is not None:
            with torch.no_grad():
                onehot = F.one_hot(y, num_classes=self.num_classes).float()
                present = onehot.sum(0) > 0
            if present.any():
                lam = self.align_lambda
                if self.align_lambda_decay_steps > 0:
                    # Linear decay of the alignment weight: strongest during the
                    # gate-warmup phase (when Q_ must be steered toward the
                    # class centroids), tapered in the endgame so the CE loss on
                    # the classification head is not diluted.
                    progress = min(
                        1.0,
                        float(step if step is not None else 0)
                        / float(self.align_lambda_decay_steps),
                    )
                    lam = self.align_lambda * (
                        1.0 - (1.0 - self.align_lambda_decay) * progress
                    )
                align_loss = lam * self._contrastive_align_loss(
                    Q_, self.proto, present, Q
                )

        return {
            "logits": logits,
            "logits_text": logits_text,
            "align_loss": align_loss,
            "V_": V_,
            "Q_": Q_,
            "gate_v": gate_v,
            "gate_t": gate_t,
            "gate_scale": gate_scale,
            "proto": self.proto,
        }

    # ------------------------------------------------------------------ #
    #  Losses                                                             #
    # ------------------------------------------------------------------ #
    def _contrastive_align_loss(
        self, Q_: Tensor, P: Tensor, present: Tensor, Q: Tensor
    ) -> Tensor:
        """InfoNCE-style alignment loss that preserves inter-class margin.

        For each class i present in the current batch:
            anchor    = Q'_i   (corrected text direction, has grad)
            positive  = P_i    (EMA video prototype, detached, no grad)
            negatives = Q_j    (j != i, *original* text directions)

        The temperature is an independent, fixed value (align_temperature,
        default 10.0) rather than the shared logit_scale: exp(logit_scale)
        ~= 100 saturates the InfoNCE logits and starves the Q_ alignment
        gradient. Using the original text directions as negatives (not Q')
        keeps the alignment objective consistent with the semantic class
        structure instead of conflicting with the classification head.
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
        neg = neg.scatter(
            1, idx.unsqueeze(1), float("-inf")
        )  # exclude the anchor's own class

        if 0 < self.align_neg_num < self.num_classes - 1:
            # Subsample a fixed number of negatives per anchor (optional; the
            # default uses all other classes, which is cheap for C <= ~101).
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

    def hard_negative_loss(
        self, logits: Tensor, y: Tensor, margin: Optional[float] = None,
        top_k: Optional[int] = None,
    ) -> Tensor:
        """Optional hard-negative hinge loss.

        For each sample, take the top-k fake classes whose logits are closest
        to the true-class logit and penalize max(0, neg - pos + margin).
        Equivalent to relu(margin - (pos - neg)) on the k smallest gaps.
        """
        if y is None:
            return torch.zeros((), device=logits.device)
        margin = self.hard_neg_margin if margin is None else float(margin)
        top_k = self.hard_neg_top_k if top_k is None else int(top_k)
        if top_k <= 0:
            return torch.zeros((), device=logits.device)

        B, C = logits.shape
        k = min(top_k, C - 1)
        if k <= 0:
            return torch.zeros((), device=logits.device)

        pos = logits.float().gather(1, y.unsqueeze(1))  # (B, 1)
        mask = torch.ones_like(logits, dtype=torch.bool)
        mask.scatter_(1, y.unsqueeze(1), False)
        neg = logits.float()[mask].view(B, C - 1)  # (B, C-1) fake classes only
        gap = pos - neg  # larger is better
        topk_gap, _ = torch.topk(gap, k=k, dim=1, largest=False)  # hardest
        return F.relu(margin - topk_gap).mean()

    def compute_loss(
        self,
        logits: Tensor,
        y: Tensor,
        align_loss: Optional[Tensor] = None,
        label_smoothing: float = 0.1,
    ) -> Tensor:
        """Main loss: CE(label_smoothing) + align_loss + hard_neg_loss."""
        ce = F.cross_entropy(logits.float(), y, label_smoothing=label_smoothing)
        total = ce
        if align_loss is not None:
            total = total + align_loss
        if self.hard_neg_weight > 0:
            total = total + self.hard_neg_weight * self.hard_negative_loss(logits, y)
        return total
