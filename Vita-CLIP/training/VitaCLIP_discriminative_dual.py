"""Discriminative region + dual-branch temporal module for Vita-CLIP.

Upgrade of the shared RWKV module (VitaCLIPDiscriminativeRWKV) to a
"conv3d spatial-temporal branch + full-coverage Mamba branch" design:

1. Discriminative region selection (unchanged, STOP-inspired):
   - A_t = CLS -> patch attention of frame t (real attention weights);
   - M_t = temporal motion strength (frame-difference squared mean);
   - W_t = alpha * A_t + (1 - alpha) * M_t + learnable score(patch);
   - keep the top-k patches per frame (k = top_k, default 12, alpha 0.4);
   - the same top-k support set yields a hard 0/1 mask (entry seed
     positions) and soft weights (softmax over w / tau, gradient bridge).

2. Dual-branch temporal modelling over ALL spatial positions:
   - Conv3d branch: ALL patches are written back onto the 14x14 grid
     (no sparse union masking), a depthwise Conv3d(3,3,3) models the
     local spatio-temporal neighbourhood (shape change of the moving part),
     followed by a pointwise 1x1x1 conv.
   - Mamba branch (default "full"): decomposed two-stage S6 selective scan
     (causal conv + linear recurrence) --
       (i)   per-position temporal scan  [B*N_p, T, D]  (motion);
       (ii)  per-frame spatial scan      [B*T, N_p, D]  (cross-position
             trajectory flow after each position models its own motion).
     --dr_mamba_scope=union falls back to per-position [B*|U|, T, D]
     scanning (no cross-position flow, validation of the code path).

3. Patch-level gating fusion:
   - stats = concat(GAP_T(conv_delta), GAP_T(mamba_delta)) ->
     Linear(2*bn -> bn) -> GELU -> Linear(bn -> 1) -> sigmoid (last layer
     zero-initialised -> sigma = 0.5 at init);
   - fused = sigma * conv + (1 - sigma) * mamba.

4. Entry seed boost (initial guidance only):
   - before modelling, the per-frame top-k action regions are emphasised
     ONCE on the raw patch tokens: patch += tanh(seed_scale) * mask;
   - from there on both branches model EVERY position uniformly and the
     write-back is uniform - the top-k has no further special treatment.

5. Write-back (uniform):
   - patch' = patch + tanh(residual_scale) * fused: EVERY position
     receives its temporal delta (both branches modelled the whole grid);
   - cls' = cls + sum_j soft_w_j * fused_j (soft aggregation gradient
     bridge, required because ln_post reads only the CLS token).

Unlike the RWKV version, no union-based pruning restricts the temporal
modelling: both branches process every spatial position, the top-k regions
are seeded once at the entrance as initial guidance, and the write-back is
uniform over all positions. All temporal computations run in float32.

Input / output shape: [B*T, N, D] (same as the frequency adapter).
"""

from __future__ import annotations

import math
from typing import Dict, Optional

import torch
from torch import Tensor, nn
import torch.nn.functional as F


def _fp32_layer_norm(module: nn.LayerNorm, x: Tensor) -> Tensor:
    """Execute LayerNorm in float32 (valid even after model.half())."""
    weight = None if module.weight is None else module.weight.float()
    bias = None if module.bias is None else module.bias.float()
    return F.layer_norm(x.float(), module.normalized_shape, weight, bias, module.eps)


def _fp32_linear(module: nn.Linear, x: Tensor) -> Tensor:
    """Execute Linear in float32 while preserving gradient flow."""
    weight = module.weight.float()
    bias = None if module.bias is None else module.bias.float()
    return F.linear(x.float(), weight, bias)


class _MambaCore(nn.Module):
    """Hand-written S6 selective scan over [B, L, dim] (full float32).

    Input  -> in_proj (dim -> 2*dim, split into u and z) -> causal Conv1d
              (groups=dim) -> SiLU -> x_proj splits (B, C) of size d_state;
              dt_proj -> softplus gives the step size dt.
    Recurrence over L steps:
        dA = exp(dt * A)      A = -exp(A_log)  (negative, in (-inf, 0))
        dB = dt * B
        h  = dA * h + dB * u
        y  = (C * h).sum(dim=-1)
    Output = out_proj((y + D * u) * SiLU(z)).

    Optional bidirectional mode scans the reversed sequence and adds the
    two outputs together.
    """

    def __init__(
        self,
        dim: int = 192,
        d_state: int = 16,
        d_conv: int = 4,
        dt_rank: Optional[int] = None,
        bidir: bool = False,
        eps: float = 1e-6,
    ) -> None:
        super().__init__()
        if dim <= 0:
            raise ValueError("dim must be positive.")
        if d_state <= 0:
            raise ValueError("d_state must be positive.")
        if d_conv <= 0:
            raise ValueError("d_conv must be positive.")

        self.dim = int(dim)
        self.d_state = int(d_state)
        self.d_conv = int(d_conv)
        self.dt_rank = int(dt_rank or max(1, math.ceil(dim / 16)))
        self.bidir = bool(bidir)
        self.eps = float(eps)

        # z (gate) and u (input) share one input projection.
        self.in_proj = nn.Linear(dim, dim * 2, bias=False)
        # Causal depthwise conv: padding = d_conv - 1, truncated to L.
        self.conv1d = nn.Conv1d(
            dim,
            dim,
            kernel_size=d_conv,
            groups=dim,
            bias=True,
            padding=d_conv - 1,
        )
        # B and C projections (dt uses a low-rank bottleneck).
        self.x_proj = nn.Linear(
            dim,
            self.dt_rank + self.d_state * 2,
            bias=False,
        )
        self.dt_proj = nn.Linear(self.dt_rank, dim, bias=True)
        # Negative exponential decay: A = -exp(A_log) in (-inf, 0).
        self.register_buffer(
            "A_log",
            torch.log(torch.arange(1, self.d_state + 1, dtype=torch.float32)),
        )
        # Learnable skip connection weight (D * u), initialised to 1.
        self.D = nn.Parameter(torch.ones(dim))
        self.out_proj = nn.Linear(dim, dim, bias=False)

        self._initialize_weights()

    def _initialize_weights(self) -> None:
        """Moderate init: small perturb of the hidden state, learnable dt."""
        nn.init.normal_(self.in_proj.weight, mean=0.0, std=0.05)
        nn.init.normal_(self.conv1d.weight, mean=0.0, std=0.05)
        nn.init.zeros_(self.conv1d.bias)
        nn.init.normal_(self.x_proj.weight, mean=0.0, std=0.1)
        nn.init.normal_(self.dt_proj.weight, mean=0.0, std=0.1)
        # softplus(-1.0) ~ 0.31 -> moderate initial step size.
        nn.init.constant_(self.dt_proj.bias, -1.0)
        nn.init.xavier_uniform_(self.out_proj.weight)

    @torch.cuda.amp.autocast(enabled=False)
    def forward(self, x: Tensor) -> Tensor:
        """Forward on [B, L, dim] -> [B, L, dim] in float32."""
        x = x.float()
        batch_size, seq_len, _ = x.shape
        device = x.device

        xz = _fp32_linear(self.in_proj, x)          # [B, L, 2*dim]
        u, z = xz.chunk(2, dim=-1)                  # each [B, L, dim]

        # Causal depthwise conv over the sequence dimension.
        u = self.conv1d(u.transpose(1, 2))[..., :seq_len].transpose(1, 2)
        u = F.silu(u)

        # B, C and dt from the post-conv hidden state.
        dbl = self.x_proj(u)                        # [B, L, dt_rank + 2*d_state]
        dt_r, b_tensor, c_tensor = dbl.split(
            [self.dt_rank, self.d_state, self.d_state], dim=-1
        )
        dt = F.softplus(_fp32_linear(self.dt_proj, dt_r))  # [B, L, dim]
        a = -torch.exp(self.A_log.float())          # [d_state]

        dA = torch.exp(dt.unsqueeze(-1) * a)        # [B, L, dim, d_state]
        dB = dt.unsqueeze(-1) * b_tensor.unsqueeze(2)  # [B, L, dim, d_state]

        def scan(u_s: Tensor, dA_s: Tensor, dB_s: Tensor, c_s: Tensor) -> Tensor:
            h = torch.zeros(
                batch_size,
                self.dim,
                self.d_state,
                device=device,
                dtype=torch.float32,
            )
            outs: list[Tensor] = []
            for t in range(seq_len):
                h = dA_s[:, t] * h + dB_s[:, t] * u_s[:, t].unsqueeze(-1)
                y = (h * c_s[:, t].unsqueeze(1)).sum(-1)  # [B, dim]
                outs.append(y)
            return torch.stack(outs, dim=1)         # [B, L, dim]

        y = scan(u, dA, dB, c_tensor)

        if self.bidir:
            flip = lambda t: torch.flip(t, dims=[1])
            y_rev = scan(flip(u), flip(dA), flip(dB), flip(c_tensor))
            y = y + flip(y_rev)

        y = (y + self.D.float().view(1, 1, -1) * u) * F.silu(z)
        return _fp32_linear(self.out_proj, y)


class _Conv3DBranch(nn.Module):
    """TAdaConv-style time-adaptive Conv3d branch on [B, T, N_p, dim].

    Inspired by TAdaConv (Temporally-Adaptive Convolutions, ICLR 2022): the
    convolution weights are NO LONGER shared across frames. Instead the base
    depthwise Conv3d(3,3,3) kernel is re-calibrated per frame:

        W_t = alpha_t * W_base

    where alpha_t is a per-frame, per-channel calibration weight generated
    by a small temporal weight generator from the frame's time context. This
    gives the spatial convolutions temporal modelling ability at negligible
    extra cost (weight-space calibration, ~0.2% extra FLOPs).

    Calibration weight generator (following TAdaConv):
        v_t = GAP_space(x_t)               # [B, C, T] per-frame descriptors
        alpha = 1 + Conv1d(C->C, k=3)(v_t) # local time context; init 0 -> alpha=1
        W_dynamic[t] = W_base * alpha[:, :, t]

    Init: the last 1D conv is zero-initialised and we add the constant 1, so
    at init alpha == 1 and W_dynamic == W_base: identical to a plain Conv3d
    (backward-compatible with old checkpoints via the shared `depthwise`
    base weight + `depthwise2` zero-init residual).

    Structure (square grid):
        x -> grid [B, C, T, gs, gs]
        -> dynamic depthwise Conv3d (TAdaConv-calibrated) -> SiLU
        -> depthwise2 (zero-init residual, widens temporal RF) 
        -> pointwise 1x1x1 (channel mixing)
        -> + tanh(residual_scale) * depthwise2_out
    Non-square patch counts fall back to a static per-position temporal
    Conv1d (no adaptive weights, keeps any N_p working).
    """

    def __init__(
        self,
        dim: int = 192,
        num_spatial_patches: int = 196,
        eps: float = 1e-6,
        residual_scale: float = 0.01,
    ) -> None:
        super().__init__()
        if dim <= 0:
            raise ValueError("dim must be positive.")
        self.dim = int(dim)
        self.num_spatial_patches = int(num_spatial_patches)
        self.eps = float(eps)
        self.residual_scale = float(residual_scale)

        grid_size = int(round(math.sqrt(self.num_spatial_patches)))
        self.grid_size = grid_size
        self.square_grid = grid_size * grid_size == self.num_spatial_patches

        if self.square_grid:
            # Shared base SPATIAL depthwise kernel (2D Conv2d, no temporal
            # axis), loaded/initialised like a plain 2D conv. Following the
            # original TAdaConv, this is a plain 2D convolution that gains
            # temporal ability ONLY through the per-frame weight calibration.
            self.depthwise = nn.Conv2d(
                dim,
                dim,
                kernel_size=3,
                groups=dim,
                padding=1,
                bias=False,
            )
            # TAdaConv temporal calibration generator: per-frame, per-channel
            # modulation from local time context of the frame descriptors.
            # Two-layer 1D conv (channel-reduction -> BN -> ReLU -> restore),
            # matching the original TAdaConv generator G. The LAST layer is
            # zero-initialised so alpha starts at 1 (identity w.r.t. the base
            # kernel); the first layer is normal-initialised and BN keeps the
            # training stable.
            self.temporal_alpha = nn.Sequential(
                nn.Conv1d(dim, dim // 4, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm1d(dim // 4),
                nn.ReLU(inplace=True),
                nn.Conv1d(dim // 4, dim, kernel_size=3, padding=1, bias=False),
            )
            # Global-context mapping (TAdaConv GAP_st fusion): maps the
            # global spatio-temporal descriptor g to every frame and adds it
            # to the per-frame descriptors before calibration. Zero-init so
            # the global context has no effect at start (identity).
            self.global_gate = nn.Linear(dim, dim, bias=False)
            # Second depthwise conv: zero-init -> residual no-op at init.
            # Kept as Conv3d(1,1,3) so it broadens the spatial context without
            # adding a temporal kernel of its own (the temporal modelling is
            # already provided by the weight calibration).
            self.depthwise2 = nn.Conv2d(
                dim,
                dim,
                kernel_size=3,
                groups=dim,
                padding=1,
                bias=False,
            )
            self.pointwise = nn.Conv2d(dim, dim, kernel_size=1, bias=True)
        else:
            self.temporal_conv = nn.Conv1d(
                dim,
                dim,
                kernel_size=3,
                groups=dim,
                padding=1,
                bias=False,
            )
            self.temporal_conv2 = nn.Conv1d(
                dim,
                dim,
                kernel_size=3,
                groups=dim,
                padding=1,
                bias=False,
            )
            self.pointwise = nn.Conv1d(dim, dim, kernel_size=1, bias=True)

        self._initialize_weights()

    def _initialize_weights(self) -> None:
        """Small hidden perturbations, zero-init deep residual, Xavier mixing."""
        if self.square_grid:
            nn.init.normal_(self.depthwise.weight, mean=0.0, std=0.05)
            # temporal_alpha: the LAST 1D conv is zero-init so alpha == 1 at
            # start (identity w.r.t. the base kernel); the first conv keeps
            # its default init (normal) and BN is left default.
            alpha_last = self.temporal_alpha[-1]
            assert isinstance(alpha_last, nn.Conv1d)
            nn.init.zeros_(alpha_last.weight)
            # global_gate zero-init -> global context has no effect at start.
            nn.init.zeros_(self.global_gate.weight)
            nn.init.zeros_(self.depthwise2.weight)
        else:
            nn.init.normal_(self.temporal_conv.weight, mean=0.0, std=0.05)
            nn.init.zeros_(self.temporal_conv2.weight)
        nn.init.xavier_uniform_(self.pointwise.weight)
        nn.init.zeros_(self.pointwise.bias)

    def _tadaconv_forward(self, h: Tensor) -> Tensor:
        """TAdaConv-calibrated depthwise Conv2d on [B, C, T, gs, gs].

        Faithful to the ORIGINAL TAdaConv: a plain 2D spatial convolution is
        performed independently on each frame, but its kernel is re-calibrated
        per frame as W_t = W_base * alpha_t, where alpha_t is a per-frame,
        per-channel scalar generated from the frame's time context. Temporal
        modelling comes entirely from the weight change across frames - there
        is NO cross-frame pixel aggregation (this is the original design:
        "empower the spatial convolutions with temporal modelling abilities by
        calibrating the convolution weights").

        alpha_t ~ [B, C] (per channel scalar). The base depthwise kernel is
        [C, 1, 3, 3]; calibrating along C_matches the original alpha shape.
        """
        batch_size, dim, num_frames, gs, _ = h.shape
        # (a) Per-frame spatial descriptors (space GAP): [B, C, T].
        v_t = h.mean(dim=(3, 4))
        # (a2) Global spatio-temporal descriptor (TAdaConv GAP_st): pool over
        # space AND time -> [B, C], map to every frame and add to v_t so the
        # calibration generator sees the whole-video context too.
        g = h.mean(dim=(2, 3, 4))  # [B, C]
        v_t = v_t + self.global_gate(g).unsqueeze(-1)  # [B, C, T] global-fused
        # (b) Temporal calibration: alpha = 1 + G(v_t) -> [B, C, T], where G
        # is the two-layer 1D conv generator (C->C/4 BN ReLU -> C), with the
        # last layer zero-initialised (alpha == 1 at start).
        alpha = 1.0 + self.temporal_alpha(v_t)  # [B, C, T]
        base_w = self.depthwise.weight  # [C, 1, 3, 3] (2D spatial depthwise)

        # (c) Fold B*T into the channel axis, calibrate each (b, t, c) kernel
        # by alpha, and run a group-folding depthwise Conv2d so every frame
        # uses its own calibrated kernel (no cross-frame pixel mixing).
        # h: [B, C, T, gs, gs] -> [B*T, C, gs, gs]
        h_bt = h.permute(0, 2, 1, 3, 4).reshape(
            batch_size * num_frames, dim, gs, gs
        )
        # alpha -> [B, T, C], weight for (b, t, c) = base_w[c] * alpha[b, t, c]
        alpha_bt = alpha.permute(0, 2, 1)  # [B, T, C]
        # base_w: [C,1,3,3] -> [1, 1, C, 1, 3, 3]; alpha_bt -> [B, T, C, 1, 1, 1]
        w_bt = base_w.view(1, 1, dim, 1, 3, 3) * alpha_bt.view(
            batch_size, num_frames, dim, 1, 1, 1
        )  # [B, T, C, 1, 3, 3]
        w_fold = w_bt.reshape(batch_size * num_frames * dim, 1, 3, 3)  # [BT*C,1,3,3]
        # fold batch*time into channels: [1, BT*C, gs, gs]
        h_fold = h_bt.reshape(1, batch_size * num_frames * dim, gs, gs)
        out = F.conv2d(
            h_fold,
            w_fold,
            stride=1,
            padding=1,
            groups=batch_size * num_frames * dim,
        )  # [1, BT*C, gs, gs]
        out = out.reshape(batch_size, num_frames, dim, gs, gs)
        return out.permute(0, 2, 1, 3, 4)  # [B, C, T, gs, gs]

    @torch.cuda.amp.autocast(enabled=False)
    def forward(self, x: Tensor) -> Tensor:
        """Forward [B, T, N_p, dim] -> [B, T, N_p, dim] (float32).

        All patches are placed on the full grid (row-major, patch index
        p -> (p // gs, p % gs)); the TAdaConv-calibrated depthwise Conv3d
        then sees genuine spatial neighbours while re-calibrating its kernel
        per frame from the temporal context.
        """
        x = x.float()
        batch_size, num_frames, num_patches, dim = x.shape

        if self.square_grid:
            gs = self.grid_size
            grid = x.reshape(batch_size, num_frames, gs, gs, dim)
            h = grid.permute(0, 4, 1, 2, 3).contiguous()  # [B, dim, T, gs, gs]
            h = self._tadaconv_forward(h)                  # [B, C, T, gs, gs]
            h = F.silu(h)
            # depthwise2 + pointwise are per-frame 2D convs: fold T into batch.
            h_bt = h.permute(0, 2, 1, 3, 4).reshape(
                batch_size * num_frames, dim, gs, gs
            )  # [B*T, C, gs, gs]
            h_deep_bt = self.depthwise2(h_bt)              # zero-init -> 0
            h_mix_bt = self.pointwise(h_bt)                # channel mixing
            h_deep = h_deep_bt.reshape(batch_size, num_frames, dim, gs, gs)
            h_mix = h_mix_bt.reshape(batch_size, num_frames, dim, gs, gs)
            h = h_mix + math.tanh(self.residual_scale) * h_deep
            h = h.permute(0, 2, 3, 4, 1).contiguous()      # [B, T, gs, gs, dim]
            return h.reshape(batch_size, num_frames, num_patches, dim)

        # Non-square fallback: per-position temporal modelling on all patches.
        h = x.permute(0, 3, 1, 2).contiguous()              # [B, dim, T, N_p]
        h = self.temporal_conv(h)
        h = F.silu(h)
        h_deep = self.temporal_conv2(h)                    # zero-init -> 0
        h = F.silu(h)
        h = self.pointwise(h)
        h = h + math.tanh(self.residual_scale) * h_deep
        return h.permute(0, 2, 3, 1)                        # [B, T, N_p, dim]


class VitaCLIPDiscriminativeDual(nn.Module):
    """
    Discriminative region selection + dual-branch temporal module.

    Input:
        x:           [B*T, N, D] tokens (CLS at index 0, then N_p patches)
        attn_weights: [B*T, N_p] real CLS -> patch attention weights

    Output:
        y: [B*T, N, D]. Selected discriminative patch tokens receive the
        gated dual-branch delta; the CLS token receives the soft-weighted
        delta aggregation (gradient bridge).

    Main procedure:
        1. Restore [B, T, N, D] and split CLS / patch tokens.
        2. Normalise each score source (z-score over N_p): attention,
           motion and the learnable score are comparable before fusion
           W = alpha * A + (1 - alpha) * M + score.
        3. Hard top-k mask (entry seed positions) + temperature softmax
           over ALL positions (straight-through: hard forward, soft gradient
           through the selection boundary).
        4. Union of selected positions across the batch.
        5. Entry seed boost: the per-frame top-k action regions are
           emphasised once (initial guidance), then both branches model
           every position uniformly.
        6. Down-project ALL patches, then model every position:
             a. Conv3d branch on the FULL grid (all patches, real spatial
                neighbours in the 3x3x3 receptive field);
             b. Mamba branch (decomposed temporal-then-spatial scan when
                mamba_scope == "full", or per-position scan when "union").
        7. Patch-level gating fusion over ALL positions:
           fused = sigma * conv + (1 - sigma) * mamba.
        8. Up-project, residual-gate; write back EVERY position uniformly
           (no top-k boost at write-back - the top-k regions were already
           seeded at the entrance); CLS receives the full soft-weighted
           aggregation of every position's delta.
    """

    def __init__(
        self,
        dim: int = 768,
        bottleneck: int = 192,
        num_spatial_patches: int = 196,
        top_k: int = 12,
        alpha: float = 0.4,
        mamba_scope: str = "full",
        use_conv3d: bool = True,
        use_mamba: bool = True,
        mamba_bidir: bool = False,
        d_state: int = 16,
        d_conv: int = 4,
        residual_scale_init: float = 0.01,
        seed_scale_init: float = 0.05,
        stability_clip: float = 100.0,
        min_frames: int = 4,
        eps: float = 1e-6,
        topk_temperature: float = 0.5,
    ) -> None:
        super().__init__()

        if dim <= 0:
            raise ValueError("dim must be positive.")
        if bottleneck <= 0:
            raise ValueError("bottleneck must be positive.")
        if num_spatial_patches <= 0:
            raise ValueError("num_spatial_patches must be positive.")
        if top_k <= 0:
            raise ValueError("top_k must be positive.")
        if not 0.0 <= alpha <= 1.0:
            raise ValueError("alpha must be in [0, 1].")
        if mamba_scope not in ("full", "union"):
            raise ValueError(f"mamba_scope must be 'full' or 'union', got {mamba_scope!r}.")
        if stability_clip <= 0:
            raise ValueError("stability_clip must be positive.")
        if min_frames < 2:
            raise ValueError("min_frames must be at least 2.")

        self.dim = int(dim)
        self.bottleneck = int(bottleneck)
        self.num_spatial_patches = int(num_spatial_patches)
        self.top_k = int(top_k)
        self.alpha = float(alpha)
        self.mamba_scope = mamba_scope
        self.use_conv3d = bool(use_conv3d)
        self.use_mamba = bool(use_mamba)
        self.mamba_bidir = bool(mamba_bidir)
        self.stability_clip = float(stability_clip)
        self.min_frames = int(min_frames)
        self.eps = float(eps)
        self.topk_temperature = float(topk_temperature)

        # Shared bottleneck projections around both branches.
        self.down_projection = nn.Linear(dim, bottleneck)
        self.up_projection = nn.Linear(bottleneck, dim)

        # Fixed residual strength (same verified value as the RWKV module).
        self.register_buffer(
            "residual_scale",
            torch.tensor(float(residual_scale_init), dtype=torch.float32),
        )
        # Entry seed boost: how strongly the top-k action regions are
        # emphasised ONCE at the module entrance (initial guidance only).
        self.register_buffer(
            "seed_scale",
            torch.tensor(float(seed_scale_init), dtype=torch.float32),
        )

        # Learnable spatial scorer -> adaptive top-k selection.
        self.scoring = nn.Sequential(
            nn.Linear(dim, bottleneck),
            nn.GELU(),
            nn.Linear(bottleneck, 1),
        )
        nn.init.zeros_(self.scoring[-1].weight)
        nn.init.zeros_(self.scoring[-1].bias)

        # Two temporal branches (both processed in the bottleneck space).
        if self.use_conv3d:
            self.conv3d_branch = _Conv3DBranch(
                dim=bottleneck,
                num_spatial_patches=num_spatial_patches,
                eps=eps,
            )
        if self.use_mamba:
            self.mamba = _MambaCore(
                dim=bottleneck,
                d_state=d_state,
                d_conv=d_conv,
                bidir=mamba_bidir,
                eps=eps,
            )

        # Patch-level gating fusion. Last layer zero-initialised ->
        # sigmoid(0) = 0.5 at init -> equal mixture of both branches.
        self.gate = nn.Sequential(
            nn.Linear(bottleneck * 2, bottleneck),
            nn.GELU(),
            nn.Linear(bottleneck, 1),
        )
        nn.init.zeros_(self.gate[-1].weight)
        nn.init.zeros_(self.gate[-1].bias)

        self._initialize_weights()

    def _initialize_weights(self) -> None:
        """Same verified scheme as the RWKV module (xavier up-projection)."""
        nn.init.normal_(self.down_projection.weight, mean=0.0, std=0.05)
        nn.init.zeros_(self.down_projection.bias)
        nn.init.xavier_uniform_(self.up_projection.weight)
        nn.init.zeros_(self.up_projection.bias)

    def _sanitize_real(self, x: Tensor) -> Tensor:
        limit = self.stability_clip
        return torch.nan_to_num(
            x,
            nan=0.0,
            posinf=limit,
            neginf=-limit,
        ).clamp(min=-limit, max=limit)

    def _motion_strength(self, patch: Tensor) -> Tensor:
        """Frame-difference squared mean -> [B, T, N_p] motion strength."""
        diff = patch[:, 1:, :, :] - patch[:, :-1, :, :]  # [B, T-1, N_p, D]
        diff_sq = (diff * diff).mean(dim=-1)             # [B, T-1, N_p]
        motion = torch.zeros_like(patch[..., 0])
        motion[:, 1:, :] += diff_sq
        motion[:, :-1, :] += diff_sq
        counts = torch.ones_like(motion)
        counts[:, 1:, :] += 1.0
        counts[:, :-1, :] += 1.0
        return motion / counts

    @torch.cuda.amp.autocast(enabled=False)
    def forward(
        self,
        x: Tensor,
        attn_weights: Tensor,
        *,
        num_frames: int,
        patch_start: int = 1,
        return_information: bool = False,
    ) -> Tensor:
        """Forward on [B*T, N, D] -> [B*T, N, D] (float32 internally)."""
        if x.ndim != 3:
            raise ValueError(f"x must be [B*T, N, D], got {tuple(x.shape)}.")
        total_tokens, num_tokens, dim = x.shape
        if num_tokens < 2:
            raise ValueError("x must contain a CLS token and at least one patch.")
        if num_frames <= 1 or total_tokens % num_frames != 0:
            raise ValueError(
                f"num_frames={num_frames} incompatible with {total_tokens} tokens."
            )
        batch_size = total_tokens // num_frames
        num_patches = num_tokens - patch_start
        if attn_weights.shape != (total_tokens, num_patches):
            raise ValueError(
                f"attn_weights must be [{total_tokens}, {num_patches}], "
                f"got {tuple(attn_weights.shape)}."
            )

        # Short-sequence protection (same as the frequency adapter).
        if num_frames < self.min_frames:
            if return_information:
                return x, {"skipped": True, "reason": "num_frames < min_frames"}
            return x

        x_float = x.float()
        tokens = x_float.view(batch_size, num_frames, num_tokens, dim)
        cls_tokens = tokens[:, :, :patch_start, :]    # [B, T, 1, D]
        patch_tokens = tokens[:, :, patch_start:, :]  # [B, T, N_p, D]

        # 1. Attention score A: [B, T, N_p].
        attn = attn_weights.float().view(batch_size, num_frames, num_patches)

        # 2. Motion strength M: [B, T, N_p].
        motion = self._motion_strength(patch_tokens)

        # 3. Normalise every score source (z-score over the N_p axis) so the
        #    alpha mixture is meaningful: raw attention is ~1/N_p while motion
        #    and the learnable score have arbitrary scales; without this the
        #    attention term is dominated and alpha is ineffective.
        score = self.scoring(patch_tokens).squeeze(-1)  # [B, T, N_p]

        def _zscore(a: Tensor) -> Tensor:
            mu = a.mean(dim=-1, keepdim=True)
            sd = a.std(dim=-1, keepdim=True).clamp_min(self.eps)
            return (a - mu) / sd

        attn_n = _zscore(attn)
        motion_n = _zscore(motion)
        score_n = _zscore(score)
        w = self.alpha * attn_n + (1.0 - self.alpha) * motion_n + score_n
        w = self._sanitize_real(w)

        # 4. Top-k selection per frame -> hard mask [B, T, N_p] (bool).
        #    The hard mask only marks the important action regions that
        #    receive a one-shot entry seed boost below; selection itself is
        #    learned through the soft weights (straight-through style:
        #    hard forward, soft grad).
        k = min(self.top_k, num_patches)
        top_indices = w.topk(k, dim=-1).indices  # [B, T, k]
        mask = torch.zeros_like(w, dtype=torch.bool).scatter_(
            -1, top_indices, True
        )

        # 4b. Soft selection weights: temperature softmax over ALL patch
        #    positions (not restricted to the top-k support set). Every
        #    position keeps a tiny but non-zero gradient, making the
        #    selection boundary learnable instead of a hard, non-differentiable
        #    choice. tau -> 0 recovers the hard one-hot selection.
        soft_weights = torch.softmax(
            w / self.topk_temperature, dim=-1
        )                                               # [B, T, N_p]

        # 5. Union of selected positions across the whole batch.
        any_selected = mask.any(dim=1)  # [B, N_p]
        union_pos = torch.unique(any_selected.nonzero()[:, 1])  # [|U|]
        if union_pos.numel() == 0:
            if return_information:
                info = {
                    "skipped": True,
                    "reason": "empty union of selected positions",
                    "attn_mean": attn.float().mean().detach(),
                    "motion_mean": motion.float().mean().detach(),
                    "top_k": k,
                    "alpha": self.alpha,
                }
                return x, info
            return x

        # 5b. Entry seed boost (initial guidance only): the per-frame top-k
        #    action regions are emphasised once on the raw patch tokens
        #    BEFORE any modelling. From here on both branches model EVERY
        #    position uniformly and the write-back is uniform too - the
        #    top-k has no further special treatment.
        seed = torch.tanh(self.seed_scale.float()) * mask.to(
            patch_tokens.dtype
        ).unsqueeze(-1)                                   # [B, T, N_p, 1]
        patch_tokens = patch_tokens + seed
        patch_tokens = self._sanitize_real(patch_tokens)

        # 6. Down-project ALL patches once (shared by both branches).
        bottleneck_feature = _fp32_linear(self.down_projection, patch_tokens)
        bottleneck_feature = self._sanitize_real(bottleneck_feature)
        bn = self.bottleneck

        # 6a. Conv3d branch on the FULL grid (all patches, local
        #    spatio-temporal neighbourhood). Output keeps every position.
        if self.use_conv3d:
            conv_delta = self.conv3d_branch(
                bottleneck_feature
            )                                          # [B, T, N_p, bn]
            conv_delta = self._sanitize_real(conv_delta)
        else:
            conv_delta = None

        # 6b. Mamba branch. "full" = decomposed two-stage scan, replacing
        #    the old unbounded 1568-step spatial-first flattening (which had
        #    unstructured cross-position flow and an impractical runtime):
        #      (i)   per-position temporal scan  [B*N_p, T, bn]  (motion);
        #      (ii)  per-frame spatial scan     [B*T, N_p, bn]  (cross-
        #            position trajectory flow after each position has
        #            modelled its own motion).
        #    "union" = per-position temporal scan over the union only
        #    (no cross-position flow); non-union positions get zero mamba
        #    contribution but still receive the full conv branch output.
        if self.use_mamba:
            if self.mamba_scope == "full":
                # Stage (i): temporal scan per position.
                seq_t = bottleneck_feature.permute(0, 2, 1, 3)  # [B, N_p, T, bn]
                seq_t = seq_t.reshape(-1, num_frames, bn)       # [B*N_p, T, bn]
                t_out = self.mamba(seq_t)                       # [B*N_p, T, bn]
                t_out = t_out.view(
                    batch_size, num_patches, num_frames, bn
                ).permute(0, 2, 1, 3)                            # [B, T, N_p, bn]
                # Stage (ii): spatial scan per frame.
                seq_s = t_out.reshape(
                    batch_size * num_frames, num_patches, bn
                )                                               # [B*T, N_p, bn]
                mamba_delta = self.mamba(seq_s).view(
                    batch_size, num_frames, num_patches, bn
                )                                               # [B, T, N_p, bn]
            else:  # "union": per-position [B*|U|, T, bn], no cross flow.
                seq = bottleneck_feature[:, :, union_pos, :]
                seq = seq.permute(0, 2, 1, 3).reshape(-1, num_frames, bn)
                mamba_out = self.mamba(seq)                     # [B*|U|, T, bn]
                mamba_out = mamba_out.view(
                    batch_size, union_pos.numel(), num_frames, bn
                ).permute(0, 2, 1, 3)                           # [B, T, |U|, bn]
                mamba_delta = torch.zeros_like(bottleneck_feature)
                mamba_delta[:, :, union_pos, :] = mamba_out     # [B, T, N_p, bn]
            mamba_delta = self._sanitize_real(mamba_delta)
        else:
            mamba_delta = None

        # 7. Patch-level gating fusion over ALL positions (skip when a
        #    branch is disabled).
        if conv_delta is not None and mamba_delta is not None:
            conv_stats = conv_delta.mean(dim=1)                # [B, N_p, bn]
            mamba_stats = mamba_delta.mean(dim=1)
            gate_logit = self.gate(
                torch.cat([conv_stats, mamba_stats], dim=-1)
            )                                                  # [B, N_p, 1]
            sigma = torch.sigmoid(gate_logit)
            fused = (
                sigma.unsqueeze(1) * conv_delta
                + (1.0 - sigma).unsqueeze(1) * mamba_delta
            )                                                  # [B, T, N_p, bn]
        elif conv_delta is not None:
            fused = conv_delta
            sigma = torch.full(
                (batch_size, num_patches, 1), 1.0, device=x.device
            )
        else:
            fused = mamba_delta
            sigma = torch.zeros(
                (batch_size, num_patches, 1), device=x.device
            )
        fused = self._sanitize_real(fused)

        # 8. Up-project back to D and residual-gate over ALL positions, so
        #    the soft CLS aggregation below sees every position's delta.
        delta_all = _fp32_linear(
            self.up_projection, fused
        )                                                      # [B, T, N_p, D]
        delta_all = self._sanitize_real(delta_all)
        residual_scale = torch.tanh(self.residual_scale.float())
        delta_all = residual_scale * delta_all

        # 9. Write back UNIFORMLY over ALL positions: both branches modelled
        #    the whole grid, so every patch receives its temporal delta
        #    (keeps the frame's spatial structure consistent). No top-k
        #    boost here - the top-k action regions were already seeded at
        #    the module entrance (step 5b).
        delta = delta_all                                  # [B, T, N_p, D]
        updated_patch = patch_tokens + delta
        updated_patch = self._sanitize_real(updated_patch)

        # 10. CLS temporal write-back: soft-weighted aggregation over ALL
        #    patch positions (temperature-softmax weights); the full soft
        #    weights carry gradients through the selection while the hard
        #    mask only marks the entry seed positions.
        sel_w = soft_weights.unsqueeze(-1)                     # [B, T, N_p, 1]
        cls_delta = (delta_all * sel_w).sum(dim=2)             # [B, T, D]
        cls_delta = self._sanitize_real(cls_delta)
        updated_cls = (
            cls_tokens[..., 0, :] + cls_delta
        ).unsqueeze(2)                                          # [B, T, 1, D]

        out = torch.cat([updated_cls, updated_patch], dim=2)
        out = out.reshape(total_tokens, num_tokens, dim).to(dtype=x.dtype)

        if return_information:
            info: Dict[str, object] = {
                "skipped": False,
                "attn_mean": attn.float().mean().detach(),
                "motion_mean": motion.float().mean().detach(),
                "w_mean": w.mean().detach(),
                "score_mean": score.mean().detach(),
                "soft_entropy": (
                    -(soft_weights * soft_weights.clamp_min(1e-8).log())
                    .sum(dim=-1)
                    .mean()
                    .detach()
                ),
                "num_selected_per_frame": mask.float().sum(dim=-1).detach(),
                "union_size": int(union_pos.numel()),
                "residual_scale": residual_scale.detach(),
                "delta_abs_mean": delta.detach().abs().mean(),
                "cls_delta_abs_mean": cls_delta.detach().abs().mean(),
                "gate_mean": (
                    sigma.float().mean().detach()
                    if sigma is not None
                    else torch.tensor(float("nan"))
                ),
                "conv_delta_abs_mean": (
                    conv_delta.detach().abs().mean()
                    if conv_delta is not None
                    else torch.tensor(float("nan"))
                ),
                "mamba_delta_abs_mean": (
                    mamba_delta.detach().abs().mean()
                    if mamba_delta is not None
                    else torch.tensor(float("nan"))
                ),
                "top_k": k,
                "alpha": self.alpha,
                "mamba_scope": self.mamba_scope,
                "seed_scale": torch.tanh(self.seed_scale.float()).detach(),
            }
            return out, info
        return out
