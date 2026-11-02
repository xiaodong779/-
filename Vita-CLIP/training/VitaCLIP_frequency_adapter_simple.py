"""Temporal frequency adapter used inside the supervised Vita-CLIP visual encoder.

For the current UCF101 configuration, the visual encoder calls this module
after blocks 3, 7 and 11 with tokens shaped ``[B*8, 197, 768]``.

The adapter:
1. restores the video layout ``[B, T, N, D]``;
2. builds a frame descriptor from CLS and spatial patches;
3. applies an FP32 temporal RFFT;
4. uses learnable soft low/middle/high-frequency boundaries;
5. applies heterogeneous processing to the three frequency bands;
6. performs an inverse RFFT;
7. updates only the CLS token and preserves every patch token.

The module is not tied to exactly eight frames, so later 8/16-frame
ablation experiments can reuse the same implementation.
"""

from __future__ import annotations

import math
from typing import Dict, Optional, Tuple, Union

import torch
from torch import Tensor, nn
import torch.nn.functional as F


FPSValue = Union[float, Tensor]


def _inverse_sigmoid(value: float) -> float:
    if not 0.0 < value < 1.0:
        raise ValueError(f"value must be in (0, 1), got {value}.")
    return math.log(value / (1.0 - value))


def _fp32_layer_norm(module: nn.LayerNorm, x: Tensor) -> Tensor:
    """
    Execute LayerNorm in float32.

    This remains valid even when the parent model has been converted by
    model.half(), because the parameters are temporarily viewed as float32
    during the functional operation.
    """
    weight = None if module.weight is None else module.weight.float()
    bias = None if module.bias is None else module.bias.float()

    return F.layer_norm(
        x.float(),
        module.normalized_shape,
        weight,
        bias,
        module.eps,
    )


def _fp32_linear(module: nn.Linear, x: Tensor) -> Tensor:
    """Execute Linear in float32 while preserving gradient flow."""
    weight = module.weight.float()
    bias = None if module.bias is None else module.bias.float()
    return F.linear(x.float(), weight, bias)


class VitaCLIPFrequencyAdapter(nn.Module):
    """
    Learnable soft-band temporal frequency adapter for Vita-CLIP.

    Input:
        x: [B*T, N, D]

    Output:
        y: [B*T, N, D]

    Main procedure:
        1. Restore [B, T, N, D].
        2. Build a frame descriptor from CLS and the mean spatial patch.
        3. Project D -> bottleneck.
        4. Apply temporal RFFT in float32.
        5. Separate DC and softly partition non-DC bins into low/mid/high bands.
        6. Apply heterogeneous processing to the three bands.
        7. Apply IRFFT and project bottleneck -> D.
        8. Update CLS only; preserve all non-CLS tokens.

    Important:
        c1 and c2 are dataset-level learnable soft boundaries. They are
        initialized from the equal-third prior, but they are not predicted
        independently for every input video.
    """

    def __init__(
        self,
        dim: int = 768,
        bottleneck: int = 192,
        temperature: float = 0.08,
        default_effective_fps: Optional[float] = None,
        reference_effective_fps: Optional[float] = None,
        num_spatial_patches: int = 196,
        low_gain_limit: float = 0.10,
        mid_gain_limit: float = 0.50,
        residual_scale_init: float = 0.2,
        stability_clip: float = 100.0,
        min_frames: int = 4,
        eps: float = 1e-6,
    ) -> None:
        super().__init__()

        if dim <= 0:
            raise ValueError("dim must be positive.")
        if bottleneck <= 0:
            raise ValueError("bottleneck must be positive.")
        if temperature <= 0:
            raise ValueError("temperature must be positive.")
        if num_spatial_patches <= 0:
            raise ValueError("num_spatial_patches must be positive.")
        if low_gain_limit <= 0:
            raise ValueError("low_gain_limit must be positive.")
        if mid_gain_limit <= 0:
            raise ValueError("mid_gain_limit must be positive.")
        if stability_clip <= 0:
            raise ValueError("stability_clip must be positive.")
        if min_frames < 2:
            raise ValueError("min_frames must be at least 2.")

        if default_effective_fps is not None and default_effective_fps <= 0:
            raise ValueError("default_effective_fps must be positive.")
        if (
            reference_effective_fps is not None
            and reference_effective_fps <= 0
        ):
            raise ValueError("reference_effective_fps must be positive.")

        self.dim = int(dim)
        self.bottleneck = int(bottleneck)
        self.temperature = float(temperature)

        self.default_effective_fps = default_effective_fps
        self.reference_effective_fps = reference_effective_fps

        self.num_spatial_patches = int(num_spatial_patches)
        self.low_gain_limit = float(low_gain_limit)
        self.mid_gain_limit = float(mid_gain_limit)

        self.stability_clip = float(stability_clip)
        self.min_frames = int(min_frames)
        self.eps = float(eps)

        # Frame descriptor and bottleneck projections.
        self.frame_norm = nn.LayerNorm(dim)
        self.down_projection = nn.Linear(dim, bottleneck)
        self.up_projection = nn.Linear(bottleneck, dim)

        # frame = CLS + sigmoid(patch_mix_logit) * mean(spatial patches)
        # sigmoid(0) = 0.5 at initialization.
        self.patch_mix_logit = nn.Parameter(
            torch.tensor(0.0, dtype=torch.float32)
        )

        # Equal-third initialization with non-overlapping valid ranges:
        # c1 in [0.20, 0.45], c2 in [0.55, 0.85].
        c1_initial = 1.0 / 3.0
        c2_initial = 2.0 / 3.0

        c1_ratio = (c1_initial - 0.20) / (0.45 - 0.20)
        c2_ratio = (c2_initial - 0.55) / (0.85 - 0.55)

        self.raw_c1 = nn.Parameter(
            torch.tensor(
                _inverse_sigmoid(c1_ratio),
                dtype=torch.float32,
            )
        )
        self.raw_c2 = nn.Parameter(
            torch.tensor(
                _inverse_sigmoid(c2_ratio),
                dtype=torch.float32,
            )
        )

        # Low-frequency processor.
        self.low_norm = nn.LayerNorm(bottleneck)
        self.low_projection = nn.Linear(
            bottleneck,
            bottleneck,
        )

        # Middle-frequency processor.
        self.mid_norm = nn.LayerNorm(bottleneck)
        self.mid_projection = nn.Linear(
            bottleneck,
            bottleneck,
        )

        # High-frequency adaptive shrinkage.
        self.high_norm = nn.LayerNorm(bottleneck)
        self.high_threshold_projection = nn.Linear(
            bottleneck,
            bottleneck,
        )
        self.high_bypass_projection = nn.Linear(
            bottleneck,
            bottleneck,
        )

        # Final residual strength:
        # CLS' = CLS + tanh(residual_scale) * frequency_delta
        self.residual_scale = nn.Parameter(
            torch.tensor(
                float(residual_scale_init),
                dtype=torch.float32,
            )
        )

        self._initialize_weights()

    def _initialize_weights(self) -> None:
        """
        Keep the adapter close to identity without blocking early gradients.

        The old version initialized both the band processors and up projection
        with a very small standard deviation. Their product made c1/c2
        gradients extremely small. Here, the up projection uses Xavier
        initialization while the final residual scale controls perturbation.
        """
        nn.init.xavier_uniform_(self.up_projection.weight)
        nn.init.zeros_(self.up_projection.bias)

        # Frequency-band projections use a moderate std so the frequency
        # delta is non-negligible at init (verified: std=1e-3 made the
        # frequency delta ~7e-4 and all gradients ~1e-10, effectively frozen;
        # in fp16 the residual was rounded away entirely and gradients became
        # exactly zero). 0.05 keeps the adapter near identity while allowing
        # early gradient flow.
        nn.init.normal_(
            self.low_projection.weight,
            mean=0.0,
            std=0.05,
        )
        nn.init.zeros_(self.low_projection.bias)

        nn.init.normal_(
            self.mid_projection.weight,
            mean=0.0,
            std=0.05,
        )
        nn.init.zeros_(self.mid_projection.bias)

        # Initially use a small threshold and preserve most raw high frequency.
        nn.init.normal_(
            self.high_threshold_projection.weight,
            mean=0.0,
            std=0.05,
        )
        nn.init.constant_(
            self.high_threshold_projection.bias,
            -3.0,
        )

        nn.init.normal_(
            self.high_bypass_projection.weight,
            mean=0.0,
            std=0.05,
        )
        nn.init.constant_(
            self.high_bypass_projection.bias,
            3.0,
        )

    def boundaries(self) -> Tuple[Tensor, Tensor]:
        """
        Return learnable normalized boundaries.

        c1 belongs to [0.20, 0.45].
        c2 belongs to [0.55, 0.85].
        """
        c1 = 0.20 + 0.25 * torch.sigmoid(self.raw_c1)
        c2 = 0.55 + 0.30 * torch.sigmoid(self.raw_c2)
        return c1, c2

    def _resolve_effective_fps(
        self,
        effective_fps: Optional[FPSValue],
        *,
        batch_size: int,
        device: torch.device,
    ) -> Optional[Tensor]:
        fps_value = (
            self.default_effective_fps
            if effective_fps is None
            else effective_fps
        )

        if fps_value is None:
            return None

        if self.reference_effective_fps is None:
            raise ValueError(
                "reference_effective_fps must be configured when "
                "effective_fps is used."
            )

        fps = torch.as_tensor(
            fps_value,
            device=device,
            dtype=torch.float32,
        )

        if fps.ndim == 0:
            fps = fps.expand(batch_size)
        elif fps.ndim == 1 and fps.numel() == batch_size:
            pass
        else:
            raise ValueError(
                "effective_fps must be a scalar or a tensor with shape [B], "
                f"got {tuple(fps.shape)} for B={batch_size}."
            )

        if torch.any(fps <= 0):
            raise ValueError("effective_fps must contain positive values.")

        return fps

    def _frequency_coordinate(
        self,
        *,
        batch_size: int,
        num_frames: int,
        effective_fps: Optional[FPSValue],
        device: torch.device,
    ) -> Tuple[Tensor, Tensor, str]:
        """
        Build coordinates for all non-DC RFFT bins.

        Relative mode:
            coordinate = frequency / Nyquist

        Physical mode:
            physical_hz = frequency_per_sample * effective_fps
            coordinate = physical_hz / reference_nyquist_hz

        torch.fft.rfftfreq handles both odd and even T correctly.
        """
        frequency_per_sample = torch.fft.rfftfreq(
            num_frames,
            d=1.0,
            device=device,
        )[1:]  # remove DC

        fps = self._resolve_effective_fps(
            effective_fps,
            batch_size=batch_size,
            device=device,
        )

        if fps is None:
            coordinate = (
                frequency_per_sample / 0.5
            ).view(1, -1).expand(batch_size, -1)

            frequency_hz = torch.full_like(
                coordinate,
                float("nan"),
            )
            mode = "relative"
        else:
            frequency_hz = (
                fps[:, None]
                * frequency_per_sample[None, :]
            )

            reference_nyquist_hz = (
                float(self.reference_effective_fps) / 2.0
            )
            coordinate = (
                frequency_hz / reference_nyquist_hz
            )
            mode = "physical"

        return coordinate, frequency_hz, mode

    def _soft_band_masks(
        self,
        coordinate: Tensor,
    ) -> Tuple[Tensor, Tensor, Tensor]:
        c1, c2 = self.boundaries()

        low = torch.sigmoid(
            (c1 - coordinate) / self.temperature
        )
        high = torch.sigmoid(
            (coordinate - c2) / self.temperature
        )
        mid = (1.0 - low) * (1.0 - high)

        masks = torch.stack(
            [low, mid, high],
            dim=-1,
        )
        masks = masks / masks.sum(
            dim=-1,
            keepdim=True,
        ).clamp_min(self.eps)

        low_mask = masks[..., 0].unsqueeze(-1)
        mid_mask = masks[..., 1].unsqueeze(-1)
        high_mask = masks[..., 2].unsqueeze(-1)

        return low_mask, mid_mask, high_mask

    def _sanitize_real(self, x: Tensor) -> Tensor:
        limit = self.stability_clip
        return torch.nan_to_num(
            x,
            nan=0.0,
            posinf=limit,
            neginf=-limit,
        ).clamp(
            min=-limit,
            max=limit,
        )

    def _sanitize_complex(self, x: Tensor) -> Tensor:
        real_and_imag = torch.view_as_real(x)
        real_and_imag = self._sanitize_real(
            real_and_imag
        )
        return torch.view_as_complex(
            real_and_imag.contiguous()
        )

    @torch.cuda.amp.autocast(enabled=False)
    def forward(
        self,
        x: Tensor,
        *,
        num_frames: int,
        effective_fps: Optional[FPSValue] = None,
        patch_start: int = 1,
        return_information: bool = False,
    ):
        """
        Args:
            x:
                Token tensor [B*T, N, D].
            num_frames:
                Number of frames T in each video.
            effective_fps:
                Sampling rate after temporal subsampling. Leave it as None
                when frames are uniformly sampled from clips with different
                durations and no reliable physical sampling rate is available.
            patch_start:
                First spatial-patch token index. For a plain CLIP ViT:
                patch_start = 1.
            return_information:
                Return debug information together with output.

        Returns:
            Tensor [B*T, N, D], or (Tensor, information).
        """
        num_frames = int(num_frames)
        patch_start = int(patch_start)

        if x.ndim != 3:
            raise ValueError(
                f"x must be [B*T, N, D], got {tuple(x.shape)}."
            )
        if not x.is_floating_point():
            raise TypeError(
                "x must be a floating-point token tensor, "
                f"got dtype={x.dtype}."
            )

        bt, num_tokens, feature_dim = x.shape

        if num_frames < self.min_frames:
            raise ValueError(
                f"num_frames must be >= {self.min_frames}, "
                f"got {num_frames}."
            )
        if bt % num_frames != 0:
            raise ValueError(
                f"B*T={bt} is not divisible by T={num_frames}."
            )
        if feature_dim != self.dim:
            raise ValueError(
                f"Expected D={self.dim}, got D={feature_dim}."
            )
        if patch_start < 1:
            raise ValueError(
                "patch_start must be >= 1 because token 0 is CLS."
            )

        patch_end = (
            patch_start + self.num_spatial_patches
        )
        if patch_end > num_tokens:
            raise ValueError(
                f"Patch range [{patch_start}:{patch_end}] "
                f"exceeds N={num_tokens}."
            )

        batch_size = bt // num_frames

        # Restore the video layout.
        video_tokens = x.reshape(
            batch_size,
            num_frames,
            num_tokens,
            feature_dim,
        )

        cls_tokens = video_tokens[:, :, 0, :]

        spatial_patches = video_tokens[
            :,
            :,
            patch_start:patch_end,
            :,
        ]

        # Patch averaging is done in float32 for stability.
        patch_mean = spatial_patches.float().mean(
            dim=2
        )

        patch_mix = torch.sigmoid(
            self.patch_mix_logit.float()
        )

        frame_feature = (
            cls_tokens.float()
            + patch_mix * patch_mean
        )
        frame_feature = _fp32_layer_norm(
            self.frame_norm,
            frame_feature,
        )

        bottleneck_feature = F.gelu(
            _fp32_linear(
                self.down_projection,
                frame_feature,
            )
        )
        bottleneck_feature = self._sanitize_real(
            bottleneck_feature
        )

        # Temporal FFT always operates in float32 -> complex64.
        spectrum = torch.fft.rfft(
            bottleneck_feature,
            dim=1,
            norm="ortho",
        )

        dc = spectrum[:, :1]
        non_dc = spectrum[:, 1:]

        coordinate, frequency_hz, frequency_mode = (
            self._frequency_coordinate(
                batch_size=batch_size,
                num_frames=num_frames,
                effective_fps=effective_fps,
                device=x.device,
            )
        )

        if coordinate.shape[1] != non_dc.shape[1]:
            raise RuntimeError(
                "Frequency-bin mismatch: "
                f"coordinate={coordinate.shape[1]}, "
                f"RFFT non-DC bins={non_dc.shape[1]}."
            )

        low_mask, mid_mask, high_mask = (
            self._soft_band_masks(coordinate)
        )

        # Keep real masks in FP32 so complex spectral operations remain
        # complex64 instead of the experimental ComplexHalf dtype.
        spectral_real_dtype = non_dc.real.dtype
        low_mask = low_mask.to(dtype=spectral_real_dtype)
        mid_mask = mid_mask.to(dtype=spectral_real_dtype)
        high_mask = high_mask.to(dtype=spectral_real_dtype)

        low_band = non_dc * low_mask
        mid_band = non_dc * mid_mask
        high_band = non_dc * high_mask

        # Low frequency: mild phase-preserving correction.
        low_statistic = torch.log1p(
            low_band.abs()
        )
        low_hidden = _fp32_layer_norm(
            self.low_norm,
            low_statistic,
        )
        low_scale = (
            self.low_gain_limit
            * torch.tanh(
                _fp32_linear(
                    self.low_projection,
                    low_hidden,
                )
            )
        )
        low_delta = low_scale * low_band

        # Middle frequency: stronger phase-preserving correction.
        mid_statistic = torch.log1p(
            mid_band.abs()
        )
        mid_hidden = _fp32_layer_norm(
            self.mid_norm,
            mid_statistic,
        )
        mid_scale = (
            self.mid_gain_limit
            * torch.tanh(
                _fp32_linear(
                    self.mid_projection,
                    mid_hidden,
                )
            )
        )
        mid_delta = mid_scale * mid_band

        # High frequency: adaptive soft shrinkage.
        high_magnitude = high_band.abs()
        high_statistic = high_magnitude.mean(
            dim=1
        )

        high_hidden = _fp32_layer_norm(
            self.high_norm,
            torch.log1p(high_statistic),
        )

        threshold_ratio = torch.sigmoid(
            _fp32_linear(
                self.high_threshold_projection,
                high_hidden,
            )
        )
        threshold = (
            threshold_ratio[:, None, :]
            * high_statistic[:, None, :].clamp_min(
                self.eps
            )
        )

        high_direction = (
            high_band
            / high_magnitude.clamp_min(self.eps)
        )
        shrunk_magnitude = F.relu(
            high_magnitude - threshold
        )
        high_shrunk = (
            high_direction * shrunk_magnitude
        )

        bypass = torch.sigmoid(
            _fp32_linear(
                self.high_bypass_projection,
                high_hidden,
            )
        )[:, None, :]

        high_processed = (
            bypass * high_band
            + (1.0 - bypass) * high_shrunk
        )
        high_delta = high_processed - high_band

        # DC is protected; only temporal variation generates a residual.
        delta_spectrum = torch.cat(
            [
                torch.zeros_like(dc),
                low_delta + mid_delta + high_delta,
            ],
            dim=1,
        )
        delta_spectrum = self._sanitize_complex(
            delta_spectrum
        )

        delta_time = torch.fft.irfft(
            delta_spectrum,
            n=num_frames,
            dim=1,
            norm="ortho",
        )
        delta_time = self._sanitize_real(
            delta_time
        )

        cls_delta = _fp32_linear(
            self.up_projection,
            delta_time,
        )

        residual_scale = torch.tanh(
            self.residual_scale.float()
        )
        cls_delta = (
            residual_scale * cls_delta
        )
        cls_delta = self._sanitize_real(
            cls_delta
        )

        updated_cls = (
            cls_tokens.float() + cls_delta
        ).to(dtype=x.dtype)

        # Only CLS changes. Every token from index 1 onward is preserved.
        output = torch.cat(
            [
                updated_cls.unsqueeze(2),
                video_tokens[:, :, 1:, :],
            ],
            dim=2,
        ).reshape(
            bt,
            num_tokens,
            feature_dim,
        ).contiguous()

        if output.shape != x.shape:
            raise RuntimeError(
                "Frequency adapter must preserve the complete token shape: "
                f"input={tuple(x.shape)}, output={tuple(output.shape)}."
            )

        if return_information:
            c1, c2 = self.boundaries()

            # Diagnostics requested for the integration report:
            #   - frequency_delta magnitude (frequency_delta_abs_mean)
            #   - overall output perturbation (output_diff_abs_mean)
            #   - high-frequency band behavior (threshold / bypass / amp ratio)
            with torch.no_grad():
                output_diff_abs_mean = (
                    output - x
                ).abs().mean()
                high_before_abs_mean = (
                    high_band.abs().mean()
                )
                high_after_abs_mean = (
                    high_processed.abs().mean()
                )
                high_ratio = (
                    high_after_abs_mean
                    / high_before_abs_mean.clamp_min(self.eps)
                )

            information: Dict[str, object] = {
                "input_shape": tuple(x.shape),
                "output_shape": tuple(output.shape),
                "batch_size": batch_size,
                "num_frames": num_frames,
                "num_tokens": num_tokens,
                "feature_dim": feature_dim,
                "patch_start": patch_start,
                "patch_end": patch_end,
                "frequency_mode": frequency_mode,
                "frequency_coordinate": coordinate.detach(),
                "frequency_hz": frequency_hz.detach(),
                "low_mask": low_mask.detach(),
                "mid_mask": mid_mask.detach(),
                "high_mask": high_mask.detach(),
                "c1": c1.detach(),
                "c2": c2.detach(),
                "raw_c1": self.raw_c1.detach().clone(),
                "raw_c2": self.raw_c2.detach().clone(),
                "patch_mix": patch_mix.detach(),
                "residual_scale": residual_scale.detach(),
                "residual_scale_raw": self.residual_scale.detach().clone(),
                "cls_delta_abs_mean": (
                    cls_delta.detach().abs().mean()
                ),
                "frequency_delta_abs_mean": (
                    delta_time.detach().abs().mean()
                ),
                "output_diff_abs_mean": output_diff_abs_mean,
                "high_threshold": threshold_ratio.detach().mean(),
                "high_bypass": bypass.detach().mean(),
                "high_before_abs_mean": high_before_abs_mean,
                "high_after_abs_mean": high_after_abs_mean,
                "high_amp_ratio": high_ratio,
            }
            return output, information

        return output

    def extra_repr(self) -> str:
        return (
            f"dim={self.dim}, "
            f"bottleneck={self.bottleneck}, "
            f"temperature={self.temperature}, "
            f"num_spatial_patches={self.num_spatial_patches}, "
            f"min_frames={self.min_frames}"
        )


# Backward-friendly alias for configurations that use a shorter class name.
FrequencyAdapter = VitaCLIPFrequencyAdapter

__all__ = [
    "VitaCLIPFrequencyAdapter",
    "FrequencyAdapter",
]