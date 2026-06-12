import copy
import math
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from models.Hybrid_mstplusplus import GatedConvFFN, MST_Plus_Plus


class SinusoidalTimeEmbedding(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        if dim < 2:
            raise ValueError("time embedding dimension must be at least 2")
        self.dim = dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        t = t.float()
        half = self.dim // 2
        exponent = math.log(10000.0) / max(half - 1, 1)
        frequencies = torch.exp(
            torch.arange(half, device=t.device, dtype=t.dtype) * -exponent
        )
        embedding = t[:, None] * frequencies[None, :]
        embedding = torch.cat([embedding.sin(), embedding.cos()], dim=-1)
        if embedding.shape[-1] < self.dim:
            embedding = F.pad(embedding, (0, self.dim - embedding.shape[-1]))
        return embedding


class ConditionalDiTBlock(nn.Module):
    """DiT block with timestep and RGB-bottleneck conditioning."""

    def __init__(self, dim: int, num_heads: int = 4, ffn_expansion: int = 2):
        super().__init__()
        if dim % num_heads != 0:
            raise ValueError(f"dim={dim} must be divisible by num_heads={num_heads}")

        self.norm1 = nn.LayerNorm(dim, elementwise_affine=False)
        self.attention = nn.MultiheadAttention(
            embed_dim=dim,
            num_heads=num_heads,
            batch_first=True,
        )
        self.norm2 = nn.LayerNorm(dim, elementwise_affine=False)
        self.ffn = GatedConvFFN(dim=dim, expansion=ffn_expansion)

        self.modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(dim, dim * 6),
        )
        nn.init.zeros_(self.modulation[-1].weight)
        nn.init.zeros_(self.modulation[-1].bias)

    def forward(
        self,
        x: torch.Tensor,
        condition: torch.Tensor,
        height: int,
        width: int,
    ) -> torch.Tensor:
        (
            shift_attn,
            scale_attn,
            gate_attn,
            shift_ffn,
            scale_ffn,
            gate_ffn,
        ) = self.modulation(condition).chunk(6, dim=-1)

        h = self.norm1(x)
        h = h * (1.0 + scale_attn[:, None, :]) + shift_attn[:, None, :]
        attention_output, _ = self.attention(h, h, h, need_weights=False)
        x = x + gate_attn[:, None, :] * attention_output

        h = self.norm2(x)
        h = h * (1.0 + scale_ffn[:, None, :]) + shift_ffn[:, None, :]
        ffn_output = self.ffn(h, height=height, width=width)
        return x + gate_ffn[:, None, :] * ffn_output


class ConditionalBottleneckDiT(nn.Module):
    """
    Predicts Gaussian noise in a noised HSI bottleneck, conditioned on the
    corresponding RGB bottleneck and diffusion timestep.

    Input/output shape: [B, feature_channels, H/4, W/4].
    """

    def __init__(
        self,
        feature_channels: int = 124,
        hidden_dim: int = 128,
        patch_size: int = 4,
        depth: int = 2,
        num_heads: int = 4,
        time_dim: int = 128,
    ):
        super().__init__()
        self.patch_size = patch_size

        self.noisy_patch_embed = nn.Conv2d(
            feature_channels,
            hidden_dim,
            kernel_size=patch_size,
            stride=patch_size,
        )
        self.condition_patch_embed = nn.Conv2d(
            feature_channels,
            hidden_dim,
            kernel_size=patch_size,
            stride=patch_size,
        )

        self.time_mlp = nn.Sequential(
            SinusoidalTimeEmbedding(time_dim),
            nn.Linear(time_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        self.blocks = nn.ModuleList(
            [
                ConditionalDiTBlock(
                    dim=hidden_dim,
                    num_heads=num_heads,
                    ffn_expansion=2,
                )
                for _ in range(depth)
            ]
        )

        self.final_norm = nn.LayerNorm(hidden_dim)
        self.unpatchify = nn.ConvTranspose2d(
            hidden_dim,
            feature_channels,
            kernel_size=patch_size,
            stride=patch_size,
        )

        # Zero initialization makes the initial network predict near-zero noise.
        nn.init.zeros_(self.unpatchify.weight)
        if self.unpatchify.bias is not None:
            nn.init.zeros_(self.unpatchify.bias)

    def _pad(self, x: torch.Tensor) -> Tuple[torch.Tensor, int, int]:
        h, w = x.shape[-2:]
        pad_h = (self.patch_size - h % self.patch_size) % self.patch_size
        pad_w = (self.patch_size - w % self.patch_size) % self.patch_size
        if pad_h or pad_w:
            x = F.pad(x, (0, pad_w, 0, pad_h), mode="reflect")
        return x, h, w

    def forward(
        self,
        noisy_target: torch.Tensor,
        t: torch.Tensor,
        rgb_condition: torch.Tensor,
    ) -> torch.Tensor:
        if noisy_target.shape != rgb_condition.shape:
            raise ValueError(
                "noisy_target and rgb_condition must have the same shape, "
                f"got {tuple(noisy_target.shape)} and {tuple(rgb_condition.shape)}"
            )

        noisy_target, original_h, original_w = self._pad(noisy_target)
        rgb_condition, _, _ = self._pad(rgb_condition)

        noisy_tokens = self.noisy_patch_embed(noisy_target)
        condition_tokens = self.condition_patch_embed(rgb_condition)
        hp, wp = noisy_tokens.shape[-2:]

        noisy_tokens = rearrange(noisy_tokens, "b c h w -> b (h w) c")
        condition_tokens = rearrange(condition_tokens, "b c h w -> b (h w) c")

        # Local token-level conditioning plus global adaLN conditioning.
        x = noisy_tokens + condition_tokens
        global_condition = condition_tokens.mean(dim=1) + self.time_mlp(t.float())

        for block in self.blocks:
            x = block(
                x,
                condition=global_condition,
                height=hp,
                width=wp,
            )

        x = self.final_norm(x)
        x = rearrange(x, "b (h w) c -> b c h w", h=hp, w=wp)
        x = self.unpatchify(x)
        return x[:, :, :original_h, :original_w]


class DiffusionScheduler:
    """Linear DDPM training schedule with DDIM sampling support."""

    def __init__(
        self,
        timesteps: int = 1000,
        beta_start: float = 1e-4,
        beta_end: float = 2e-2,
    ):
        if timesteps < 2:
            raise ValueError("timesteps must be at least 2")
        self.timesteps = timesteps
        self.betas = torch.linspace(beta_start, beta_end, timesteps)
        self.alphas = 1.0 - self.betas
        self.alpha_bars = torch.cumprod(self.alphas, dim=0)

    def sample_timesteps(self, batch_size: int, device: torch.device) -> torch.Tensor:
        return torch.randint(0, self.timesteps, (batch_size,), device=device)

    def add_noise(
        self,
        clean: torch.Tensor,
        t: torch.Tensor,
        noise: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if noise is None:
            noise = torch.randn_like(clean)
        t = t.to(clean.device, dtype=torch.long)
        alpha_bars = self.alpha_bars.to(clean.device, dtype=clean.dtype)
        alpha_bar_t = alpha_bars[t].view(-1, 1, 1, 1)
        noisy = (
            torch.sqrt(alpha_bar_t) * clean
            + torch.sqrt(torch.clamp(1.0 - alpha_bar_t, min=0.0)) * noise
        )
        return noisy, noise

    def predict_clean(
        self,
        noisy: torch.Tensor,
        predicted_noise: torch.Tensor,
        t: torch.Tensor,
    ) -> torch.Tensor:
        t = t.to(noisy.device, dtype=torch.long)
        alpha_bars = self.alpha_bars.to(noisy.device, dtype=noisy.dtype)
        alpha_bar_t = alpha_bars[t].view(-1, 1, 1, 1)
        return (
            noisy
            - torch.sqrt(torch.clamp(1.0 - alpha_bar_t, min=0.0))
            * predicted_noise
        ) / torch.sqrt(torch.clamp(alpha_bar_t, min=1e-8))

    def inference_timesteps(self, num_steps: int, device: torch.device) -> torch.Tensor:
        if not 1 <= num_steps <= self.timesteps:
            raise ValueError(f"num_steps must be in [1, {self.timesteps}]")
        ascending = torch.linspace(0, self.timesteps - 1, num_steps, device=device)
        ascending = torch.unique_consecutive(ascending.round().long())
        return torch.flip(ascending, dims=[0])


@dataclass
class RGBContext:
    base_feature: torch.Tensor
    stage_input: torch.Tensor
    rgb_bottleneck: torch.Tensor
    skip_features: Tuple[torch.Tensor, ...]
    original_size: Tuple[int, int]


class MSTPlusPlusBottleneckDiffusion(nn.Module):
    """
    Uses the first two MST++ stages as a deterministic RGB condition encoder.
    The final MST stage bottleneck is replaced by a conditional DiT denoiser.

    During training, a frozen copy of the pretrained final stage maps the
    ground-truth HSI to a stable clean bottleneck target.
    """

    def __init__(
        self,
        backbone: MST_Plus_Plus,
        hidden_dim: int = 128,
        patch_size: int = 4,
        depth: int = 2,
        num_heads: int = 4,
        time_dim: int = 128,
        build_teacher: bool = True,
    ):
        super().__init__()
        if len(backbone.body) < 1:
            raise ValueError("MST++ backbone must contain at least one MST stage")

        self.backbone = backbone
        final_stage = self.backbone.body[-1]

        feature_channels = final_stage.dim * (2 ** final_stage.stage)

        self.teacher_stage = copy.deepcopy(final_stage) if build_teacher else None
        if self.teacher_stage is not None:
            self.teacher_stage.eval()
            for parameter in self.teacher_stage.parameters():
                parameter.requires_grad_(False)

        # The original deterministic final-stage bottleneck is replaced.
        final_stage.bottleneck = nn.Identity()

        self.denoiser = ConditionalBottleneckDiT(
            feature_channels=feature_channels,
            hidden_dim=hidden_dim,
            patch_size=patch_size,
            depth=depth,
            num_heads=num_heads,
            time_dim=time_dim,
        )

    @staticmethod
    def _pad_input(x: torch.Tensor) -> Tuple[torch.Tensor, Tuple[int, int]]:
        h, w = x.shape[-2:]
        pad_h = (8 - h % 8) % 8
        pad_w = (8 - w % 8) % 8
        if pad_h or pad_w:
            x = F.pad(x, (0, pad_w, 0, pad_h), mode="reflect")
        return x, (h, w)

    @staticmethod
    def _encode_stage(stage: nn.Module, x: torch.Tensor, use_bottleneck: bool):
        feature = stage.embedding(x)
        skips = []
        for block, downsample in stage.encoder_layers:
            feature = block(feature)
            skips.append(feature)
            feature = downsample(feature)
        if use_bottleneck:
            feature = stage.bottleneck(feature)
        return feature, tuple(skips)

    @staticmethod
    def _decode_stage(
        stage: nn.Module,
        bottleneck: torch.Tensor,
        skips: Tuple[torch.Tensor, ...],
        stage_input: torch.Tensor,
    ) -> torch.Tensor:
        feature = bottleneck
        for index, (upsample, fusion, block) in enumerate(stage.decoder_layers):
            feature = upsample(feature)
            skip = skips[stage.stage - 1 - index]
            feature = fusion(torch.cat([feature, skip], dim=1))
            feature = block(feature)
        return stage.mapping(feature) + stage_input

    def encode_rgb(self, rgb: torch.Tensor, detach: bool = False) -> RGBContext:
        rgb, original_size = self._pad_input(rgb)
        base_feature = self.backbone.conv_in(rgb)

        stage_input = base_feature
        for stage in self.backbone.body[:-1]:
            stage_input = stage(stage_input)

        final_stage = self.backbone.body[-1]
        rgb_bottleneck, skips = self._encode_stage(
            final_stage,
            stage_input,
            use_bottleneck=False,
        )

        if detach:
            base_feature = base_feature.detach()
            stage_input = stage_input.detach()
            rgb_bottleneck = rgb_bottleneck.detach()
            skips = tuple(skip.detach() for skip in skips)

        return RGBContext(
            base_feature=base_feature,
            stage_input=stage_input,
            rgb_bottleneck=rgb_bottleneck,
            skip_features=skips,
            original_size=original_size,
        )

    @torch.no_grad()
    def encode_hsi_target(self, hsi: torch.Tensor) -> torch.Tensor:
        if self.teacher_stage is None:
            raise RuntimeError("Teacher stage is not available in inference-only mode")
        hsi, _ = self._pad_input(hsi)
        target, _ = self._encode_stage(
            self.teacher_stage,
            hsi,
            use_bottleneck=True,
        )
        return target.detach()

    def decode_bottleneck(
        self,
        bottleneck: torch.Tensor,
        context: RGBContext,
    ) -> torch.Tensor:
        final_stage = self.backbone.body[-1]
        final_feature = self._decode_stage(
            final_stage,
            bottleneck,
            context.skip_features,
            context.stage_input,
        )
        output = self.backbone.conv_out(final_feature) + context.base_feature
        h, w = context.original_size
        return output[:, :, :h, :w]

    def freeze_condition_encoder(self, train_final_decoder: bool = True) -> None:
        for parameter in self.backbone.parameters():
            parameter.requires_grad_(False)

        if train_final_decoder:
            final_stage = self.backbone.body[-1]
            for parameter in final_stage.decoder_layers.parameters():
                parameter.requires_grad_(True)
            for parameter in final_stage.mapping.parameters():
                parameter.requires_grad_(True)
            for parameter in self.backbone.conv_out.parameters():
                parameter.requires_grad_(True)

        for parameter in self.denoiser.parameters():
            parameter.requires_grad_(True)

    @torch.no_grad()
    def sample_ddim(
        self,
        rgb: torch.Tensor,
        scheduler: DiffusionScheduler,
        num_steps: int = 20,
        eta: float = 0.0,
        initial_noise: Optional[torch.Tensor] = None,
        target_mean: Optional[torch.Tensor] = None,
        target_std: Optional[torch.Tensor] = None,
        condition_mean: Optional[torch.Tensor] = None,
        condition_std: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        self.eval()
        context = self.encode_rgb(rgb, detach=True)
        raw_condition = context.rgb_bottleneck

        if condition_mean is None:
            condition_mean = torch.zeros(
                1, raw_condition.size(1), 1, 1,
                device=raw_condition.device, dtype=raw_condition.dtype
            )
        if condition_std is None:
            condition_std = torch.ones_like(condition_mean)
        condition = (raw_condition - condition_mean) / condition_std.clamp_min(1e-6)

        if initial_noise is None:
            current = torch.randn_like(condition)
        else:
            current = initial_noise.to(condition.device, dtype=condition.dtype)
            if current.shape != condition.shape:
                raise ValueError("initial_noise shape must match the RGB bottleneck")

        alpha_bars = scheduler.alpha_bars.to(condition.device, dtype=condition.dtype)
        timesteps = scheduler.inference_timesteps(num_steps, condition.device)

        for index, timestep in enumerate(timesteps):
            step = int(timestep.item())
            t_batch = torch.full(
                (condition.size(0),),
                step,
                device=condition.device,
                dtype=torch.long,
            )
            predicted_noise = self.denoiser(current, t_batch, condition)
            alpha_bar_t = alpha_bars[step]

            if index + 1 < len(timesteps):
                previous_step = int(timesteps[index + 1].item())
                alpha_bar_previous = alpha_bars[previous_step]
            else:
                alpha_bar_previous = torch.ones(
                    (), device=condition.device, dtype=condition.dtype
                )

            predicted_clean = (
                current
                - torch.sqrt(torch.clamp(1.0 - alpha_bar_t, min=0.0))
                * predicted_noise
            ) / torch.sqrt(torch.clamp(alpha_bar_t, min=1e-8))

            variance_ratio = (
                (1.0 - alpha_bar_previous)
                / torch.clamp(1.0 - alpha_bar_t, min=1e-8)
            )
            transition_ratio = (
                1.0
                - alpha_bar_t
                / torch.clamp(alpha_bar_previous, min=1e-8)
            )
            sigma = eta * torch.sqrt(
                torch.clamp(variance_ratio * transition_ratio, min=0.0)
            )
            direction = torch.sqrt(
                torch.clamp(
                    1.0 - alpha_bar_previous - sigma.square(),
                    min=0.0,
                )
            ) * predicted_noise

            current = torch.sqrt(alpha_bar_previous) * predicted_clean + direction
            if eta > 0.0 and index + 1 < len(timesteps):
                current = current + sigma * torch.randn_like(current)

        if target_mean is None:
            target_mean = torch.zeros(
                1, current.size(1), 1, 1,
                device=current.device, dtype=current.dtype
            )
        if target_std is None:
            target_std = torch.ones_like(target_mean)

        sampled_bottleneck = current * target_std + target_mean
        return self.decode_bottleneck(sampled_bottleneck, context)


def load_pretrained_backbone(
    checkpoint_path: str,
    device: torch.device,
) -> MST_Plus_Plus:
    backbone = MST_Plus_Plus(
        in_channels=3,
        out_channels=31,
        n_feat=31,
        stage=3,
    ).to(device)

    checkpoint = torch.load(checkpoint_path, map_location=device)
    state = checkpoint.get("model", checkpoint) if isinstance(checkpoint, dict) else checkpoint

    # Handle checkpoints saved from DataParallel.
    if any(key.startswith("module.") for key in state):
        state = {key.removeprefix("module."): value for key, value in state.items()}

    backbone.load_state_dict(state, strict=True)
    return backbone
