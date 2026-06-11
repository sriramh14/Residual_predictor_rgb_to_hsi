import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class SinusoidalTimeEmbedding(nn.Module):
    def __init__(self, dim: int):
        super().__init__()

        if dim < 2:
            raise ValueError("Time embedding dimension must be at least 2.")

        self.dim = dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        t = t.float()

        half_dim = self.dim // 2

        if half_dim == 1:
            frequencies = torch.ones(
                1,
                device=t.device,
                dtype=t.dtype
            )
        else:
            exponent = math.log(10000.0) / (half_dim - 1)

            frequencies = torch.exp(
                torch.arange(
                    half_dim,
                    device=t.device,
                    dtype=t.dtype
                ) * -exponent
            )

        embedding = t[:, None] * frequencies[None, :]

        embedding = torch.cat(
            [embedding.sin(), embedding.cos()],
            dim=-1
        )

        if embedding.shape[-1] < self.dim:
            embedding = F.pad(
                embedding,
                (0, self.dim - embedding.shape[-1])
            )

        return embedding


class ResidualDiffusionModel(nn.Module):
    def __init__(
        self,
        latent_dim: int = 8,
        hidden_dim: int = 64,
        time_dim: int = 128
    ):
        super().__init__()

        self.time_mlp = nn.Sequential(
            SinusoidalTimeEmbedding(time_dim),
            nn.Linear(time_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )

        self.conv1 = nn.Conv2d(
            latent_dim * 2,
            hidden_dim,
            kernel_size=3,
            padding=1
        )

        self.conv2 = nn.Conv2d(
            hidden_dim,
            hidden_dim,
            kernel_size=3,
            padding=1
        )

        self.conv3 = nn.Conv2d(
            hidden_dim,
            hidden_dim,
            kernel_size=3,
            padding=1
        )

        self.conv4 = nn.Conv2d(
            hidden_dim,
            hidden_dim,
            kernel_size=3,
            padding=1
        )

        self.out = nn.Conv2d(
            hidden_dim,
            latent_dim,
            kernel_size=3,
            padding=1
        )

        self.norm1 = nn.GroupNorm(8, hidden_dim)
        self.norm2 = nn.GroupNorm(8, hidden_dim)
        self.norm3 = nn.GroupNorm(8, hidden_dim)
        self.norm4 = nn.GroupNorm(8, hidden_dim)

        self.act = nn.SiLU()

    def forward(
        self,
        noisy_residual: torch.Tensor,
        t: torch.Tensor,
        z_rgb: torch.Tensor
    ) -> torch.Tensor:
        t_embedding = self.time_mlp(t)

        x = torch.cat(
            [noisy_residual, z_rgb],
            dim=1
        )

        x = self.conv1(x)
        x = self.norm1(x)
        x = x + t_embedding[:, :, None, None]
        x = self.act(x)

        x = self.act(self.norm2(self.conv2(x)))
        x = self.act(self.norm3(self.conv3(x)))
        x = self.act(self.norm4(self.conv4(x)))

        return self.out(x)


class DiffusionScheduler:
    """
    Forward diffusion scheduler compatible with DDIM inference.

    Important:
    The same schedule and number of training timesteps must be used during
    training and inference.
    """

    def __init__(
        self,
        timesteps: int = 20,
        schedule: str = "cosine",
        beta_start: float = 1e-4,
        beta_end: float = 2e-2,
        cosine_s: float = 0.008,
        max_beta: float = 0.999
    ):
        if timesteps < 2:
            raise ValueError("timesteps must be at least 2.")

        if schedule not in {"linear", "cosine"}:
            raise ValueError(
                "schedule must be 'linear' or 'cosine'."
            )

        self.timesteps = timesteps
        self.schedule = schedule

        if schedule == "linear":
            betas = torch.linspace(
                beta_start,
                beta_end,
                timesteps,
                dtype=torch.float32
            )
        else:
            betas = self._cosine_beta_schedule(
                timesteps=timesteps,
                s=cosine_s,
                max_beta=max_beta
            )

        self.betas = betas
        self.alphas = 1.0 - betas
        self.alpha_bars = torch.cumprod(
            self.alphas,
            dim=0
        )

    @staticmethod
    def _cosine_beta_schedule(
        timesteps: int,
        s: float = 0.008,
        max_beta: float = 0.999
    ) -> torch.Tensor:
        steps = timesteps + 1

        x = torch.linspace(
            0,
            timesteps,
            steps,
            dtype=torch.float64
        )

        alpha_bars = torch.cos(
            (
                (x / timesteps + s)
                / (1.0 + s)
            )
            * math.pi
            * 0.5
        ).pow(2)

        alpha_bars = alpha_bars / alpha_bars[0]

        betas = 1.0 - (
            alpha_bars[1:]
            / alpha_bars[:-1]
        )

        return betas.clamp(
            min=1e-8,
            max=max_beta
        ).float()

    def sample_timesteps(
        self,
        batch_size: int,
        device: torch.device
    ) -> torch.Tensor:
        return torch.randint(
            low=0,
            high=self.timesteps,
            size=(batch_size,),
            device=device,
            dtype=torch.long
        )

    def add_noise(
        self,
        clean_residual: torch.Tensor,
        t: torch.Tensor,
        noise: Optional[torch.Tensor] = None
    ):
        if noise is None:
            noise = torch.randn_like(clean_residual)

        t = t.to(
            device=clean_residual.device,
            dtype=torch.long
        )

        alpha_bars = self.alpha_bars.to(
            device=clean_residual.device,
            dtype=clean_residual.dtype
        )

        alpha_bar_t = alpha_bars[t].view(
            -1, 1, 1, 1
        )

        noisy_residual = (
            torch.sqrt(alpha_bar_t) * clean_residual
            + torch.sqrt(
                torch.clamp(
                    1.0 - alpha_bar_t,
                    min=0.0
                )
            ) * noise
        )

        return noisy_residual, noise

    def ddim_timesteps(
        self,
        num_inference_steps: Optional[int],
        device: torch.device
    ) -> torch.Tensor:
        if num_inference_steps is None:
            num_inference_steps = self.timesteps

        if not 1 <= num_inference_steps <= self.timesteps:
            raise ValueError(
                "num_inference_steps must be in the interval "
                f"[1, {self.timesteps}]."
            )

        if num_inference_steps == self.timesteps:
            return torch.arange(
                self.timesteps - 1,
                -1,
                -1,
                device=device,
                dtype=torch.long
            )

        # Uniformly select training timesteps and traverse them in reverse.
        ascending = torch.linspace(
            0,
            self.timesteps - 1,
            num_inference_steps,
            device=device
        ).round().long()

        ascending = torch.unique_consecutive(ascending)

        return torch.flip(
            ascending,
            dims=[0]
        )


def diffusion_loss(
    model: nn.Module,
    scheduler: DiffusionScheduler,
    z_rgb: torch.Tensor,
    z_hsi: torch.Tensor
) -> torch.Tensor:
    residual_gt = z_hsi - z_rgb

    t = scheduler.sample_timesteps(
        batch_size=z_rgb.size(0),
        device=z_rgb.device
    )

    noisy_residual, noise = scheduler.add_noise(
        clean_residual=residual_gt,
        t=t
    )

    predicted_noise = model(
        noisy_residual,
        t.float(),
        z_rgb
    )

    return F.mse_loss(
        predicted_noise,
        noise
    )


def _dynamic_threshold(
    x: torch.Tensor,
    percentile: float = 0.995,
    minimum_scale: float = 1.0
) -> torch.Tensor:
    """
    Prevent extreme x0 estimates from exploding during DDIM sampling.

    The operation is applied independently to each sample.
    """
    batch_size = x.shape[0]

    flattened = x.abs().reshape(batch_size, -1)

    scale = torch.quantile(
        flattened,
        percentile,
        dim=1
    )

    scale = torch.maximum(
        scale,
        torch.full_like(scale, minimum_scale)
    )

    scale = scale.view(
        batch_size,
        1,
        1,
        1
    )

    return torch.clamp(
        x,
        -scale,
        scale
    ) / scale


@torch.no_grad()
def sample_residual_ddim(
    model: nn.Module,
    scheduler: DiffusionScheduler,
    z_rgb: torch.Tensor,
    num_inference_steps: Optional[int] = None,
    eta: float = 0.0,
    initial_noise: Optional[torch.Tensor] = None,
    use_dynamic_threshold: bool = True,
    dynamic_threshold_percentile: float = 0.995
) -> torch.Tensor:
    """
    Perform the complete DDIM reverse process.

    Defaults
    --------
    num_inference_steps=None:
        Uses all 20 training timesteps by default.
    eta=0:
        Deterministic DDIM reverse transitions after the initial noise.
    use_dynamic_threshold=True:
        Prevents extremely large clean-residual estimates that can cause the
        decoder output, PSNR, and SSIM to become invalid.

    The model must have been trained using the same scheduler configuration.
    """
    if eta < 0.0:
        raise ValueError("eta must be non-negative.")

    model.eval()

    device = z_rgb.device
    dtype = z_rgb.dtype
    batch_size = z_rgb.size(0)

    alpha_bars = scheduler.alpha_bars.to(
        device=device,
        dtype=dtype
    )

    inference_timesteps = scheduler.ddim_timesteps(
        num_inference_steps=num_inference_steps,
        device=device
    )

    if initial_noise is None:
        x_t = torch.randn_like(z_rgb)
    else:
        if initial_noise.shape != z_rgb.shape:
            raise ValueError(
                "initial_noise and z_rgb must have identical shapes."
            )

        x_t = initial_noise.to(
            device=device,
            dtype=dtype
        )

    for index, timestep in enumerate(inference_timesteps):
        step = int(timestep.item())

        t_batch = torch.full(
            (batch_size,),
            step,
            device=device,
            dtype=torch.long
        )

        predicted_noise = model(
            x_t,
            t_batch.float(),
            z_rgb
        )

        alpha_bar_t = alpha_bars[step]

        if index + 1 < len(inference_timesteps):
            previous_step = int(
                inference_timesteps[index + 1].item()
            )
            alpha_bar_previous = alpha_bars[previous_step]
        else:
            alpha_bar_previous = torch.ones(
                (),
                device=device,
                dtype=dtype
            )

        sqrt_alpha_bar_t = torch.sqrt(
            torch.clamp(
                alpha_bar_t,
                min=1e-8
            )
        )

        sqrt_one_minus_alpha_bar_t = torch.sqrt(
            torch.clamp(
                1.0 - alpha_bar_t,
                min=0.0
            )
        )

        predicted_clean_residual = (
            x_t
            - sqrt_one_minus_alpha_bar_t * predicted_noise
        ) / sqrt_alpha_bar_t

        if use_dynamic_threshold:
            predicted_clean_residual = _dynamic_threshold(
                predicted_clean_residual,
                percentile=dynamic_threshold_percentile
            )

        variance_ratio = (
            (1.0 - alpha_bar_previous)
            / torch.clamp(
                1.0 - alpha_bar_t,
                min=1e-8
            )
        )

        transition_ratio = (
            1.0
            - alpha_bar_t
            / torch.clamp(
                alpha_bar_previous,
                min=1e-8
            )
        )

        sigma_t = eta * torch.sqrt(
            torch.clamp(
                variance_ratio * transition_ratio,
                min=0.0
            )
        )

        direction_scale = torch.sqrt(
            torch.clamp(
                1.0
                - alpha_bar_previous
                - sigma_t.square(),
                min=0.0
            )
        )

        x_previous = (
            torch.sqrt(alpha_bar_previous)
            * predicted_clean_residual
            + direction_scale
            * predicted_noise
        )

        if (
            eta > 0.0
            and index + 1 < len(inference_timesteps)
        ):
            x_previous = (
                x_previous
                + sigma_t * torch.randn_like(x_t)
            )

        x_t = x_previous

    return x_t


@torch.no_grad()
def sample_residual(
    model: nn.Module,
    scheduler: DiffusionScheduler,
    z_rgb: torch.Tensor,
    num_inference_steps: Optional[int] = None,
    eta: float = 0.0,
    initial_noise: Optional[torch.Tensor] = None,
    use_dynamic_threshold: bool = True,
    dynamic_threshold_percentile: float = 0.995
) -> torch.Tensor:
    """
    Backward-compatible alias used by the existing inference.py.
    """
    return sample_residual_ddim(
        model=model,
        scheduler=scheduler,
        z_rgb=z_rgb,
        num_inference_steps=num_inference_steps,
        eta=eta,
        initial_noise=initial_noise,
        use_dynamic_threshold=use_dynamic_threshold,
        dynamic_threshold_percentile=dynamic_threshold_percentile
    )
