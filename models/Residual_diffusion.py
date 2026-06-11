import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class SinusoidalTimeEmbedding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, t):
        half_dim = self.dim // 2

        if half_dim < 1:
            raise ValueError("Time embedding dimension must be at least 2.")

        if half_dim == 1:
            frequencies = torch.ones(
                1,
                device=t.device,
                dtype=t.dtype
            )
        else:
            scale = math.log(10000) / (half_dim - 1)
            frequencies = torch.exp(
                torch.arange(
                    half_dim,
                    device=t.device,
                    dtype=t.dtype
                ) * -scale
            )

        emb = t[:, None] * frequencies[None, :]
        emb = torch.cat([emb.sin(), emb.cos()], dim=-1)

        # Support an odd requested embedding dimension.
        if emb.shape[-1] < self.dim:
            emb = F.pad(emb, (0, self.dim - emb.shape[-1]))

        return emb


class ResidualDiffusionModel(nn.Module):
    def __init__(
        self,
        latent_dim=8,
        hidden_dim=64,
        time_dim=128
    ):
        super().__init__()

        self.time_mlp = nn.Sequential(
            SinusoidalTimeEmbedding(time_dim),
            nn.Linear(time_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )

        self.conv1 = nn.Conv2d(
            latent_dim * 2,
            hidden_dim,
            3,
            padding=1
        )

        self.conv2 = nn.Conv2d(
            hidden_dim,
            hidden_dim,
            3,
            padding=1
        )

        self.conv3 = nn.Conv2d(
            hidden_dim,
            hidden_dim,
            3,
            padding=1
        )

        self.conv4 = nn.Conv2d(
            hidden_dim,
            hidden_dim,
            3,
            padding=1
        )

        self.out = nn.Conv2d(
            hidden_dim,
            latent_dim,
            3,
            padding=1
        )

        self.act = nn.ReLU(inplace=True)

    def forward(
        self,
        noisy_residual,
        t,
        z_rgb
    ):
        t_emb = self.time_mlp(t)

        x = torch.cat(
            [noisy_residual, z_rgb],
            dim=1
        )

        x = self.conv1(x)
        x = x + t_emb[:, :, None, None]
        x = self.act(x)

        x = self.act(self.conv2(x))
        x = self.act(self.conv3(x))
        x = self.act(self.conv4(x))

        return self.out(x)


class DiffusionScheduler:
    """
    Forward diffusion scheduler.

    Training remains standard epsilon-prediction training. DDIM changes only
    the reverse inference procedure, so the same trained model can be sampled
    with DDIM without retraining.
    """

    def __init__(
        self,
        timesteps=100,
        beta_start=1e-4,
        beta_end=2e-2
    ):
        if timesteps < 2:
            raise ValueError("timesteps must be at least 2.")

        self.timesteps = timesteps

        self.betas = torch.linspace(
            beta_start,
            beta_end,
            timesteps,
            dtype=torch.float32
        )

        self.alphas = 1.0 - self.betas

        self.alpha_bars = torch.cumprod(
            self.alphas,
            dim=0
        )

    def sample_timesteps(
        self,
        batch_size,
        device
    ):
        return torch.randint(
            0,
            self.timesteps,
            (batch_size,),
            device=device,
            dtype=torch.long
        )

    def add_noise(
        self,
        x,
        t
    ):
        noise = torch.randn_like(x)

        t = t.to(
            device=x.device,
            dtype=torch.long
        )

        alpha_bars = self.alpha_bars.to(x.device)
        alpha_bar = alpha_bars[t].view(-1, 1, 1, 1)

        noisy = (
            torch.sqrt(alpha_bar) * x
            + torch.sqrt(1.0 - alpha_bar) * noise
        )

        return noisy, noise

    def get_ddim_timesteps(
        self,
        num_inference_steps,
        device
    ):
        """
        Return descending DDIM timesteps, for example:
        [99, 94, 89, ..., 4, 0] for a reduced-step schedule.
        """
        if not 1 <= num_inference_steps <= self.timesteps:
            raise ValueError(
                "num_inference_steps must be between 1 and "
                f"{self.timesteps}, but received {num_inference_steps}."
            )

        timesteps = torch.linspace(
            self.timesteps - 1,
            0,
            num_inference_steps,
            device=device
        ).round().long()

        # Rounding may theoretically create duplicates. Preserve descending
        # order while removing duplicate entries.
        ordered_unique = []
        seen = set()

        for step in timesteps.tolist():
            if step not in seen:
                ordered_unique.append(step)
                seen.add(step)

        return ordered_unique


def diffusion_loss(
    model,
    scheduler,
    z_rgb,
    z_hsi
):
    """
    Standard epsilon-prediction loss.

    DDIM uses the same training process as DDPM. Only sampling differs.
    """
    residual_gt = z_hsi - z_rgb

    t = scheduler.sample_timesteps(
        z_rgb.size(0),
        z_rgb.device
    )

    noisy_residual, noise = scheduler.add_noise(
        residual_gt,
        t
    )

    pred_noise = model(
        noisy_residual,
        t.float(),
        z_rgb
    )

    return F.mse_loss(
        pred_noise,
        noise
    )


@torch.no_grad()
def sample_residual_ddim(
    model,
    scheduler,
    z_rgb,
    num_inference_steps=50,
    eta=0.0,
    initial_noise=None,
    clip_denoised=False,
    clip_range=1.0
):
    """
    Generate the latent residual using the complete DDIM reverse process.

    Parameters
    ----------
    model:
        Trained residual noise-prediction model.
    scheduler:
        DiffusionScheduler used during training.
    z_rgb:
        RGB latent condition, shaped [B, C, H, W].
    num_inference_steps:
        Number of DDIM reverse steps. This can be lower than the number of
        training timesteps. A common starting value is 50.
    eta:
        DDIM stochasticity parameter.
        eta=0.0 gives deterministic DDIM after the initial noise is fixed.
        eta>0.0 adds stochasticity.
    initial_noise:
        Optional starting Gaussian noise. Supplying the same tensor makes
        repeated runs exactly reproducible.
    clip_denoised:
        Whether to clamp the estimated clean residual x_0. Keep False unless
        normalized residuals are known to lie in a bounded interval.
    clip_range:
        Clamp interval used when clip_denoised=True.

    Returns
    -------
    residual:
        Predicted clean latent residual r_0, shaped like z_rgb.
    """
    if eta < 0.0:
        raise ValueError("eta must be non-negative.")

    model.eval()

    device = z_rgb.device
    batch_size = z_rgb.size(0)

    alpha_bars = scheduler.alpha_bars.to(
        device=device,
        dtype=z_rgb.dtype
    )

    ddim_timesteps = scheduler.get_ddim_timesteps(
        num_inference_steps,
        device
    )

    if initial_noise is None:
        residual = torch.randn_like(z_rgb)
    else:
        if initial_noise.shape != z_rgb.shape:
            raise ValueError(
                "initial_noise must have the same shape as z_rgb. "
                f"Got {initial_noise.shape} and {z_rgb.shape}."
            )
        residual = initial_noise.to(
            device=device,
            dtype=z_rgb.dtype
        )

    for index, step in enumerate(ddim_timesteps):
        t = torch.full(
            (batch_size,),
            step,
            device=device,
            dtype=torch.long
        )

        pred_noise = model(
            residual,
            t.float(),
            z_rgb
        )

        alpha_bar_t = alpha_bars[step]

        if index + 1 < len(ddim_timesteps):
            previous_step = ddim_timesteps[index + 1]
            alpha_bar_prev = alpha_bars[previous_step]
        else:
            # At the final reverse step, alpha_bar_prev = 1 gives x_prev = x_0.
            alpha_bar_prev = torch.ones(
                (),
                device=device,
                dtype=z_rgb.dtype
            )

        sqrt_alpha_bar_t = torch.sqrt(
            torch.clamp(alpha_bar_t, min=1e-12)
        )
        sqrt_one_minus_alpha_bar_t = torch.sqrt(
            torch.clamp(1.0 - alpha_bar_t, min=0.0)
        )

        # Estimate the clean residual x_0 from x_t and predicted epsilon.
        pred_clean_residual = (
            residual
            - sqrt_one_minus_alpha_bar_t * pred_noise
        ) / sqrt_alpha_bar_t

        if clip_denoised:
            pred_clean_residual = pred_clean_residual.clamp(
                -clip_range,
                clip_range
            )

        # DDIM variance:
        # sigma_t = eta * sqrt(
        #   ((1-a_prev)/(1-a_t)) * (1-a_t/a_prev)
        # )
        variance_term = (
            (1.0 - alpha_bar_prev)
            / torch.clamp(1.0 - alpha_bar_t, min=1e-12)
        ) * (
            1.0
            - alpha_bar_t
            / torch.clamp(alpha_bar_prev, min=1e-12)
        )

        sigma_t = eta * torch.sqrt(
            torch.clamp(variance_term, min=0.0)
        )

        # Direction pointing from x_0 toward x_t.
        direction_coefficient = torch.sqrt(
            torch.clamp(
                1.0 - alpha_bar_prev - sigma_t.square(),
                min=0.0
            )
        )

        residual = (
            torch.sqrt(alpha_bar_prev) * pred_clean_residual
            + direction_coefficient * pred_noise
        )

        if eta > 0.0 and index + 1 < len(ddim_timesteps):
            residual = residual + sigma_t * torch.randn_like(residual)

    return residual


# Backward-compatible name. Existing inference.py code that imports and calls
# sample_residual(...) will now use DDIM automatically.
@torch.no_grad()
def sample_residual(
    model,
    scheduler,
    z_rgb,
    num_inference_steps=50,
    eta=0.0,
    initial_noise=None,
    clip_denoised=False,
    clip_range=1.0
):
    return sample_residual_ddim(
        model=model,
        scheduler=scheduler,
        z_rgb=z_rgb,
        num_inference_steps=num_inference_steps,
        eta=eta,
        initial_noise=initial_noise,
        clip_denoised=clip_denoised,
        clip_range=clip_range
    )
