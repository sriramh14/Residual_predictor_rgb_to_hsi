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

        emb = math.log(10000) / (half_dim - 1)

        emb = torch.exp(
            torch.arange(
                half_dim,
                device=t.device
            ) * -emb
        )

        emb = t[:, None] * emb[None, :]

        emb = torch.cat(
            [emb.sin(), emb.cos()],
            dim=-1
        )

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

    def __init__(self, timesteps=100):

        self.timesteps = timesteps

        self.betas = torch.linspace(
            1e-4,
            2e-2,
            timesteps
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
            device=device
        )

    def add_noise(
        self,
        x,
        t
    ):

        noise = torch.randn_like(x)

        # Move the schedule to the data device before indexing.
        # This avoids indexing a CPU tensor with CUDA timestep indices.
        t = t.to(device=x.device, dtype=torch.long)
        alpha_bar = self.alpha_bars.to(x.device)[t]

        alpha_bar = alpha_bar.view(
            -1, 1, 1, 1
        )

        noisy = (
            torch.sqrt(alpha_bar) * x +
            torch.sqrt(1 - alpha_bar) * noise
        )

        return noisy, noise


def diffusion_loss(
    model,
    scheduler,
    z_rgb,
    z_hsi
):

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
def sample_residual(
    model,
    scheduler,
    z_rgb
):

    residual = torch.randn_like(z_rgb)

    for step in reversed(
        range(scheduler.timesteps)
    ):

        t = torch.full(
            (z_rgb.size(0),),
            step,
            device=z_rgb.device
        ).float()

        pred_noise = model(
            residual,
            t,
            z_rgb
        )

        alpha = scheduler.alphas[step].to(z_rgb.device)
        alpha_bar = scheduler.alpha_bars[step].to(z_rgb.device)
        beta = scheduler.betas[step].to(z_rgb.device)

        residual = (
            1 / torch.sqrt(alpha)
        ) * (
            residual -
            ((1 - alpha) /
             torch.sqrt(1 - alpha_bar))
            * pred_noise
        )

        if step > 0:
            residual += (
                torch.sqrt(beta)
                * torch.randn_like(residual)
            )

    return residual
