import torch
import torch.nn as nn
from .resblock import ResidualBlock
class HSIEncoder(nn.Module):
    def __init__(
        self,
        in_channels=31,
        latent_channels=8
    ):
        super().__init__()

        self.backbone = nn.Sequential(

            nn.Conv2d(in_channels, 64, 3, padding=1),
            nn.GELU(),
            ResidualBlock(64),

            nn.Conv2d(
                64,
                128,
                kernel_size=4,
                stride=2,
                padding=1
            ),
            nn.GELU(),
            ResidualBlock(128),

            nn.Conv2d(
                128,
                256,
                kernel_size=4,
                stride=2,
                padding=1
            ),
            nn.GELU(),
            ResidualBlock(256),
        )

        self.mu = nn.Conv2d(
            256,
            latent_channels,
            kernel_size=1
        )

        self.logvar = nn.Conv2d(
            256,
            latent_channels,
            kernel_size=1
        )

    def forward(self, x):

        h = self.backbone(x)

        mu = self.mu(h)
        logvar = self.logvar(h)

        return mu, logvar
