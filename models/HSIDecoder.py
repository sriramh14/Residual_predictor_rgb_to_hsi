import torch
import torch.nn as nn
from .resblock import ResidualBlock
class HSIDecoder(nn.Module):
    def __init__(
        self,
        latent_channels=8,
        out_channels=31
    ):
        super().__init__()

        self.decoder = nn.Sequential(

            nn.Conv2d(
                latent_channels,
                256,
                kernel_size=3,
                padding=1
            ),
            nn.GELU(),
            ResidualBlock(256),

            nn.ConvTranspose2d(
                256,
                128,
                kernel_size=4,
                stride=2,
                padding=1
            ),
            nn.GELU(),
            ResidualBlock(128),

            nn.ConvTranspose2d(
                128,
                64,
                kernel_size=4,
                stride=2,
                padding=1
            ),
            nn.GELU(),
            ResidualBlock(64),

            nn.Conv2d(
                64,
                out_channels,
                kernel_size=3,
                padding=1
            )
        )

    def forward(self, z):
        return self.decoder(z)
