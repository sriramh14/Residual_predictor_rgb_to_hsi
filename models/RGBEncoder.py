import torch
import torch.nn as nn

from .resblock import ResidualBlock


class RGBEncoder(nn.Module):

    def __init__(
        self,
        in_channels=3,
        latent_channels=8
    ):
        super().__init__()

        self.net = nn.Sequential(

            nn.Conv2d(
                in_channels,
                64,
                kernel_size=4,
                stride=2,
                padding=1
            ),
            nn.ReLU(inplace=True),

            ResidualBlock(64),

            nn.Conv2d(
                64,
                128,
                kernel_size=4,
                stride=2,
                padding=1
            ),
            nn.ReLU(inplace=True),

            ResidualBlock(128),

            nn.Conv2d(
                128,
                latent_channels,
                kernel_size=3,
                padding=1
            )
        )

    def forward(self, x):

        return self.net(x)
