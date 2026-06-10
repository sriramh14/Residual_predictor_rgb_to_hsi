import torch
import torch.nn as nn


class ResidualPredictor(nn.Module):
    def __init__(self, latent_dim=8):
        super().__init__()

        self.net = nn.Sequential(
            nn.Conv2d(latent_dim, 64, 3, padding=1),
            nn.ReLU(inplace=True),

            nn.Conv2d(64, 64, 3, padding=1),
            nn.ReLU(inplace=True),

            nn.Conv2d(64, 64, 3, padding=1),
            nn.ReLU(inplace=True),

            nn.Conv2d(64, latent_dim, 3, padding=1)
        )

    def forward(self, z_rgb):
        delta_z = self.net(z_rgb)
        return delta_z
