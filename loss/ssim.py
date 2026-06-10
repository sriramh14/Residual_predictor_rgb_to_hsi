# loss/ssim.py

import torch
import torch.nn.functional as F


def ssim(
    pred,
    target,
    C1=0.01**2,
    C2=0.03**2
):
    """
    pred   : [B, C, H, W]
    target : [B, C, H, W]

    Returns mean SSIM.
    Higher is better.
    """

    mu_x = pred.mean(dim=(-2, -1), keepdim=True)
    mu_y = target.mean(dim=(-2, -1), keepdim=True)

    sigma_x = (
        (pred - mu_x) ** 2
    ).mean(dim=(-2, -1), keepdim=True)

    sigma_y = (
        (target - mu_y) ** 2
    ).mean(dim=(-2, -1), keepdim=True)

    sigma_xy = (
        (pred - mu_x)
        * (target - mu_y)
    ).mean(dim=(-2, -1), keepdim=True)

    numerator = (
        (2 * mu_x * mu_y + C1)
        * (2 * sigma_xy + C2)
    )

    denominator = (
        (mu_x**2 + mu_y**2 + C1)
        * (sigma_x + sigma_y + C2)
    )

    ssim_map = numerator / (
        denominator + 1e-8
    )

    return ssim_map.mean()
