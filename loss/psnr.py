import math
import torch
def psnr(pred, gt):

    mse = torch.mean(
        (pred - gt) ** 2
    )

    return 10 * torch.log10(
        1.0 / mse
    )
