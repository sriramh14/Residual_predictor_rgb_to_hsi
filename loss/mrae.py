import torch
def mrae(pred, gt, eps=1e-6):
    return torch.mean(
        torch.abs(pred - gt) /
        (gt + eps)
    )
