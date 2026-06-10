import torch
def rmse(pred, gt):
    return torch.sqrt(
        torch.mean(
            (pred - gt) ** 2
        )
    )
