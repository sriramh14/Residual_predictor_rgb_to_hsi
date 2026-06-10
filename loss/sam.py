import torch
def sam(pred, gt, eps=1e-8):

    pred = pred.permute(0,2,3,1)
    gt   = gt.permute(0,2,3,1)

    dot = torch.sum(
        pred * gt,
        dim=-1
    )

    pred_norm = torch.norm(
        pred,
        dim=-1
    )

    gt_norm = torch.norm(
        gt,
        dim=-1
    )

    cos_theta = dot / (
        pred_norm * gt_norm + eps
    )

    cos_theta = torch.clamp(
        cos_theta,
        -1.0,
        1.0
    )

    angles = torch.acos(
        cos_theta
    )

    return angles.mean()
