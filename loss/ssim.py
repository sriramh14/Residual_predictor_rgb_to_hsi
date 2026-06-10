# loss/ssim.py

from torchmetrics.image import StructuralSimilarityIndexMeasure


def ssim(
    pred,
    target
):
    metric = StructuralSimilarityIndexMeasure(
        data_range=1.0
    ).to(pred.device)

    return metric(
        pred,
        target
    )
