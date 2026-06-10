# Eval/SSIM.py

from torchmetrics.image import StructuralSimilarityIndexMeasure


def ssim(
    pred,
    target
):
    return StructuralSimilarityIndexMeasure(
        data_range=1.0
    ).to(pred.device)(
        pred,
        target
    )
