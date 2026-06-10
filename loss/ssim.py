import torchmetrics
from torchmetrics.image import StructuralSimilarityIndexMeasure
def ssim:
    ssim_metric = (
        StructuralSimilarityIndexMeasure(
            data_range=1.0
        ).cuda()
    )

    ssim_value = ssim_metric(
        pred,
        gt
    )
    return ssim

