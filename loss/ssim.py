import torchmetrics
from torchmetrics.image import StructuralSimilarityIndexMeasure
ssim_metric = (
    StructuralSimilarityIndexMeasure(
        data_range=1.0
    ).cuda()
)

ssim_value = ssim_metric(
    pred,
    gt
)
