"""
Train the deterministic MST++ model with the DiT-style bottleneck.

Expected dataset output:
    rgb: [B, 3, H, W]
    hsi: [B, 31, H, W]

Both tensors should be float tensors scaled consistently, preferably to [0, 1].
"""

import os
import random
from contextlib import nullcontext

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

# Adjust these two imports to match your repository.
from models.mst_plus_plus_dit import MST_Plus_Plus
from dataset.dataset_loader import ARADDataset


# ---------------------------------------------------------------------
# CONFIGURATION
# ---------------------------------------------------------------------

SEED = 42
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

EPOCHS = 100
BATCH_SIZE = 4
LEARNING_RATE = 4e-4
MIN_LEARNING_RATE = 1e-6
WEIGHT_DECAY = 1e-4
GRAD_CLIP_NORM = 1.0

USE_AMP = DEVICE.type == "cuda"
USE_PAIRED_AUGMENTATION = True

# "balanced": EMA-normalized 0.5*L1 + 0.4*MRAE + 0.1*SAM
# "mrae": MRAE only, closer to the original MST++ objective
LOSS_MODE = "balanced"
LOSS_WEIGHTS = (0.5, 0.4, 0.1)
LOSS_EMA_MOMENTUM = 0.99

CHECKPOINT_DIR = "checkpoints_mstpp_dit"
BEST_CHECKPOINT = os.path.join(CHECKPOINT_DIR, "best_mstpp_dit.pth")
LAST_CHECKPOINT = os.path.join(CHECKPOINT_DIR, "last_mstpp_dit.pth")
os.makedirs(CHECKPOINT_DIR, exist_ok=True)


# ---------------------------------------------------------------------
# REPRODUCIBILITY
# ---------------------------------------------------------------------

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.benchmark = True


set_seed(SEED)


# ---------------------------------------------------------------------
# LOSSES AND METRICS
# ---------------------------------------------------------------------

def mrae(pred, target, eps=1e-3):
    # Use FP32 for numerical stability under AMP.
    pred = pred.float()
    target = target.float()
    denominator = target.abs().clamp_min(eps)
    return ((pred - target).abs() / denominator).mean()


def sam(pred, target, eps=1e-8):
    """Mean spectral angle in radians for [B, C, H, W]."""
    pred = pred.float()
    target = target.float()

    dot = torch.sum(pred * target, dim=1)
    pred_norm = torch.linalg.vector_norm(pred, dim=1)
    target_norm = torch.linalg.vector_norm(target, dim=1)

    cosine = dot / (pred_norm * target_norm).clamp_min(eps)
    cosine = cosine.clamp(-1.0 + 1e-7, 1.0 - 1e-7)
    return torch.acos(cosine).mean()


def psnr(pred, target, data_range=1.0, eps=1e-10):
    pred = pred.float()
    target = target.float()

    mse_per_image = (pred - target).square().mean(dim=(1, 2, 3))
    return (
        10.0
        * torch.log10(
            (data_range ** 2) / mse_per_image.clamp_min(eps)
        )
    ).mean()


class BalancedReconstructionLoss(nn.Module):
    """
    Normalize each raw loss by its detached running EMA before weighting.

    This makes 0.5/0.4/0.1 reflect relative influence instead of letting
    whichever raw loss has the largest numerical magnitude dominate.
    """

    def __init__(
        self,
        weights=(0.5, 0.4, 0.1),
        momentum=0.99,
        eps=1e-8
    ):
        super().__init__()
        self.weights = tuple(float(v) for v in weights)
        self.momentum = float(momentum)
        self.eps = float(eps)

        self.register_buffer("ema_l1", torch.tensor(0.0))
        self.register_buffer("ema_mrae", torch.tensor(0.0))
        self.register_buffer("ema_sam", torch.tensor(0.0))
        self.register_buffer(
            "initialized",
            torch.tensor(False, dtype=torch.bool)
        )

    @torch.no_grad()
    def _update_scales(self, l1_value, mrae_value, sam_value):
        if not bool(self.initialized.item()):
            self.ema_l1.copy_(l1_value.detach())
            self.ema_mrae.copy_(mrae_value.detach())
            self.ema_sam.copy_(sam_value.detach())
            self.initialized.fill_(True)
            return

        alpha = 1.0 - self.momentum
        self.ema_l1.mul_(self.momentum).add_(l1_value.detach(), alpha=alpha)
        self.ema_mrae.mul_(self.momentum).add_(mrae_value.detach(), alpha=alpha)
        self.ema_sam.mul_(self.momentum).add_(sam_value.detach(), alpha=alpha)

    def forward(self, pred, target, update_scales=True):
        pred = pred.float()
        target = target.float()

        loss_l1 = F.l1_loss(pred, target)
        loss_mrae = mrae(pred, target)
        loss_sam = sam(pred, target)

        raw = {
            "l1": loss_l1,
            "mrae": loss_mrae,
            "sam": loss_sam,
        }

        if LOSS_MODE == "mrae":
            return loss_mrae, raw

        if LOSS_MODE != "balanced":
            raise ValueError("LOSS_MODE must be 'balanced' or 'mrae'.")

        if update_scales or not bool(self.initialized.item()):
            self._update_scales(loss_l1, loss_mrae, loss_sam)

        scaled_l1 = loss_l1 / self.ema_l1.detach().clamp_min(self.eps)
        scaled_mrae = loss_mrae / self.ema_mrae.detach().clamp_min(self.eps)
        scaled_sam = loss_sam / self.ema_sam.detach().clamp_min(self.eps)

        w_l1, w_mrae, w_sam = self.weights
        total = (
            w_l1 * scaled_l1
            + w_mrae * scaled_mrae
            + w_sam * scaled_sam
        )

        return total, raw

    def scales(self):
        return {
            "l1": float(self.ema_l1.item()),
            "mrae": float(self.ema_mrae.item()),
            "sam": float(self.ema_sam.item()),
        }


def evaluation_metrics(pred, target):
    # Same metric preprocessing for train and validation.
    pred_eval = pred.float().clamp(0.0, 1.0)
    target_eval = target.float().clamp(0.0, 1.0)

    return {
        "mrae": mrae(pred_eval, target_eval),
        "sam": sam(pred_eval, target_eval),
        "psnr": psnr(pred_eval, target_eval, data_range=1.0),
    }


# ---------------------------------------------------------------------
# DATA HELPERS
# ---------------------------------------------------------------------

def unpack_batch(batch):
    """
    Supports either:
        (rgb, hsi)
    or:
        {"rgb": rgb, "hsi": hsi}
    """
    if isinstance(batch, (tuple, list)) and len(batch) >= 2:
        return batch[0], batch[1]

    if isinstance(batch, dict):
        return batch["rgb"], batch["hsi"]

    raise TypeError(
        "Each dataset batch must be (rgb, hsi) or "
        "{'rgb': rgb, 'hsi': hsi}."
    )


def paired_augmentation(rgb, hsi):
    """
    Applies the same random spatial transform to RGB and HSI.
    """
    if torch.rand(1).item() < 0.5:
        rgb = torch.flip(rgb, dims=[-1])
        hsi = torch.flip(hsi, dims=[-1])

    if torch.rand(1).item() < 0.5:
        rgb = torch.flip(rgb, dims=[-2])
        hsi = torch.flip(hsi, dims=[-2])

    k = int(torch.randint(0, 4, (1,)).item())

    if k:
        rgb = torch.rot90(rgb, k, dims=(-2, -1))
        hsi = torch.rot90(hsi, k, dims=(-2, -1))

    return rgb, hsi


# ---------------------------------------------------------------------
# TRAIN AND VALIDATION
# ---------------------------------------------------------------------

def train_one_epoch(
    model,
    loader,
    optimizer,
    scaler,
    loss_function
):
    model.train()

    totals = {
        "loss": 0.0,
        "l1": 0.0,
        "mrae": 0.0,
        "sam": 0.0,
        "psnr": 0.0,
    }
    sample_count = 0

    for batch in loader:
        rgb, hsi = unpack_batch(batch)

        rgb = rgb.to(DEVICE, dtype=torch.float32, non_blocking=True)
        hsi = hsi.to(DEVICE, dtype=torch.float32, non_blocking=True)

        if USE_PAIRED_AUGMENTATION:
            rgb, hsi = paired_augmentation(rgb, hsi)

        optimizer.zero_grad(set_to_none=True)

        autocast_context = (
            torch.amp.autocast(device_type="cuda", dtype=torch.float16)
            if USE_AMP
            else nullcontext()
        )

        with autocast_context:
            pred = model(rgb)

        # Composite spectral losses are evaluated in FP32.
        loss, raw_components = loss_function(
            pred,
            hsi,
            update_scales=True
        )

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)

        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            max_norm=GRAD_CLIP_NORM
        )

        scaler.step(optimizer)
        scaler.update()

        with torch.no_grad():
            metrics = evaluation_metrics(pred, hsi)

        batch_size = rgb.size(0)
        sample_count += batch_size

        totals["loss"] += loss.detach().item() * batch_size
        totals["l1"] += raw_components["l1"].item() * batch_size
        totals["mrae"] += metrics["mrae"].item() * batch_size
        totals["sam"] += metrics["sam"].item() * batch_size
        totals["psnr"] += metrics["psnr"].item() * batch_size

    return {key: value / sample_count for key, value in totals.items()}


@torch.inference_mode()
def validate(
    model,
    loader,
    loss_function
):
    model.eval()

    totals = {
        "loss": 0.0,
        "l1": 0.0,
        "mrae": 0.0,
        "sam": 0.0,
        "psnr": 0.0,
    }
    sample_count = 0

    for batch in loader:
        rgb, hsi = unpack_batch(batch)

        rgb = rgb.to(DEVICE, dtype=torch.float32, non_blocking=True)
        hsi = hsi.to(DEVICE, dtype=torch.float32, non_blocking=True)

        autocast_context = (
            torch.amp.autocast(device_type="cuda", dtype=torch.float16)
            if USE_AMP
            else nullcontext()
        )

        with autocast_context:
            pred = model(rgb)

        # Freeze EMA scales during validation.
        loss, raw_components = loss_function(
            pred,
            hsi,
            update_scales=False
        )
        metrics = evaluation_metrics(pred, hsi)

        batch_size = rgb.size(0)
        sample_count += batch_size

        totals["loss"] += loss.item() * batch_size
        totals["l1"] += raw_components["l1"].item() * batch_size
        totals["mrae"] += metrics["mrae"].item() * batch_size
        totals["sam"] += metrics["sam"].item() * batch_size
        totals["psnr"] += metrics["psnr"].item() * batch_size

    return {key: value / sample_count for key, value in totals.items()}


def save_checkpoint(
    path,
    model,
    optimizer,
    scheduler,
    scaler,
    loss_function,
    epoch,
    best_mrae
):
    torch.save(
        {
            "epoch": epoch,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "scaler": scaler.state_dict(),
            "loss_function": loss_function.state_dict(),
            "loss_scales": loss_function.scales(),
            "best_mrae": best_mrae,
            "config": {
                "epochs": EPOCHS,
                "batch_size": BATCH_SIZE,
                "learning_rate": LEARNING_RATE,
                "loss_mode": LOSS_MODE,
                "loss_weights": LOSS_WEIGHTS,
                "loss_ema_momentum": LOSS_EMA_MOMENTUM,
            },
        },
        path
    )


# ---------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------

def main():
    train_dataset = ARADDataset(train=True)
    val_dataset = ARADDataset(train=False)

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        pin_memory=DEVICE.type == "cuda",
        drop_last=False
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        pin_memory=DEVICE.type == "cuda",
        drop_last=False
    )

    model = MST_Plus_Plus(
        in_channels=3,
        out_channels=31,
        n_feat=31,
        stage=3
    ).to(DEVICE)

    loss_function = BalancedReconstructionLoss(
        weights=LOSS_WEIGHTS,
        momentum=LOSS_EMA_MOMENTUM
    ).to(DEVICE)

    parameter_count = sum(
        parameter.numel()
        for parameter in model.parameters()
    )

    print(f"Device: {DEVICE}")
    print(f"Parameters: {parameter_count:,}")
    print(f"Train samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
        betas=(0.9, 0.999)
    )

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=EPOCHS,
        eta_min=MIN_LEARNING_RATE
    )

    scaler = torch.amp.GradScaler(
        "cuda",
        enabled=USE_AMP
    )

    best_mrae = float("inf")

    for epoch in range(1, EPOCHS + 1):
        train_metrics = train_one_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            scaler=scaler,
            loss_function=loss_function
        )

        val_metrics = validate(
            model=model,
            loader=val_loader,
            loss_function=loss_function
        )

        scheduler.step()

        current_lr = optimizer.param_groups[0]["lr"]

        print(
            f"Epoch {epoch:03d}/{EPOCHS} | "
            f"LR {current_lr:.2e} | "
            f"Train Loss {train_metrics['loss']:.6f} | "
            f"Train MRAE {train_metrics['mrae']:.6f} | "
            f"Train SAM {train_metrics['sam']:.6f} | "
            f"Train PSNR {train_metrics['psnr']:.4f} | "
            f"Val Loss {val_metrics['loss']:.6f} | "
            f"Val MRAE {val_metrics['mrae']:.6f} | "
            f"Val SAM {val_metrics['sam']:.6f} | "
            f"Val PSNR {val_metrics['psnr']:.4f}"
        )

        scales = loss_function.scales()
        print(
            "  Loss EMA scales | "
            f"L1 {scales['l1']:.6f} | "
            f"MRAE {scales['mrae']:.6f} | "
            f"SAM {scales['sam']:.6f}"
        )

        if val_metrics["mrae"] < best_mrae:
            best_mrae = val_metrics["mrae"]

            save_checkpoint(
                path=BEST_CHECKPOINT,
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                scaler=scaler,
                loss_function=loss_function,
                epoch=epoch,
                best_mrae=best_mrae
            )

            print(
                f"Saved best checkpoint: "
                f"MRAE={best_mrae:.6f}"
            )

        save_checkpoint(
            path=LAST_CHECKPOINT,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            loss_function=loss_function,
            epoch=epoch,
            best_mrae=best_mrae
        )


if __name__ == "__main__":
    main()
