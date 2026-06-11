"""
Train MST++ with the deterministic DiT-style bottleneck.

Dataset output:
    rgb: [B, 3, H, W]
    hsi: [B, 31, H, W]

Both should use the same normalization, preferably [0, 1].
The script automatically uses both GPUs through nn.DataParallel when available.
"""

import os
import random
from contextlib import nullcontext

import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.utils.data import DataLoader

# Adjust these imports to match your repository.
from models.mst_plus_plus_dit import MST_Plus_Plus
from dataset.dataset_loader import ARADDataset


# ---------------------------------------------------------------------
# CONFIGURATION
# ---------------------------------------------------------------------

SEED = 42
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
NUM_GPUS = torch.cuda.device_count() if torch.cuda.is_available() else 0

EPOCHS = 100

# Global batch size becomes:
# BATCH_SIZE_PER_GPU × number of visible GPUs.
BATCH_SIZE_PER_GPU = 4
BATCH_SIZE = BATCH_SIZE_PER_GPU * max(NUM_GPUS, 1)

LEARNING_RATE = 4e-4
MIN_LEARNING_RATE = 1e-6
WEIGHT_DECAY = 1e-4
GRAD_CLIP_NORM = 1.0

NUM_WORKERS = 8 if NUM_GPUS > 1 else 4
USE_AMP = DEVICE.type == "cuda"
USE_PAIRED_AUGMENTATION = True

# "mixed": 0.5*L1 + 0.4*MRAE + 0.1*SAM
# "mrae": MRAE only
LOSS_MODE = "mixed"

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
    denominator = torch.clamp(target.abs(), min=eps)
    return torch.mean(torch.abs(pred - target) / denominator)


def sam(pred, target, eps=1e-8):
    """Mean spectral angle in radians for [B, C, H, W]."""
    dot = torch.sum(pred * target, dim=1)
    pred_norm = torch.linalg.vector_norm(pred, dim=1)
    target_norm = torch.linalg.vector_norm(target, dim=1)

    cosine = dot / torch.clamp(
        pred_norm * target_norm,
        min=eps
    )
    cosine = torch.clamp(
        cosine,
        min=-1.0 + 1e-7,
        max=1.0 - 1e-7
    )

    return torch.acos(cosine).mean()


def psnr(pred, target, data_range=1.0, eps=1e-10):
    """Mean per-image PSNR."""
    mse_per_image = torch.mean(
        (pred - target).square(),
        dim=(1, 2, 3)
    )

    return (
        10.0
        * torch.log10(
            (data_range ** 2)
            / torch.clamp(mse_per_image, min=eps)
        )
    ).mean()


def reconstruction_loss(pred, target):
    """
    Training/validation objective computed on raw model outputs.
    Reported reconstruction metrics are calculated separately after clamping.
    """
    loss_mrae = mrae(pred, target)

    if LOSS_MODE == "mrae":
        return loss_mrae

    if LOSS_MODE != "mixed":
        raise ValueError(
            "LOSS_MODE must be either 'mixed' or 'mrae'."
        )

    loss_l1 = F.l1_loss(pred, target)
    loss_sam = sam(pred, target)

    return (
        0.5 * loss_l1
        + 0.4 * loss_mrae
        + 0.1 * loss_sam
    )


def evaluation_metrics(pred, target):
    """
    Train and validation metrics use exactly the same preprocessing.

    This assumes the HSI data are normalized to [0, 1].
    Clamping is used only for reporting metrics, not for the training loss.
    """
    pred_eval = pred.float().clamp(0.0, 1.0)
    target_eval = target.float().clamp(0.0, 1.0)

    return {
        "mrae": mrae(pred_eval, target_eval),
        "sam": sam(pred_eval, target_eval),
        "psnr": psnr(
            pred_eval,
            target_eval,
            data_range=1.0
        ),
    }


# ---------------------------------------------------------------------
# DATA HELPERS
# ---------------------------------------------------------------------

def unpack_batch(batch):
    if isinstance(batch, (tuple, list)) and len(batch) >= 2:
        return batch[0], batch[1]

    if isinstance(batch, dict):
        return batch["rgb"], batch["hsi"]

    raise TypeError(
        "Each batch must be (rgb, hsi) or "
        "{'rgb': rgb, 'hsi': hsi}."
    )


def paired_augmentation(rgb, hsi):
    """Apply identical random transforms to RGB and HSI."""
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
# MULTI-GPU AND CHECKPOINT HELPERS
# ---------------------------------------------------------------------

def unwrap_model(model):
    """Return the underlying model when nn.DataParallel is active."""
    return model.module if isinstance(model, torch.nn.DataParallel) else model


def save_checkpoint(
    path,
    model,
    optimizer,
    scheduler,
    scaler,
    epoch,
    best_mrae
):
    torch.save(
        {
            "epoch": epoch,
            # Save without the DataParallel "module." prefix.
            "model": unwrap_model(model).state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "scaler": scaler.state_dict(),
            "best_mrae": best_mrae,
            "config": {
                "epochs": EPOCHS,
                "batch_size_per_gpu": BATCH_SIZE_PER_GPU,
                "global_batch_size": BATCH_SIZE,
                "learning_rate": LEARNING_RATE,
                "loss_mode": LOSS_MODE,
                "num_gpus": NUM_GPUS,
            },
        },
        path
    )


# ---------------------------------------------------------------------
# TRAINING AND VALIDATION
# ---------------------------------------------------------------------

def run_epoch(
    model,
    loader,
    optimizer=None,
    scaler=None
):
    is_training = optimizer is not None

    if is_training:
        model.train()
    else:
        model.eval()

    totals = {
        "loss": 0.0,
        "mrae": 0.0,
        "sam": 0.0,
        "psnr": 0.0,
    }
    sample_count = 0

    grad_context = nullcontext() if is_training else torch.inference_mode()

    with grad_context:
        for batch in loader:
            rgb, hsi = unpack_batch(batch)

            rgb = rgb.to(
                DEVICE,
                dtype=torch.float32,
                non_blocking=True
            )
            hsi = hsi.to(
                DEVICE,
                dtype=torch.float32,
                non_blocking=True
            )

            if is_training and USE_PAIRED_AUGMENTATION:
                rgb, hsi = paired_augmentation(rgb, hsi)

            if is_training:
                optimizer.zero_grad(set_to_none=True)

            autocast_context = (
                torch.amp.autocast(
                    device_type="cuda",
                    dtype=torch.float16
                )
                if USE_AMP
                else nullcontext()
            )

            with autocast_context:
                pred = model(rgb)
                loss = reconstruction_loss(pred, hsi)

            if is_training:
                scaler.scale(loss).backward()

                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(),
                    max_norm=GRAD_CLIP_NORM
                )

                scaler.step(optimizer)
                scaler.update()

            # Use identical clamped metric evaluation for train and validation.
            with torch.no_grad():
                metrics = evaluation_metrics(pred, hsi)

            batch_size = rgb.size(0)
            sample_count += batch_size

            totals["loss"] += loss.detach().item() * batch_size
            totals["mrae"] += metrics["mrae"].item() * batch_size
            totals["sam"] += metrics["sam"].item() * batch_size
            totals["psnr"] += metrics["psnr"].item() * batch_size

    return {
        key: value / sample_count
        for key, value in totals.items()
    }


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
        num_workers=NUM_WORKERS,
        pin_memory=DEVICE.type == "cuda",
        persistent_workers=NUM_WORKERS > 0,
        drop_last=False
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=DEVICE.type == "cuda",
        persistent_workers=NUM_WORKERS > 0,
        drop_last=False
    )

    base_model = MST_Plus_Plus(
        in_channels=3,
        out_channels=31,
        n_feat=31,
        stage=3
    ).to(DEVICE)

    parameter_count = sum(
        parameter.numel()
        for parameter in base_model.parameters()
    )

    # Automatically use both visible GPUs.
    if NUM_GPUS > 1:
        model = torch.nn.DataParallel(
            base_model,
            device_ids=list(range(NUM_GPUS))
        )
    else:
        model = base_model

    print(f"Primary device: {DEVICE}")
    print(f"Visible GPUs: {NUM_GPUS}")
    print(f"Multi-GPU enabled: {NUM_GPUS > 1}")
    print(f"Batch size per GPU: {BATCH_SIZE_PER_GPU}")
    print(f"Global batch size: {BATCH_SIZE}")
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
        train_metrics = run_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            scaler=scaler
        )

        val_metrics = run_epoch(
            model=model,
            loader=val_loader
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

        if val_metrics["mrae"] < best_mrae:
            best_mrae = val_metrics["mrae"]

            save_checkpoint(
                path=BEST_CHECKPOINT,
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                scaler=scaler,
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
            epoch=epoch,
            best_mrae=best_mrae
        )


if __name__ == "__main__":
    main()
