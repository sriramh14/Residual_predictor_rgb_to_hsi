"""
Train the final MST++ bottleneck as a true conditional diffusion model.

Required first stage:
    Train the deterministic MST++-DiT model with main2.py and keep its best
    checkpoint. This script uses that model as a pretrained backbone and a
    frozen HSI bottleneck teacher.

Diffusion target:
    A frozen copy of the pretrained final MST stage encodes the ground-truth
    31-band HSI into a clean 124-channel bottleneck.

Condition:
    The RGB image is processed by conv_in, the first two MST stages, and the
    final-stage encoder up to (but not through) its bottleneck.
"""

import os
import random
from contextlib import nullcontext

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from dataset.dataset_loader import ARADDataset
from models.Bottleneck_diffusion_mst import (
    DiffusionScheduler,
    MSTPlusPlusBottleneckDiffusion,
    load_pretrained_backbone,
)


# ---------------------------------------------------------------------
# CONFIGURATION
# ---------------------------------------------------------------------

SEED = 42
VAL_SEED = 1234
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

PRETRAINED_MST_CHECKPOINT = "checkpoints_mstpp_dit/best_mstpp_dit.pth"
CHECKPOINT_DIR = "checkpoints_bottleneck_diffusion"
BEST_CHECKPOINT = os.path.join(CHECKPOINT_DIR, "best_bottleneck_diffusion.pth")
LAST_CHECKPOINT = os.path.join(CHECKPOINT_DIR, "last_bottleneck_diffusion.pth")
STATS_PATH = os.path.join(CHECKPOINT_DIR, "bottleneck_stats.pth")
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

EPOCHS = 100
BATCH_SIZE = 2
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-4
GRAD_CLIP_NORM = 1.0

TRAIN_TIMESTEPS = 20
VALIDATION_DDIM_STEPS = 20
DDIM_ETA = 0.0

DIT_HIDDEN_DIM = 128
DIT_PATCH_SIZE = 4
DIT_DEPTH = 2
DIT_HEADS = 4
DIT_TIME_DIM = 128

# Loss = noise + FEATURE_WEIGHT * feature + RECON_WEIGHT * HSI reconstruction.
FEATURE_WEIGHT = 0.05
RECON_WEIGHT = 0.02

# Keep the RGB condition encoder frozen. The final decoder/output head remain
# trainable so they can adapt to sampled bottlenecks.
TRAIN_FINAL_DECODER = True

USE_AMP = DEVICE.type == "cuda"
USE_PAIRED_AUGMENTATION = True
EARLY_STOPPING_PATIENCE = 15


# ---------------------------------------------------------------------
# REPRODUCIBILITY
# ---------------------------------------------------------------------


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True


set_seed(SEED)


# ---------------------------------------------------------------------
# METRICS / AUXILIARY RECONSTRUCTION LOSS
# ---------------------------------------------------------------------


def mrae(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-3):
    pred = pred.float()
    target = target.float()
    return ((pred - target).abs() / target.abs().clamp_min(eps)).mean()


def sam(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-8):
    pred = pred.float()
    target = target.float()
    dot = (pred * target).sum(dim=1)
    denominator = (
        torch.linalg.vector_norm(pred, dim=1)
        * torch.linalg.vector_norm(target, dim=1)
    ).clamp_min(eps)
    cosine = (dot / denominator).clamp(-1.0 + 1e-7, 1.0 - 1e-7)
    return torch.acos(cosine).mean()


def psnr(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-10):
    mse = (pred.float() - target.float()).square().mean(dim=(1, 2, 3))
    return (10.0 * torch.log10(1.0 / mse.clamp_min(eps))).mean()


def spectral_reconstruction_loss(pred: torch.Tensor, target: torch.Tensor):
    pred = pred.float()
    target = target.float()
    return (
        0.5 * F.l1_loss(pred, target)
        + 0.4 * mrae(pred, target)
        + 0.1 * sam(pred, target)
    )


def evaluation_metrics(pred: torch.Tensor, target: torch.Tensor):
    pred = pred.float().clamp(0.0, 1.0)
    target = target.float().clamp(0.0, 1.0)
    return {
        "mrae": mrae(pred, target),
        "sam": sam(pred, target),
        "psnr": psnr(pred, target),
    }


# ---------------------------------------------------------------------
# DATA AUGMENTATION
# ---------------------------------------------------------------------


def paired_augmentation(rgb: torch.Tensor, hsi: torch.Tensor):
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
# BOTTLENECK STATISTICS
# ---------------------------------------------------------------------


@torch.no_grad()
def compute_bottleneck_stats(model, loader):
    """Compute fixed channel-wise statistics for target and condition spaces."""
    model.eval()

    target_sum = target_sq_sum = None
    condition_sum = condition_sq_sum = None
    count = 0

    print("Computing bottleneck normalization statistics...")

    for rgb, hsi in loader:
        rgb = rgb.to(DEVICE, dtype=torch.float32)
        hsi = hsi.to(DEVICE, dtype=torch.float32)

        context = model.encode_rgb(rgb, detach=True)
        target = model.encode_hsi_target(hsi)
        condition = context.rgb_bottleneck

        reduce_dims = (0, 2, 3)
        if target_sum is None:
            channels = target.size(1)
            target_sum = torch.zeros(channels, device=DEVICE, dtype=torch.float64)
            target_sq_sum = torch.zeros_like(target_sum)
            condition_sum = torch.zeros_like(target_sum)
            condition_sq_sum = torch.zeros_like(target_sum)

        target64 = target.double()
        condition64 = condition.double()
        target_sum += target64.sum(dim=reduce_dims)
        target_sq_sum += target64.square().sum(dim=reduce_dims)
        condition_sum += condition64.sum(dim=reduce_dims)
        condition_sq_sum += condition64.square().sum(dim=reduce_dims)
        count += target.size(0) * target.size(2) * target.size(3)

    target_mean = target_sum / count
    target_var = target_sq_sum / count - target_mean.square()
    condition_mean = condition_sum / count
    condition_var = condition_sq_sum / count - condition_mean.square()

    stats = {
        "target_mean": target_mean.float().view(1, -1, 1, 1).cpu(),
        "target_std": target_var.clamp_min(1e-8).sqrt().float().view(1, -1, 1, 1).cpu(),
        "condition_mean": condition_mean.float().view(1, -1, 1, 1).cpu(),
        "condition_std": condition_var.clamp_min(1e-8).sqrt().float().view(1, -1, 1, 1).cpu(),
    }
    torch.save(stats, STATS_PATH)
    print(f"Saved bottleneck statistics to {STATS_PATH}")
    return stats


def move_stats(stats):
    return {
        key: value.to(DEVICE, dtype=torch.float32)
        for key, value in stats.items()
    }


# ---------------------------------------------------------------------
# TRAIN / VALIDATE
# ---------------------------------------------------------------------


def train_one_epoch(model, scheduler, loader, optimizer, scaler, stats):
    model.train()
    if model.teacher_stage is not None:
        model.teacher_stage.eval()

    totals = {"loss": 0.0, "noise": 0.0, "feature": 0.0, "recon": 0.0}
    sample_count = 0

    for rgb, hsi in loader:
        rgb = rgb.to(DEVICE, dtype=torch.float32)
        hsi = hsi.to(DEVICE, dtype=torch.float32)

        if USE_PAIRED_AUGMENTATION:
            rgb, hsi = paired_augmentation(rgb, hsi)

        context = model.encode_rgb(rgb, detach=True)
        with torch.no_grad():
            target_raw = model.encode_hsi_target(hsi)

        condition = (
            context.rgb_bottleneck - stats["condition_mean"]
        ) / stats["condition_std"].clamp_min(1e-6)
        target = (
            target_raw - stats["target_mean"]
        ) / stats["target_std"].clamp_min(1e-6)

        t = scheduler.sample_timesteps(target.size(0), target.device)
        noisy_target, noise = scheduler.add_noise(target, t)

        optimizer.zero_grad(set_to_none=True)
        autocast_context = (
            torch.amp.autocast(device_type="cuda", dtype=torch.float16)
            if USE_AMP
            else nullcontext()
        )

        with autocast_context:
            predicted_noise = model.denoiser(noisy_target, t, condition)

        noise_loss = F.mse_loss(predicted_noise.float(), noise.float())
        predicted_clean = scheduler.predict_clean(
            noisy_target.float(), predicted_noise.float(), t
        )

        # Smooth L1 is less sensitive to high-timestep x0 errors.
        feature_loss = F.smooth_l1_loss(predicted_clean, target.float())

        # Use a bounded x0 estimate only for the weak decoded reconstruction
        # auxiliary. The diffusion/noise target itself remains unclipped.
        clean_for_decode = predicted_clean.clamp(-5.0, 5.0)
        bottleneck_for_decode = (
            clean_for_decode * stats["target_std"] + stats["target_mean"]
        )
        predicted_hsi = model.decode_bottleneck(bottleneck_for_decode, context)
        reconstruction_loss = spectral_reconstruction_loss(predicted_hsi, hsi)

        alpha_bar_t = scheduler.alpha_bars.to(DEVICE)[t].mean().detach()
        loss = (
            noise_loss
            + FEATURE_WEIGHT * feature_loss
            + RECON_WEIGHT * alpha_bar_t * reconstruction_loss
        )

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP_NORM)
        scaler.step(optimizer)
        scaler.update()

        batch_size = rgb.size(0)
        sample_count += batch_size
        totals["loss"] += loss.item() * batch_size
        totals["noise"] += noise_loss.item() * batch_size
        totals["feature"] += feature_loss.item() * batch_size
        totals["recon"] += reconstruction_loss.item() * batch_size

    return {key: value / max(sample_count, 1) for key, value in totals.items()}


@torch.inference_mode()
def validate(model, scheduler, loader, stats):
    model.eval()
    totals = {"noise": 0.0, "mrae": 0.0, "sam": 0.0, "psnr": 0.0}
    sample_count = 0

    devices = [torch.cuda.current_device()] if torch.cuda.is_available() else []
    with torch.random.fork_rng(devices=devices, enabled=True):
        torch.manual_seed(VAL_SEED)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(VAL_SEED)

        for rgb, hsi in loader:
            rgb = rgb.to(DEVICE, dtype=torch.float32)
            hsi = hsi.to(DEVICE, dtype=torch.float32)

            context = model.encode_rgb(rgb, detach=True)
            target_raw = model.encode_hsi_target(hsi)
            condition = (
                context.rgb_bottleneck - stats["condition_mean"]
            ) / stats["condition_std"].clamp_min(1e-6)
            target = (
                target_raw - stats["target_mean"]
            ) / stats["target_std"].clamp_min(1e-6)

            t = scheduler.sample_timesteps(target.size(0), target.device)
            noisy_target, noise = scheduler.add_noise(target, t)
            predicted_noise = model.denoiser(noisy_target, t, condition)
            noise_loss = F.mse_loss(predicted_noise.float(), noise.float())

            predicted_hsi = model.sample_ddim(
                rgb=rgb,
                scheduler=scheduler,
                num_steps=VALIDATION_DDIM_STEPS,
                eta=DDIM_ETA,
                target_mean=stats["target_mean"],
                target_std=stats["target_std"],
                condition_mean=stats["condition_mean"],
                condition_std=stats["condition_std"],
            )
            metrics = evaluation_metrics(predicted_hsi, hsi)

            batch_size = rgb.size(0)
            sample_count += batch_size
            totals["noise"] += noise_loss.item() * batch_size
            totals["mrae"] += metrics["mrae"].item() * batch_size
            totals["sam"] += metrics["sam"].item() * batch_size
            totals["psnr"] += metrics["psnr"].item() * batch_size

    return {key: value / max(sample_count, 1) for key, value in totals.items()}


# ---------------------------------------------------------------------
# CHECKPOINT
# ---------------------------------------------------------------------


def save_checkpoint(path, model, optimizer, lr_scheduler, scaler, stats, epoch, best_mrae):
    torch.save(
        {
            "epoch": epoch,
            "backbone": model.backbone.state_dict(),
            "denoiser": model.denoiser.state_dict(),
            "optimizer": optimizer.state_dict(),
            "lr_scheduler": lr_scheduler.state_dict(),
            "scaler": scaler.state_dict(),
            "stats": {key: value.cpu() for key, value in stats.items()},
            "best_mrae": best_mrae,
            "config": {
                "train_timesteps": TRAIN_TIMESTEPS,
                "ddim_steps": VALIDATION_DDIM_STEPS,
                "dit_hidden_dim": DIT_HIDDEN_DIM,
                "dit_patch_size": DIT_PATCH_SIZE,
                "dit_depth": DIT_DEPTH,
                "dit_heads": DIT_HEADS,
                "dit_time_dim": DIT_TIME_DIM,
            },
        },
        path,
    )


# ---------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------


def main():
    if not os.path.exists(PRETRAINED_MST_CHECKPOINT):
        raise FileNotFoundError(
            f"Pretrained deterministic MST++ checkpoint not found: "
            f"{PRETRAINED_MST_CHECKPOINT}. Train main2.py first or update "
            "PRETRAINED_MST_CHECKPOINT."
        )

    train_dataset = ARADDataset(train=True)
    val_dataset = ARADDataset(train=False)

    # num_workers is intentionally omitted so PyTorch uses the normal,
    # notebook-safe default num_workers=0.
    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        pin_memory=DEVICE.type == "cuda",
    )
    stats_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        pin_memory=DEVICE.type == "cuda",
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        pin_memory=DEVICE.type == "cuda",
    )

    backbone = load_pretrained_backbone(PRETRAINED_MST_CHECKPOINT, DEVICE)
    model = MSTPlusPlusBottleneckDiffusion(
        backbone=backbone,
        hidden_dim=DIT_HIDDEN_DIM,
        patch_size=DIT_PATCH_SIZE,
        depth=DIT_DEPTH,
        num_heads=DIT_HEADS,
        time_dim=DIT_TIME_DIM,
        build_teacher=True,
    ).to(DEVICE)
    model.freeze_condition_encoder(train_final_decoder=TRAIN_FINAL_DECODER)

    scheduler = DiffusionScheduler(timesteps=TRAIN_TIMESTEPS)

    if os.path.exists(STATS_PATH):
        stats = torch.load(STATS_PATH, map_location="cpu")
        print(f"Loaded bottleneck statistics from {STATS_PATH}")
    else:
        stats = compute_bottleneck_stats(model, stats_loader)
    stats = move_stats(stats)

    trainable_parameters = sum(
        parameter.numel() for parameter in model.parameters() if parameter.requires_grad
    )
    print(f"Device: {DEVICE}")
    print(f"Trainable parameters: {trainable_parameters:,}")
    print(f"Train samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")

    optimizer = torch.optim.AdamW(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )
    lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=5,
        min_lr=1e-6,
    )
    scaler = torch.amp.GradScaler("cuda", enabled=USE_AMP)

    best_mrae = float("inf")
    epochs_without_improvement = 0

    for epoch in range(1, EPOCHS + 1):
        train_metrics = train_one_epoch(
            model, scheduler, train_loader, optimizer, scaler, stats
        )
        val_metrics = validate(model, scheduler, val_loader, stats)
        lr_scheduler.step(val_metrics["mrae"])

        print(
            f"Epoch {epoch:03d}/{EPOCHS} | "
            f"Train Total {train_metrics['loss']:.6f} | "
            f"Noise {train_metrics['noise']:.6f} | "
            f"Feature {train_metrics['feature']:.6f} | "
            f"Recon {train_metrics['recon']:.6f} | "
            f"Val Noise {val_metrics['noise']:.6f} | "
            f"Val MRAE {val_metrics['mrae']:.6f} | "
            f"Val SAM {val_metrics['sam']:.6f} | "
            f"Val PSNR {val_metrics['psnr']:.4f} | "
            f"LR {optimizer.param_groups[0]['lr']:.2e}"
        )

        save_checkpoint(
            LAST_CHECKPOINT,
            model,
            optimizer,
            lr_scheduler,
            scaler,
            stats,
            epoch,
            best_mrae,
        )

        if val_metrics["mrae"] < best_mrae:
            best_mrae = val_metrics["mrae"]
            epochs_without_improvement = 0
            save_checkpoint(
                BEST_CHECKPOINT,
                model,
                optimizer,
                lr_scheduler,
                scaler,
                stats,
                epoch,
                best_mrae,
            )
            print(f"Saved best bottleneck diffusion model: MRAE={best_mrae:.6f}")
        else:
            epochs_without_improvement += 1

        if epochs_without_improvement >= EARLY_STOPPING_PATIENCE:
            print(f"Early stopping. Best validation MRAE: {best_mrae:.6f}")
            break


if __name__ == "__main__":
    main()
