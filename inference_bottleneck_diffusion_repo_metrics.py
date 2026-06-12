"""Inference for the conditional MST++ bottleneck diffusion model."""

import csv
import os

import scipy.io as sio
import torch
from torch.utils.data import DataLoader

from loss.mrae import mrae
from loss.sam import sam
from loss.psnr import psnr

from dataset.random_arad_loader import load_random_arad1k_samples
from models.Bottleneck_diffusion_mst import (
    DiffusionScheduler,
    MSTPlusPlusBottleneckDiffusion,
)
from models.Hybrid_mstplusplus import MST_Plus_Plus


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CHECKPOINT = "checkpoints_bottleneck_diffusion/best_bottleneck_diffusion.pth"
OUTPUT_DIR = "inference_bottleneck_diffusion"
DATA_ROOT = "data"

BATCH_SIZE = 1
NUM_RANDOM_IMAGES = 50
SEED = 42
DDIM_STEPS = 20
DDIM_ETA = 0.0
SAVE_PREDICTIONS = True

os.makedirs(OUTPUT_DIR, exist_ok=True)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)



def main():
    if not os.path.exists(CHECKPOINT):
        raise FileNotFoundError(f"Checkpoint not found: {CHECKPOINT}")

    checkpoint = torch.load(CHECKPOINT, map_location=DEVICE)
    config = checkpoint["config"]

    backbone = MST_Plus_Plus(
        in_channels=3,
        out_channels=31,
        n_feat=31,
        stage=3,
    ).to(DEVICE)

    model = MSTPlusPlusBottleneckDiffusion(
        backbone=backbone,
        hidden_dim=config["dit_hidden_dim"],
        patch_size=config["dit_patch_size"],
        depth=config["dit_depth"],
        num_heads=config["dit_heads"],
        time_dim=config["dit_time_dim"],
        build_teacher=False,
    ).to(DEVICE)

    model.backbone.load_state_dict(checkpoint["backbone"], strict=True)
    model.denoiser.load_state_dict(checkpoint["denoiser"], strict=True)
    model.eval()

    scheduler = DiffusionScheduler(timesteps=config["train_timesteps"])
    stats = {
        key: value.to(DEVICE, dtype=torch.float32)
        for key, value in checkpoint["stats"].items()
    }

    dataset, selected_samples = load_random_arad1k_samples(
        root_dir=DATA_ROOT,
        num_samples=NUM_RANDOM_IMAGES,
        seed=SEED,
        total_images=1000,
        download=True,
    )

    # num_workers is omitted intentionally (normal default = 0).
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)

    totals = {"mrae": 0.0, "sam": 0.0, "psnr": 0.0}
    sample_count = 0
    rows = []

    with torch.inference_mode():
        for rgb, hsi in loader:
            rgb = rgb.to(DEVICE, dtype=torch.float32)
            hsi = hsi.to(DEVICE, dtype=torch.float32)

            prediction = model.sample_ddim(
                rgb=rgb,
                scheduler=scheduler,
                num_steps=DDIM_STEPS,
                eta=DDIM_ETA,
                target_mean=stats["target_mean"],
                target_std=stats["target_std"],
                condition_mean=stats["condition_mean"],
                condition_std=stats["condition_std"],
            )

            pred_eval = prediction.clamp(0.0, 1.0)
            hsi_eval = hsi.clamp(0.0, 1.0)

            batch_mrae = mrae(pred_eval, hsi_eval)
            batch_sam = sam(pred_eval, hsi_eval)
            batch_psnr = psnr(pred_eval, hsi_eval)

            batch_size = rgb.size(0)
            totals["mrae"] += batch_mrae.item() * batch_size
            totals["sam"] += batch_sam.item() * batch_size
            totals["psnr"] += batch_psnr.item() * batch_size

            for index in range(batch_size):
                sample_count += 1
                sample_info = selected_samples[sample_count - 1]

                if SAVE_PREDICTIONS:
                    cube = prediction[index].cpu().numpy().transpose(1, 2, 0)
                    target_cube = hsi[index].cpu().numpy().transpose(1, 2, 0)
                    sio.savemat(
                        os.path.join(OUTPUT_DIR, f"prediction_{sample_count:04d}.mat"),
                        {"cube": cube, "ground_truth": target_cube},
                    )

                rows.append(
                    {
                        "sample": sample_count,
                        "dataset_index": sample_info["dataset_index"],
                        "rgb_filename": sample_info["rgb_filename"],
                        "hsi_filename": sample_info["hsi_filename"],
                        "batch_mrae": batch_mrae.item(),
                        "batch_sam": batch_sam.item(),
                        "batch_psnr": batch_psnr.item(),
                    }
                )

            print(
                f"Processed {sample_count}/{len(dataset)} | "
                f"MRAE {batch_mrae.item():.6f} | "
                f"SAM {batch_sam.item():.6f} | "
                f"PSNR {batch_psnr.item():.4f}"
            )

    means = {key: value / max(sample_count, 1) for key, value in totals.items()}
    print("Inference complete")
    print(f"MRAE: {means['mrae']:.6f}")
    print(f"SAM:  {means['sam']:.6f}")
    print(f"PSNR: {means['psnr']:.4f}")

    with open(os.path.join(OUTPUT_DIR, "metrics.csv"), "w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
