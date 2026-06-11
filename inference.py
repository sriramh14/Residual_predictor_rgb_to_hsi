import os
import csv
import numpy as np
import scipy.io as sio
import torch
from torch.utils.data import DataLoader

from models.HSIencoder import HSIEncoder
from models.HSIDecoder import HSIDecoder
from models.RGBEncoder import RGBEncoder
from models.Residual_diffusion import (
    ResidualDiffusionModel,
    DiffusionScheduler,
    sample_residual
)

from loss.mrae import mrae
from loss.sam import sam
from loss.psnr import psnr
from loss.ssim import ssim

from dataset.dataset_loader import ARADDataset

# --------------------------------------------------
# CONFIG
# --------------------------------------------------

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

BATCH_SIZE = 1
NUM_WORKERS = 4

LATENT_CHANNELS = 8
HIDDEN_DIM = 64
TIME_DIM = 128
DIFFUSION_TIMESTEPS = 100

RGB_CHECKPOINT = "checkpoints/rgb_to_hsi_best.pth"
VAE_CHECKPOINT = "checkpoints/best_model.pth"
DIFFUSION_CHECKPOINT = "checkpoints/residual_diffusion_best.pth"

OUTPUT_DIR = "inference_results"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Set this to True to save every reconstructed HSI cube as a .mat file.
SAVE_PREDICTIONS = True

# Fixing the seed makes stochastic reverse diffusion reproducible.
SEED = 42
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

# --------------------------------------------------
# LOAD VALIDATION / TEST DATA
# --------------------------------------------------

# ARADDataset currently uses train=False for the held-out 30 samples.
test_dataset = ARADDataset(
    train=False
)

test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=NUM_WORKERS
)

# --------------------------------------------------
# INSTANTIATE PRETRAINED MODELS
# --------------------------------------------------

hsi_encoder = HSIEncoder(
    in_channels=31,
    latent_channels=LATENT_CHANNELS
).to(DEVICE)

hsi_decoder = HSIDecoder(
    latent_channels=LATENT_CHANNELS,
    out_channels=31
).to(DEVICE)

rgb_encoder = RGBEncoder(
    in_channels=3,
    latent_channels=LATENT_CHANNELS
).to(DEVICE)

residual_net = ResidualDiffusionModel(
    latent_dim=LATENT_CHANNELS,
    hidden_dim=HIDDEN_DIM,
    time_dim=TIME_DIM
).to(DEVICE)

noise_scheduler = DiffusionScheduler(
    timesteps=DIFFUSION_TIMESTEPS
)

# --------------------------------------------------
# LOAD CHECKPOINTS
# --------------------------------------------------

rgb_ckpt = torch.load(
    RGB_CHECKPOINT,
    map_location=DEVICE
)

rgb_encoder.load_state_dict(
    rgb_ckpt["rgb_encoder"]
)

vae_ckpt = torch.load(
    VAE_CHECKPOINT,
    map_location=DEVICE
)

hsi_encoder.load_state_dict(
    vae_ckpt["encoder"]
)

hsi_decoder.load_state_dict(
    vae_ckpt["decoder"]
)

# main.py saves only residual_net.state_dict().
diffusion_ckpt = torch.load(
    DIFFUSION_CHECKPOINT,
    map_location=DEVICE
)

# Also supports a future checkpoint dictionary containing "residual_net".
if isinstance(diffusion_ckpt, dict) and "residual_net" in diffusion_ckpt:
    diffusion_ckpt = diffusion_ckpt["residual_net"]

residual_net.load_state_dict(
    diffusion_ckpt
)

print("Loaded RGB encoder, HSI VAE, and residual diffusion model")

# --------------------------------------------------
# EVALUATION MODE
# --------------------------------------------------

rgb_encoder.eval()
hsi_encoder.eval()
hsi_decoder.eval()
residual_net.eval()

# --------------------------------------------------
# INFERENCE
# --------------------------------------------------

running_mrae = 0.0
running_sam = 0.0
running_psnr = 0.0
running_ssim = 0.0

rows = []
sample_index = 0

with torch.inference_mode():

    for rgb, hsi in test_loader:

        rgb = rgb.to(DEVICE)
        hsi = hsi.to(DEVICE)

        # Obtain the RGB latent condition.
        z_rgb = rgb_encoder(rgb)

        # Sample the latent residual by reverse diffusion.
        delta_pred = sample_residual(
            residual_net,
            noise_scheduler,
            z_rgb
        )

        # Convert the RGB latent into the predicted HSI latent.
        z_hsi_pred = z_rgb + delta_pred

        # Decode the predicted HSI latent into a spectral cube.
        hsi_pred = hsi_decoder(
            z_hsi_pred
        )

        batch_mrae = mrae(hsi_pred, hsi).item()
        batch_sam = sam(hsi_pred, hsi).item()
        batch_psnr = psnr(hsi_pred, hsi).item()
        batch_ssim = ssim(hsi_pred, hsi).item()

        running_mrae += batch_mrae
        running_sam += batch_sam
        running_psnr += batch_psnr
        running_ssim += batch_ssim

        batch_size = rgb.size(0)

        for b in range(batch_size):

            sample_index += 1

            pred_cube = hsi_pred[b].detach().cpu().numpy()
            gt_cube = hsi[b].detach().cpu().numpy()

            # Save in H x W x C format, which is common for MATLAB HSI files.
            pred_cube_hwc = np.transpose(pred_cube, (1, 2, 0))
            gt_cube_hwc = np.transpose(gt_cube, (1, 2, 0))

            if SAVE_PREDICTIONS:

                output_path = os.path.join(
                    OUTPUT_DIR,
                    f"prediction_{sample_index:04d}.mat"
                )

                sio.savemat(
                    output_path,
                    {
                        "cube": pred_cube_hwc,
                        "ground_truth": gt_cube_hwc
                    }
                )

            rows.append(
                {
                    "sample": sample_index,
                    "batch_mrae": batch_mrae,
                    "batch_sam": batch_sam,
                    "batch_psnr": batch_psnr,
                    "batch_ssim": batch_ssim
                }
            )

        print(
            f"Processed {sample_index}/{len(test_dataset)} samples "
            f"| MRAE {batch_mrae:.6f} "
            f"| SAM {batch_sam:.6f} "
            f"| PSNR {batch_psnr:.4f} "
            f"| SSIM {batch_ssim:.6f}"
        )

# --------------------------------------------------
# FINAL METRICS
# --------------------------------------------------

n_batches = len(test_loader)

mean_mrae = running_mrae / n_batches
mean_sam = running_sam / n_batches
mean_psnr = running_psnr / n_batches
mean_ssim = running_ssim / n_batches

metrics_path = os.path.join(
    OUTPUT_DIR,
    "metrics.csv"
)

with open(metrics_path, "w", newline="") as csv_file:

    fieldnames = [
        "sample",
        "batch_mrae",
        "batch_sam",
        "batch_psnr",
        "batch_ssim"
    ]

    writer = csv.DictWriter(
        csv_file,
        fieldnames=fieldnames
    )

    writer.writeheader()
    writer.writerows(rows)

summary_path = os.path.join(
    OUTPUT_DIR,
    "summary.txt"
)

with open(summary_path, "w") as summary_file:
    summary_file.write(f"Number of samples: {len(test_dataset)}\n")
    summary_file.write(f"MRAE: {mean_mrae:.6f}\n")
    summary_file.write(f"SAM: {mean_sam:.6f}\n")
    summary_file.write(f"PSNR: {mean_psnr:.4f}\n")
    summary_file.write(f"SSIM: {mean_ssim:.6f}\n")

print("\nInference complete")
print(f"MRAE: {mean_mrae:.6f}")
print(f"SAM: {mean_sam:.6f}")
print(f"PSNR: {mean_psnr:.4f}")
print(f"SSIM: {mean_ssim:.6f}")
print(f"Results saved in: {OUTPUT_DIR}")
