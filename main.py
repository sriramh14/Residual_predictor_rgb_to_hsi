import os
import random

import torch
import torch.nn.functional as F
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

SEED = 42
VAL_SEED = 1234

BATCH_SIZE = 8
NUM_EPOCHS = 100
LR = 1e-4
WEIGHT_DECAY = 1e-4
GRAD_CLIP_NORM = 1.0

LATENT_CHANNELS = 8
HIDDEN_DIM = 64
TIME_DIM = 128
DIFFUSION_TIMESTEPS = 20

# Validation MRAE is used for checkpointing and early stopping because
# reconstruction quality, rather than random-timestep noise MSE, is the
# final objective.
EARLY_STOPPING_PATIENCE = 15
LR_PATIENCE = 5
LR_FACTOR = 0.5
MIN_LR = 1e-6

CHECKPOINT_DIR = "checkpoints"
BEST_MODEL_PATH = os.path.join(
    CHECKPOINT_DIR,
    "residual_diffusion_best.pth"
)
BEST_NOISE_MODEL_PATH = os.path.join(
    CHECKPOINT_DIR,
    "residual_diffusion_best_noise_loss.pth"
)

os.makedirs(CHECKPOINT_DIR, exist_ok=True)


# --------------------------------------------------
# REPRODUCIBILITY
# --------------------------------------------------

random.seed(SEED)
torch.manual_seed(SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)


# --------------------------------------------------
# METRIC PREPARATION
# --------------------------------------------------

def prepare_metric_tensors(pred, target):
    """
    Clamp predictions only when the target batch is already normalized
    to [0, 1]. This prevents invalid PSNR/SSIM values caused by decoder
    overshoot while avoiding an incorrect range assumption for
    unnormalized data.
    """
    target_min = target.detach().amin().item()
    target_max = target.detach().amax().item()

    if target_min >= -1e-6 and target_max <= 1.0 + 1e-6:
        pred = pred.clamp(0.0, 1.0)
        target = target.clamp(0.0, 1.0)

    return pred, target


# --------------------------------------------------
# LOAD DATA
# --------------------------------------------------

train_dataset = ARADDataset(
    train=True
)

val_dataset = ARADDataset(
    train=False
)

pin_memory = DEVICE == "cuda"

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=4,
    pin_memory=pin_memory
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=4,
    pin_memory=pin_memory
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


# --------------------------------------------------
# LOAD PRETRAINED MODELS
# --------------------------------------------------

rgb_ckpt = torch.load(
    "checkpoints/rgb_to_hsi_best.pth",
    map_location=DEVICE
)

rgb_encoder.load_state_dict(
    rgb_ckpt["rgb_encoder"]
)

vae_ckpt = torch.load(
    "checkpoints/best_model.pth",
    map_location=DEVICE
)

hsi_encoder.load_state_dict(
    vae_ckpt["encoder"]
)

hsi_decoder.load_state_dict(
    vae_ckpt["decoder"]
)

print("Loaded pretrained RGB encoder and HSI VAE")

rgb_encoder.eval()
hsi_encoder.eval()
hsi_decoder.eval()

for parameter in rgb_encoder.parameters():
    parameter.requires_grad = False

for parameter in hsi_encoder.parameters():
    parameter.requires_grad = False

for parameter in hsi_decoder.parameters():
    parameter.requires_grad = False


# --------------------------------------------------
# RESIDUAL DIFFUSION MODEL
# --------------------------------------------------

residual_net = ResidualDiffusionModel(
    latent_dim=LATENT_CHANNELS,
    hidden_dim=HIDDEN_DIM,
    time_dim=TIME_DIM
).to(DEVICE)

noise_scheduler = DiffusionScheduler(
    timesteps=DIFFUSION_TIMESTEPS
)

# AdamW adds mild regularization and is safer than unregularized Adam
# for a small training set.
optimizer = torch.optim.AdamW(
    residual_net.parameters(),
    lr=LR,
    weight_decay=WEIGHT_DECAY
)

lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    mode="min",
    factor=LR_FACTOR,
    patience=LR_PATIENCE,
    min_lr=MIN_LR
)


# --------------------------------------------------
# TRAIN
# --------------------------------------------------

best_val_mrae = float("inf")
best_val_noise_loss = float("inf")
epochs_without_improvement = 0

for epoch in range(NUM_EPOCHS):

    residual_net.train()

    running_loss = 0.0
    train_count = 0

    for rgb, hsi in train_loader:

        rgb = rgb.to(
            DEVICE,
            non_blocking=pin_memory
        )

        hsi = hsi.to(
            DEVICE,
            non_blocking=pin_memory
        )

        batch_size = rgb.size(0)

        with torch.no_grad():

            z_rgb = rgb_encoder(rgb)

            # Use the deterministic HSI latent mean as the target.
            z_hsi, _ = hsi_encoder(hsi)

        residual_gt = z_hsi - z_rgb

        t = noise_scheduler.sample_timesteps(
            z_rgb.size(0),
            z_rgb.device
        )

        noisy_residual, noise = noise_scheduler.add_noise(
            residual_gt,
            t
        )

        pred_noise = residual_net(
            noisy_residual,
            t.float(),
            z_rgb
        )

        loss = F.mse_loss(
            pred_noise,
            noise
        )

        optimizer.zero_grad(
            set_to_none=True
        )

        loss.backward()

        torch.nn.utils.clip_grad_norm_(
            residual_net.parameters(),
            max_norm=GRAD_CLIP_NORM
        )

        optimizer.step()

        running_loss += loss.item() * batch_size
        train_count += batch_size

    train_loss = running_loss / max(train_count, 1)


    # --------------------------------------
    # VALIDATION
    # --------------------------------------
    #
    # Do not compute training PSNR/MRAE from a one-step x0 estimate.
    # Such estimates are teacher-forced because noisy_residual contains
    # the ground-truth residual, and they are not comparable with full
    # reverse-diffusion validation.
    #
    # Validation is evaluated with the complete sampler. The RNG state is
    # temporarily fixed so every epoch uses the same timesteps, Gaussian
    # noise targets and reverse-sampling noise sequence.

    residual_net.eval()

    val_loss_sum = 0.0
    val_mrae_sum = 0.0
    val_sam_sum = 0.0
    val_psnr_sum = 0.0
    val_ssim_sum = 0.0
    val_count = 0

    cuda_devices = []

    if torch.cuda.is_available():
        cuda_devices = [torch.cuda.current_device()]

    with torch.random.fork_rng(
        devices=cuda_devices,
        enabled=True
    ):

        torch.manual_seed(VAL_SEED)

        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(VAL_SEED)

        with torch.no_grad():

            for rgb, hsi in val_loader:

                rgb = rgb.to(
                    DEVICE,
                    non_blocking=pin_memory
                )

                hsi = hsi.to(
                    DEVICE,
                    non_blocking=pin_memory
                )

                batch_size = rgb.size(0)

                z_rgb = rgb_encoder(rgb)

                z_hsi, _ = hsi_encoder(hsi)

                residual_gt = z_hsi - z_rgb

                # Validation noise-prediction loss, evaluated with a
                # reproducible random sequence.
                t = noise_scheduler.sample_timesteps(
                    z_rgb.size(0),
                    z_rgb.device
                )

                noisy_residual, noise = noise_scheduler.add_noise(
                    residual_gt,
                    t
                )

                pred_noise = residual_net(
                    noisy_residual,
                    t.float(),
                    z_rgb
                )

                batch_val_loss = F.mse_loss(
                    pred_noise,
                    noise
                )

                # Full reverse-diffusion reconstruction.
                delta_pred = sample_residual(
                    residual_net,
                    noise_scheduler,
                    z_rgb
                )

                z_final = z_rgb + delta_pred

                hsi_pred = hsi_decoder(
                    z_final
                )

                hsi_pred_eval, hsi_eval = prepare_metric_tensors(
                    hsi_pred,
                    hsi
                )

                batch_mrae = mrae(
                    hsi_pred_eval,
                    hsi_eval
                )

                batch_sam = sam(
                    hsi_pred_eval,
                    hsi_eval
                )

                batch_psnr = psnr(
                    hsi_pred_eval,
                    hsi_eval
                )

                batch_ssim = ssim(
                    hsi_pred_eval,
                    hsi_eval
                )

                # Sample-weighted accumulation prevents the smaller final
                # batch from receiving the same weight as a full batch.
                val_loss_sum += (
                    batch_val_loss.item()
                    * batch_size
                )

                val_mrae_sum += (
                    batch_mrae.item()
                    * batch_size
                )

                val_sam_sum += (
                    batch_sam.item()
                    * batch_size
                )

                val_psnr_sum += (
                    batch_psnr.item()
                    * batch_size
                )

                val_ssim_sum += (
                    batch_ssim.item()
                    * batch_size
                )

                val_count += batch_size

    val_loss = val_loss_sum / max(val_count, 1)
    val_mrae = val_mrae_sum / max(val_count, 1)
    val_sam = val_sam_sum / max(val_count, 1)
    val_psnr = val_psnr_sum / max(val_count, 1)
    val_ssim = val_ssim_sum / max(val_count, 1)

    lr_scheduler.step(
        val_mrae
    )

    current_lr = optimizer.param_groups[0]["lr"]

    print(
        f"Epoch {epoch + 1}/{NUM_EPOCHS} "
        f"| Train Noise Loss {train_loss:.6f} "
        f"| Val Noise Loss {val_loss:.6f} "
        f"| Val MRAE {val_mrae:.6f} "
        f"| Val SAM {val_sam:.6f} "
        f"| Val PSNR {val_psnr:.4f} "
        f"| Val SSIM {val_ssim:.6f} "
        f"| LR {current_lr:.2e}"
    )

    # Save a separate checkpoint for the lowest validation noise loss.
    # This is diagnostic only; the primary model is selected by MRAE.
    if val_loss < best_val_noise_loss:

        best_val_noise_loss = val_loss

        torch.save(
            residual_net.state_dict(),
            BEST_NOISE_MODEL_PATH
        )

    # Primary checkpoint criterion: full-sampling HSI reconstruction.
    if val_mrae < best_val_mrae:

        best_val_mrae = val_mrae
        epochs_without_improvement = 0

        torch.save(
            residual_net.state_dict(),
            BEST_MODEL_PATH
        )

        print(
            f"Saved best reconstruction model "
            f"(Val MRAE: {best_val_mrae:.6f})"
        )

    else:

        epochs_without_improvement += 1

        print(
            f"No validation MRAE improvement for "
            f"{epochs_without_improvement}/"
            f"{EARLY_STOPPING_PATIENCE} epochs"
        )

    if epochs_without_improvement >= EARLY_STOPPING_PATIENCE:

        print(
            "Early stopping triggered. "
            f"Best validation MRAE: {best_val_mrae:.6f}"
        )

        break
