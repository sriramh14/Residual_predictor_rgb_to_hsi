# RGB-to-HSI Latent Residual Diffusion Pipeline

## 1. Purpose

This project reconstructs a 31-band hyperspectral image (HSI) from a 3-channel RGB image through a shared latent space and a conditional residual-diffusion model.

The complete system is trained in three sequential stages:

1. **HSI beta-VAE pretraining:** learn an HSI encoder and HSI decoder.
2. **RGB latent encoder training:** map RGB images into the pretrained HSI latent space.
3. **Latent residual diffusion training:** model the remaining difference between the RGB latent and the ground-truth HSI latent.

At inference time, only an RGB image is required. The RGB encoder produces a base latent, the diffusion model generates a corrective latent residual, and the pretrained HSI decoder reconstructs the 31-band cube.

---

## 2. Repository roles

### Repository A: `vae_sriram-main`

This repository trains the reusable latent-space components.

- `main.py`: trains the HSI beta-VAE.
- `main2.py`: freezes the HSI encoder/decoder and trains the RGB encoder.
- `models/HSIencoder.py`: HSI encoder.
- `models/HSIDecoder.py`: HSI decoder.
- `models/HSIBetaVAE.py`: combines encoder, reparameterization, and decoder.
- `models/RGBEncoder.py`: RGB-to-latent encoder.
- `dataset_loader/Dataset_loader.py`: HSI-only ARAD loader.
- `dataset_loader/dataset_loader_rgbandhsi.py`: paired RGB-HSI loader.

### Repository B: `Residual_predictor_rgb_to_hsi-main`

This repository trains and evaluates the conditional latent residual-diffusion model.

- `main.py`: trains `ResidualDiffusionModel` with frozen pretrained components.
- `inference.py`: reconstructs HSI cubes for 50 random ARAD samples.
- `models/Residual_diffusion.py`: time-conditioned residual-noise predictor, scheduler, and reverse sampler.
- `dataset/dataset_loader.py`: paired RGB-HSI loader.
- `dataset/random_arad_loader.py`: reproducible random subset selection.
- `loss/`: MRAE, SAM, PSNR, SSIM, and RMSE implementations.

---

## 3. End-to-end data flow

```text
Training pair: RGB image x_rgb and HSI cube x_hsi

                  Stage 1: HSI beta-VAE
x_hsi [B,31,256,256]
        |
        v
HSI encoder E_hsi
        |
        +--> mu_hsi [B,8,64,64]
        +--> logvar_hsi [B,8,64,64]
        |
        v
sample z_hsi = mu_hsi + eps * exp(0.5*logvar_hsi)
        |
        v
HSI decoder D_hsi
        |
        v
reconstructed HSI [B,31,256,256]

                  Stage 2: RGB latent alignment
x_rgb [B,3,256,256] --> RGB encoder E_rgb --> z_rgb [B,8,64,64]
                                               |
Ground-truth target: mu_hsi <------------------+

                  Stage 3: residual diffusion
r_0 = mu_hsi - z_rgb
r_t = sqrt(alpha_bar_t) * r_0 + sqrt(1-alpha_bar_t) * noise

(noisy residual r_t, timestep t, RGB condition z_rgb)
                         |
                         v
              ResidualDiffusionModel
                         |
                         v
                 predicted noise

                  Inference
RGB --> E_rgb --> z_rgb
random Gaussian residual r_T
        |
        v
reverse diffusion conditioned on z_rgb
        |
        v
predicted residual r_hat
        |
z_hsi_hat = z_rgb + r_hat
        |
        v
D_hsi --> predicted 31-band HSI
```

---

## 4. Dataset preparation

The loaders retrieve paired files from the Hugging Face dataset repository `mhmdjouni/arad_hsdb` and place them under:

```text
data/
├── NTIRE2020_Train_Spectral/
│   └── *.mat
└── NTIRE2020_Train_RealWorld/
    └── *.jpg
```

### HSI loading

- MATLAB variable key: `cube` by default.
- Original layout is converted from `[H, W, 31]` to `[31, H, W]`.
- The cube is resized to `[31, 256, 256]` using bilinear interpolation.
- The provided code does not apply active HSI normalization.

### RGB loading

- RGB images are converted to three channels.
- Pixel values are divided by 255.
- Layout is converted from `[H, W, 3]` to `[3, H, W]`.
- Images are resized to `[3, 256, 256]` using bilinear interpolation.

### Pair construction

The loader associates an HSI file with an RGB file by matching the common filename stem after removing `_RealWorld.jpg` from the RGB filename.

### Default split in training scripts

The current loaders use an ordered split rather than a randomized split:

- first 200 paired samples: training;
- next 30 paired samples: validation;
- total requested samples: 230.

Changing `train_images` or `total_images` must be done consistently across all stages.

---

## 5. Stage 1: HSI beta-VAE training

### Entry point

```bash
cd vae_sriram-main
python main.py
```

### Model

The beta-VAE is:

```text
HSIBetaVAE = HSIEncoder + Reparameterization + HSIDecoder
```

For a 256 x 256 HSI input and `LATENT_CHANNELS = 8`:

| Operation | Output shape |
|---|---|
| Input HSI | `[B, 31, 256, 256]` |
| First convolution + residual block | `[B, 64, 256, 256]` |
| Strided convolution | `[B, 128, 128, 128]` |
| Strided convolution | `[B, 256, 64, 64]` |
| Mean head | `[B, 8, 64, 64]` |
| Log-variance head | `[B, 8, 64, 64]` |
| Reparameterized latent | `[B, 8, 64, 64]` |
| Decoder output | `[B, 31, 256, 256]` |

### Objective

The reconstruction objective is:

```text
L_recon = 0.5 L1 + 0.4 MRAE + 0.1 SAM
L_total = L_recon + beta(epoch) * KL(mu, logvar)
```

The KL weight is linearly warmed up:

```text
beta(epoch) = BETA_MAX * min(epoch / WARMUP_EPOCHS, 1)
```

Default values:

- epochs: 100;
- batch size: 1;
- learning rate: `1e-4`;
- AdamW weight decay: `1e-4`;
- latent channels: 8;
- maximum beta: `1e-5`;
- warm-up: 20 epochs.

### Produced checkpoints

```text
checkpoints/
├── best_model.pth
├── encoder_final.pth
└── decoder_final.pth
```

`best_model.pth` contains:

```python
{
    "epoch": ...,
    "vae": model.state_dict(),
    "encoder": model.encoder.state_dict(),
    "decoder": model.decoder.state_dict(),
    "optimizer": optimizer.state_dict(),
    "val_loss": ...
}
```

The later stages specifically expect the keys `encoder` and `decoder`.

---

## 6. Stage 2: RGB encoder training

### Dependency

Copy or retain the Stage 1 checkpoint at:

```text
vae_sriram-main/checkpoints/best_model.pth
```

### Entry point

```bash
cd vae_sriram-main
python main2.py
```

### Training logic

The pretrained HSI encoder and HSI decoder are loaded and frozen. For each paired sample:

```text
mu_hsi = E_hsi(x_hsi)
z_rgb  = E_rgb(x_rgb)
x_hat  = D_hsi(z_rgb)
```

The RGB encoder is optimized with:

```text
L_latent = MSE(z_rgb, mu_hsi)
L_recon  = L1(x_hat, x_hsi)
L_total  = L_latent + 0.1 L_recon
```

Using `mu_hsi` rather than a stochastic VAE sample gives the RGB encoder a deterministic latent target.

Default values:

- epochs: 100;
- batch size: 4;
- learning rate: `1e-4`;
- latent channels: 8.

### Produced checkpoint

```text
checkpoints_rgb/rgb_to_hsi_best.pth
```

It contains:

```python
{
    "rgb_encoder": rgb_encoder.state_dict(),
    "hsi_decoder": hsi_decoder.state_dict(),
    "val_mrae": ...,
    "epoch": ...
}
```

The residual-diffusion repository currently expects this file under its own `checkpoints/` directory:

```bash
mkdir -p ../Residual_predictor_rgb_to_hsi-main/checkpoints
cp checkpoints/best_model.pth \
   ../Residual_predictor_rgb_to_hsi-main/checkpoints/best_model.pth
cp checkpoints_rgb/rgb_to_hsi_best.pth \
   ../Residual_predictor_rgb_to_hsi-main/checkpoints/rgb_to_hsi_best.pth
```

---

## 7. Stage 3: latent residual-diffusion training

### Entry point

```bash
cd Residual_predictor_rgb_to_hsi-main
python main.py
```

### Frozen components

The script loads and freezes:

- `RGBEncoder` from `checkpoints/rgb_to_hsi_best.pth`;
- `HSIEncoder` from `checkpoints/best_model.pth`;
- `HSIDecoder` from `checkpoints/best_model.pth`.

Only `ResidualDiffusionModel` is optimized.

### Residual target

For each paired sample:

```text
z_rgb = E_rgb(x_rgb)
mu_hsi, _ = E_hsi(x_hsi)
r_0 = mu_hsi - z_rgb
```

The model therefore learns the latent correction not already captured by the RGB encoder.

### Forward diffusion

A timestep `t` is sampled uniformly from `[0, T-1]`. Gaussian noise is added using:

```text
r_t = sqrt(alpha_bar_t) r_0 + sqrt(1-alpha_bar_t) epsilon
```

where:

```text
beta_t  linearly increases from 1e-4 to 2e-2
alpha_t = 1 - beta_t
alpha_bar_t = product(alpha_0 ... alpha_t)
```

### Conditional noise predictor

`ResidualDiffusionModel` receives:

- noisy residual `r_t`: `[B, 8, 64, 64]`;
- RGB latent condition `z_rgb`: `[B, 8, 64, 64]`;
- timestep `t`: `[B]`.

The two latent tensors are concatenated to `[B, 16, 64, 64]`. A sinusoidal timestep embedding is projected to the hidden dimension and added after the first convolution.

Default architecture:

- latent channels: 8;
- hidden channels: 64;
- time embedding dimension: 128;
- four hidden 3 x 3 convolutions;
- one output 3 x 3 convolution producing 8 noise channels.

### Training objective

The model predicts the exact Gaussian noise used to create `r_t`:

```text
L_diffusion = MSE(epsilon_theta(r_t, t, z_rgb), epsilon)
```

Default values:

- epochs: 100;
- batch size: 8;
- learning rate: `1e-4`;
- diffusion timesteps: 100;
- optimizer: Adam.

### Training-time reconstruction estimate

To avoid a full 100-step reverse process for every training batch, the code estimates the clean residual directly:

```text
r_0_hat = (r_t - sqrt(1-alpha_bar_t) epsilon_hat) / sqrt(alpha_bar_t)
z_hsi_hat = z_rgb + r_0_hat
x_hsi_hat = D_hsi(z_hsi_hat)
```

MRAE, SAM, PSNR, and SSIM are reported from this estimate. These metrics do not contribute to the diffusion loss.

### Validation

Validation computes:

1. a random-timestep noise-prediction MSE;
2. a complete reverse-diffusion sample;
3. reconstruction metrics after decoding `z_rgb + r_hat`.

### Produced checkpoint

```text
checkpoints/residual_diffusion_best.pth
```

The current training script stores only:

```python
residual_net.state_dict()
```

The best checkpoint is selected by validation noise-prediction loss, not by MRAE, SAM, PSNR, or SSIM.

---

## 8. Inference on 50 random ARAD samples

### Entry point

```bash
cd Residual_predictor_rgb_to_hsi-main
python inference.py
```

### Required checkpoints

```text
checkpoints/
├── best_model.pth
├── rgb_to_hsi_best.pth
└── residual_diffusion_best.pth
```

All architecture constants in inference must match training:

```python
LATENT_CHANNELS = 8
HIDDEN_DIM = 64
TIME_DIM = 128
DIFFUSION_TIMESTEPS = 100
```

### Random selection

`dataset/random_arad_loader.py` exposes:

```python
load_random_arad1k_samples(
    root_dir="data",
    num_samples=50,
    seed=42,
    total_images=1000,
    cube_key="cube",
    download=True
)
```

It:

1. constructs the complete paired pool requested from the base loader;
2. samples unique indices without replacement using `random.Random(seed)`;
3. returns a PyTorch `Subset` and filename metadata.

Keeping the same seed reproduces the selected image list. Reverse diffusion is also stochastic, so PyTorch and CUDA seeds are set in `inference.py`.

### Inference equations

```text
z_rgb = E_rgb(x_rgb)
r_T ~ N(0, I)
r_hat = reverse_diffusion(r_T | z_rgb)
z_hsi_hat = z_rgb + r_hat
x_hsi_hat = D_hsi(z_hsi_hat)
```

### Outputs

```text
inference_results/
├── selected_samples.csv
├── metrics.csv
├── summary.txt
└── prediction_XXXX_<sample-name>.mat
```

Each MAT file contains:

```python
{
    "cube": predicted_hsi_hwc,
    "ground_truth": ground_truth_hsi_hwc
}
```

Both arrays use `[height, width, spectral_channels]` order.

---

## 9. Checkpoint dependency map

| Stage | Trained component | Loaded checkpoint | Output checkpoint |
|---|---|---|---|
| 1 | HSI encoder + decoder | none | `checkpoints/best_model.pth` |
| 2 | RGB encoder | `checkpoints/best_model.pth` | `checkpoints_rgb/rgb_to_hsi_best.pth` |
| 3 | Residual diffusion | VAE and RGB checkpoints | `checkpoints/residual_diffusion_best.pth` |
| Inference | none | all three checkpoints | reconstructed cubes and metrics |

The same values of `LATENT_CHANNELS`, input size, spectral-band count, and architecture definitions must be used in every stage.

---

## 10. Recommended environment

A minimal installation is:

```bash
python -m venv .venv
source .venv/bin/activate        # Linux/macOS
# .venv\Scripts\activate         # Windows

pip install torch torchvision numpy scipy pillow huggingface_hub
```

A CUDA-enabled PyTorch build should be installed according to the local CUDA driver. The scripts automatically choose CUDA when available.

Suggested reproducibility additions:

```python
import random
import numpy as np
import torch

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
```

For stricter deterministic behavior, enable deterministic PyTorch algorithms where supported, noting that this may reduce performance.

---

## 11. Important consistency and evaluation notes

### 11.1 Data-scale consistency

The RGB input is normalized to `[0,1]`, while active HSI normalization is absent in the provided loaders. Verify that the MATLAB cubes already have an appropriate and consistent scale. Otherwise, MRAE and reconstruction losses can become unstable or misleading.

### 11.2 Dataset identity

Although helper names use “ARAD1K,” the folders and downloaded files are named `NTIRE2020_Train_Spectral` and `NTIRE2020_Train_RealWorld`. Confirm that this is the intended ARAD/NTIRE subset for the experiment and describe it consistently in publications.

### 11.3 Train/test overlap risk

The inference helper randomly selects from the full sorted pool of up to 1000 samples. It does not automatically exclude the first 200 training samples or the following validation samples used by the default training scripts. For an unbiased test, construct an explicit held-out index range or exclude all indices used during Stages 1-3 before random sampling.

### 11.4 Shared split across stages

All three training stages should use the same paired-file ordering and split definition. A change to one loader but not the others can produce leakage or incompatible evaluation sets.

### 11.5 Stochastic inference

Reverse diffusion begins from Gaussian noise. Even for the same RGB image, predictions can vary unless all random seeds and deterministic settings are fixed. Multiple samples per RGB image can also be generated to study reconstruction uncertainty.

### 11.6 Checkpoint selection criterion

The diffusion model is currently saved according to validation noise MSE. If final HSI quality is the primary goal, consider selecting by validation MRAE or a composite reconstruction score, while still reporting noise MSE.

### 11.7 Metric aggregation

The inference script uses batch size 1, so averaging batch metrics is equivalent to averaging per-image metrics. If batch size is increased, ensure that each metric implementation returns a true per-image average before accumulating it.

### 11.8 Spectral assumptions

The decoder output is fixed at 31 channels. The wavelength range and spectral spacing are not encoded in the architecture. They must match the dataset used to train the HSI VAE.

---

## 12. Suggested unified project layout

```text
rgb_hsi_pipeline/
├── data/
│   ├── NTIRE2020_Train_Spectral/
│   └── NTIRE2020_Train_RealWorld/
├── checkpoints/
│   ├── best_model.pth
│   ├── rgb_to_hsi_best.pth
│   └── residual_diffusion_best.pth
├── vae_sriram-main/
├── Residual_predictor_rgb_to_hsi-main/
└── inference_results/
```

The current scripts use relative paths, so either copy checkpoints into each expected location or replace hard-coded paths with one shared configuration file.

---

## 13. Complete execution order

```bash
# Stage 1: train HSI beta-VAE
cd vae_sriram-main
python main.py

# Stage 2: train RGB encoder against the frozen HSI latent space
python main2.py

# Copy required pretrained checkpoints
mkdir -p ../Residual_predictor_rgb_to_hsi-main/checkpoints
cp checkpoints/best_model.pth \
   ../Residual_predictor_rgb_to_hsi-main/checkpoints/best_model.pth
cp checkpoints_rgb/rgb_to_hsi_best.pth \
   ../Residual_predictor_rgb_to_hsi-main/checkpoints/rgb_to_hsi_best.pth

# Stage 3: train conditional residual diffusion
cd ../Residual_predictor_rgb_to_hsi-main
python main.py

# Inference on 50 reproducibly selected random samples
python inference.py
```

---

## 14. Troubleshooting

### `KeyError: 'encoder'` or `KeyError: 'decoder'`

The VAE checkpoint does not follow the expected dictionary format. Use `checkpoints/best_model.pth` produced by Stage 1 rather than `encoder_final.pth` or `decoder_final.pth` directly.

### State-dictionary size mismatch

At least one architecture constant differs between training and loading. Check latent channels, hidden dimension, time dimension, spectral channels, and model source files.

### Empty validation loader or division by zero

The number of locally paired files is not larger than `train_images`. Download more samples or reduce the split boundary.

### No paired samples found

Check that both directories exist and that HSI and RGB filenames share the expected stem.

### CUDA out-of-memory

Reduce batch size first. Reverse diffusion is particularly expensive because it invokes the residual network once per diffusion timestep.

### Very slow validation or inference

`sample_residual` performs all 100 reverse steps. Reduce `DIFFUSION_TIMESTEPS` only if the model is retrained with the same schedule, or implement a compatible accelerated sampler.

### NaN or extreme MRAE

Inspect HSI scaling and near-zero ground-truth values. The metric implementation should include a stable denominator epsilon.

---

## 15. Summary

The pipeline decomposes RGB-to-HSI reconstruction into a deterministic base estimate and a learned stochastic correction:

```text
predicted HSI latent = RGB latent + diffusion-generated latent residual
```

The HSI beta-VAE first establishes a compact, decodable spectral latent space. The RGB encoder learns the portion of that latent predictable directly from RGB. The conditional diffusion model then learns the remaining latent discrepancy. This modular training strategy allows the decoder and both encoders to remain fixed while the residual distribution is learned separately.
