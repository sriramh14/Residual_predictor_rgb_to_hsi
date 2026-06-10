import os
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from models.HSIencoder import HSIEncoder
from models.HSIDecoder import HSIDecoder
from models.RGBEncoder import RGBEncoder

from dataset.dataset_loader import ARADDataset
from models.Residual_predictor import ResidualPredictor

# --------------------------------------------------
# CONFIG
# --------------------------------------------------

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

BATCH_SIZE = 8
NUM_EPOCHS = 100
LR = 1e-4

LATENT_CHANNELS = 8

CHECKPOINT_DIR = "checkpoints"
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

# --------------------------------------------------
# LOAD DATA
# --------------------------------------------------

train_dataset = ARADDataset(
    train=True
)

val_dataset = ARADDataset(
    train=False
)

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=4
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=4
)
# --------------------------------------------------
# Instantiate models 
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
    in_channels = 3,
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

#Loading HSI VAE
vae_ckpt = torch.load(
    "checkpoints/best_model.pth"
)
hsi_encoder.load_state_dict(
    vae_ckpt["encoder"]
)

hsi_decoder.load_state_dict(
    vae_ckpt["decoder"]
)

print("Loaded pretrained RGB encoder and HSI VAE")

rgb_encoder.to(DEVICE)
hsi_encoder.to(DEVICE)
hsi_decoder.to(DEVICE)

rgb_encoder.eval()
hsi_encoder.eval()
hsi_decoder.eval()

for p in rgb_encoder.parameters():
    p.requires_grad = False

for p in hsi_encoder.parameters():
    p.requires_grad = False

for p in hsi_decoder.parameters():
    p.requires_grad = False

# --------------------------------------------------
# RESIDUAL MODEL
# --------------------------------------------------

residual_net = ResidualPredictor(
    latent_dim=LATENT_CHANNELS
).to(DEVICE)

optimizer = torch.optim.Adam(
    residual_net.parameters(),
    lr=LR
)

# --------------------------------------------------
# TRAIN
# --------------------------------------------------

best_loss = float("inf")

for epoch in range(NUM_EPOCHS):

    residual_net.train()

    running_loss = 0.0

    for rgb, hsi in train_loader:

        rgb = rgb.to(DEVICE)
        hsi = hsi.to(DEVICE)

        with torch.no_grad():

            z_rgb = rgb_encoder(rgb)

            z_hsi = hsi_encoder(hsi)

        delta_pred = residual_net(
            z_rgb
        )

        z_final = z_rgb + delta_pred

        hsi_pred = hsi_decoder(
            z_final
        )

        loss_latent = F.l1_loss(
            z_final,
            z_hsi
        )

        loss_recon = F.l1_loss(
            hsi_pred,
            hsi
        )

        loss = (
            loss_latent
            + loss_recon
        )

        optimizer.zero_grad()

        loss.backward()

        optimizer.step()

        running_loss += loss.item()

    train_loss = (
        running_loss
        / len(train_loader)
    )

    # --------------------------------------
    # VALIDATION
    # --------------------------------------

    residual_net.eval()

    val_loss = 0.0

    with torch.no_grad():

        for rgb, hsi in val_loader:

            rgb = rgb.to(DEVICE)
            hsi = hsi.to(DEVICE)

            z_rgb = rgb_encoder(rgb)

            z_hsi = hsi_encoder(hsi)

            delta_pred = residual_net(
                z_rgb
            )

            z_final = (
                z_rgb
                + delta_pred
            )

            hsi_pred = hsi_decoder(
                z_final
            )

            loss_latent = F.l1_loss(
                z_final,
                z_hsi
            )

            loss_recon = F.l1_loss(
                hsi_pred,
                hsi
            )

            loss = (
                loss_latent
                + loss_recon
            )

            val_loss += loss.item()

    val_loss /= len(val_loader)

    print(
        f"Epoch {epoch+1}/{NUM_EPOCHS} "
        f"| Train {train_loss:.6f} "
        f"| Val {val_loss:.6f}"
    )

    if val_loss < best_loss:

        best_loss = val_loss

        torch.save(
            residual_net.state_dict(),
            os.path.join(
                CHECKPOINT_DIR,
                "residual_predictor_best.pth"
            )
        )

        print(
            "Saved best model"
        )
