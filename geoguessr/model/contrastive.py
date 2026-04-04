"""
Phase 1.5: Contrastive pretraining with geographic captions.

Teaches StreetCLIP's embedding space to encode geographic proximity:
  - Generate captions from GPS metadata (country, region, city, climate, etc.)
  - InfoNCE contrastive loss aligns image embeddings with caption embeddings
  - Only DoRA adapter parameters are trainable (frozen backbone)

Usage:
    python -m GEOPROJECT.geoguessr.model.contrastive \
        --metadata GEOPROJECT/data/osv5m_50k/metadata.csv \
        --images GEOPROJECT/data/osv5m_50k/images \
        --epochs 2 --batch-size 6
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from peft import LoraConfig, get_peft_model
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from transformers import CLIPModel, CLIPProcessor, CLIPTokenizer


# ── Country code to full name mapping ─────────────────────────────────

COUNTRY_NAMES = {
    "US": "United States", "CA": "Canada", "MX": "Mexico", "GB": "United Kingdom",
    "FR": "France", "DE": "Germany", "IT": "Italy", "ES": "Spain", "PT": "Portugal",
    "NL": "Netherlands", "BE": "Belgium", "CH": "Switzerland", "AT": "Austria",
    "SE": "Sweden", "NO": "Norway", "DK": "Denmark", "FI": "Finland", "IS": "Iceland",
    "IE": "Ireland", "PL": "Poland", "CZ": "Czech Republic", "SK": "Slovakia",
    "HU": "Hungary", "RO": "Romania", "BG": "Bulgaria", "HR": "Croatia",
    "RS": "Serbia", "BA": "Bosnia and Herzegovina", "SI": "Slovenia", "ME": "Montenegro",
    "MK": "North Macedonia", "AL": "Albania", "GR": "Greece", "CY": "Cyprus",
    "MT": "Malta", "LU": "Luxembourg", "LI": "Liechtenstein", "EE": "Estonia",
    "LV": "Latvia", "LT": "Lithuania", "UA": "Ukraine", "BY": "Belarus",
    "MD": "Moldova", "RU": "Russia", "GE": "Georgia", "AM": "Armenia", "AZ": "Azerbaijan",
    "TR": "Turkey", "IL": "Israel", "JO": "Jordan", "LB": "Lebanon", "SA": "Saudi Arabia",
    "AE": "United Arab Emirates", "QA": "Qatar", "KW": "Kuwait", "BH": "Bahrain",
    "OM": "Oman", "YE": "Yemen", "IQ": "Iraq", "IR": "Iran", "SY": "Syria",
    "PS": "Palestine", "EG": "Egypt", "LY": "Libya", "TN": "Tunisia",
    "DZ": "Algeria", "MA": "Morocco", "SD": "Sudan",
    "ZA": "South Africa", "NG": "Nigeria", "KE": "Kenya", "ET": "Ethiopia",
    "GH": "Ghana", "TZ": "Tanzania", "UG": "Uganda", "SN": "Senegal",
    "CM": "Cameroon", "CI": "Ivory Coast", "MG": "Madagascar", "MZ": "Mozambique",
    "ZM": "Zambia", "ZW": "Zimbabwe", "BW": "Botswana", "NA": "Namibia",
    "AO": "Angola", "CD": "DR Congo", "CG": "Congo", "GA": "Gabon",
    "RW": "Rwanda", "BI": "Burundi", "MW": "Malawi", "ML": "Mali",
    "BF": "Burkina Faso", "NE": "Niger", "TD": "Chad", "SO": "Somalia",
    "DJ": "Djibouti", "ER": "Eritrea", "MR": "Mauritania", "GM": "Gambia",
    "GW": "Guinea-Bissau", "SL": "Sierra Leone", "LR": "Liberia",
    "TG": "Togo", "BJ": "Benin", "CF": "Central African Republic",
    "SS": "South Sudan", "MU": "Mauritius", "SC": "Seychelles",
    "CV": "Cape Verde", "LS": "Lesotho", "SZ": "Eswatini", "GQ": "Equatorial Guinea",
    "IN": "India", "PK": "Pakistan", "BD": "Bangladesh", "LK": "Sri Lanka",
    "NP": "Nepal", "BT": "Bhutan", "MV": "Maldives",
    "CN": "China", "JP": "Japan", "KR": "South Korea", "TW": "Taiwan",
    "HK": "Hong Kong", "MO": "Macau", "KP": "North Korea", "MN": "Mongolia",
    "KZ": "Kazakhstan", "UZ": "Uzbekistan", "TM": "Turkmenistan",
    "KG": "Kyrgyzstan", "TJ": "Tajikistan", "AF": "Afghanistan",
    "TH": "Thailand", "VN": "Vietnam", "MY": "Malaysia", "ID": "Indonesia",
    "PH": "Philippines", "SG": "Singapore", "MM": "Myanmar", "KH": "Cambodia",
    "LA": "Laos", "BN": "Brunei", "TL": "Timor-Leste",
    "AU": "Australia", "NZ": "New Zealand", "PG": "Papua New Guinea",
    "FJ": "Fiji", "WS": "Samoa", "TO": "Tonga", "VU": "Vanuatu",
    "SB": "Solomon Islands", "KI": "Kiribati",
    "BR": "Brazil", "AR": "Argentina", "CL": "Chile", "CO": "Colombia",
    "PE": "Peru", "VE": "Venezuela", "EC": "Ecuador", "BO": "Bolivia",
    "PY": "Paraguay", "UY": "Uruguay", "GY": "Guyana", "SR": "Suriname",
    "GT": "Guatemala", "BZ": "Belize", "SV": "El Salvador", "HN": "Honduras",
    "NI": "Nicaragua", "CR": "Costa Rica", "PA": "Panama",
    "CU": "Cuba", "JM": "Jamaica", "HT": "Haiti", "DO": "Dominican Republic",
    "PR": "Puerto Rico", "TT": "Trinidad and Tobago", "BB": "Barbados",
    "BS": "Bahamas",
}

# Climate code to human-readable name
CLIMATE_CODE_NAMES = {
    0: "tropical rainforest", 1: "tropical monsoon", 2: "tropical savanna",
    3: "hot desert", 4: "cold desert", 5: "hot semi-arid", 6: "cold semi-arid",
    7: "Mediterranean hot summer", 8: "Mediterranean warm summer",
    9: "Mediterranean cold summer", 10: "humid subtropical dry winter",
    11: "subtropical highland", 12: "cold subtropical highland",
    13: "humid subtropical", 14: "oceanic", 15: "subpolar oceanic",
    16: "continental hot dry summer", 17: "continental warm dry summer",
    18: "continental cold dry summer", 19: "continental very cold dry summer",
    21: "continental warm dry winter", 22: "continental cold dry winter",
    23: "continental very cold dry winter",
    25: "continental warm humid", 26: "continental cold humid",
    27: "continental very cold humid", 29: "tundra",
}

LAND_COVER_NAMES = {
    0: "urban", 1: "urban", 2: "cropland", 3: "cropland",
    4: "forest", 5: "forest", 6: "sparse vegetation", 7: "sparse vegetation",
    8: "wetland", 9: "wetland", 10: "snow or ice", 11: "snow or ice",
}

DRIVE_SIDE_NAMES = {0: "right", 1: "left"}


# ── Caption Generation ────────────────────────────────────────────────

def generate_caption(row: pd.Series) -> str:
    """Generate a geographic caption from metadata fields."""
    parts = ["A street view photo"]

    # Location: city, region, country
    location_parts = []
    city = row.get("city", "")
    if pd.notna(city) and city and city != "":
        location_parts.append(str(city))
    region = row.get("region", "")
    if pd.notna(region) and region and region != "":
        location_parts.append(str(region))
    country_code = row.get("country", "")
    country_name = COUNTRY_NAMES.get(country_code, country_code)
    if country_name:
        location_parts.append(country_name)

    if location_parts:
        parts.append("in " + ", ".join(location_parts))

    # Climate
    climate = row.get("climate", -1)
    if pd.notna(climate) and climate >= 0:
        climate_name = CLIMATE_CODE_NAMES.get(int(climate), "")
        if climate_name:
            parts.append(f"with {climate_name} climate")

    # Scene type
    land_cover = row.get("land_cover", -1)
    if pd.notna(land_cover) and land_cover >= 0:
        scene = LAND_COVER_NAMES.get(int(land_cover), "")
        if scene:
            parts.append(f"in an {scene} area" if scene[0] in "aeiou" else f"in a {scene} area")

    # Driving side
    drive = row.get("drive_side", -1)
    if pd.notna(drive) and drive >= 0:
        side = DRIVE_SIDE_NAMES.get(int(drive), "")
        if side:
            parts.append(f"with {side}-hand traffic")

    return ". ".join(parts) + "."


# ── Contrastive Dataset ───────────────────────────────────────────────

CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
CLIP_STD = (0.26862954, 0.26130258, 0.27577711)
CLIP_SIZE = 336


class ContrastiveDataset(Dataset):
    """Dataset for contrastive pretraining: returns (image, caption)."""

    def __init__(self, df: pd.DataFrame, images_dir: str):
        self.images_dir = Path(images_dir)
        self.image_ids = df["id"].astype(str).values

        # Pre-generate all captions
        self.captions = [generate_caption(row) for _, row in df.iterrows()]

        self.transform = transforms.Compose([
            transforms.Resize(CLIP_SIZE, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.CenterCrop(CLIP_SIZE),
            transforms.ToTensor(),
            transforms.Normalize(mean=CLIP_MEAN, std=CLIP_STD),
        ])

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):
        img_path = self.images_dir / f"{self.image_ids[idx]}.jpg"
        image = Image.open(img_path).convert("RGB")
        image = self.transform(image)
        return image, self.captions[idx]


# ── Model Setup ───────────────────────────────────────────────────────

def create_contrastive_model(device: torch.device):
    """Load StreetCLIP with DoRA adapters on vision encoder."""
    print("Loading StreetCLIP...")
    model = CLIPModel.from_pretrained("geolocal/StreetCLIP")
    tokenizer = CLIPTokenizer.from_pretrained("geolocal/StreetCLIP")

    # Freeze everything
    for param in model.parameters():
        param.requires_grad = False

    # Apply DoRA to vision encoder (last 8 layers, Q and V projections)
    dora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["q_proj", "v_proj"],
        use_dora=True,
        layers_to_transform=list(range(16, 24)),
    )
    model.vision_model = get_peft_model(model.vision_model, dora_config)

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"  Trainable: {trainable:,} / {total:,} ({100*trainable/total:.2f}%)")

    model = model.to(device)
    return model, tokenizer


# ── InfoNCE Loss ──────────────────────────────────────────────────────

def contrastive_loss(image_embeds: torch.Tensor, text_embeds: torch.Tensor, logit_scale: float):
    """Symmetric InfoNCE loss (same as CLIP's training objective).

    Args:
        image_embeds: (B, D) L2-normalized image embeddings
        text_embeds: (B, D) L2-normalized text embeddings
        logit_scale: learned temperature (exp of raw parameter)

    Returns:
        loss: scalar
    """
    # Similarity matrix: (B, B)
    logits = logit_scale * image_embeds @ text_embeds.T
    # Labels: diagonal (each image matches its own caption)
    labels = torch.arange(len(logits), device=logits.device)
    # Symmetric loss
    loss_i2t = F.cross_entropy(logits, labels)
    loss_t2i = F.cross_entropy(logits.T, labels)
    return (loss_i2t + loss_t2i) / 2


# ── Training Loop ─────────────────────────────────────────────────────

def train_contrastive(
    model: CLIPModel,
    tokenizer: CLIPTokenizer,
    train_loader: DataLoader,
    device: torch.device,
    epochs: int = 2,
    lr: float = 1e-4,
    accumulation_steps: int = 10,
    checkpoint_dir: str = "GEOPROJECT/checkpoints",
):
    """Train contrastive pretraining with InfoNCE loss."""
    os.makedirs(checkpoint_dir, exist_ok=True)

    # Optimizer: only DoRA parameters
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable_params, lr=lr, weight_decay=0.01)

    # Cosine scheduler
    total_steps = len(train_loader) * epochs // accumulation_steps
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps)

    model.train()
    global_step = 0
    best_loss = float("inf")

    for epoch in range(epochs):
        epoch_loss = 0.0
        epoch_steps = 0
        optimizer.zero_grad()
        t0 = time.time()

        for batch_idx, (images, captions) in enumerate(train_loader):
            images = images.to(device)

            # Tokenize captions
            text_inputs = tokenizer(
                captions, return_tensors="pt", padding=True,
                truncation=True, max_length=77,
            ).to(device)

            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                # Forward: get embeddings using pooler_output (post-LayerNorm CLS)
                # Note: can't use get_image_features() because PEFT wrapper
                # changes the return type
                vision_outputs = model.vision_model(pixel_values=images)
                image_embeds = model.visual_projection(vision_outputs.pooler_output)
                image_embeds = F.normalize(image_embeds, dim=-1)

                text_outputs = model.text_model(
                    input_ids=text_inputs["input_ids"],
                    attention_mask=text_inputs["attention_mask"],
                )
                text_embeds = model.text_projection(text_outputs.pooler_output)
                text_embeds = F.normalize(text_embeds, dim=-1)

                logit_scale = model.logit_scale.exp()
                loss = contrastive_loss(image_embeds, text_embeds, logit_scale)
                loss = loss / accumulation_steps

            loss.backward()

            if (batch_idx + 1) % accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=1.0)
                optimizer.step()
                optimizer.zero_grad()
                scheduler.step()
                global_step += 1

            epoch_loss += loss.item() * accumulation_steps
            epoch_steps += 1

            # Progress every 100 batches
            if (batch_idx + 1) % 100 == 0:
                avg = epoch_loss / epoch_steps
                elapsed = time.time() - t0
                rate = (batch_idx + 1) / elapsed
                eta = (len(train_loader) - batch_idx - 1) / rate
                print(
                    f"  Epoch {epoch+1} [{batch_idx+1}/{len(train_loader)}] "
                    f"loss={avg:.4f} lr={scheduler.get_last_lr()[0]:.2e} "
                    f"{rate:.1f} batch/s ETA: {eta/60:.1f}min"
                )

        avg_loss = epoch_loss / epoch_steps
        elapsed = time.time() - t0
        print(f"Epoch {epoch+1}/{epochs} done: loss={avg_loss:.4f} time={elapsed/60:.1f}min")

        # Save checkpoint
        ckpt_path = Path(checkpoint_dir) / f"contrastive_epoch{epoch+1}.pt"
        save_dora_checkpoint(model, optimizer, scheduler, epoch, avg_loss, ckpt_path)
        print(f"  Saved checkpoint: {ckpt_path}")

        if avg_loss < best_loss:
            best_loss = avg_loss
            best_path = Path(checkpoint_dir) / "contrastive_best.pt"
            save_dora_checkpoint(model, optimizer, scheduler, epoch, avg_loss, best_path)
            print(f"  New best! Saved: {best_path}")

    return model


# ── Checkpoint Save/Load ──────────────────────────────────────────────

def save_dora_checkpoint(model, optimizer, scheduler, epoch, loss, path):
    """Save only DoRA adapter weights + training state."""
    # Extract DoRA state dict (only trainable params)
    dora_state = {
        k: v for k, v in model.state_dict().items()
        if "lora_" in k or "dora_" in k or "magnitude" in k
    }
    torch.save({
        "epoch": epoch,
        "loss": loss,
        "dora_state_dict": dora_state,
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
    }, path)


def load_dora_checkpoint(model, path, optimizer=None, scheduler=None):
    """Load DoRA adapter weights."""
    ckpt = torch.load(path, map_location="cpu", weights_only=True)
    # Load DoRA weights into model
    missing, unexpected = model.load_state_dict(ckpt["dora_state_dict"], strict=False)
    print(f"  Loaded DoRA weights from {path}")
    print(f"  Epoch: {ckpt['epoch']+1}, Loss: {ckpt['loss']:.4f}")
    if optimizer and "optimizer_state_dict" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
    if scheduler and "scheduler_state_dict" in ckpt:
        scheduler.load_state_dict(ckpt["scheduler_state_dict"])
    return ckpt["epoch"]


# ── Evaluation: Geographic NN Quality ─────────────────────────────────

def evaluate_geo_nn(
    model: CLIPModel,
    dataset: ContrastiveDataset,
    df: pd.DataFrame,
    device: torch.device,
    n_queries: int = 100,
    k: int = 10,
):
    """Check if nearest neighbors in embedding space are geographically close.

    Embeds a random sample, finds k-NN, reports median haversine distance
    between query and its neighbors.
    """
    from GEOPROJECT.geoguessr.data.geocells import haversine_km

    model.eval()
    n = min(n_queries * 10, len(dataset))  # embed more for a good pool
    n_queries = min(n_queries, n)  # guard against tiny datasets
    indices = np.random.RandomState(42).choice(len(dataset), n, replace=False)

    # Extract embeddings
    embeddings = []
    with torch.no_grad():
        for i in range(0, n, 32):
            batch_idx = indices[i:i+32]
            imgs = torch.stack([dataset[j][0] for j in batch_idx]).to(device)
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                vis_out = model.vision_model(pixel_values=imgs)
                emb = model.visual_projection(vis_out.pooler_output)
                emb = F.normalize(emb, dim=-1)
            embeddings.append(emb.cpu())

    embeddings = torch.cat(embeddings, dim=0).numpy()  # (n, 768)

    lats = df["latitude"].values[indices]
    lons = df["longitude"].values[indices]

    # For first n_queries images, find k-NN and compute geo distance
    distances = []
    for qi in range(n_queries):
        # Cosine similarity to all others
        sims = embeddings[qi] @ embeddings.T
        sims[qi] = -1  # exclude self
        nn_idx = np.argsort(-sims)[:k]

        for ni in nn_idx:
            d = haversine_km(lats[qi], lons[qi], lats[ni], lons[ni])
            distances.append(d)

    distances = np.array(distances)
    print(f"\nGeo-NN Evaluation ({n_queries} queries, k={k}):")
    print(f"  Median NN distance:  {np.median(distances):.0f} km")
    print(f"  Mean NN distance:    {np.mean(distances):.0f} km")
    print(f"  25th percentile:     {np.percentile(distances, 25):.0f} km")
    print(f"  75th percentile:     {np.percentile(distances, 75):.0f} km")
    print(f"  <100km:              {(distances < 100).mean()*100:.1f}%")
    print(f"  <500km:              {(distances < 500).mean()*100:.1f}%")
    print(f"  <1000km:             {(distances < 1000).mean()*100:.1f}%")
    model.train()
    return np.median(distances)


# ── Main ──────────────────────────────────────────────────────────────

def main():
    if sys.platform == "win32":
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    parser = argparse.ArgumentParser(description="Phase 1.5: Contrastive pretraining")
    parser.add_argument("--metadata", default="GEOPROJECT/data/osv5m_50k/metadata.csv")
    parser.add_argument("--images", default="GEOPROJECT/data/osv5m_50k/images")
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=6)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--accumulation", type=int, default=10)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--checkpoint-dir", default="GEOPROJECT/checkpoints")
    parser.add_argument("--resume", type=str, default=None, help="Path to checkpoint to resume from")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"  GPU: {torch.cuda.get_device_name()}")
        print(f"  VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    print()

    # Load data
    print("Loading metadata...")
    df = pd.read_csv(args.metadata, dtype={"id": str})
    print(f"  {len(df):,} images")

    # Create dataset
    print("Creating contrastive dataset...")
    dataset = ContrastiveDataset(df, args.images)
    print(f"  Sample caption: {dataset.captions[0]}")
    print(f"  Sample caption: {dataset.captions[100]}")
    print()

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
    )

    # Create model
    model, tokenizer = create_contrastive_model(device)

    if args.resume:
        load_dora_checkpoint(model, args.resume)

    # Evaluate BEFORE training (baseline)
    print("\n--- Baseline (before contrastive training) ---")
    evaluate_geo_nn(model, dataset, df, device)

    # Train
    print(f"\n{'='*60}")
    print(f"Starting contrastive pretraining")
    print(f"  Epochs: {args.epochs}")
    print(f"  Batch size: {args.batch_size} x {args.accumulation} = {args.batch_size * args.accumulation} effective")
    print(f"  Learning rate: {args.lr}")
    print(f"  Total batches/epoch: {len(loader)}")
    print(f"{'='*60}\n")

    model = train_contrastive(
        model, tokenizer, loader, device,
        epochs=args.epochs, lr=args.lr,
        accumulation_steps=args.accumulation,
        checkpoint_dir=args.checkpoint_dir,
    )

    # Evaluate AFTER training
    print("\n--- After contrastive training ---")
    evaluate_geo_nn(model, dataset, df, device)

    print("\nPhase 1.5 complete!")


if __name__ == "__main__":
    main()
