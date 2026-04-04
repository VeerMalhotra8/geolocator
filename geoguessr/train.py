"""
Phase 2: Training loop for GeoLocator classification model.

Features:
  - bfloat16 autocast (RTX 4060, compute cap 8.9)
  - Gradient accumulation (batch 6 x accum 10 = effective 60)
  - KL-div geo loss + weighted CE auxiliary losses
  - Differential LR: DoRA=1e-4, heads=5e-4
  - Cosine scheduler with linear warmup
  - Checkpoint every N steps + end of epoch
  - Early stopping on validation median haversine distance
  - Validation with haversine distance metrics at multiple thresholds

Usage:
    python -m GEOPROJECT.geoguessr.train --epochs 10 --batch-size 6 --accumulation 10
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from GEOPROJECT.geoguessr.data.dataset import create_dataloaders
from GEOPROJECT.geoguessr.data.geocells import haversine_km
from GEOPROJECT.geoguessr.model.geolocator import GeoLocator


# ── Loss Function ────────────────────────────────────────────────────

def compute_loss(
    outputs: dict[str, torch.Tensor],
    smooth_targets: torch.Tensor,
    aux_labels: dict[str, torch.Tensor],
    geo_weight: float = 1.0,
    scene_weight: float = 0.3,
    climate_weight: float = 0.2,
    driving_weight: float = 0.2,
    region_weight: float = 0.3,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Compute total loss: KL-div for geo + weighted CE for auxiliaries.

    Returns:
        total_loss: scalar tensor
        loss_dict: dict of individual loss values (for logging)
    """
    # Geo loss: KL-divergence with haversine-smoothed soft targets
    # Upcast to float32 for numerical precision with ~9K sparse classes
    geo_log_probs = F.log_softmax(outputs["geo"].float(), dim=-1)
    geo_loss = F.kl_div(geo_log_probs, smooth_targets.float(), reduction="batchmean")

    # Auxiliary losses: cross-entropy with hard labels
    scene_loss = F.cross_entropy(outputs["scene"], aux_labels["scene"])
    climate_loss = F.cross_entropy(outputs["climate"], aux_labels["climate"])
    driving_loss = F.cross_entropy(outputs["driving"], aux_labels["drive_side"])
    region_loss = F.cross_entropy(outputs["region"], aux_labels["region"])

    total = (
        geo_weight * geo_loss
        + scene_weight * scene_loss
        + climate_weight * climate_loss
        + driving_weight * driving_loss
        + region_weight * region_loss
    )

    loss_dict = {
        "geo": geo_loss.item(),
        "scene": scene_loss.item(),
        "climate": climate_loss.item(),
        "driving": driving_loss.item(),
        "region": region_loss.item(),
        "total": total.item(),
    }
    return total, loss_dict


# ── Learning Rate Scheduler with Warmup ──────────────────────────────

class CosineWithWarmup(torch.optim.lr_scheduler._LRScheduler):
    """Cosine annealing with linear warmup."""

    def __init__(self, optimizer, warmup_steps: int, total_steps: int, last_epoch=-1):
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        step = self.last_epoch
        if step < self.warmup_steps:
            # Linear warmup
            scale = step / max(1, self.warmup_steps)
        else:
            # Cosine decay (clamp progress to [0,1] to prevent negative LR on resume)
            progress = min(1.0, (step - self.warmup_steps) / max(1, self.total_steps - self.warmup_steps))
            scale = 0.5 * (1 + np.cos(np.pi * progress))
        return [base_lr * scale for base_lr in self.base_lrs]


# ── Validation ───────────────────────────────────────────────────────

@torch.no_grad()
def validate(
    model: GeoLocator,
    val_loader: DataLoader,
    centroids: np.ndarray,
    device: torch.device,
) -> dict[str, float]:
    """Run validation: compute loss and haversine distance metrics.

    Predicts the centroid of the top-1 geocell for each image,
    then measures haversine distance to ground truth GPS.
    """
    model.eval()
    try:
        total_loss = 0.0
        total_samples = 0
        all_distances = []

        for images, smooth_targets, aux_labels in val_loader:
            images = images.to(device)
            smooth_targets = smooth_targets.to(device)
            aux_labels = {k: v.to(device) for k, v in aux_labels.items()}

            with torch.amp.autocast(device.type, dtype=torch.bfloat16, enabled=(device.type == "cuda")):
                outputs = model(images)
                loss, _ = compute_loss(outputs, smooth_targets, aux_labels)

            total_loss += loss.item() * images.size(0)
            total_samples += images.size(0)

            # Get predicted cell index (argmax of geo logits)
            pred_cells = outputs["geo"].argmax(dim=-1).cpu().numpy()

            # Get ground truth GPS from dataset
            batch_start = total_samples - images.size(0)
            ds = val_loader.dataset
            for i, pred_cell in enumerate(pred_cells):
                ds_idx = batch_start + i
                true_lat = ds.latitudes[ds_idx]
                true_lon = ds.longitudes[ds_idx]
                pred_lat, pred_lon = centroids[pred_cell]
                dist = haversine_km(true_lat, true_lon, pred_lat, pred_lon)
                all_distances.append(dist)

        distances = np.array(all_distances)
        metrics = {
            "val_loss": total_loss / total_samples,
            "median_km": float(np.median(distances)),
            "mean_km": float(np.mean(distances)),
            "pct_25km": float((distances < 25).mean() * 100),
            "pct_200km": float((distances < 200).mean() * 100),
            "pct_750km": float((distances < 750).mean() * 100),
            "pct_2500km": float((distances < 2500).mean() * 100),
        }
        return metrics
    finally:
        model.train()


# ── Checkpoint Save/Load ──────────────────────────────────────────────

def save_checkpoint(
    model: GeoLocator,
    optimizer: torch.optim.Optimizer,
    scheduler,
    epoch: int,
    global_step: int,
    metrics: dict,
    path: str,
):
    """Save full training checkpoint."""
    # Save only trainable parameters (DoRA + heads + unfrozen backbone) by checking requires_grad
    trainable_keys = {n for n, p in model.named_parameters() if p.requires_grad}
    trainable_state = {k: v for k, v in model.state_dict().items() if k in trainable_keys}
    torch.save({
        "epoch": epoch,
        "global_step": global_step,
        "model_state_dict": trainable_state,
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "metrics": metrics,
        "num_cells": model.num_cells,
        "unfreeze_layers": model.unfreeze_layers,
    }, path)


def load_checkpoint(model, optimizer, scheduler, path, device):
    """Load training checkpoint and resume."""
    ckpt = torch.load(path, map_location=device, weights_only=True)
    if "num_cells" in ckpt and ckpt["num_cells"] != model.num_cells:
        raise ValueError(
            f"Checkpoint num_cells ({ckpt['num_cells']}) != model num_cells ({model.num_cells}). "
            f"Cannot resume from a checkpoint trained with different geocells."
        )
    if "unfreeze_layers" in ckpt and ckpt["unfreeze_layers"] != model.unfreeze_layers:
        raise ValueError(
            f"Checkpoint unfreeze_layers ({ckpt['unfreeze_layers']}) != "
            f"model unfreeze_layers ({model.unfreeze_layers}). "
            f"Cannot resume with different backbone unfreeze config."
        )
    missing, unexpected = model.load_state_dict(ckpt["model_state_dict"], strict=False)
    if optimizer and "optimizer_state_dict" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
    if scheduler and "scheduler_state_dict" in ckpt:
        scheduler.load_state_dict(ckpt["scheduler_state_dict"])
    print(f"Resumed from {path} (epoch {ckpt['epoch']+1}, step {ckpt['global_step']})")
    return ckpt["epoch"], ckpt["global_step"]


# ── Training Loop ────────────────────────────────────────────────────

def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"  GPU: {torch.cuda.get_device_name()}")
        print(f"  VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    print()

    # Create data loaders
    print("Creating data loaders...")
    train_loader, val_loader, num_cells = create_dataloaders(
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        geocell_dir=args.geocell_dir,
    )
    print(f"  Num cells: {num_cells}")
    print(f"  Train batches: {len(train_loader)}, Val batches: {len(val_loader)}")
    print()

    # Load geocell centroids for distance evaluation (works for both S2 and semantic)
    centroids = np.load(Path(args.geocell_dir) / "cell_centroids.npy")

    # Create model
    print("Creating GeoLocator model...")
    model = GeoLocator(
        num_cells=num_cells,
        unfreeze_layers=args.unfreeze_layers,
    ).to(device)
    print(model.trainable_summary())

    # Load Phase 1.5 contrastive checkpoint
    contrastive_ckpt = Path(args.contrastive_checkpoint)
    if contrastive_ckpt.exists():
        model.load_contrastive_checkpoint(str(contrastive_ckpt))
    else:
        print(f"  WARNING: No contrastive checkpoint at {contrastive_ckpt}, training from scratch")
    print()

    # Optimizer with differential LR
    param_groups = model.get_parameter_groups(
        lr_dora=args.lr_dora, lr_heads=args.lr_heads, lr_backbone=args.lr_backbone,
    )
    optimizer = torch.optim.AdamW(param_groups, weight_decay=args.weight_decay)

    # Scheduler: cosine with warmup
    steps_per_epoch = len(train_loader) // args.accumulation
    total_steps = steps_per_epoch * args.epochs
    scheduler = CosineWithWarmup(optimizer, warmup_steps=args.warmup_steps, total_steps=total_steps)

    # Resume from checkpoint if provided
    # Note: resumes at the NEXT epoch — any partial epoch is skipped.
    # Use epoch checkpoints for clean resume; step checkpoints are for safety only.
    start_epoch = 0
    global_step = 0
    if args.resume:
        start_epoch, global_step = load_checkpoint(model, optimizer, scheduler, args.resume, device)
        start_epoch += 1  # resume from next epoch

    # Training state
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    best_median_km = float("inf")
    patience_counter = 0
    training_log = []

    print(f"{'='*60}")
    print(f"Training GeoLocator")
    print(f"  Epochs: {args.epochs}")
    print(f"  Batch: {args.batch_size} x {args.accumulation} = {args.batch_size * args.accumulation} effective")
    lr_info = f"  LR: DoRA={args.lr_dora}, Heads={args.lr_heads}"
    if args.unfreeze_layers > 0:
        lr_info += f", Backbone={args.lr_backbone}"
    print(lr_info)
    if args.unfreeze_layers > 0:
        print(f"  Unfreeze: last {args.unfreeze_layers} ViT layers")
    print(f"  Warmup: {args.warmup_steps} steps")
    print(f"  Total optimizer steps: {total_steps}")
    print(f"  Early stopping: patience={args.patience}")
    print(f"{'='*60}\n")

    model.train()
    for epoch in range(start_epoch, args.epochs):
        epoch_loss = 0.0
        epoch_steps = 0
        epoch_losses = {"geo": 0, "scene": 0, "climate": 0, "driving": 0, "region": 0}
        optimizer.zero_grad()
        t0 = time.time()

        for batch_idx, (images, smooth_targets, aux_labels) in enumerate(train_loader):
            images = images.to(device)
            smooth_targets = smooth_targets.to(device)
            aux_labels = {k: v.to(device) for k, v in aux_labels.items()}

            with torch.amp.autocast(device.type, dtype=torch.bfloat16, enabled=(device.type == "cuda")):
                outputs = model(images)
                loss, loss_dict = compute_loss(outputs, smooth_targets, aux_labels)
                loss = loss / args.accumulation

            loss.backward()

            if (batch_idx + 1) % args.accumulation == 0:
                torch.nn.utils.clip_grad_norm_(
                    [p for p in model.parameters() if p.requires_grad], max_norm=1.0
                )
                optimizer.step()
                optimizer.zero_grad()
                scheduler.step()
                global_step += 1

            epoch_loss += loss_dict["total"]
            epoch_steps += 1
            for k in epoch_losses:
                epoch_losses[k] += loss_dict[k]

            # Progress logging
            if (batch_idx + 1) % args.log_interval == 0:
                avg_loss = epoch_loss / epoch_steps
                elapsed = time.time() - t0
                rate = (batch_idx + 1) / elapsed
                eta = (len(train_loader) - batch_idx - 1) / rate
                lr_current = scheduler.get_last_lr()[0]
                print(
                    f"  Epoch {epoch+1} [{batch_idx+1}/{len(train_loader)}] "
                    f"loss={avg_loss:.4f} geo={loss_dict['geo']:.4f} "
                    f"lr={lr_current:.2e} {rate:.1f} b/s ETA:{eta/60:.1f}m"
                )

            # Step checkpoint
            if args.checkpoint_steps > 0 and global_step > 0 and global_step % args.checkpoint_steps == 0:
                ckpt_path = f"{args.checkpoint_dir}/step_{global_step}.pt"
                save_checkpoint(model, optimizer, scheduler, epoch, global_step, {}, ckpt_path)
                print(f"  [Checkpoint saved: {ckpt_path}]")

        # Flush leftover accumulated gradients at end of epoch
        if (batch_idx + 1) % args.accumulation != 0:
            torch.nn.utils.clip_grad_norm_(
                [p for p in model.parameters() if p.requires_grad], max_norm=1.0
            )
            optimizer.step()
            optimizer.zero_grad()
            scheduler.step()
            global_step += 1

        # End of epoch
        avg_loss = epoch_loss / epoch_steps
        avg_losses = {k: v / epoch_steps for k, v in epoch_losses.items()}
        elapsed = time.time() - t0
        print(f"\nEpoch {epoch+1}/{args.epochs} done in {elapsed/60:.1f}min")
        print(f"  Train loss: {avg_loss:.4f} (geo={avg_losses['geo']:.4f} scene={avg_losses['scene']:.4f} "
              f"climate={avg_losses['climate']:.4f} driving={avg_losses['driving']:.4f} region={avg_losses['region']:.4f})")

        # Validation
        print("  Validating...")
        metrics = validate(model, val_loader, centroids, device)
        print(f"  Val loss: {metrics['val_loss']:.4f}")
        print(f"  Median distance: {metrics['median_km']:.0f} km")
        print(f"  Mean distance: {metrics['mean_km']:.0f} km")
        print(f"  <25km: {metrics['pct_25km']:.1f}% | <200km: {metrics['pct_200km']:.1f}% | "
              f"<750km: {metrics['pct_750km']:.1f}% | <2500km: {metrics['pct_2500km']:.1f}%")

        # Log
        log_entry = {
            "epoch": epoch + 1,
            "train_loss": avg_loss,
            "train_losses": avg_losses,
            **metrics,
            "lr": scheduler.get_last_lr()[0],
            "time_min": elapsed / 60,
        }
        training_log.append(log_entry)

        # Save training log
        with open(f"{args.checkpoint_dir}/training_log.json", "w") as f:
            json.dump(training_log, f, indent=2)

        # Save epoch checkpoint
        epoch_path = f"{args.checkpoint_dir}/geolocator_epoch{epoch+1}.pt"
        save_checkpoint(model, optimizer, scheduler, epoch, global_step, metrics, epoch_path)
        print(f"  Saved: {epoch_path}")

        # Best model check
        if metrics["median_km"] < best_median_km:
            best_median_km = metrics["median_km"]
            best_path = f"{args.checkpoint_dir}/geolocator_best.pt"
            save_checkpoint(model, optimizer, scheduler, epoch, global_step, metrics, best_path)
            print(f"  New best! median={best_median_km:.0f}km → {best_path}")
            patience_counter = 0
        else:
            patience_counter += 1
            print(f"  No improvement (patience {patience_counter}/{args.patience})")

        # Early stopping
        if patience_counter >= args.patience:
            print(f"\nEarly stopping triggered after {epoch+1} epochs")
            break

        print()

    print(f"\n{'='*60}")
    print(f"Training complete!")
    print(f"  Best median distance: {best_median_km:.0f} km")
    print(f"  Best checkpoint: {args.checkpoint_dir}/geolocator_best.pt")
    print(f"{'='*60}")

    return training_log


# ── Main ─────────────────────────────────────────────────────────────

def main():
    if sys.platform == "win32":
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    parser = argparse.ArgumentParser(description="Phase 2: Train GeoLocator")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=6)
    parser.add_argument("--accumulation", type=int, default=10)
    parser.add_argument("--lr-dora", type=float, default=1e-4)
    parser.add_argument("--lr-heads", type=float, default=5e-4)
    parser.add_argument("--lr-backbone", type=float, default=2e-5,
                        help="LR for unfrozen backbone layers (only used with --unfreeze-layers)")
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--warmup-steps", type=int, default=1000)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument("--unfreeze-layers", type=int, default=0,
                        help="Fully unfreeze last N ViT encoder layers (0=DoRA-only, 4=recommended for A100)")
    parser.add_argument("--log-interval", type=int, default=100)
    parser.add_argument("--checkpoint-steps", type=int, default=1000)
    parser.add_argument("--checkpoint-dir", default="GEOPROJECT/checkpoints")
    parser.add_argument("--contrastive-checkpoint", default="GEOPROJECT/checkpoints/contrastive_best.pt")
    parser.add_argument("--geocell-dir", type=str, default="GEOPROJECT/data/osv5m_50k",
                        help="Directory with geocell config (S2 or semantic)")
    parser.add_argument("--resume", type=str, default=None)
    args = parser.parse_args()

    train(args)


if __name__ == "__main__":
    main()
