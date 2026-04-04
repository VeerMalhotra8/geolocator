"""
Evaluate GeoLocator on external benchmarks (Im2GPS3k, YFCC4k).

No data leakage possible — these images were never in training.

Usage:
    # Coarse-only (top-1 cell centroid):
    python -m GEOPROJECT.geoguessr.eval_benchmark \
        --benchmark im2gps3k \
        --checkpoint GEOPROJECT/checkpoints/geolocator_best.pt \
        --geocell-dir GEOPROJECT/data/osv5m_50k/semantic_cells

    # With Test-Time Augmentation (3 views: center, flip, zoom-out):
    python -m GEOPROJECT.geoguessr.eval_benchmark \
        --benchmark im2gps3k --tta

    # With per-continent error analysis:
    python -m GEOPROJECT.geoguessr.eval_benchmark \
        --benchmark im2gps3k --tta --error-analysis

    # Two-stage with dense FAISS (4-tier metrics):
    python -m GEOPROJECT.geoguessr.eval_benchmark \
        --benchmark im2gps3k \
        --two-stage \
        --faiss-dir GEOPROJECT/faiss_index
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from transformers import CLIPProcessor

from GEOPROJECT.geoguessr.data.dataset import CLIP_MEAN, CLIP_STD, CLIP_SIZE
from GEOPROJECT.geoguessr.data.geocells import haversine_km
from GEOPROJECT.geoguessr.model.geolocator import GeoLocator


class BenchmarkDataset(Dataset):
    """Simple dataset for benchmark evaluation — images + GPS labels."""

    def __init__(self, image_dir: str, labels_csv: str, processor):
        self.image_dir = Path(image_dir)
        self.processor = processor

        df = pd.read_csv(labels_csv)
        self.filenames = df["filename"].tolist()
        self.latitudes = df["latitude"].values
        self.longitudes = df["longitude"].values

        # Filter to only existing images
        valid = []
        for i, fn in enumerate(self.filenames):
            if (self.image_dir / fn).exists():
                valid.append(i)
        if len(valid) < len(self.filenames):
            print(f"  Warning: {len(self.filenames) - len(valid)} images not found, using {len(valid)}")
        self.valid_indices = valid

    def __len__(self):
        return len(self.valid_indices)

    def __getitem__(self, idx):
        real_idx = self.valid_indices[idx]
        img_path = self.image_dir / self.filenames[real_idx]
        image = Image.open(img_path).convert("RGB")
        pixel_values = self.processor(images=image, return_tensors="pt")["pixel_values"].squeeze(0)
        lat = self.latitudes[real_idx]
        lon = self.longitudes[real_idx]
        return pixel_values, lat, lon


# ── TTA transforms ───────────────────────────────────────────────────────


def get_tta_transforms():
    """Return 3 deterministic TTA transforms for geolocation evaluation.

    View 0: Standard center crop (same as val transform — baseline).
    View 1: Horizontal flip (scene structure preserved, text mirrored).
    View 2: Zoomed-out center crop (resize to 448 then crop 336 — 25% more context).
    """
    return [
        transforms.Compose([
            transforms.Resize(CLIP_SIZE, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.CenterCrop(CLIP_SIZE),
            transforms.ToTensor(),
            transforms.Normalize(mean=CLIP_MEAN, std=CLIP_STD),
        ]),
        transforms.Compose([
            transforms.Resize(CLIP_SIZE, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.CenterCrop(CLIP_SIZE),
            transforms.RandomHorizontalFlip(p=1.0),
            transforms.ToTensor(),
            transforms.Normalize(mean=CLIP_MEAN, std=CLIP_STD),
        ]),
        transforms.Compose([
            transforms.Resize(448, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.CenterCrop(CLIP_SIZE),
            transforms.ToTensor(),
            transforms.Normalize(mean=CLIP_MEAN, std=CLIP_STD),
        ]),
    ]


# ── Continent assignment ─────────────────────────────────────────────────


def assign_continent(lat, lon):
    """Approximate continent from lat/lon. Good enough for error analysis."""
    if lat < -60:
        return "Antarctica"
    if lon < -30:
        return "North America" if lat > 12 else "South America"
    if lon < 60:
        if lat > 35:
            return "Europe"
        if lon > 50 and lat > 0:
            return "Asia"
        return "Africa"
    if lat < -10:
        return "Oceania"
    return "Asia"


# ── Error analysis ───────────────────────────────────────────────────────


def compute_error_analysis(true_lats, true_lons, distances):
    """Per-continent breakdown of prediction errors.

    Returns dict of {continent: {n, median_km, pct_25km, ...}}.
    """
    continents = [assign_continent(lat, lon) for lat, lon in zip(true_lats, true_lons)]
    distances = np.asarray(distances)

    continent_order = ["Europe", "Asia", "North America", "South America",
                       "Africa", "Oceania", "Antarctica", "Other"]
    stats = {}
    for cont in continent_order:
        mask = np.array([c == cont for c in continents])
        if mask.sum() == 0:
            continue
        d = distances[mask]
        stats[cont] = {
            "n": int(mask.sum()),
            "median_km": float(np.median(d)),
            "mean_km": float(np.mean(d)),
            "pct_25km": float((d < 25).mean() * 100),
            "pct_200km": float((d < 200).mean() * 100),
            "pct_750km": float((d < 750).mean() * 100),
            "pct_2500km": float((d < 2500).mean() * 100),
        }

    return stats, continents


def print_error_analysis(true_lats, true_lons, distances):
    """Print per-continent error breakdown and worst predictions."""
    stats, continents = compute_error_analysis(true_lats, true_lons, distances)
    distances = np.asarray(distances)

    print(f"\n{'='*75}")
    print("ERROR ANALYSIS BY CONTINENT")
    print(f"{'='*75}")
    print(f"{'Continent':<16} {'N':>5} {'Median km':>10} "
          f"{'<25km':>7} {'<200km':>7} {'<750km':>7} {'<2500km':>8}")
    print(f"{'-'*75}")

    for cont, s in stats.items():
        print(f"{cont:<16} {s['n']:>5} {s['median_km']:>10.1f} "
              f"{s['pct_25km']:>6.1f}% {s['pct_200km']:>6.1f}% "
              f"{s['pct_750km']:>6.1f}% {s['pct_2500km']:>6.1f}%")

    print(f"{'-'*75}")
    print(f"{'Overall':<16} {len(distances):>5} {float(np.median(distances)):>10.1f} "
          f"{float((distances < 25).mean() * 100):>6.1f}% "
          f"{float((distances < 200).mean() * 100):>6.1f}% "
          f"{float((distances < 750).mean() * 100):>6.1f}% "
          f"{float((distances < 2500).mean() * 100):>6.1f}%")
    print(f"{'='*75}")

    # Top 10 worst predictions
    worst_idx = np.argsort(distances)[-10:][::-1]
    print(f"\nTOP 10 WORST PREDICTIONS:")
    print(f"{'Rank':<5} {'Distance':>10} {'True Lat':>9} {'True Lon':>9} {'Continent':<14}")
    print(f"{'-'*55}")
    for rank, idx in enumerate(worst_idx, 1):
        print(f"{rank:<5} {distances[idx]:>9.0f}km "
              f"{true_lats[idx]:>9.2f} {true_lons[idx]:>9.2f} {continents[idx]:<14}")

    return stats


# ── Model loading ────────────────────────────────────────────────────────


def load_model(checkpoint_path: str, num_cells: int, device: torch.device) -> GeoLocator:
    """Load trained GeoLocator for evaluation."""
    model = GeoLocator(num_cells=num_cells).to(device)
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)

    if "num_cells" in ckpt and ckpt["num_cells"] != num_cells:
        raise ValueError(
            f"Checkpoint num_cells ({ckpt['num_cells']}) != geocell config ({num_cells})"
        )

    missing, unexpected = model.load_state_dict(ckpt["model_state_dict"], strict=False)
    print(f"Loaded checkpoint: {checkpoint_path}")
    if missing:
        print(f"  Missing keys: {len(missing)}")
    if unexpected:
        print(f"  Unexpected keys: {len(unexpected)}")

    model.eval()
    return model


# ── Evaluation functions ─────────────────────────────────────────────────


@torch.no_grad()
def evaluate(
    model: GeoLocator,
    dataloader: DataLoader,
    centroids: np.ndarray,
    device: torch.device,
) -> dict:
    """Run coarse evaluation on benchmark dataset (top-1 + weighted coarse)."""
    from GEOPROJECT.geoguessr.model.faiss_refinement import predict_weighted_coarse

    top1_distances = []
    weighted_distances = []
    all_true_lats, all_true_lons = [], []
    all_pred_lats, all_pred_lons = [], []
    n_processed = 0
    t0 = time.time()

    for batch_idx, (images, lats, lons) in enumerate(dataloader):
        images = images.to(device)

        with torch.amp.autocast(device.type, dtype=torch.bfloat16, enabled=(device.type == "cuda")):
            outputs = model(images)

        geo_probs = torch.nn.functional.softmax(outputs["geo"].float(), dim=-1).cpu().numpy()

        for i in range(images.size(0)):
            true_lat = lats[i].item()
            true_lon = lons[i].item()

            # Top-1 coarse
            pred_cell = np.argmax(geo_probs[i])
            pred_lat, pred_lon = centroids[pred_cell]
            top1_distances.append(haversine_km(true_lat, true_lon, pred_lat, pred_lon))

            # Weighted coarse
            w_lat, w_lon, _ = predict_weighted_coarse(geo_probs[i], centroids, top_n=10)
            weighted_distances.append(haversine_km(true_lat, true_lon, w_lat, w_lon))

            all_true_lats.append(true_lat)
            all_true_lons.append(true_lon)
            all_pred_lats.append(float(pred_lat))
            all_pred_lons.append(float(pred_lon))

        n_processed += images.size(0)
        if (batch_idx + 1) % 50 == 0:
            elapsed = time.time() - t0
            print(f"  {n_processed}/{len(dataloader.dataset)} images, "
                  f"{elapsed:.0f}s elapsed, "
                  f"{n_processed/elapsed:.1f} img/s")

    top1_arr = np.array(top1_distances)
    weighted_arr = np.array(weighted_distances)
    elapsed = time.time() - t0

    metrics = {
        "num_images": len(top1_arr),
        "median_km": float(np.median(top1_arr)),
        "mean_km": float(np.mean(top1_arr)),
        "weighted_median_km": float(np.median(weighted_arr)),
        "weighted_mean_km": float(np.mean(weighted_arr)),
        "pct_1km": float((top1_arr < 1).mean() * 100),
        "pct_25km": float((top1_arr < 25).mean() * 100),
        "pct_200km": float((top1_arr < 200).mean() * 100),
        "pct_750km": float((top1_arr < 750).mean() * 100),
        "pct_2500km": float((top1_arr < 2500).mean() * 100),
        "weighted_pct_25km": float((weighted_arr < 25).mean() * 100),
        "weighted_pct_200km": float((weighted_arr < 200).mean() * 100),
        "weighted_pct_750km": float((weighted_arr < 750).mean() * 100),
        "time_seconds": elapsed,
        "images_per_second": len(top1_arr) / elapsed,
    }

    # Per-image data for error analysis (excluded from JSON save)
    metrics["_per_image"] = {
        "true_lats": all_true_lats,
        "true_lons": all_true_lons,
        "pred_lats": all_pred_lats,
        "pred_lons": all_pred_lons,
        "distances": top1_distances,
    }

    return metrics


@torch.no_grad()
def evaluate_tta(
    model: GeoLocator,
    dataset: BenchmarkDataset,
    centroids: np.ndarray,
    device: torch.device,
) -> dict:
    """TTA evaluation: average softmax over 3 augmented views per image.

    Processes one image at a time, batching the 3 TTA views together.
    Returns both top-1 baseline and TTA metrics for direct comparison.
    """
    tta_xforms = get_tta_transforms()
    n_views = len(tta_xforms)

    top1_dists = []
    tta_dists = []
    true_lats, true_lons = [], []
    tta_pred_lats, tta_pred_lons = [], []
    n_changed = 0
    t0 = time.time()

    n_skipped = 0
    for idx in range(len(dataset)):
        real_idx = dataset.valid_indices[idx]
        img_path = dataset.image_dir / dataset.filenames[real_idx]
        try:
            image = Image.open(img_path).convert("RGB")
        except (OSError, IOError) as e:
            print(f"  Warning: skipping {img_path.name} ({e})")
            n_skipped += 1
            continue
        t_lat = float(dataset.latitudes[real_idx])
        t_lon = float(dataset.longitudes[real_idx])

        # Apply all TTA transforms and stack into a batch
        views = torch.stack([xf(image) for xf in tta_xforms]).to(device)

        with torch.amp.autocast(device.type, dtype=torch.bfloat16, enabled=(device.type == "cuda")):
            outputs = model(views)

        geo_probs = torch.nn.functional.softmax(outputs["geo"].float(), dim=-1)

        # Baseline: first view only (center crop, matches standard eval)
        base_cell = geo_probs[0].argmax().item()
        base_lat, base_lon = centroids[base_cell]

        # TTA: average probabilities across views, then argmax
        avg_probs = geo_probs.mean(dim=0)
        tta_cell = avg_probs.argmax().item()
        tta_lat, tta_lon = centroids[tta_cell]

        top1_dists.append(haversine_km(t_lat, t_lon, float(base_lat), float(base_lon)))
        tta_dists.append(haversine_km(t_lat, t_lon, float(tta_lat), float(tta_lon)))
        true_lats.append(t_lat)
        true_lons.append(t_lon)
        tta_pred_lats.append(float(tta_lat))
        tta_pred_lons.append(float(tta_lon))

        if tta_cell != base_cell:
            n_changed += 1

        if (idx + 1) % 500 == 0:
            elapsed = time.time() - t0
            rate = (idx + 1) / elapsed
            print(f"  {idx+1}/{len(dataset)} images, {elapsed:.0f}s, {rate:.1f} img/s")

    top1_arr = np.array(top1_dists)
    tta_arr = np.array(tta_dists)
    elapsed = time.time() - t0

    if n_skipped:
        print(f"  Skipped {n_skipped} images due to load errors")

    if len(top1_arr) == 0:
        return {"num_images": 0, "error": "no valid images processed"}

    def _pct(arr, prefix):
        return {
            f"{prefix}_median_km": float(np.median(arr)),
            f"{prefix}_mean_km": float(np.mean(arr)),
            f"{prefix}_pct_25km": float((arr < 25).mean() * 100),
            f"{prefix}_pct_200km": float((arr < 200).mean() * 100),
            f"{prefix}_pct_750km": float((arr < 750).mean() * 100),
            f"{prefix}_pct_2500km": float((arr < 2500).mean() * 100),
        }

    metrics = {
        "mode": "tta",
        "num_images": len(top1_arr),
        "num_views": n_views,
        "n_skipped": n_skipped,
        "n_predictions_changed": n_changed,
        "pct_predictions_changed": float(n_changed / len(top1_arr) * 100),
        **_pct(top1_arr, "top1"),
        **_pct(tta_arr, "tta"),
        "time_seconds": elapsed,
        "images_per_second": len(top1_arr) / elapsed,
    }

    # Per-image data for error analysis
    metrics["_per_image"] = {
        "true_lats": true_lats,
        "true_lons": true_lons,
        "pred_lats": tta_pred_lats,
        "pred_lons": tta_pred_lons,
        "distances": tta_dists,
        "top1_distances": top1_dists,
    }

    return metrics


# ── Main ─────────────────────────────────────────────────────────────────


def main():
    if sys.platform == "win32":
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    parser = argparse.ArgumentParser(description="Evaluate on external benchmark")
    parser.add_argument("--benchmark", type=str, required=True,
                        choices=["im2gps3k", "yfcc4k"],
                        help="Which benchmark to evaluate on")
    parser.add_argument("--checkpoint", type=str,
                        default="GEOPROJECT/checkpoints/geolocator_best.pt")
    parser.add_argument("--geocell-dir", type=str,
                        default="GEOPROJECT/data/osv5m_50k/semantic_cells")
    parser.add_argument("--two-stage", action="store_true",
                        help="Enable two-stage evaluation with dense FAISS (4-tier metrics)")
    parser.add_argument("--faiss-dir", type=str, default="GEOPROJECT/faiss_index",
                        help="Directory containing dense FAISS index (used with --two-stage)")
    parser.add_argument("--tta", action="store_true",
                        help="Enable Test-Time Augmentation (3 views: center, flip, zoom-out)")
    parser.add_argument("--error-analysis", action="store_true",
                        help="Run per-continent error breakdown")
    parser.add_argument("--batch-size", type=int, default=6)
    parser.add_argument("--num-workers", type=int, default=2)
    args = parser.parse_args()

    if args.tta and args.two_stage:
        parser.error("--tta and --two-stage are mutually exclusive")
    if args.error_analysis and args.two_stage:
        print("Warning: --error-analysis is not supported with --two-stage, ignoring.")
        args.error_analysis = False

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Set benchmark paths
    if args.benchmark == "im2gps3k":
        image_dir = "GEOPROJECT/data/im2gps3k/im2gps3ktest"
        labels_csv = "GEOPROJECT/data/im2gps3k/im2gps3k_labels.csv"
    elif args.benchmark == "yfcc4k":
        image_dir = "GEOPROJECT/data/yfcc4k/images"
        labels_csv = "GEOPROJECT/data/yfcc4k/yfcc4k_labels.csv"

    # Load geocell config
    geocell_path = Path(args.geocell_dir)
    with open(geocell_path / "geocell_config.json") as f:
        geocell_config = json.load(f)
    num_cells = geocell_config["num_cells"]
    centroids = np.load(geocell_path / "cell_centroids.npy")
    print(f"Geocells: {num_cells} cells")

    # Load model
    model = load_model(args.checkpoint, num_cells, device)

    # Load benchmark dataset
    processor = CLIPProcessor.from_pretrained("geolocal/StreetCLIP")
    dataset = BenchmarkDataset(image_dir, labels_csv, processor)
    print(f"Benchmark: {args.benchmark} -- {len(dataset)} images")

    if args.two_stage:
        # Two-stage dense FAISS evaluation (4-tier)
        from GEOPROJECT.geoguessr.model.faiss_refinement import (
            evaluate_two_stage_v2,
            load_dense_faiss_index,
        )

        dataloader = DataLoader(
            dataset, batch_size=args.batch_size, shuffle=False,
            num_workers=args.num_workers, pin_memory=True,
        )

        print(f"\nLoading dense FAISS index from {args.faiss_dir}...")
        dense_data = load_dense_faiss_index(args.faiss_dir)

        print(f"Running two-stage evaluation...")
        t0 = time.time()
        metrics = evaluate_two_stage_v2(
            model, dataloader, dense_data, centroids, device,
            gps_source="batch",
        )
        elapsed = time.time() - t0

        # Print 4-tier results
        print(f"\n{'='*80}")
        print(f"BENCHMARK: {args.benchmark.upper()} (TWO-STAGE DENSE)")
        print(f"{'='*80}")
        print(f"Images: {metrics['num_images']}  Time: {elapsed:.0f}s")
        print(f"{'Metric':<15} {'Top-1':>12} {'Weighted':>12} {'k-NN Raw':>12} {'Gated':>12}")
        print(f"{'-'*80}")

        for label, key in [("Median km", "median_km"), ("<25km", "pct_25km"),
                            ("<200km", "pct_200km"), ("<750km", "pct_750km"), ("<2500km", "pct_2500km")]:
            t1 = metrics[f"top1_{key}"]
            wt = metrics[f"weighted_{key}"]
            kn = metrics[f"knn_raw_{key}"]
            gt = metrics[f"gated_{key}"]
            if "pct" in key:
                print(f"{label:<15} {t1:>11.1f}% {wt:>11.1f}% {kn:>11.1f}% {gt:>11.1f}%")
            else:
                print(f"{label:<15} {t1:>12.1f} {wt:>12.1f} {kn:>12.1f} {gt:>12.1f}")

        print(f"{'-'*80}")
        print(f"k-NN gate rate: {metrics['knn_gate_rate']:.1f}%")
        print(f"{'='*80}")

        metrics["time_seconds"] = elapsed

    elif args.tta:
        # TTA evaluation (3 views per image)
        print(f"\nRunning TTA evaluation (3 views per image)...")
        metrics = evaluate_tta(model, dataset, centroids, device)

        print(f"\n{'='*65}")
        print(f"BENCHMARK: {args.benchmark.upper()} (TTA, {metrics['num_views']} views)")
        print(f"{'='*65}")
        print(f"Images: {metrics['num_images']}  "
              f"Time: {metrics['time_seconds']:.0f}s ({metrics['images_per_second']:.1f} img/s)")
        print(f"Predictions changed by TTA: {metrics['n_predictions_changed']} "
              f"({metrics['pct_predictions_changed']:.1f}%)")
        print(f"")
        print(f"  {'Metric':<20} {'Top-1':>12} {'TTA':>12} {'Delta':>10}")
        print(f"  {'-'*54}")

        for label, key in [("Median km", "median_km"), ("<25km", "pct_25km"),
                            ("<200km", "pct_200km"), ("<750km", "pct_750km"),
                            ("<2500km", "pct_2500km")]:
            t1 = metrics[f"top1_{key}"]
            tta = metrics[f"tta_{key}"]
            delta = tta - t1
            sign = "+" if delta >= 0 else ""
            if "pct" in key:
                better = delta > 0
                marker = " *" if abs(delta) > 0.5 else ""
                print(f"  {label:<20} {t1:>11.1f}% {tta:>11.1f}% {sign}{delta:>8.1f}pp{marker}")
            else:
                better = delta < 0
                marker = " *" if abs(delta) > 10 else ""
                print(f"  {label:<20} {t1:>12.1f} {tta:>12.1f} {sign}{delta:>9.1f}{marker}")

        print(f"  {'-'*54}")
        print(f"  (* = meaningful change)")
        print(f"{'='*65}")

        if args.error_analysis and "_per_image" in metrics:
            pi = metrics["_per_image"]
            continent_stats = print_error_analysis(pi["true_lats"], pi["true_lons"], pi["distances"])
            metrics["continent_stats"] = continent_stats

    else:
        # Coarse evaluation (top-1 + weighted coarse)
        dataloader = DataLoader(
            dataset, batch_size=args.batch_size, shuffle=False,
            num_workers=args.num_workers, pin_memory=True,
        )

        print(f"\nRunning evaluation...")
        metrics = evaluate(model, dataloader, centroids, device)

        print(f"\n{'='*60}")
        print(f"BENCHMARK: {args.benchmark.upper()}")
        print(f"{'='*60}")
        print(f"Images evaluated: {metrics['num_images']}")
        print(f"Time: {metrics['time_seconds']:.1f}s ({metrics['images_per_second']:.1f} img/s)")
        print(f"")
        print(f"  {'Metric':<20} {'Top-1':>12} {'Weighted':>12}")
        print(f"  {'-'*44}")
        print(f"  {'Median km':<20} {metrics['median_km']:>12.1f} {metrics['weighted_median_km']:>12.1f}")
        print(f"  {'<1 km (street)':<20} {metrics['pct_1km']:>11.1f}%")
        print(f"  {'<25 km (city)':<20} {metrics['pct_25km']:>11.1f}% {metrics['weighted_pct_25km']:>11.1f}%")
        print(f"  {'<200 km (region)':<20} {metrics['pct_200km']:>11.1f}% {metrics['weighted_pct_200km']:>11.1f}%")
        print(f"  {'<750 km (country)':<20} {metrics['pct_750km']:>11.1f}% {metrics['weighted_pct_750km']:>11.1f}%")
        print(f"  {'<2500 km (cont.)':<20} {metrics['pct_2500km']:>11.1f}%")
        print(f"{'='*60}")

        if args.error_analysis and "_per_image" in metrics:
            pi = metrics["_per_image"]
            continent_stats = print_error_analysis(pi["true_lats"], pi["true_lons"], pi["distances"])
            metrics["continent_stats"] = continent_stats

    # Save results — exclude _per_image (large, transient)
    save_metrics = {k: v for k, v in metrics.items() if not k.startswith("_")}
    suffix = "_tta" if args.tta else ("_v2" if args.two_stage else "")
    results_path = f"GEOPROJECT/checkpoints/{args.benchmark}{suffix}_results.json"
    with open(results_path, "w") as f:
        json.dump(save_metrics, f, indent=2)
    print(f"\nResults saved to {results_path}")


if __name__ == "__main__":
    main()
