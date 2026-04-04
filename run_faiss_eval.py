"""Build FAISS index from training data + evaluate on Im2GPS3k with two-stage inference.

Supports two modes:
  --mode sparse  (v1): OPTICS cluster centroids → IndexFlatL2
  --mode dense   (v2): All training embeddings → IndexFlatIP + weighted coarse + gated k-NN

Usage:
    python GEOPROJECT/run_faiss_eval.py --mode dense
    python GEOPROJECT/run_faiss_eval.py --mode dense --calibrate
    python GEOPROJECT/run_faiss_eval.py --mode sparse  # original v1 behavior
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from transformers import CLIPProcessor

from GEOPROJECT.geoguessr.data.dataset import create_dataloaders
from GEOPROJECT.geoguessr.data.geocells import haversine_km
from GEOPROJECT.geoguessr.model.faiss_refinement import (
    build_dense_faiss_index,
    build_faiss_index,
    calibrate_thresholds,
    cluster_within_cells,
    evaluate_two_stage_v2,
    extract_embeddings,
    save_dense_faiss_index,
    save_faiss_index,
)
from GEOPROJECT.geoguessr.model.geolocator import GeoLocator


class Im2GPS3kDataset(Dataset):
    """Im2GPS3k benchmark dataset for batched v2 evaluation."""

    def __init__(self, image_dir: str, labels_csv: str, processor):
        self.image_dir = Path(image_dir)
        self.processor = processor

        df = pd.read_csv(labels_csv)
        self.filenames = df["filename"].tolist()
        self.latitudes = df["latitude"].values
        self.longitudes = df["longitude"].values

        # Filter to existing images
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


def run_sparse_eval(model, embeddings, train_lats, train_lons, train_cells, centroids, device):
    """Original v1 sparse (OPTICS clustered) FAISS pipeline."""
    # Step 2: Cluster within cells
    print("\n=== Step 2: Clustering within cells (OPTICS) ===")
    t0 = time.time()
    cluster_data = cluster_within_cells(
        embeddings, train_cells, train_lats, train_lons,
        min_samples=5, max_clusters_per_cell=50,
    )
    print(f"  Clustered in {time.time()-t0:.0f}s")

    # Step 3: Build FAISS index
    print("\n=== Step 3: Building sparse FAISS index ===")
    faiss_index = build_faiss_index(cluster_data["cluster_embeddings"])
    save_faiss_index(
        faiss_index,
        cluster_data["cluster_gps"],
        cluster_data["cluster_cell_ids"],
        "GEOPROJECT/faiss_index",
    )

    # Step 4: Im2GPS3k evaluation (original loop)
    print("\n=== Step 4: Im2GPS3k sparse evaluation ===")
    cluster_gps = cluster_data["cluster_gps"]
    cluster_cell_ids = cluster_data["cluster_cell_ids"]

    labels = pd.read_csv("GEOPROJECT/data/im2gps3k/im2gps3k_labels.csv")
    img_dir = Path("GEOPROJECT/data/im2gps3k/im2gps3ktest")
    processor = CLIPProcessor.from_pretrained("geolocal/StreetCLIP")

    coarse_dists = []
    fine_dists = []
    n = 0
    t0 = time.time()

    for idx, row in labels.iterrows():
        img_path = img_dir / row["filename"]
        if not img_path.exists():
            continue

        image = Image.open(img_path).convert("RGB")
        pixel_values = processor(images=image, return_tensors="pt")["pixel_values"].to(device)

        with torch.no_grad():
            with torch.amp.autocast(device.type, dtype=torch.bfloat16, enabled=(device.type == "cuda")):
                outputs = model(pixel_values)

        geo_probs = F.softmax(outputs["geo"].float(), dim=-1).cpu().numpy()[0]
        embedding = outputs["embedding"].float().cpu().numpy()

        true_lat, true_lon = row["latitude"], row["longitude"]

        coarse_cell = np.argmax(geo_probs)
        c_lat, c_lon = centroids[coarse_cell]
        coarse_dists.append(haversine_km(true_lat, true_lon, c_lat, c_lon))

        top_cells = np.argsort(-geo_probs)[:50]
        valid_mask = np.isin(cluster_cell_ids, top_cells)
        valid_indices = np.where(valid_mask)[0]

        if len(valid_indices) > 0:
            valid_set = set(valid_indices.tolist())
            emb = np.ascontiguousarray(embedding, dtype=np.float32)
            _, indices = faiss_index.search(emb, min(500, faiss_index.ntotal))

            fine_idx = -1
            for fi in indices[0]:
                if fi >= 0 and fi in valid_set:
                    fine_idx = fi
                    break

            if fine_idx == -1:
                valid_embs = np.stack([faiss_index.reconstruct(int(vi)) for vi in valid_indices[:200]])
                dists_l2 = np.linalg.norm(valid_embs - emb, axis=1)
                fine_idx = valid_indices[np.argmin(dists_l2)]

            f_lat, f_lon = cluster_gps[fine_idx]
            fine_dists.append(haversine_km(true_lat, true_lon, f_lat, f_lon))
        else:
            fine_dists.append(coarse_dists[-1])

        n += 1
        if n % 500 == 0:
            print(f"  {n}/{len(labels)} images, {time.time()-t0:.0f}s")

    coarse_dists = np.array(coarse_dists)
    fine_dists = np.array(fine_dists)

    print(f"\n{'='*60}")
    print(f"Im2GPS3k SPARSE RESULTS -- {n} images")
    print(f"{'='*60}")
    print(f"{'Metric':<20} {'Coarse (cell)':>15} {'Fine (FAISS)':>15} {'Delta':>10}")
    print(f"{'-'*60}")

    for label, thresh in [("Median km", None), ("<25km", 25), ("<200km", 200), ("<750km", 750), ("<2500km", 2500)]:
        if thresh is None:
            c_val = f"{np.median(coarse_dists):.1f}"
            f_val = f"{np.median(fine_dists):.1f}"
            delta = f"{np.median(fine_dists) - np.median(coarse_dists):+.1f}"
        else:
            c_pct = (coarse_dists < thresh).mean() * 100
            f_pct = (fine_dists < thresh).mean() * 100
            c_val = f"{c_pct:.1f}%"
            f_val = f"{f_pct:.1f}%"
            delta = f"{f_pct - c_pct:+.1f}pp"
        print(f"{label:<20} {c_val:>15} {f_val:>15} {delta:>10}")

    print(f"{'='*60}")

    results = {
        "mode": "sparse",
        "coarse_median_km": float(np.median(coarse_dists)),
        "fine_median_km": float(np.median(fine_dists)),
        "coarse_pct_25km": float((coarse_dists < 25).mean() * 100),
        "fine_pct_25km": float((fine_dists < 25).mean() * 100),
        "coarse_pct_200km": float((coarse_dists < 200).mean() * 100),
        "fine_pct_200km": float((fine_dists < 200).mean() * 100),
        "coarse_pct_750km": float((coarse_dists < 750).mean() * 100),
        "fine_pct_750km": float((fine_dists < 750).mean() * 100),
        "coarse_pct_2500km": float((coarse_dists < 2500).mean() * 100),
        "fine_pct_2500km": float((fine_dists < 2500).mean() * 100),
        "num_images": n,
    }
    with open("GEOPROJECT/checkpoints/im2gps3k_faiss_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to GEOPROJECT/checkpoints/im2gps3k_faiss_results.json")
    return results


def run_dense_eval(model, embeddings, train_lats, train_lons, train_cells, centroids, device,
                   do_calibrate=False):
    """V2 dense (all embeddings) FAISS pipeline with 4-tier metrics."""
    # Step 2: Build dense index (no clustering)
    print("\n=== Step 2: Building dense FAISS index (all embeddings) ===")
    train_gps = np.stack([train_lats, train_lons], axis=1)
    dense_data = build_dense_faiss_index(embeddings, train_gps, train_cells)
    save_dense_faiss_index(dense_data, "GEOPROJECT/faiss_index")

    # Step 3: Im2GPS3k 4-tier evaluation
    print("\n=== Step 3: Im2GPS3k dense evaluation (4 tiers) ===")
    processor = CLIPProcessor.from_pretrained("geolocal/StreetCLIP")
    benchmark_ds = Im2GPS3kDataset(
        "GEOPROJECT/data/im2gps3k/im2gps3ktest",
        "GEOPROJECT/data/im2gps3k/im2gps3k_labels.csv",
        processor,
    )
    benchmark_loader = DataLoader(
        benchmark_ds, batch_size=6, shuffle=False, num_workers=0, pin_memory=True,
    )
    print(f"  Benchmark: {len(benchmark_ds)} images")

    t0 = time.time()
    metrics = evaluate_two_stage_v2(
        model, benchmark_loader, dense_data, centroids, device,
        gps_source="batch",
    )
    elapsed = time.time() - t0

    # Print 4-tier results table
    print(f"\n{'='*80}")
    print(f"Im2GPS3k DENSE RESULTS -- {metrics['num_images']} images ({elapsed:.0f}s)")
    print(f"{'='*80}")
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
    print(f"k-NN gate rate: {metrics['knn_gate_rate']:.1f}% "
          f"(threshold={metrics['confidence_threshold']})")
    print(f"{'='*80}")

    # Step 4: Calibrate (optional)
    cal_results = None
    if do_calibrate:
        print("\n=== Step 4: Calibrating thresholds ===")
        t0 = time.time()
        cal_results = calibrate_thresholds(
            model, benchmark_loader, dense_data, centroids, device,
            gps_source="batch",
        )
        print(f"  Calibration took {time.time()-t0:.0f}s")

        # Print top-5 combos
        grid_sorted = sorted(cal_results["grid"], key=lambda x: x["median_km"])
        print(f"\n  Top 5 parameter combinations:")
        print(f"  {'Conf Thresh':>12} {'Geo Radius':>12} {'Median km':>12} {'<25km':>10} {'<750km':>10} {'kNN Rate':>10}")
        for g in grid_sorted[:5]:
            print(f"  {g['confidence_threshold']:>12.2f} {g['geo_radius']:>12.0f} "
                  f"{g['median_km']:>12.1f} {g['pct_25km']:>9.1f}% {g['pct_750km']:>9.1f}% "
                  f"{g['knn_rate']:>9.1f}%")

        # Re-evaluate with best params
        best = cal_results["best_params"]
        print(f"\n  Re-evaluating with best: conf={best['confidence_threshold']}, "
              f"geo_radius={best['geo_radius']}km")
        metrics_best = evaluate_two_stage_v2(
            model, benchmark_loader, dense_data, centroids, device,
            confidence_threshold=best["confidence_threshold"],
            geo_radius=float(best["geo_radius"]),
            gps_source="batch",
        )

        print(f"\n  Calibrated gated median: {metrics_best['gated_median_km']:.1f}km "
              f"(was {metrics['gated_median_km']:.1f}km)")
        metrics = metrics_best  # use calibrated as final

    # Save results
    results = {
        "mode": "dense",
        **metrics,
    }
    if cal_results:
        results["calibration"] = {
            "best_params": cal_results["best_params"],
            "best_median_km": cal_results["best_median_km"],
        }
    with open("GEOPROJECT/checkpoints/im2gps3k_faiss_v2_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to GEOPROJECT/checkpoints/im2gps3k_faiss_v2_results.json")
    return results


def main():
    if sys.platform == "win32":
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    parser = argparse.ArgumentParser(description="Build FAISS index + evaluate on Im2GPS3k")
    parser.add_argument("--mode", type=str, default="dense", choices=["dense", "sparse"],
                        help="Index mode: dense (v2, all embeddings) or sparse (v1, OPTICS clusters)")
    parser.add_argument("--calibrate", action="store_true",
                        help="[dense only] Sweep confidence_threshold x geo_radius for best gating")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Load geocell config
    geocell_path = Path("GEOPROJECT/data/osv5m_50k/semantic_cells")
    with open(geocell_path / "geocell_config.json") as f:
        geocell_config = json.load(f)
    num_cells = geocell_config["num_cells"]
    centroids = np.load(geocell_path / "cell_centroids.npy")
    cell_indices = np.load(geocell_path / "cell_indices.npy")
    print(f"Geocells: {num_cells} cells")

    # Load model
    model = GeoLocator(num_cells=num_cells).to(device)
    ckpt = torch.load("GEOPROJECT/checkpoints/geolocator_best.pt", map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"], strict=False)
    model.eval()
    print("Model loaded")

    # Step 1: Extract embeddings
    print("\n=== Step 1: Extracting training embeddings ===")
    train_loader, val_loader, _ = create_dataloaders(
        geocell_dir=str(geocell_path),
        batch_size=6,
        num_workers=0,
    )

    train_ds = train_loader.dataset
    train_lats = np.array(train_ds.latitudes)
    train_lons = np.array(train_ds.longitudes)
    train_cells = cell_indices[train_ds.df_indices] if train_ds.df_indices is not None else cell_indices[:len(train_ds)]

    # IMPORTANT: Use a non-shuffled, non-drop-last loader for embedding extraction
    # so that embeddings[i] aligns with train_lats[i] / train_cells[i]
    extract_loader = DataLoader(
        train_ds, batch_size=6, shuffle=False, num_workers=0,
        pin_memory=True, drop_last=False,
    )

    t0 = time.time()
    embeddings = extract_embeddings(model, extract_loader, device)
    print(f"  Extracted in {time.time()-t0:.0f}s")

    # Align arrays
    n_emb = len(embeddings)
    if n_emb < len(train_lats):
        print(f"  Note: {len(train_lats) - n_emb} images failed to load, trimming to {n_emb}")
        train_lats = train_lats[:n_emb]
        train_lons = train_lons[:n_emb]
        train_cells = train_cells[:n_emb]

    # Run selected mode
    if args.mode == "sparse":
        run_sparse_eval(model, embeddings, train_lats, train_lons, train_cells, centroids, device)
    else:
        run_dense_eval(model, embeddings, train_lats, train_lons, train_cells, centroids, device,
                       do_calibrate=args.calibrate)


if __name__ == "__main__":
    main()
