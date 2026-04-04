"""
Geocell partitioning and haversine label smoothing.

Phase 1: S2 cells (level 8, ~9K cells for 50K images).
Phase 3 will replace with semantic geocells (OPTICS + Voronoi).

Usage:
    # Build geocells from training data:
    python -m geoguessr.data.geocells --metadata data/osv5m_50k/metadata.csv --output data/osv5m_50k

    # Then use in training via updated dataset.py
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import s2sphere

EARTH_RADIUS_KM = 6371.0
DEFAULT_S2_LEVEL = 8
DEFAULT_TAU_KM = 200.0
DEFAULT_TOP_K = 200


# ── S2 Cell Partitioning ──────────────────────────────────────────────

def latlon_to_cell_id(lat: float, lon: float, level: int) -> int:
    """Convert (lat, lon) in degrees to S2 cell ID at given level."""
    ll = s2sphere.LatLng.from_degrees(lat, lon)
    return s2sphere.CellId.from_lat_lng(ll).parent(level).id()


def cell_id_to_centroid(cell_id: int) -> tuple[float, float]:
    """Get (lat, lon) centroid of an S2 cell in degrees."""
    cell = s2sphere.Cell(s2sphere.CellId(cell_id))
    center = cell.get_center()
    ll = s2sphere.LatLng.from_point(center)
    return ll.lat().degrees, ll.lng().degrees


def build_s2_cells(
    latitudes: np.ndarray,
    longitudes: np.ndarray,
    level: int = DEFAULT_S2_LEVEL,
) -> tuple[np.ndarray, np.ndarray, dict[int, int]]:
    """Assign GPS coordinates to S2 cells and compute centroids.

    Args:
        latitudes: (N,) array of latitudes in degrees
        longitudes: (N,) array of longitudes in degrees
        level: S2 cell level (8 gives ~9K cells for 50K images)

    Returns:
        cell_indices: (N,) array of integer class labels per image
        centroids: (num_cells, 2) array of cell centroids [lat, lon] in degrees
        cell_id_to_idx: mapping from S2 cell ID to integer index
    """
    n = len(latitudes)
    # S2 cell IDs can be unsigned 64-bit, store as Python ints in a list
    raw_cell_ids = []
    for i in range(n):
        raw_cell_ids.append(latlon_to_cell_id(latitudes[i], longitudes[i], level))

    # Build mapping: unique S2 cell ID → contiguous integer index
    unique_ids = sorted(set(raw_cell_ids))
    cell_id_to_idx = {cid: idx for idx, cid in enumerate(unique_ids)}
    num_cells = len(unique_ids)

    # Assign each image its cell index
    cell_indices = np.array([cell_id_to_idx[cid] for cid in raw_cell_ids], dtype=np.int64)

    # Compute centroids
    centroids = np.empty((num_cells, 2), dtype=np.float64)
    for idx, cid in enumerate(unique_ids):
        centroids[idx] = cell_id_to_centroid(int(cid))

    return cell_indices, centroids, cell_id_to_idx


# ── Haversine Distance ────────────────────────────────────────────────

def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Haversine distance between two points in km. Inputs in degrees."""
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    return 2 * EARTH_RADIUS_KM * np.arcsin(np.sqrt(np.clip(a, 0, 1)))


def haversine_to_all_centroids(
    lat: float, lon: float, centroids: np.ndarray,
) -> np.ndarray:
    """Compute haversine distance from (lat, lon) to all centroids.

    Args:
        lat, lon: query point in degrees
        centroids: (num_cells, 2) array of [lat, lon] in degrees

    Returns:
        distances: (num_cells,) array of distances in km
    """
    lat1 = np.radians(lat)
    lon1 = np.radians(lon)
    lat2 = np.radians(centroids[:, 0])
    lon2 = np.radians(centroids[:, 1])

    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    return 2 * EARTH_RADIUS_KM * np.arcsin(np.sqrt(np.clip(a, 0, 1)))


# ── Haversine Label Smoothing (Sparse Precomputation) ─────────────────

def precompute_smooth_targets(
    latitudes: np.ndarray,
    longitudes: np.ndarray,
    centroids: np.ndarray,
    tau_km: float = DEFAULT_TAU_KM,
    top_k: int = DEFAULT_TOP_K,
    output_dir: str = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Precompute sparse haversine-smoothed targets for all images.

    For each image, stores the top-K nearest cell indices and their
    softmax probabilities. All other cells get probability ~ 0.

    Args:
        latitudes: (N,) latitudes in degrees
        longitudes: (N,) longitudes in degrees
        centroids: (num_cells, 2) cell centroids [lat, lon] in degrees
        tau_km: temperature for smoothing (default: 200 km)
        top_k: number of nearest cells to store (default: 200)
        output_dir: if set, save as .npy files

    Returns:
        top_k_indices: (N, top_k) int32 array of nearest cell indices
        top_k_probs: (N, top_k) float32 array of normalized probabilities
    """
    n = len(latitudes)
    num_cells = len(centroids)
    k = min(top_k, num_cells)

    top_k_indices = np.empty((n, k), dtype=np.int32)
    top_k_probs = np.empty((n, k), dtype=np.float32)

    report_interval = max(1, n // 20)
    t0 = time.time()

    for i in range(n):
        dists = haversine_to_all_centroids(latitudes[i], longitudes[i], centroids)

        # Find top-K nearest cells
        if k < num_cells:
            idx = np.argpartition(dists, k)[:k]
        else:
            idx = np.arange(num_cells)
        idx = idx[np.argsort(dists[idx])]  # sort by distance

        # Compute softmax probabilities: p_i = exp(-d_i / tau) / sum
        top_dists = dists[idx]
        log_probs = -top_dists / tau_km
        log_probs -= log_probs.max()  # numerical stability
        probs = np.exp(log_probs)
        probs /= probs.sum()

        top_k_indices[i] = idx
        top_k_probs[i] = probs.astype(np.float32)

        if (i + 1) % report_interval == 0:
            elapsed = time.time() - t0
            rate = (i + 1) / elapsed
            eta = (n - i - 1) / rate
            print(f"  [{i+1:,}/{n:,}] {rate:.0f} img/s, ETA: {eta:.0f}s")

    if output_dir:
        out = Path(output_dir)
        np.save(out / "smooth_indices.npy", top_k_indices)
        np.save(out / "smooth_probs.npy", top_k_probs)
        print(f"  Saved to {out / 'smooth_indices.npy'} and {out / 'smooth_probs.npy'}")

    return top_k_indices, top_k_probs


# ── Save / Load Geocell Config ────────────────────────────────────────

def save_geocell_config(
    output_dir: str,
    centroids: np.ndarray,
    cell_id_to_idx: dict[int, int],
    level: int,
    tau_km: float,
    top_k: int,
):
    """Save geocell configuration for reproducibility."""
    out = Path(output_dir)
    np.save(out / "cell_centroids.npy", centroids)

    config = {
        "s2_level": level,
        "num_cells": len(centroids),
        "tau_km": tau_km,
        "top_k": top_k,
        "cell_id_to_idx": {str(k): v for k, v in cell_id_to_idx.items()},
    }
    with open(out / "geocell_config.json", "w") as f:
        json.dump(config, f, indent=2)

    print(f"  Saved geocell config: {len(centroids)} cells, level={level}")


def load_geocell_config(config_dir: str) -> dict:
    """Load geocell configuration."""
    p = Path(config_dir)
    centroids = np.load(p / "cell_centroids.npy")

    with open(p / "geocell_config.json") as f:
        config = json.load(f)

    # Restore int keys
    config["cell_id_to_idx"] = {int(k): v for k, v in config["cell_id_to_idx"].items()}
    config["centroids"] = centroids
    return config


def assign_cell_index(lat: float, lon: float, level: int, cell_id_to_idx: dict) -> int:
    """Assign a single (lat, lon) to its cell index. For inference/val."""
    cid = latlon_to_cell_id(lat, lon, level)
    return cell_id_to_idx.get(cid, -1)


# ── CLI Entry Point ───────────────────────────────────────────────────

def main():
    import sys
    if sys.platform == "win32":
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    parser = argparse.ArgumentParser(description="Build S2 geocells + haversine smoothing")
    parser.add_argument("--metadata", type=str, default="GEOPROJECT/data/osv5m_50k/metadata.csv")
    parser.add_argument("--output", type=str, default="GEOPROJECT/data/osv5m_50k")
    parser.add_argument("--level", type=int, default=DEFAULT_S2_LEVEL)
    parser.add_argument("--tau", type=float, default=DEFAULT_TAU_KM)
    parser.add_argument("--top-k", type=int, default=DEFAULT_TOP_K)
    args = parser.parse_args()

    import pandas as pd

    print("=" * 60)
    print("Building S2 Geocells + Haversine Label Smoothing")
    print(f"  S2 Level: {args.level}")
    print(f"  Tau: {args.tau} km")
    print(f"  Top-K: {args.top_k}")
    print("=" * 60)
    print()

    # Load metadata
    print("Loading metadata...")
    df = pd.read_csv(args.metadata, dtype={"id": str})
    lats = df["latitude"].values
    lons = df["longitude"].values
    print(f"  {len(df):,} images")
    print()

    # Build S2 cells
    print(f"Building S2 cells (level {args.level})...")
    t0 = time.time()
    cell_indices, centroids, cell_id_to_idx = build_s2_cells(lats, lons, args.level)
    print(f"  {len(centroids):,} unique cells in {time.time()-t0:.1f}s")

    # Save cell assignments per image
    np.save(Path(args.output) / "cell_indices.npy", cell_indices)
    print(f"  Saved cell_indices.npy")

    # Cell distribution stats
    unique, counts = np.unique(cell_indices, return_counts=True)
    print(f"  Images per cell: min={counts.min()}, median={int(np.median(counts))}, max={counts.max()}")
    print()

    # Save geocell config
    save_geocell_config(args.output, centroids, cell_id_to_idx, args.level, args.tau, args.top_k)
    print()

    # Precompute haversine-smoothed targets
    print(f"Precomputing haversine-smoothed targets (tau={args.tau}km, top-{args.top_k})...")
    t0 = time.time()
    top_k_indices, top_k_probs = precompute_smooth_targets(
        lats, lons, centroids, args.tau, args.top_k, args.output,
    )
    print(f"  Done in {time.time()-t0:.1f}s")
    print()

    # Verify smoothing
    print("Verification:")
    sample_idx = 0
    print(f"  Image 0: lat={lats[sample_idx]:.4f}, lon={lons[sample_idx]:.4f}")
    print(f"  Assigned cell: {cell_indices[sample_idx]}")
    print(f"  Top-5 nearest cells: {top_k_indices[sample_idx, :5]}")
    print(f"  Top-5 probabilities:  {top_k_probs[sample_idx, :5]}")
    print(f"  Probability sum: {top_k_probs[sample_idx].sum():.6f}")
    print(f"  Max prob cell == assigned cell: {top_k_indices[sample_idx, 0] == cell_indices[sample_idx]}")

    print()
    print("=" * 60)
    print("Done! Files created:")
    out = Path(args.output)
    for f in ["cell_indices.npy", "cell_centroids.npy", "geocell_config.json",
              "smooth_indices.npy", "smooth_probs.npy"]:
        size = (out / f).stat().st_size / 1024
        unit = "KB"
        if size > 1024:
            size /= 1024
            unit = "MB"
        print(f"  {f}: {size:.1f} {unit}")
    print("=" * 60)


if __name__ == "__main__":
    main()
