"""
Phase 3: Semantic geocells via OPTICS clustering + Voronoi tessellation.

Replaces uniform S2 cells with data-driven geocells that adapt to image density:
  - Dense areas (cities) → small, numerous cells
  - Sparse areas (rural) → large cells
  - OPTICS finds clusters of varying density from training GPS coordinates
  - Noise points assigned to nearest cluster centroid
  - Voronoi tessellation gives complete Earth coverage (nearest centroid assignment)

For 1M+ images, use --subsample to run OPTICS on a geographic subsample,
then Voronoi-assign all images to the discovered centroids.

Architecture doc reference: GEOGUESSR_ARCHITECTURE.md Section 2.2

Usage:
    # 50K (direct OPTICS on all points):
    python -m GEOPROJECT.geoguessr.data.semantic_geocells \
        --metadata GEOPROJECT/data/osv5m_50k/metadata.csv \
        --output GEOPROJECT/data/osv5m_50k/semantic_cells \
        --min-samples 20 --max-eps 0.0087

    # 1M (subsample OPTICS + Voronoi extend):
    python -m GEOPROJECT.geoguessr.data.semantic_geocells \
        --metadata GEOPROJECT/data/osv5m_1m/metadata.csv \
        --output GEOPROJECT/data/osv5m_1m/semantic_cells \
        --subsample 200000 --min-samples 50 --tau 100

    # Then retrain with: --geocell-dir <output>/semantic_cells
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
from sklearn.cluster import OPTICS

from GEOPROJECT.geoguessr.data.geocells import (
    EARTH_RADIUS_KM,
    DEFAULT_TAU_KM,
    DEFAULT_TOP_K,
    haversine_km,
    haversine_to_all_centroids,
    precompute_smooth_targets,
)


# ── Vectorized Voronoi Assignment ────────────────────────────────────

def voronoi_assign_batched(
    latitudes: np.ndarray,
    longitudes: np.ndarray,
    centroids: np.ndarray,
    batch_size: int = 5000,
) -> np.ndarray:
    """Assign each point to its nearest centroid using batched haversine.

    Fully vectorized: each batch computes distances to ALL centroids at once.
    Memory: batch_size * num_centroids * 8 bytes per batch.

    Args:
        latitudes: (N,) latitudes in degrees
        longitudes: (N,) longitudes in degrees
        centroids: (C, 2) array of [lat, lon] in degrees
        batch_size: points per batch (controls peak memory)

    Returns:
        assignments: (N,) array of nearest centroid indices
    """
    n = len(latitudes)
    num_centroids = len(centroids)
    assignments = np.empty(n, dtype=np.int64)

    # Precompute centroid radians
    c_lat_rad = np.radians(centroids[:, 0])  # (C,)
    c_lon_rad = np.radians(centroids[:, 1])  # (C,)

    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        batch_lat = np.radians(latitudes[start:end])  # (B,)
        batch_lon = np.radians(longitudes[start:end])  # (B,)

        # Broadcast: (B, 1) vs (1, C) → (B, C)
        dlat = c_lat_rad[np.newaxis, :] - batch_lat[:, np.newaxis]
        dlon = c_lon_rad[np.newaxis, :] - batch_lon[:, np.newaxis]

        a = (np.sin(dlat / 2) ** 2 +
             np.cos(batch_lat[:, np.newaxis]) * np.cos(c_lat_rad[np.newaxis, :]) *
             np.sin(dlon / 2) ** 2)
        # Don't need actual km — just argmin, so skip the 2*R*arcsin
        # arcsin is monotonic, so argmin(a) == argmin(haversine)
        assignments[start:end] = np.argmin(a, axis=1)

        if end % 50000 < batch_size and end < n:
            print(f"    Assigned {end:,}/{n:,} points...")

    return assignments


# ── OPTICS Semantic Geocells ─────────────────────────────────────────

def _run_optics(
    latitudes: np.ndarray,
    longitudes: np.ndarray,
    min_samples: int,
    max_eps: float,
    min_cluster_size: int,
) -> tuple[np.ndarray, int, int]:
    """Run OPTICS clustering and return labels + stats.

    Returns:
        raw_labels: (N,) cluster labels (-1 = noise)
        num_clusters: number of clusters found
        num_noise: number of noise points
    """
    n = len(latitudes)
    coords_rad = np.column_stack([
        np.radians(latitudes),
        np.radians(longitudes),
    ])

    print(f"  Running OPTICS (min_samples={min_samples}, max_eps={max_eps:.4f} rad, "
          f"n={n:,})...")
    t0 = time.time()
    optics = OPTICS(
        min_samples=min_samples,
        max_eps=max_eps,
        metric="haversine",
        cluster_method="xi",
        min_cluster_size=min_cluster_size,
        n_jobs=-1,
    )
    optics.fit(coords_rad)
    raw_labels = optics.labels_
    elapsed = time.time() - t0
    print(f"  OPTICS done in {elapsed:.1f}s")

    num_clusters = len(set(raw_labels)) - (1 if -1 in raw_labels else 0)
    num_noise = int((raw_labels == -1).sum())
    print(f"  Clusters: {num_clusters}, Noise points: {num_noise:,} ({100*num_noise/n:.1f}%)")

    if num_clusters == 0:
        raise ValueError(
            "OPTICS found 0 clusters. Try reducing min_samples or increasing max_eps. "
            f"Current: min_samples={min_samples}, max_eps={max_eps}"
        )

    return raw_labels, num_clusters, num_noise


def _compute_centroids(
    raw_labels: np.ndarray,
    latitudes: np.ndarray,
    longitudes: np.ndarray,
) -> np.ndarray:
    """Compute median centroid for each cluster label (excluding noise)."""
    unique_labels = sorted(set(raw_labels.tolist()) - {-1})
    centroids = np.empty((len(unique_labels), 2), dtype=np.float64)

    for new_idx, old_label in enumerate(unique_labels):
        mask = raw_labels == old_label
        centroids[new_idx, 0] = np.median(latitudes[mask])
        centroids[new_idx, 1] = np.median(longitudes[mask])

    return centroids


def build_semantic_geocells(
    latitudes: np.ndarray,
    longitudes: np.ndarray,
    min_samples: int = 20,
    max_eps: float = 0.0087,
    min_cluster_size: int = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Build semantic geocells using OPTICS on ALL points + vectorized Voronoi.

    Best for datasets up to ~200K points. For larger datasets, use
    build_semantic_geocells_subsampled() instead.

    Args:
        latitudes: (N,) latitudes in degrees
        longitudes: (N,) longitudes in degrees
        min_samples: OPTICS min_samples (min points to form a cluster core)
        max_eps: OPTICS max_eps in RADIANS (~0.0087 rad ≈ 0.5 degrees ≈ 50km)
        min_cluster_size: minimum cluster size for extraction (default: min_samples)

    Returns:
        cell_indices: (N,) array of integer cell labels per image (0-indexed)
        centroids: (num_cells, 2) array of cell centroids [lat, lon] in degrees
    """
    if min_cluster_size is None:
        min_cluster_size = min_samples

    n = len(latitudes)
    print(f"  Building semantic geocells from {n:,} GPS points...")

    # Step 1: OPTICS
    raw_labels, num_clusters, num_noise = _run_optics(
        latitudes, longitudes, min_samples, max_eps, min_cluster_size,
    )

    # Step 2: Compute centroids
    centroids = _compute_centroids(raw_labels, latitudes, longitudes)

    # Step 3: Voronoi-assign ALL points to nearest centroid (vectorized)
    print(f"  Voronoi-assigning all {n:,} points to {len(centroids)} centroids...")
    t0 = time.time()
    cell_indices = voronoi_assign_batched(latitudes, longitudes, centroids)
    print(f"  Voronoi assignment done in {time.time()-t0:.1f}s")

    num_cells = len(centroids)
    print(f"  Final: {num_cells} semantic geocells")

    # Verify balance
    unique, counts = np.unique(cell_indices, return_counts=True)
    print(f"  Images per cell: min={counts.min()}, median={int(np.median(counts))}, "
          f"max={counts.max()}, mean={counts.mean():.0f}")

    return cell_indices, centroids


def build_semantic_geocells_subsampled(
    latitudes: np.ndarray,
    longitudes: np.ndarray,
    subsample_n: int = 200_000,
    min_samples: int = 50,
    max_eps: float = 0.0087,
    min_cluster_size: int = None,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """Build semantic geocells by running OPTICS on a geographic subsample.

    Strategy for 1M+ datasets:
      1. Geographically stratified subsample (S2 level-4 cells for even coverage)
      2. Run OPTICS on subsample to discover cluster centroids
      3. Voronoi-assign ALL images to nearest centroid (vectorized, fast)

    This avoids OPTICS O(n²) memory/time on the full dataset.

    Args:
        latitudes: (N,) ALL latitudes in degrees
        longitudes: (N,) ALL longitudes in degrees
        subsample_n: number of points to subsample for OPTICS
        min_samples: OPTICS min_samples (higher for denser data)
        max_eps: OPTICS max_eps in RADIANS
        min_cluster_size: minimum cluster size (default: min_samples)
        seed: random seed for reproducibility

    Returns:
        cell_indices: (N,) array of cell labels for ALL images
        centroids: (num_cells, 2) array of cell centroids [lat, lon] in degrees
    """
    if min_cluster_size is None:
        min_cluster_size = min_samples

    n = len(latitudes)
    print(f"  Building semantic geocells: subsample {subsample_n:,} from {n:,} points...")

    # Step 1: Geographically stratified subsample using grid cells
    rng = np.random.RandomState(seed)

    if n <= subsample_n:
        print(f"  Dataset smaller than subsample — using all {n:,} points for OPTICS")
        sub_lats = latitudes
        sub_lons = longitudes
    else:
        print(f"  Creating geographically stratified subsample...")
        t0 = time.time()

        # Grid into 1-degree cells for stratified sampling
        grid_lat = np.floor(latitudes).astype(np.int32)
        grid_lon = np.floor(longitudes).astype(np.int32)
        cell_keys = grid_lat * 1000 + grid_lon

        unique_cells, cell_inverse = np.unique(cell_keys, return_inverse=True)
        num_grid_cells = len(unique_cells)

        # Target per grid cell (even geographic distribution in subsample)
        base_per_cell = subsample_n // num_grid_cells
        remainder = subsample_n - base_per_cell * num_grid_cells

        # Count images per grid cell
        cell_counts = np.bincount(cell_inverse)

        # Allocate: min(available, target) per cell
        targets = np.minimum(cell_counts, base_per_cell)

        # Redistribute surplus from small cells to larger cells
        deficit = subsample_n - targets.sum()
        if deficit > 0:
            headroom = cell_counts - targets
            cells_with_room = np.where(headroom > 0)[0]
            if len(cells_with_room) > 0:
                # Proportional redistribution
                total_headroom = headroom[cells_with_room].sum()
                for ci in cells_with_room:
                    extra = int(deficit * headroom[ci] / total_headroom) if total_headroom > 0 else 0
                    extra = min(extra, headroom[ci])
                    targets[ci] += extra

        # Sample from each grid cell (argsort + searchsorted for O(n log n))
        order = np.argsort(cell_inverse)
        sorted_inverse = cell_inverse[order]
        splits = np.searchsorted(sorted_inverse, np.arange(num_grid_cells + 1))

        sub_indices = []
        for ci in range(num_grid_cells):
            cell_idx = order[splits[ci]:splits[ci + 1]]
            n_take = min(int(targets[ci]), len(cell_idx))
            if n_take > 0:
                chosen = rng.choice(cell_idx, size=n_take, replace=False)
                sub_indices.append(chosen)

        sub_indices = np.concatenate(sub_indices)
        sub_lats = latitudes[sub_indices]
        sub_lons = longitudes[sub_indices]
        print(f"  Subsample: {len(sub_indices):,} points from {num_grid_cells:,} grid cells "
              f"({time.time()-t0:.1f}s)")

    # Step 2: Run OPTICS on subsample
    raw_labels, num_clusters, num_noise = _run_optics(
        sub_lats, sub_lons, min_samples, max_eps, min_cluster_size,
    )

    # Step 3: Compute centroids from subsample clusters
    centroids = _compute_centroids(raw_labels, sub_lats, sub_lons)
    print(f"  Discovered {len(centroids)} centroids from subsample")

    # Step 4: Voronoi-assign ALL points to nearest centroid (vectorized)
    print(f"  Voronoi-assigning all {n:,} points to {len(centroids)} centroids...")
    t0 = time.time()
    cell_indices = voronoi_assign_batched(latitudes, longitudes, centroids)
    print(f"  Voronoi assignment done in {time.time()-t0:.1f}s")

    num_cells = len(centroids)
    print(f"  Final: {num_cells} semantic geocells")

    # Verify balance
    unique, counts = np.unique(cell_indices, return_counts=True)
    print(f"  Images per cell: min={counts.min()}, median={int(np.median(counts))}, "
          f"max={counts.max()}, mean={counts.mean():.0f}")

    return cell_indices, centroids


# Keep legacy alias for backward compatibility
build_semantic_geocells_fast = build_semantic_geocells


def build_semantic_geocells_country_constrained(
    latitudes: np.ndarray,
    longitudes: np.ndarray,
    countries: np.ndarray,
    subsample_n: int = 500_000,
    min_samples: int = 20,
    max_eps: float = 0.0087,
    min_cluster_size: int = None,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """Build semantic geocells with OPTICS run per-country so clusters never span borders.

    Strategy:
      1. Geographically stratified subsample (same as subsampled version)
      2. For each country in subsample:
           - >= min_samples points: run OPTICS, extract cluster centroids
           - < min_samples points: single centroid at median lat/lon
      3. Voronoi-assign all images to nearest centroid *within same country*
         (fallback to global nearest for countries missing from subsample)

    This ensures geocell boundaries respect country borders — road markings,
    driving side, and language all change at borders, so this is semantically
    meaningful (same motivation as PIGEOTTO's admin boundary approach).

    Args:
        latitudes: (N,) latitudes in degrees
        longitudes: (N,) longitudes in degrees
        countries: (N,) ISO country code strings
        subsample_n: number of points for geographically stratified subsample
        min_samples: OPTICS min_samples per country
        max_eps: OPTICS max_eps in RADIANS (~0.0087 rad ≈ 50km)
        min_cluster_size: minimum cluster size (default: min_samples)
        seed: random seed for reproducibility

    Returns:
        cell_indices: (N,) array of integer cell labels (0-indexed)
        centroids: (num_cells, 2) array of cell centroids [lat, lon] in degrees
    """
    if min_cluster_size is None:
        min_cluster_size = min_samples

    n = len(latitudes)
    print(f"  Building country-constrained geocells: subsample {subsample_n:,} from {n:,} points...")

    # ── Step 1: Geographically stratified subsample ───────────────────
    rng = np.random.RandomState(seed)

    if n <= subsample_n:
        print(f"  Dataset smaller than subsample — using all {n:,} points")
        sub_lats = latitudes
        sub_lons = longitudes
        sub_countries = countries
    else:
        print(f"  Creating geographically stratified subsample...")
        t0 = time.time()

        grid_lat = np.floor(latitudes).astype(np.int32)
        grid_lon = np.floor(longitudes).astype(np.int32)
        cell_keys = grid_lat * 1000 + grid_lon

        unique_cells, cell_inverse = np.unique(cell_keys, return_inverse=True)
        num_grid_cells = len(unique_cells)

        base_per_cell = subsample_n // num_grid_cells
        cell_counts = np.bincount(cell_inverse)
        targets = np.minimum(cell_counts, base_per_cell)

        deficit = subsample_n - targets.sum()
        if deficit > 0:
            headroom = cell_counts - targets
            cells_with_room = np.where(headroom > 0)[0]
            if len(cells_with_room) > 0:
                total_headroom = headroom[cells_with_room].sum()
                for ci in cells_with_room:
                    extra = int(deficit * headroom[ci] / total_headroom) if total_headroom > 0 else 0
                    extra = min(extra, headroom[ci])
                    targets[ci] += extra

        order = np.argsort(cell_inverse)
        sorted_inverse = cell_inverse[order]
        splits = np.searchsorted(sorted_inverse, np.arange(num_grid_cells + 1))

        sub_indices_list = []
        for ci in range(num_grid_cells):
            cell_idx = order[splits[ci]:splits[ci + 1]]
            n_take = min(int(targets[ci]), len(cell_idx))
            if n_take > 0:
                chosen = rng.choice(cell_idx, size=n_take, replace=False)
                sub_indices_list.append(chosen)

        sub_indices = np.concatenate(sub_indices_list)
        sub_lats = latitudes[sub_indices]
        sub_lons = longitudes[sub_indices]
        sub_countries = countries[sub_indices]
        print(f"  Subsample: {len(sub_indices):,} points from {num_grid_cells:,} grid cells "
              f"({time.time()-t0:.1f}s)")

    # ── Step 2: Per-country OPTICS ────────────────────────────────────
    unique_sub_countries = np.unique(sub_countries)
    print(f"  Running per-country OPTICS on {len(unique_sub_countries)} countries "
          f"(min_samples={min_samples})...")
    t0 = time.time()

    all_centroids = []
    all_centroid_countries = []
    optics_countries = 0
    fallback_countries = 0

    for i, country in enumerate(unique_sub_countries):
        mask = sub_countries == country
        c_lats = sub_lats[mask]
        c_lons = sub_lons[mask]
        n_c = len(c_lats)

        if n_c >= min_samples:
            coords_rad = np.column_stack([np.radians(c_lats), np.radians(c_lons)])
            optics = OPTICS(
                min_samples=min_samples,
                max_eps=max_eps,
                metric="haversine",
                cluster_method="xi",
                min_cluster_size=min_cluster_size,
                n_jobs=-1,
            )
            optics.fit(coords_rad)
            raw_labels = optics.labels_

            num_clusters = len(set(raw_labels.tolist())) - (1 if -1 in raw_labels else 0)

            if num_clusters > 0:
                country_centroids = _compute_centroids(raw_labels, c_lats, c_lons)
                all_centroids.extend(country_centroids.tolist())
                all_centroid_countries.extend([country] * len(country_centroids))
                optics_countries += 1
                if (i + 1) % 20 == 0 or n_c > 10_000:
                    elapsed = time.time() - t0
                    print(f"  [{i+1}/{len(unique_sub_countries)}] {country}: "
                          f"{n_c:,} pts → {num_clusters} clusters ({elapsed:.0f}s elapsed)")
                continue
            else:
                print(f"  WARNING: {country} has {n_c:,} subsample pts but OPTICS found 0 clusters "
                      f"→ collapsing to single centroid (try increasing max_eps)")

        # Fallback: single centroid at median position
        all_centroids.append([float(np.median(c_lats)), float(np.median(c_lons))])
        all_centroid_countries.append(country)
        fallback_countries += 1

    elapsed = time.time() - t0
    print(f"  Per-country OPTICS done in {elapsed:.1f}s")
    print(f"  OPTICS countries: {optics_countries}, single-centroid fallback: {fallback_countries}")

    centroids_arr = np.array(all_centroids, dtype=np.float64)
    centroid_countries_arr = np.array(all_centroid_countries)
    print(f"  Total centroids discovered: {len(centroids_arr)}")

    # ── Step 3: Country-constrained Voronoi assignment ─────────────────
    # Build mapping: country → list of global centroid indices
    country_centroid_map: dict[str, list[int]] = {}
    for idx, c in enumerate(centroid_countries_arr):
        country_centroid_map.setdefault(c, []).append(idx)

    full_unique_countries = np.unique(countries)
    missing = set(full_unique_countries.tolist()) - set(country_centroid_map.keys())
    if missing:
        print(f"  WARNING: {len(missing)} countries in full dataset have no subsample centroid "
              f"→ using global nearest fallback: {sorted(missing)}")

    print(f"  Country-constrained Voronoi assigning {n:,} points...")
    t0 = time.time()
    assignments = np.empty(n, dtype=np.int64)

    for country in full_unique_countries:
        mask = countries == country
        cidx = country_centroid_map.get(country, None)

        if cidx is None:
            # Fallback: nearest centroid globally
            cidx = list(range(len(centroids_arr)))

        country_centroids = centroids_arr[cidx]
        local_assignments = voronoi_assign_batched(
            latitudes[mask], longitudes[mask], country_centroids
        )
        assignments[mask] = np.array(cidx, dtype=np.int64)[local_assignments]

    print(f"  Voronoi assignment done in {time.time()-t0:.1f}s")

    num_cells = len(centroids_arr)
    unique_assigned, counts = np.unique(assignments, return_counts=True)
    print(f"  Final: {num_cells} semantic geocells")
    print(f"  Images per cell: min={counts.min()}, median={int(np.median(counts))}, "
          f"max={counts.max()}, mean={counts.mean():.0f}")

    return assignments, centroids_arr


# ── Save / Load Semantic Geocell Config ───────────────────────────────

def save_semantic_config(
    output_dir: str,
    centroids: np.ndarray,
    cell_indices: np.ndarray,
    tau_km: float,
    top_k: int,
    optics_params: dict,
):
    """Save semantic geocell configuration."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    np.save(out / "cell_centroids.npy", centroids)
    np.save(out / "cell_indices.npy", cell_indices)

    config = {
        "type": "semantic_optics",
        "num_cells": len(centroids),
        "tau_km": tau_km,
        "top_k": top_k,
        "optics_params": optics_params,
    }
    with open(out / "geocell_config.json", "w") as f:
        json.dump(config, f, indent=2)

    print(f"  Saved semantic geocell config: {len(centroids)} cells")


def load_semantic_config(config_dir: str) -> dict:
    """Load semantic geocell configuration."""
    p = Path(config_dir)
    centroids = np.load(p / "cell_centroids.npy")

    with open(p / "geocell_config.json") as f:
        config = json.load(f)

    config["centroids"] = centroids
    return config


def assign_to_nearest_centroid(lat: float, lon: float, centroids: np.ndarray) -> int:
    """Assign a single (lat, lon) to its nearest semantic geocell (Voronoi)."""
    dists = haversine_to_all_centroids(lat, lon, centroids)
    return int(np.argmin(dists))


# ── CLI Entry Point ──────────────────────────────────────────────────

def main():
    if sys.platform == "win32":
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    parser = argparse.ArgumentParser(description="Build semantic geocells (OPTICS + Voronoi)")
    parser.add_argument("--metadata", type=str, default="GEOPROJECT/data/osv5m_50k/metadata.csv",
                        help="Path to metadata CSV (should be TRAIN-ONLY to avoid leakage)")
    parser.add_argument("--output", type=str, default="GEOPROJECT/data/osv5m_50k/semantic_cells")
    parser.add_argument("--subsample", type=int, default=None,
                        help="Subsample N points for OPTICS (recommended for >200K images)")
    parser.add_argument("--min-samples", type=int, default=20,
                        help="OPTICS min_samples (use 50 for 1M+)")
    parser.add_argument("--max-eps", type=float, default=0.0087,
                        help="OPTICS max_eps in RADIANS (~0.0087 rad = ~50km)")
    parser.add_argument("--min-cluster-size", type=int, default=None)
    parser.add_argument("--tau", type=float, default=DEFAULT_TAU_KM,
                        help="Label smoothing temperature in km (use 100 for 2000+ cells)")
    parser.add_argument("--top-k", type=int, default=DEFAULT_TOP_K)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--country-aware", action="store_true",
                        help="Run OPTICS per-country so clusters never span borders")
    args = parser.parse_args()

    import pandas as pd

    print("=" * 60)
    print("Building Semantic Geocells (OPTICS + Voronoi)")
    print(f"  min_samples: {args.min_samples}")
    print(f"  max_eps: {args.max_eps} rad ({np.degrees(args.max_eps):.2f} deg)")
    print(f"  tau: {args.tau} km, top-k: {args.top_k}")
    print(f"  country-aware: {args.country_aware}")
    if args.subsample:
        print(f"  subsample: {args.subsample:,} (OPTICS on subsample, Voronoi-assign all)")
    else:
        print(f"  subsample: None (OPTICS on all points)")
    print("=" * 60)
    print()

    # Load metadata
    print("Loading metadata...")
    df = pd.read_csv(args.metadata, dtype={"id": str})
    lats = df["latitude"].values
    lons = df["longitude"].values
    print(f"  {len(df):,} images")
    print(f"  WARNING: Ensure this is TRAIN-ONLY data to avoid leakage!")

    # Drop NaN coordinates — OPTICS crashes on NaN input
    nan_mask = np.isnan(lats) | np.isnan(lons)
    if nan_mask.any():
        print(f"  WARNING: Dropping {nan_mask.sum():,} rows with NaN lat/lon")
        df = df[~nan_mask].reset_index(drop=True)
        lats = df["latitude"].values
        lons = df["longitude"].values
    print()

    # Build semantic geocells
    t0 = time.time()
    if args.country_aware:
        if "country" not in df.columns:
            raise ValueError("--country-aware requires a 'country' column in metadata CSV")
        if args.subsample is None:
            print(f"  WARNING: --country-aware without --subsample will run OPTICS on all "
                  f"{len(df):,} points per country. This may be very slow or OOM on large datasets. "
                  f"Consider adding --subsample 500000.")
        country_codes = df["country"].fillna("XX").values.astype(str)
        cell_indices, centroids = build_semantic_geocells_country_constrained(
            lats, lons, country_codes,
            subsample_n=args.subsample or len(df),
            min_samples=args.min_samples,
            max_eps=args.max_eps,
            min_cluster_size=args.min_cluster_size,
            seed=args.seed,
        )
    elif args.subsample and len(df) > args.subsample:
        cell_indices, centroids = build_semantic_geocells_subsampled(
            lats, lons,
            subsample_n=args.subsample,
            min_samples=args.min_samples,
            max_eps=args.max_eps,
            min_cluster_size=args.min_cluster_size,
            seed=args.seed,
        )
    else:
        cell_indices, centroids = build_semantic_geocells(
            lats, lons,
            min_samples=args.min_samples,
            max_eps=args.max_eps,
            min_cluster_size=args.min_cluster_size,
        )
    print(f"  Total geocell build time: {time.time()-t0:.1f}s")
    print()

    # Save config
    optics_params = {
        "min_samples": args.min_samples,
        "max_eps": args.max_eps,
        "min_cluster_size": args.min_cluster_size or args.min_samples,
        "subsample": args.subsample,
        "country_aware": args.country_aware,
    }
    save_semantic_config(args.output, centroids, cell_indices, args.tau, args.top_k, optics_params)
    print()

    # Precompute haversine-smoothed targets
    print(f"Precomputing haversine-smoothed targets (tau={args.tau}km, top-{args.top_k})...")
    t0 = time.time()
    precompute_smooth_targets(
        lats, lons, centroids, args.tau, args.top_k, args.output,
    )
    print(f"  Done in {time.time()-t0:.1f}s")
    print()

    # Summary
    print("=" * 60)
    print("Done! Files created:")
    out = Path(args.output)
    for f_name in ["cell_centroids.npy", "cell_indices.npy", "geocell_config.json",
                    "smooth_indices.npy", "smooth_probs.npy"]:
        fp = out / f_name
        if fp.exists():
            size = fp.stat().st_size / 1024
            unit = "KB"
            if size > 1024:
                size /= 1024
                unit = "MB"
            print(f"  {f_name}: {size:.1f} {unit}")
    print("=" * 60)


if __name__ == "__main__":
    main()
