"""
Download a stratified subset of OSV-5M from HuggingFace.

Two sampling strategies:

  spatial (default, recommended for 1M+):
    1. Download train.csv (4.9M metadata) — select 1M via S2 density balancing
    2. Download shards one at a time, extract ONLY selected images
    Peak disk: ~100GB (selected images) + 2.5GB (one shard)

  country (legacy, used for 50K prototype):
    1. Download N shards, extract all images
    2. Stratified sample by country with min/max caps
    3. Delete unselected images

Usage:
    # 1M with spatial density balancing (select-first, download-selectively):
    python -m GEOPROJECT.geoguessr.data.download --total 1000000 --sampling-strategy spatial

    # 50K legacy (download-first, sample-after):
    python -m GEOPROJECT.geoguessr.data.download --total 50000 --sampling-strategy country
"""

import argparse
import math
import os
import shutil
import sys
import tempfile
import time
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
from huggingface_hub import hf_hub_download

from GEOPROJECT.geoguessr.data.geocells import latlon_to_cell_id


REPO_ID = "osv5m/osv5m"
NUM_TRAIN_SHARDS = 98

# 10 spread-out shards for geographic diversity (legacy default)
DEFAULT_SHARDS = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90]

S2_SAMPLE_LEVEL = 4  # ~800 equal-area cells globally, each ~60,000 km²


def download_csv() -> pd.DataFrame:
    """Download and load train.csv from HuggingFace."""
    print("Downloading train.csv (~2.9 GB, cached after first run)...")
    t0 = time.time()

    csv_path = hf_hub_download(
        REPO_ID, "train.csv", repo_type="dataset",
    )

    print(f"  Downloaded in {time.time()-t0:.0f}s")
    print(f"  Path: {csv_path}")
    print("  Loading into pandas...")

    df = pd.read_csv(csv_path, dtype={
        "id": str, "latitude": float, "longitude": float,
        "country": str, "region": str, "sub-region": str, "city": str,
        "climate": float, "drive_side": float, "land_cover": float,
        "soil": float, "road_index": float, "dist_sea": float,
    }, usecols=[
        "id", "latitude", "longitude", "country", "region", "sub-region",
        "city", "climate", "drive_side", "land_cover", "soil",
        "road_index", "dist_sea",
    ])

    print(f"  Loaded {len(df):,} rows, {df['country'].nunique()} countries")
    print()
    return df


# ── Spatial density-balanced sampling (S2 cells) ─────────────────────


def spatial_sample(
    df: pd.DataFrame,
    total: int,
    seed: int = 42,
) -> pd.DataFrame:
    """Density-balanced sampling using S2 level-4 cells.

    Ensures roughly equal images per unit area of Earth:
      - Each image is assigned to an S2 level-4 cell (~60,000 km² each)
      - Target per cell = total / num_occupied_cells
      - Cells with fewer images → take ALL (boosts underrepresented regions)
      - Surplus redistributed to cells with headroom
      - Within each cell, sample proportionally across countries for diversity
    """
    print(f"Spatial sampling: selecting {total:,} from {len(df):,} images...")
    rng = np.random.RandomState(seed)

    if len(df) <= total:
        print(f"  Pool smaller than target — using all {len(df):,}")
        return df.copy()

    # Drop rows with NaN coordinates
    n_before = len(df)
    df = df.dropna(subset=["latitude", "longitude"])
    if len(df) < n_before:
        print(f"  Dropped {n_before - len(df):,} rows with NaN coordinates")

    # Assign each image to an S2 level-4 cell
    print(f"  Assigning to S2 level-{S2_SAMPLE_LEVEL} cells...", end="", flush=True)
    t0 = time.time()
    cell_ids = np.array([
        latlon_to_cell_id(lat, lon, S2_SAMPLE_LEVEL)
        for lat, lon in zip(df["latitude"].values, df["longitude"].values)
    ])
    print(f" done ({time.time()-t0:.0f}s)")

    df = df.copy()
    df["_s2_cell"] = cell_ids

    # Count images per cell
    cell_counts = df["_s2_cell"].value_counts()
    num_cells = len(cell_counts)
    print(f"  {num_cells} occupied S2 cells")

    # Compute target per cell: even distribution
    if total < num_cells:
        # Fewer requested than cells — randomly select which cells get 1 image
        print(f"  WARNING: target ({total:,}) < occupied cells ({num_cells}). "
              f"Selecting {total} random cells.")
        selected_cells = rng.choice(cell_counts.index.values, size=total, replace=False)
        sampled_parts = []
        for cell in selected_cells:
            cell_df = df[df["_s2_cell"] == cell]
            sampled_parts.append(cell_df.sample(n=1, random_state=rng))
        result = pd.concat(sampled_parts, ignore_index=True).drop(columns="_s2_cell")
        _print_spatial_stats(df, result, cell_counts,
                             {c: 1 for c in selected_cells})
        return result

    base_target = math.ceil(total / num_cells)

    # Pass 1: cells with fewer images than target take all; track surplus
    allocations = {}
    surplus = 0
    cells_with_headroom = []

    for cell, count in cell_counts.items():
        if count <= base_target:
            allocations[cell] = count  # take all
            surplus += base_target - count
        else:
            allocations[cell] = base_target
            cells_with_headroom.append((cell, count))

    # Pass 2: redistribute surplus to cells with headroom (proportionally)
    if surplus > 0 and cells_with_headroom:
        total_headroom = sum(c - allocations[cell] for cell, c in cells_with_headroom)
        for cell, count in cells_with_headroom:
            headroom = count - allocations[cell]
            extra = int(surplus * headroom / total_headroom) if total_headroom > 0 else 0
            extra = min(extra, headroom)
            allocations[cell] += extra

    # Pass 3: adjust to hit exact total (greedy trim/add)
    current_total = sum(allocations.values())
    if current_total > total:
        # Trim from largest allocations first
        for cell in sorted(allocations, key=lambda c: -allocations[c]):
            if current_total <= total:
                break
            trim = min(allocations[cell] - 1, current_total - total)
            if trim > 0:
                allocations[cell] -= trim
                current_total -= trim
    elif current_total < total:
        # Add to cells with headroom
        for cell, count in sorted(cells_with_headroom, key=lambda x: -x[1]):
            if current_total >= total:
                break
            headroom = count - allocations[cell]
            add = min(headroom, total - current_total)
            if add > 0:
                allocations[cell] += add
                current_total += add

    # Sample within each cell, proportionally across countries
    print(f"  Sampling within cells (country-proportional)...")
    sampled_parts = []
    grouped = df.groupby("_s2_cell")
    for cell, n_target in allocations.items():
        if n_target <= 0:
            continue
        cell_df = grouped.get_group(cell)
        if len(cell_df) <= n_target:
            sampled_parts.append(cell_df)
            continue

        # Within cell: sample proportionally by country
        country_counts = cell_df["country"].value_counts()
        cell_sampled = []
        remaining = n_target

        for country, c_count in country_counts.items():
            proportion = c_count / len(cell_df)
            n_country = max(1, int(proportion * n_target))
            n_country = min(n_country, c_count, remaining)
            if n_country <= 0:
                continue
            country_df = cell_df[cell_df["country"] == country]
            cell_sampled.append(country_df.sample(n=n_country, random_state=rng))
            remaining -= n_country

        # Top up remainder (int() truncation causes systematic undershoot)
        if remaining > 0 and cell_sampled:
            already_idx = pd.concat(cell_sampled).index
            leftover = cell_df.drop(already_idx)
            if len(leftover) > 0:
                n_extra = min(remaining, len(leftover))
                cell_sampled.append(leftover.sample(n=n_extra, random_state=rng))

        if cell_sampled:
            sampled_parts.append(pd.concat(cell_sampled))

    result = pd.concat(sampled_parts, ignore_index=True).drop(columns="_s2_cell")

    # Print distribution stats
    _print_spatial_stats(df, result, cell_counts, allocations)

    return result


def _print_spatial_stats(full_df, sampled_df, cell_counts, allocations):
    """Print spatial sampling distribution statistics."""
    alloc_values = np.array(list(allocations.values()))
    print(f"\n  Selected: {len(sampled_df):,} images, "
          f"{sampled_df['country'].nunique()} countries")
    print(f"  Per-cell allocation: min={alloc_values.min()}, "
          f"med={int(np.median(alloc_values))}, "
          f"max={alloc_values.max()}, mean={alloc_values.mean():.0f}")

    # Top/bottom cells — use groupby for efficient lookup
    grouped = full_df.groupby("_s2_cell")
    sorted_allocs = sorted(allocations.items(), key=lambda x: -x[1])
    print(f"  Top 5 cells by allocation:")
    for cell, n in sorted_allocs[:5]:
        available = cell_counts[cell]
        sample_row = grouped.get_group(cell).iloc[0]
        print(f"    {n:6d}/{available:6d} | ~({sample_row['latitude']:.0f}, "
              f"{sample_row['longitude']:.0f}) | {sample_row['country']}")
    print(f"  Bottom 5 cells:")
    for cell, n in sorted_allocs[-5:]:
        available = cell_counts[cell]
        sample_row = grouped.get_group(cell).iloc[0]
        print(f"    {n:6d}/{available:6d} | ~({sample_row['latitude']:.0f}, "
              f"{sample_row['longitude']:.0f}) | {sample_row['country']}")
    print()


# ── Shard download with selective extraction ──────────────────────────


def download_shards(
    shard_indices: list[int],
    output_dir: Path,
    selected_ids: set[str] | None = None,
) -> set[str]:
    """Download ZIP shards, extract images, delete shards. Returns extracted IDs.

    Args:
        shard_indices: which shard indices to download
        output_dir: where to write images/
        selected_ids: if provided, extract ONLY these image IDs (selective mode).
                      If None, extract all images (legacy mode).
    """
    images_dir = output_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)

    # Check existing images (jpg or png)
    existing_ids = set(p.stem for p in images_dir.iterdir()
                       if p.suffix.lower() in (".jpg", ".jpeg", ".png"))
    if existing_ids:
        print(f"  Found {len(existing_ids):,} existing images")

    # In selective mode, skip images we already have
    if selected_ids is not None:
        remaining = selected_ids - existing_ids
        if not remaining:
            print(f"  All {len(selected_ids):,} selected images already extracted")
            return existing_ids & selected_ids
        print(f"  Need {len(remaining):,} of {len(selected_ids):,} selected images "
              f"({len(existing_ids):,} already on disk)")
    else:
        remaining = None

    extracted_ids = set(existing_ids)
    start_time = time.time()
    mode = "selective" if selected_ids is not None else "full"

    # Use a temp dir for shard downloads to avoid polluting HF cache
    tmp_dir = tempfile.mkdtemp(prefix="osv5m_shards_")

    print(f"Downloading {len(shard_indices)} shards (~2.5 GB each, {mode} extraction)...")
    print(f"  Temp dir for shards: {tmp_dir}")
    print()

    for i, shard_idx in enumerate(shard_indices):
        shard_name = f"images/train/{shard_idx:02d}.zip"
        label = f"[{i+1}/{len(shard_indices)}] Shard {shard_idx:02d}"

        # In selective mode, skip shard if we already have all needed images
        if remaining is not None and not remaining:
            print(f"  {label}: All selected images found — skipping remaining shards")
            break

        print(f"  {label}: Downloading...", end="", flush=True)
        t0 = time.time()

        try:
            zip_path = hf_hub_download(
                REPO_ID, shard_name, repo_type="dataset",
                local_dir=tmp_dir,
            )
        except Exception as e:
            print(f" FAILED: {e}")
            continue

        dl_time = time.time() - t0
        print(f" done ({dl_time:.0f}s). Extracting...", end="", flush=True)

        # Extract images from this shard
        count = 0
        with zipfile.ZipFile(zip_path, "r") as zf:
            for name in zf.namelist():
                if not name.lower().endswith((".jpg", ".jpeg", ".png")):
                    continue
                img_id = os.path.splitext(os.path.basename(name))[0]
                if not img_id or img_id in extracted_ids:
                    continue
                # Selective mode: skip images not in our selection
                if selected_ids is not None and img_id not in selected_ids:
                    continue

                img_data = zf.read(name)
                ext = os.path.splitext(name)[1].lower() or ".jpg"
                (images_dir / f"{img_id}{ext}").write_bytes(img_data)
                extracted_ids.add(img_id)
                count += 1

                if remaining is not None:
                    remaining.discard(img_id)

        elapsed = time.time() - start_time

        # Delete the downloaded shard ZIP to free disk (temp dir, not HF cache)
        try:
            Path(zip_path).unlink(missing_ok=True)
        except OSError:
            pass

        if selected_ids is not None:
            found_so_far = len(selected_ids) - len(remaining)
            print(f" {count:,} extracted. "
                  f"Progress: {found_so_far:,}/{len(selected_ids):,} "
                  f"[{elapsed/60:.1f}min]")
        else:
            print(f" {count:,} images. Total: {len(extracted_ids):,} "
                  f"[{elapsed/60:.1f}min]")

    # Clean up temp dir
    shutil.rmtree(tmp_dir, ignore_errors=True)

    print(f"\n  Total extracted: {len(extracted_ids):,}")
    print()
    return extracted_ids


# ── Legacy country-stratified sampling ────────────────────────────────


def stratified_sample(
    df: pd.DataFrame,
    available_ids: set[str],
    total: int,
    max_per_country: int,
) -> pd.DataFrame:
    """Legacy stratified sample from available images by country."""
    print(f"Country-stratified sampling: {total:,} from {len(available_ids):,} images...")

    pool = df[df["id"].isin(available_ids)].copy()
    print(f"  Pool: {len(pool):,} images, {pool['country'].nunique()} countries")

    if len(pool) <= total:
        print(f"  Pool is smaller than target — using all {len(pool):,} images")
        return pool

    country_counts = pool["country"].value_counts()
    num_countries = len(country_counts)

    min_per_country = max(3, total // (num_countries * 5))
    allocations = {}

    for country, available in country_counts.items():
        proportion = available / len(pool)
        alloc = int(proportion * total)
        alloc = max(min_per_country, min(alloc, max_per_country))
        alloc = min(alloc, available)
        allocations[country] = alloc

    current_total = sum(allocations.values())
    if current_total > total:
        sorted_allocs = sorted(allocations.items(), key=lambda x: -x[1])
        for country, alloc in sorted_allocs:
            if current_total <= total:
                break
            trim = min(alloc - min_per_country, current_total - total)
            if trim > 0:
                allocations[country] -= trim
                current_total -= trim
    elif current_total < total:
        sorted_allocs = sorted(allocations.items(), key=lambda x: -country_counts[x[0]])
        for country, alloc in sorted_allocs:
            if current_total >= total:
                break
            available = country_counts[country]
            headroom = min(available, max_per_country) - alloc
            add = min(headroom, total - current_total)
            if add > 0:
                allocations[country] += add
                current_total += add

    sampled_parts = []
    for country, n in allocations.items():
        if n <= 0:
            continue
        country_df = pool[pool["country"] == country]
        sampled = country_df.sample(n=min(n, len(country_df)), random_state=42)
        sampled_parts.append(sampled)

    result = pd.concat(sampled_parts, ignore_index=True)
    print(f"  Selected {len(result):,} images from {len(sampled_parts)} countries")
    print()
    return result


def cleanup_unselected(selected_ids: set[str], output_dir: Path) -> int:
    """Delete images that weren't selected to save disk space."""
    images_dir = output_dir / "images"
    removed = 0
    for img_file in images_dir.iterdir():
        if img_file.suffix.lower() in (".jpg", ".jpeg", ".png") and img_file.stem not in selected_ids:
            img_file.unlink()
            removed += 1
    return removed


# ── Main ──────────────────────────────────────────────────────────────


def main():
    if sys.platform == "win32":
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    parser = argparse.ArgumentParser(description="Download stratified OSV-5M subset")
    parser.add_argument(
        "--total", type=int, default=50000,
        help="Target number of images (default: 50000)",
    )
    parser.add_argument(
        "--num-shards", type=int, default=None,
        help="Number of shards to download (default: 10 for country, 98 for spatial)",
    )
    parser.add_argument(
        "--output", type=str, default="GEOPROJECT/data/osv5m_50k",
        help="Output directory (default: GEOPROJECT/data/osv5m_50k)",
    )
    parser.add_argument(
        "--max-per-country", type=int, default=500,
        help="Max images per country (country strategy only, default: 500)",
    )
    parser.add_argument(
        "--sampling-strategy", type=str, default="spatial",
        choices=["spatial", "country"],
        help="Sampling strategy: 'spatial' (S2 density-balanced) or 'country' (legacy)",
    )
    args = parser.parse_args()

    # Default num_shards depends on strategy
    if args.num_shards is None:
        args.num_shards = NUM_TRAIN_SHARDS if args.sampling_strategy == "spatial" else 10

    # Pick spread-out shard indices
    if args.num_shards >= NUM_TRAIN_SHARDS:
        shard_indices = list(range(NUM_TRAIN_SHARDS))
    else:
        step = NUM_TRAIN_SHARDS // args.num_shards
        shard_indices = [i * step for i in range(args.num_shards)]

    output_path = Path(args.output)
    output_path.mkdir(parents=True, exist_ok=True)
    metadata_out = output_path / "metadata.csv"

    print("=" * 60)
    print("OSV-5M Downloader")
    print(f"  Strategy: {args.sampling_strategy}")
    print(f"  Target: {args.total:,} images")
    print(f"  Shards: {args.num_shards} (all)" if args.num_shards == NUM_TRAIN_SHARDS
          else f"  Shards: {args.num_shards} (indices: {shard_indices})")
    if args.sampling_strategy == "country":
        print(f"  Max/country: {args.max_per_country}")
    print(f"  Output: {output_path.resolve()}")
    est_gb = args.num_shards * 2.5 + 2.9
    print(f"  Est. download bandwidth: ~{est_gb:.0f} GB")
    if args.sampling_strategy == "spatial":
        print(f"  Est. final disk: ~{args.total * 100 / 1_000_000:.0f} GB (images only)")
    print("=" * 60)
    print()

    # Step 1: Download metadata
    df = download_csv()

    if args.sampling_strategy == "spatial":
        # ── Spatial: select first, download selectively ──
        print("Step 1/3: Spatial density-balanced selection...")
        sampled = spatial_sample(df, args.total)
        selected_ids = set(sampled["id"].astype(str))

        # Save metadata
        sampled.to_csv(metadata_out, index=False)
        print(f"  Metadata saved to {metadata_out}")

        # Step 2/3: Download shards, extract only selected images
        print("Step 2/3: Downloading shards (selective extraction)...")
        extracted_ids = download_shards(shard_indices, output_path, selected_ids)

        # Verify
        found = selected_ids & extracted_ids
        missing = selected_ids - extracted_ids
        if missing:
            print(f"  WARNING: {len(missing):,} selected images not found in shards")
            # Update metadata to only include images we actually have
            sampled = sampled[sampled["id"].isin(found)]
            sampled.to_csv(metadata_out, index=False)
            print(f"  Updated metadata: {len(sampled):,} images")

        print("Step 3/3: Done (no cleanup needed — only selected images on disk)")

    else:
        # ── Country: legacy flow (download all, sample, cleanup) ──
        print("Step 1/4: Downloading shards...")
        available_ids = download_shards(shard_indices, output_path)

        print("Step 2/4: Country-stratified sampling...")
        sampled = stratified_sample(df, available_ids, args.total, args.max_per_country)
        selected_ids = set(sampled["id"].astype(str))

        sampled.to_csv(metadata_out, index=False)
        print(f"  Metadata saved to {metadata_out}")

        print("Step 3/4: Cleaning up unselected images...")
        removed = cleanup_unselected(selected_ids, output_path)
        print(f"  Removed {removed:,} unselected images")

    # Final summary
    country_dist = sampled["country"].value_counts()
    print()
    print("=" * 60)
    print("DONE!")
    print(f"  Images: {len(sampled):,}")
    print(f"  Countries: {len(country_dist)}")
    print(f"  Lat range: {sampled['latitude'].min():.1f} to {sampled['latitude'].max():.1f}")
    print(f"  Lon range: {sampled['longitude'].min():.1f} to {sampled['longitude'].max():.1f}")
    print(f"  Directory: {output_path.resolve()}")
    print()
    print("Top 10 countries:")
    for country, count in country_dist.head(10).items():
        print(f"  {country}: {count}")
    print()
    print("Bottom 5 countries:")
    for country, count in country_dist.tail(5).items():
        print(f"  {country}: {count}")
    print("=" * 60)


if __name__ == "__main__":
    main()
