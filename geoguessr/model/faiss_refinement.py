"""
Phase 3: FAISS-based two-stage geolocation refinement.

Stage 2 (Coarse): GeoLocator predicts top-K geocells from image
Stage 3 (Fine): FAISS retrieves nearest location cluster within those cells

Pipeline:
  1. After training, extract embeddings for all training images
  2. Within each geocell, cluster embeddings with OPTICS
  3. Build FAISS IndexFlatL2 over cluster centroid embeddings
  4. At inference: top-K cells → filter clusters → nearest centroid → GPS

Architecture doc reference: GEOGUESSR_ARCHITECTURE.md Section 4

Usage:
    # Build FAISS index from trained model:
    python -m GEOPROJECT.geoguessr.model.faiss_refinement build \
        --checkpoint GEOPROJECT/checkpoints/geolocator_best.pt \
        --output GEOPROJECT/faiss_index

    # Run inference:
    python -m GEOPROJECT.geoguessr.model.faiss_refinement predict \
        --image path/to/image.jpg \
        --index GEOPROJECT/faiss_index
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

import faiss
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from sklearn.cluster import OPTICS
from torchvision import transforms

from GEOPROJECT.geoguessr.data.geocells import haversine_km


# ── Constants ────────────────────────────────────────────────────────

EARTH_RADIUS_KM = 6371.0

CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
CLIP_STD = (0.26862954, 0.26130258, 0.27577711)
CLIP_SIZE = 336
EMBED_DIM = 768


# ── Embedding Extraction ─────────────────────────────────────────────

@torch.no_grad()
def extract_embeddings(
    model,
    dataloader,
    device: torch.device,
) -> np.ndarray:
    """Extract L2-normalized embeddings for all images in dataloader.

    Args:
        model: GeoLocator model (in eval mode)
        dataloader: DataLoader yielding (images, smooth_targets, aux_labels)
        device: torch device

    Returns:
        embeddings: (N, 768) float32 numpy array, L2-normalized
    """
    model.eval()
    all_embeddings = []
    total = 0

    for batch_idx, (images, _, _) in enumerate(dataloader):
        images = images.to(device)

        with torch.amp.autocast(device.type, dtype=torch.bfloat16, enabled=(device.type == "cuda")):
            outputs = model(images)

        # Already L2-normalized in model.forward()
        emb = outputs["embedding"].float().cpu().numpy()
        all_embeddings.append(emb)
        total += len(emb)

        if (batch_idx + 1) % 500 == 0:
            print(f"  Extracted {total:,} embeddings...")

    embeddings = np.concatenate(all_embeddings, axis=0)
    print(f"  Total: {embeddings.shape[0]:,} embeddings, dim={embeddings.shape[1]}")
    return embeddings


# ── Within-Cell Clustering ───────────────────────────────────────────

def cluster_within_cells(
    embeddings: np.ndarray,
    cell_indices: np.ndarray,
    latitudes: np.ndarray,
    longitudes: np.ndarray,
    min_samples: int = 5,
    max_clusters_per_cell: int = 50,
) -> dict:
    """Cluster embeddings within each geocell using OPTICS.

    For each cell, clusters the embeddings and stores:
      - Centroid embedding (mean of cluster)
      - Centroid GPS (median lat/lon of cluster members)
      - Cell ID

    Args:
        embeddings: (N, 768) L2-normalized embeddings
        cell_indices: (N,) geocell assignment per image
        latitudes: (N,) latitude per image
        longitudes: (N,) longitude per image
        min_samples: OPTICS min_samples for within-cell clustering
        max_clusters_per_cell: cap clusters per cell to prevent explosion

    Returns:
        dict with keys:
            cluster_embeddings: (M, 768) centroid embeddings
            cluster_gps: (M, 2) centroid GPS [lat, lon]
            cluster_cell_ids: (M,) cell ID for each cluster
    """
    unique_cells = np.unique(cell_indices)
    num_cells = len(unique_cells)

    all_cluster_embeddings = []
    all_cluster_gps = []
    all_cluster_cell_ids = []

    t0 = time.time()
    for ci, cell_id in enumerate(unique_cells):
        mask = cell_indices == cell_id
        cell_embeds = embeddings[mask]
        cell_lats = latitudes[mask]
        cell_lons = longitudes[mask]
        n_in_cell = mask.sum()

        if n_in_cell < min_samples:
            # Too few images: treat entire cell as one cluster
            centroid_emb = cell_embeds.mean(axis=0, keepdims=True)
            centroid_emb /= np.linalg.norm(centroid_emb, axis=1, keepdims=True) + 1e-8
            centroid_gps = np.array([[np.median(cell_lats), np.median(cell_lons)]])
            all_cluster_embeddings.append(centroid_emb)
            all_cluster_gps.append(centroid_gps)
            all_cluster_cell_ids.append(np.array([cell_id]))
            continue

        # OPTICS clustering on embeddings (euclidean ≈ cosine for L2-normalized)
        try:
            optics = OPTICS(
                min_samples=min(min_samples, n_in_cell),
                metric="euclidean",
                cluster_method="xi",
                n_jobs=1,
            )
            optics.fit(cell_embeds)
            labels = optics.labels_
        except Exception as e:
            # Fallback: single cluster (log warning for debugging)
            print(f"    Warning: OPTICS failed for cell {cell_id} ({n_in_cell} pts): {e}")
            labels = np.zeros(n_in_cell, dtype=int)

        cluster_ids = sorted(set(labels) - {-1})

        if len(cluster_ids) == 0:
            # All points are noise: treat as one cluster
            cluster_ids = [0]
            labels = np.zeros(n_in_cell, dtype=int)

        # Cap clusters per cell (keep largest by member count)
        if len(cluster_ids) > max_clusters_per_cell:
            cluster_ids = sorted(cluster_ids, key=lambda c: (labels == c).sum(), reverse=True)
            cluster_ids = cluster_ids[:max_clusters_per_cell]

        for cid in cluster_ids:
            cmask = labels == cid
            if cmask.sum() == 0:
                continue
            centroid_emb = cell_embeds[cmask].mean(axis=0, keepdims=True)
            centroid_emb /= np.linalg.norm(centroid_emb, axis=1, keepdims=True) + 1e-8
            centroid_gps = np.array([[
                np.median(cell_lats[cmask]),
                np.median(cell_lons[cmask]),
            ]])
            all_cluster_embeddings.append(centroid_emb)
            all_cluster_gps.append(centroid_gps)
            all_cluster_cell_ids.append(np.array([cell_id]))

        # Assign noise points as individual micro-clusters (optional, capped)
        noise_mask = labels == -1
        n_noise = noise_mask.sum()
        if n_noise > 0 and n_noise <= 20:
            # Small number of noise: add as individual entries
            for ni in np.where(noise_mask)[0]:
                emb = cell_embeds[ni:ni+1].copy()
                emb /= np.linalg.norm(emb, axis=1, keepdims=True) + 1e-8
                gps = np.array([[cell_lats[ni], cell_lons[ni]]])
                all_cluster_embeddings.append(emb)
                all_cluster_gps.append(gps)
                all_cluster_cell_ids.append(np.array([cell_id]))

        if (ci + 1) % 1000 == 0:
            elapsed = time.time() - t0
            print(f"  Clustered {ci+1}/{num_cells} cells ({elapsed:.0f}s)")

    cluster_embeddings = np.concatenate(all_cluster_embeddings, axis=0).astype(np.float32)
    cluster_gps = np.concatenate(all_cluster_gps, axis=0).astype(np.float64)
    cluster_cell_ids = np.concatenate(all_cluster_cell_ids, axis=0).astype(np.int64)

    print(f"  Total location clusters: {len(cluster_embeddings):,}")
    print(f"  Average clusters per cell: {len(cluster_embeddings)/num_cells:.1f}")

    return {
        "cluster_embeddings": cluster_embeddings,
        "cluster_gps": cluster_gps,
        "cluster_cell_ids": cluster_cell_ids,
    }


# ── FAISS Index ──────────────────────────────────────────────────────

def build_faiss_index(cluster_embeddings: np.ndarray) -> faiss.IndexFlatL2:
    """Build FAISS IndexFlatL2 over cluster centroid embeddings.

    For L2-normalized vectors, L2 distance is monotonically related to
    cosine similarity: ||a - b||^2 = 2 - 2*cos(a,b).
    So IndexFlatL2 gives equivalent ranking to cosine similarity search.

    Args:
        cluster_embeddings: (M, 768) float32, L2-normalized

    Returns:
        FAISS IndexFlatL2 with M vectors
    """
    dim = cluster_embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    # Ensure contiguous float32
    embeddings = np.ascontiguousarray(cluster_embeddings, dtype=np.float32)
    index.add(embeddings)
    print(f"  FAISS index built: {index.ntotal:,} vectors, dim={dim}")
    mem_mb = index.ntotal * dim * 4 / (1024 * 1024)
    print(f"  Index memory: ~{mem_mb:.1f} MB")
    return index


def save_faiss_index(
    index: faiss.IndexFlatL2,
    cluster_gps: np.ndarray,
    cluster_cell_ids: np.ndarray,
    output_dir: str,
):
    """Save FAISS index and cluster metadata."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    faiss.write_index(index, str(out / "faiss.index"))
    np.save(out / "cluster_gps.npy", cluster_gps)
    np.save(out / "cluster_cell_ids.npy", cluster_cell_ids)

    config = {
        "num_clusters": index.ntotal,
        "embed_dim": index.d,
    }
    with open(out / "faiss_config.json", "w") as f:
        json.dump(config, f, indent=2)

    print(f"  Saved FAISS index to {out}")


def load_faiss_index(index_dir: str) -> tuple:
    """Load FAISS index and cluster metadata.

    Returns:
        (index, cluster_gps, cluster_cell_ids)
    """
    p = Path(index_dir)
    index = faiss.read_index(str(p / "faiss.index"))
    cluster_gps = np.load(p / "cluster_gps.npy")
    cluster_cell_ids = np.load(p / "cluster_cell_ids.npy")
    print(f"  Loaded FAISS index: {index.ntotal:,} clusters")
    return index, cluster_gps, cluster_cell_ids


# ── Two-Stage Inference ──────────────────────────────────────────────

def predict_two_stage(
    model,
    image: torch.Tensor,
    faiss_index: faiss.IndexFlatL2,
    cluster_gps: np.ndarray,
    cluster_cell_ids: np.ndarray,
    centroids: np.ndarray,
    device: torch.device,
    top_k_cells: int = 50,
) -> dict:
    """Two-stage geolocation prediction.

    Stage 2 (Coarse): Predict top-K geocells from classification head
    Stage 3 (Fine): Retrieve nearest location cluster within those cells

    Args:
        model: GeoLocator model
        image: (1, 3, 336, 336) preprocessed image tensor
        faiss_index: FAISS index over cluster embeddings
        cluster_gps: (M, 2) GPS for each cluster
        cluster_cell_ids: (M,) cell ID for each cluster
        centroids: (num_cells, 2) geocell centroids for coarse prediction
        device: torch device
        top_k_cells: number of top cells to consider (default: 50)

    Returns:
        dict with:
            coarse_lat, coarse_lon: Stage 2 prediction (cell centroid)
            fine_lat, fine_lon: Stage 3 prediction (nearest cluster)
            coarse_cell: predicted cell index
            top_cells: top-K cell indices with probabilities
            confidence: max cell probability
            embedding: (768,) image embedding
    """
    model.eval()
    image = image.to(device)

    with torch.no_grad():
        with torch.amp.autocast(device.type, dtype=torch.bfloat16, enabled=(device.type == "cuda")):
            outputs = model(image)

    # Stage 2: Coarse cell prediction
    geo_probs = F.softmax(outputs["geo"].float(), dim=-1).cpu().numpy()[0]
    top_k_idx = np.argsort(-geo_probs)[:top_k_cells]
    top_k_probs = geo_probs[top_k_idx]

    coarse_cell = top_k_idx[0]
    coarse_lat, coarse_lon = centroids[coarse_cell]

    # Stage 3: Fine retrieval within top-K cells
    embedding = outputs["embedding"].float().cpu().numpy()  # (1, 768)

    # Filter clusters belonging to top-K cells
    valid_mask = np.isin(cluster_cell_ids, top_k_idx)
    valid_indices = np.where(valid_mask)[0]
    valid_set = set(valid_indices.tolist())  # O(1) membership check

    if len(valid_set) == 0:
        # Fallback: use coarse prediction
        return {
            "coarse_lat": float(coarse_lat),
            "coarse_lon": float(coarse_lon),
            "fine_lat": float(coarse_lat),
            "fine_lon": float(coarse_lon),
            "coarse_cell": int(coarse_cell),
            "top_cells": list(zip(top_k_idx.tolist(), top_k_probs.tolist())),
            "confidence": float(top_k_probs[0]),
            "embedding": embedding[0],
        }

    # Search FAISS for nearest cluster among valid ones
    # Search enough to likely find a valid hit; fallback to brute-force if needed
    k_search = min(faiss_index.ntotal, max(len(valid_set) * 5, 500))
    distances, indices = faiss_index.search(
        np.ascontiguousarray(embedding, dtype=np.float32), k_search
    )

    # Find best match among valid clusters (FAISS returns sorted by distance)
    best_idx = -1
    for idx in indices[0]:
        if idx >= 0 and idx in valid_set:
            best_idx = idx
            break

    if best_idx == -1:
        # FAISS didn't return any valid cluster in top-k;
        # reconstruct valid embeddings and compute distances directly
        valid_embs = np.stack([
            faiss_index.reconstruct(int(vi)) for vi in valid_indices
        ])
        dists = np.linalg.norm(valid_embs - embedding.astype(np.float32), axis=1)
        best_idx = valid_indices[np.argmin(dists)]

    fine_lat, fine_lon = cluster_gps[best_idx]

    return {
        "coarse_lat": float(coarse_lat),
        "coarse_lon": float(coarse_lon),
        "fine_lat": float(fine_lat),
        "fine_lon": float(fine_lon),
        "coarse_cell": int(coarse_cell),
        "top_cells": list(zip(top_k_idx[:10].tolist(), top_k_probs[:10].tolist())),
        "confidence": float(top_k_probs[0]),
        "embedding": embedding[0],
    }


# ── Batch Evaluation ─────────────────────────────────────────────────

@torch.no_grad()
def evaluate_two_stage(
    model,
    val_loader,
    faiss_index: faiss.IndexFlatL2,
    cluster_gps: np.ndarray,
    cluster_cell_ids: np.ndarray,
    centroids: np.ndarray,
    device: torch.device,
    top_k_cells: int = 50,
) -> dict:
    """Evaluate two-stage pipeline on validation set.

    Returns metrics for both coarse (cell-only) and fine (FAISS) predictions.
    """
    model.eval()
    coarse_distances = []
    fine_distances = []
    ds = val_loader.dataset
    sample_idx = 0

    for images, smooth_targets, aux_labels in val_loader:
        images = images.to(device)
        batch_size = images.size(0)

        with torch.amp.autocast(device.type, dtype=torch.bfloat16, enabled=(device.type == "cuda")):
            outputs = model(images)

        geo_probs = F.softmax(outputs["geo"].float(), dim=-1).cpu().numpy()
        embeddings = outputs["embedding"].float().cpu().numpy()

        for i in range(batch_size):
            true_lat = ds.latitudes[sample_idx]
            true_lon = ds.longitudes[sample_idx]

            # Coarse: top-1 cell centroid
            coarse_cell = np.argmax(geo_probs[i])
            c_lat, c_lon = centroids[coarse_cell]
            coarse_dist = haversine_km(true_lat, true_lon, c_lat, c_lon)
            coarse_distances.append(coarse_dist)

            # Fine: FAISS retrieval within top-K cells
            top_cells_arr = np.argsort(-geo_probs[i])[:top_k_cells]
            valid_mask = np.isin(cluster_cell_ids, top_cells_arr)
            valid_indices = np.where(valid_mask)[0]
            valid_set = set(valid_indices.tolist())

            if len(valid_set) > 0:
                emb = np.ascontiguousarray(embeddings[i:i+1], dtype=np.float32)
                _, indices = faiss_index.search(emb, min(500, faiss_index.ntotal))

                fine_idx = -1
                for idx in indices[0]:
                    if idx >= 0 and idx in valid_set:
                        fine_idx = idx
                        break

                if fine_idx == -1:
                    # Fallback: brute-force nearest among valid clusters
                    valid_embs = np.stack([faiss_index.reconstruct(int(vi)) for vi in valid_indices])
                    dists = np.linalg.norm(valid_embs - emb, axis=1)
                    fine_idx = valid_indices[np.argmin(dists)]

                f_lat, f_lon = cluster_gps[fine_idx]
                fine_dist = haversine_km(true_lat, true_lon, f_lat, f_lon)
            else:
                fine_dist = coarse_dist

            fine_distances.append(fine_dist)
            sample_idx += 1

    coarse_distances = np.array(coarse_distances)
    fine_distances = np.array(fine_distances)

    def compute_metrics(distances, prefix):
        return {
            f"{prefix}_median_km": float(np.median(distances)),
            f"{prefix}_mean_km": float(np.mean(distances)),
            f"{prefix}_pct_25km": float((distances < 25).mean() * 100),
            f"{prefix}_pct_200km": float((distances < 200).mean() * 100),
            f"{prefix}_pct_750km": float((distances < 750).mean() * 100),
            f"{prefix}_pct_2500km": float((distances < 2500).mean() * 100),
        }

    metrics = {
        **compute_metrics(coarse_distances, "coarse"),
        **compute_metrics(fine_distances, "fine"),
    }
    return metrics


# ── Image Preprocessing ──────────────────────────────────────────────

def preprocess_image(image_path: str) -> torch.Tensor:
    """Load and preprocess a single image for inference."""
    transform = transforms.Compose([
        transforms.Resize(CLIP_SIZE, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop(CLIP_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(mean=CLIP_MEAN, std=CLIP_STD),
    ])
    image = Image.open(image_path).convert("RGB")
    return transform(image).unsqueeze(0)  # (1, 3, 336, 336)


# ══════════════════════════════════════════════════════════════════════
# V2: Dense k-NN FAISS Refinement
# ══════════════════════════════════════════════════════════════════════
#
# Key differences from v1 (clustered):
#   - Index ALL ~45K training embeddings, not OPTICS cluster centroids
#   - Weighted coarse: probability-weighted cell centroid average
#   - Dense k-NN: inverse-distance GPS weighting over k nearest neighbors
#   - Confidence gating: only use k-NN when coarse confidence is low
#   - Antimeridian-safe spherical averaging
# ══════════════════════════════════════════════════════════════════════


def weighted_centroid_spherical(
    lats: np.ndarray,
    lons: np.ndarray,
    weights: np.ndarray,
) -> tuple:
    """Compute weighted centroid on the sphere. Antimeridian-safe.

    Converts to Cartesian, averages with weights, converts back.

    Args:
        lats: (N,) latitudes in degrees
        lons: (N,) longitudes in degrees
        weights: (N,) non-negative weights (will be normalized)

    Returns:
        (lat, lon) in degrees
    """
    weights = np.asarray(weights, dtype=np.float64)
    w_sum = weights.sum()
    if w_sum < 1e-12:
        # Uniform weights fallback — still use Cartesian path for antimeridian safety
        weights = np.ones_like(weights) / len(weights)
    else:
        weights = weights / w_sum

    lat_r = np.radians(lats.astype(np.float64))
    lon_r = np.radians(lons.astype(np.float64))

    x = np.sum(weights * np.cos(lat_r) * np.cos(lon_r))
    y = np.sum(weights * np.cos(lat_r) * np.sin(lon_r))
    z = np.sum(weights * np.sin(lat_r))

    lon_out = np.degrees(np.arctan2(y, x))
    hyp = np.sqrt(x ** 2 + y ** 2)
    lat_out = np.degrees(np.arctan2(z, hyp))

    return float(lat_out), float(lon_out)


def build_dense_faiss_index(
    embeddings: np.ndarray,
    gps: np.ndarray,
    cell_ids: np.ndarray,
) -> dict:
    """Build FAISS IndexFlatIP over ALL training embeddings (dense, no clustering).

    Uses inner product (IP) on L2-normalized vectors = cosine similarity.

    Args:
        embeddings: (N, 768) L2-normalized float32
        gps: (N, 2) [lat, lon] for each training image
        cell_ids: (N,) geocell assignment per training image

    Returns:
        dict with keys:
            index: faiss.IndexFlatIP
            gps: (N, 2) GPS array
            cell_ids: (N,) cell IDs
            num_embeddings: int
    """
    embeddings = np.ascontiguousarray(embeddings, dtype=np.float32)
    gps = np.asarray(gps, dtype=np.float64)
    cell_ids = np.asarray(cell_ids, dtype=np.int64)

    assert embeddings.shape[0] == gps.shape[0] == cell_ids.shape[0], \
        f"Length mismatch: emb={embeddings.shape[0]}, gps={gps.shape[0]}, cells={cell_ids.shape[0]}"

    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)

    n = index.ntotal
    mem_mb = n * dim * 4 / (1024 * 1024)
    print(f"  Dense FAISS index: {n:,} vectors, dim={dim}, ~{mem_mb:.1f} MB")

    return {
        "index": index,
        "gps": gps,
        "cell_ids": cell_ids,
        "num_embeddings": n,
    }


def predict_weighted_coarse(
    geo_probs: np.ndarray,
    centroids: np.ndarray,
    top_n: int = 10,
) -> tuple:
    """Probability-weighted cell centroid average (spherical).

    Args:
        geo_probs: (num_cells,) softmax probabilities
        centroids: (num_cells, 2) cell centroids [lat, lon]
        top_n: number of top cells to average over

    Returns:
        (lat, lon, confidence) — confidence is top-1 probability
    """
    top_idx = np.argsort(-geo_probs)[:top_n]
    top_probs = geo_probs[top_idx]
    top_lats = centroids[top_idx, 0]
    top_lons = centroids[top_idx, 1]

    lat, lon = weighted_centroid_spherical(top_lats, top_lons, top_probs)
    confidence = float(geo_probs[top_idx[0]])

    return lat, lon, confidence


def predict_dense_knn(
    embedding: np.ndarray,
    dense_index: faiss.IndexFlatIP,
    gps: np.ndarray,
    cell_ids: np.ndarray,
    top_cells: np.ndarray,
    k: int = 20,
    geo_radius: float = 0.0,
) -> tuple:
    """Dense k-NN retrieval with cell filtering and inverse-distance GPS weighting.

    1. Search FAISS for k*5 nearest neighbors (IP = cosine similarity)
    2. Filter to neighbors in top_cells
    3. Take top-k filtered neighbors
    4. Optionally filter by geo_radius (km from neighbor group center)
    5. Inverse-distance weighted GPS average (spherical)

    Args:
        embedding: (1, 768) query embedding, L2-normalized
        dense_index: FAISS IndexFlatIP with all training embeddings
        gps: (N, 2) GPS for all training images
        cell_ids: (N,) cell ID for each training image
        top_cells: array of cell IDs to restrict search to
        k: number of nearest neighbors to use for GPS averaging
        geo_radius: geographic radius filter in km (0=disabled)

    Returns:
        (lat, lon, n_neighbors, mean_nn_dist_km)
        - lat, lon: predicted GPS
        - n_neighbors: how many neighbors were used
        - mean_nn_dist_km: mean haversine distance among used neighbors (spread indicator)
    """
    emb = np.ascontiguousarray(embedding, dtype=np.float32)
    if emb.ndim == 1:
        emb = emb.reshape(1, -1)

    # Search for more than k to allow for cell filtering
    k_search = min(dense_index.ntotal, k * 5)
    similarities, indices = dense_index.search(emb, k_search)
    sims = similarities[0]
    idxs = indices[0]

    # Filter to top cells
    top_cell_set = set(top_cells.tolist())
    valid = [(s, i) for s, i in zip(sims, idxs) if i >= 0 and int(cell_ids[i]) in top_cell_set]

    if len(valid) == 0:
        # No neighbors in top cells — search unrestricted
        valid = [(s, i) for s, i in zip(sims, idxs) if i >= 0]

    if len(valid) == 0:
        return 0.0, 0.0, 0, float("inf")

    # Take top-k by similarity
    valid = valid[:k]

    nn_sims = np.array([v[0] for v in valid])
    nn_idxs = np.array([v[1] for v in valid], dtype=int)
    nn_lats = gps[nn_idxs, 0]
    nn_lons = gps[nn_idxs, 1]

    # Compute similarity-based weights (used for both geo center and final average)
    # similarity in [-1, 1] for normalized vectors; map to [0, 1] then square
    weights = ((nn_sims + 1.0) / 2.0) ** 2

    # Geo radius filter (optional): remove neighbors too far from group center
    if geo_radius > 0 and len(nn_idxs) > 3:
        center_lat, center_lon = weighted_centroid_spherical(nn_lats, nn_lons, weights)
        keep = []
        for j in range(len(nn_idxs)):
            d = haversine_km(center_lat, center_lon, nn_lats[j], nn_lons[j])
            if d <= geo_radius:
                keep.append(j)
        if len(keep) >= 3:
            nn_sims = nn_sims[keep]
            nn_idxs = nn_idxs[keep]
            nn_lats = nn_lats[keep]
            nn_lons = nn_lons[keep]
            weights = weights[keep]

    lat, lon = weighted_centroid_spherical(nn_lats, nn_lons, weights)

    # Compute spread: mean pairwise haversine among neighbors
    if len(nn_idxs) > 1:
        dists = [haversine_km(nn_lats[0], nn_lons[0], nn_lats[j], nn_lons[j])
                 for j in range(1, len(nn_idxs))]
        mean_nn_dist = float(np.mean(dists))
    else:
        mean_nn_dist = 0.0

    return lat, lon, len(nn_idxs), mean_nn_dist


def predict_two_stage_v2(
    model,
    image: torch.Tensor,
    dense_data: dict,
    centroids: np.ndarray,
    device: torch.device,
    top_n_coarse: int = 10,
    top_k_cells: int = 50,
    k_neighbors: int = 20,
    confidence_threshold: float = 0.3,
    geo_radius: float = 0.0,
) -> dict:
    """V2 two-stage prediction: weighted coarse + gated dense k-NN.

    Decision logic:
    - Always compute weighted coarse (probability-weighted cell centroids)
    - If top-1 confidence >= confidence_threshold: use weighted coarse (model is sure)
    - Else: use dense k-NN refinement (model is uncertain, let neighbors vote)

    Args:
        model: GeoLocator model
        image: (1, 3, 336, 336) preprocessed image tensor
        dense_data: dict from build_dense_faiss_index() with index, gps, cell_ids
        centroids: (num_cells, 2) geocell centroids
        device: torch device
        top_n_coarse: cells for weighted coarse average
        top_k_cells: cells for k-NN search scope
        k_neighbors: k for dense k-NN
        confidence_threshold: above this, skip k-NN (model confident enough)
        geo_radius: geographic radius filter for k-NN neighbors (0=disabled)

    Returns:
        dict with:
            top1_lat, top1_lon: raw top-1 cell centroid
            weighted_lat, weighted_lon: probability-weighted coarse
            knn_lat, knn_lon: dense k-NN prediction (always computed)
            final_lat, final_lon: gated output (weighted or knn based on confidence)
            confidence: top-1 softmax probability
            used_knn: bool — whether final prediction used k-NN
            n_neighbors: number of k-NN neighbors used
            nn_spread_km: mean distance among neighbors
            embedding: (768,) image embedding
    """
    model.eval()
    image = image.to(device)

    with torch.no_grad():
        with torch.amp.autocast(device.type, dtype=torch.bfloat16, enabled=(device.type == "cuda")):
            outputs = model(image)

    geo_probs = F.softmax(outputs["geo"].float(), dim=-1).cpu().numpy()[0]
    embedding = outputs["embedding"].float().cpu().numpy()  # (1, 768)

    # Top-1 coarse
    top1_cell = np.argmax(geo_probs)
    top1_lat, top1_lon = centroids[top1_cell]

    # Weighted coarse
    w_lat, w_lon, confidence = predict_weighted_coarse(geo_probs, centroids, top_n_coarse)

    # Dense k-NN (always compute for comparison, even if gated out)
    top_cells = np.argsort(-geo_probs)[:top_k_cells]
    knn_lat, knn_lon, n_neighbors, nn_spread = predict_dense_knn(
        embedding, dense_data["index"], dense_data["gps"],
        dense_data["cell_ids"], top_cells,
        k=k_neighbors, geo_radius=geo_radius,
    )

    # Gating: use top-1 coarse when confident (preserves fine precision),
    # k-NN when uncertain (improves median via neighbor voting).
    # Note: weighted coarse was empirically worse than top-1 on Im2GPS3k
    # (783km vs 672km) because averaging dilutes accurate single-cell predictions.
    used_knn = confidence < confidence_threshold and n_neighbors >= 3
    if used_knn:
        final_lat, final_lon = knn_lat, knn_lon
    else:
        final_lat, final_lon = float(top1_lat), float(top1_lon)

    return {
        "top1_lat": float(top1_lat),
        "top1_lon": float(top1_lon),
        "weighted_lat": float(w_lat),
        "weighted_lon": float(w_lon),
        "knn_lat": float(knn_lat),
        "knn_lon": float(knn_lon),
        "final_lat": float(final_lat),
        "final_lon": float(final_lon),
        "confidence": float(confidence),
        "used_knn": used_knn,
        "n_neighbors": n_neighbors,
        "nn_spread_km": nn_spread,
        "top1_cell": int(top1_cell),
        "embedding": embedding[0],
    }


@torch.no_grad()
def evaluate_two_stage_v2(
    model,
    dataloader,
    dense_data: dict,
    centroids: np.ndarray,
    device: torch.device,
    top_n_coarse: int = 10,
    top_k_cells: int = 50,
    k_neighbors: int = 20,
    confidence_threshold: float = 0.3,
    geo_radius: float = 0.0,
    gps_source: str = "dataset",
) -> dict:
    """Evaluate v2 two-stage pipeline. Reports 4-tier metrics.

    Tiers:
        1. top1: raw top-1 cell centroid
        2. weighted: probability-weighted coarse
        3. knn_raw: dense k-NN (always applied, ignoring gate)
        4. gated: confidence-gated (weighted when confident, k-NN when not)

    Args:
        model: GeoLocator model
        dataloader: benchmark or val DataLoader (must have shuffle=False)
        dense_data: dict from build_dense_faiss_index()
        centroids: (num_cells, 2) geocell centroids
        device: torch device
        top_n_coarse: cells for weighted coarse
        top_k_cells: cells for k-NN scope
        k_neighbors: k for dense k-NN
        confidence_threshold: gating threshold
        geo_radius: geographic radius filter (0=disabled)
        gps_source: "dataset" for val loader (uses ds.latitudes), "batch" for benchmark (yields lat/lon)

    Returns:
        dict with 4-tier metrics + gating stats
    """
    model.eval()
    index = dense_data["index"]
    all_gps = dense_data["gps"]
    all_cell_ids = dense_data["cell_ids"]

    top1_dists = []
    weighted_dists = []
    knn_dists = []
    gated_dists = []
    n_used_knn = 0
    n_total = 0

    if gps_source == "dataset":
        ds = dataloader.dataset
        sample_idx = 0

    for batch in dataloader:
        if gps_source == "dataset":
            images, _, _ = batch
        else:
            images, batch_lats, batch_lons = batch

        images = images.to(device)
        batch_size = images.size(0)

        with torch.amp.autocast(device.type, dtype=torch.bfloat16, enabled=(device.type == "cuda")):
            outputs = model(images)

        geo_probs = F.softmax(outputs["geo"].float(), dim=-1).cpu().numpy()
        embeddings = outputs["embedding"].float().cpu().numpy()

        for i in range(batch_size):
            if gps_source == "dataset":
                true_lat = ds.latitudes[sample_idx]
                true_lon = ds.longitudes[sample_idx]
                sample_idx += 1
            else:
                true_lat = batch_lats[i].item()
                true_lon = batch_lons[i].item()

            probs = geo_probs[i]
            emb = embeddings[i:i + 1]

            # Tier 1: Top-1 coarse
            top1_cell = np.argmax(probs)
            t1_lat, t1_lon = centroids[top1_cell]
            top1_dists.append(haversine_km(true_lat, true_lon, t1_lat, t1_lon))

            # Tier 2: Weighted coarse
            w_lat, w_lon, confidence = predict_weighted_coarse(probs, centroids, top_n_coarse)
            weighted_dists.append(haversine_km(true_lat, true_lon, w_lat, w_lon))

            # Tier 3: Dense k-NN (always)
            top_cells = np.argsort(-probs)[:top_k_cells]
            knn_lat, knn_lon, n_neighbors, _ = predict_dense_knn(
                emb, index, all_gps, all_cell_ids, top_cells,
                k=k_neighbors, geo_radius=geo_radius,
            )
            knn_dists.append(haversine_km(true_lat, true_lon, knn_lat, knn_lon))

            # Tier 4: Gated (top-1 when confident, k-NN when uncertain)
            used_knn = confidence < confidence_threshold and n_neighbors >= 3
            if used_knn:
                gated_dists.append(knn_dists[-1])
                n_used_knn += 1
            else:
                gated_dists.append(top1_dists[-1])

            n_total += 1

        if n_total % 500 == 0 and n_total > 0:
            print(f"  Evaluated {n_total} images...")

    def _metrics(distances, prefix):
        d = np.array(distances)
        return {
            f"{prefix}_median_km": float(np.median(d)),
            f"{prefix}_mean_km": float(np.mean(d)),
            f"{prefix}_pct_25km": float((d < 25).mean() * 100),
            f"{prefix}_pct_200km": float((d < 200).mean() * 100),
            f"{prefix}_pct_750km": float((d < 750).mean() * 100),
            f"{prefix}_pct_2500km": float((d < 2500).mean() * 100),
        }

    metrics = {
        **_metrics(top1_dists, "top1"),
        **_metrics(weighted_dists, "weighted"),
        **_metrics(knn_dists, "knn_raw"),
        **_metrics(gated_dists, "gated"),
        "num_images": n_total,
        "n_used_knn": n_used_knn,
        "knn_gate_rate": float(n_used_knn / max(n_total, 1) * 100),
        "confidence_threshold": confidence_threshold,
    }
    return metrics


@torch.no_grad()
def calibrate_thresholds(
    model,
    dataloader,
    dense_data: dict,
    centroids: np.ndarray,
    device: torch.device,
    top_n_coarse: int = 10,
    top_k_cells: int = 50,
    k_neighbors: int = 20,
    gps_source: str = "dataset",
) -> dict:
    """Sweep confidence_threshold x geo_radius to find best gating params.

    Tests a grid of:
      - confidence_threshold: [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]
      - geo_radius: [0, 200, 500, 1000] km

    Returns best params and full results grid.
    """
    # First, collect per-image predictions for all tiers (one forward pass)
    model.eval()
    index = dense_data["index"]
    all_gps = dense_data["gps"]
    all_cell_ids = dense_data["cell_ids"]

    records = []  # list of dicts per image

    if gps_source == "dataset":
        ds = dataloader.dataset
        sample_idx = 0

    print("  Collecting predictions for calibration...")
    for batch in dataloader:
        if gps_source == "dataset":
            images, _, _ = batch
        else:
            images, batch_lats, batch_lons = batch

        images = images.to(device)
        batch_size = images.size(0)

        with torch.amp.autocast(device.type, dtype=torch.bfloat16, enabled=(device.type == "cuda")):
            outputs = model(images)

        geo_probs = F.softmax(outputs["geo"].float(), dim=-1).cpu().numpy()
        embeddings_batch = outputs["embedding"].float().cpu().numpy()

        for i in range(batch_size):
            if gps_source == "dataset":
                true_lat = ds.latitudes[sample_idx]
                true_lon = ds.longitudes[sample_idx]
                sample_idx += 1
            else:
                true_lat = batch_lats[i].item()
                true_lon = batch_lons[i].item()

            probs = geo_probs[i]
            emb = embeddings_batch[i:i + 1]

            # Top-1 coarse
            top1_cell = np.argmax(probs)
            t1_lat, t1_lon = centroids[top1_cell]
            top1_dist = haversine_km(true_lat, true_lon, t1_lat, t1_lon)

            # Weighted coarse (for comparison)
            _, _, confidence = predict_weighted_coarse(probs, centroids, top_n_coarse)

            # Dense k-NN for each geo_radius
            top_cells = np.argsort(-probs)[:top_k_cells]
            knn_results = {}
            for gr in [0, 200, 500, 1000]:
                knn_lat, knn_lon, n_nb, _ = predict_dense_knn(
                    emb, index, all_gps, all_cell_ids, top_cells,
                    k=k_neighbors, geo_radius=float(gr),
                )
                knn_dist = haversine_km(true_lat, true_lon, knn_lat, knn_lon)
                knn_results[gr] = {"dist": knn_dist, "n_nb": n_nb}

            records.append({
                "confidence": confidence,
                "top1_dist": top1_dist,
                "knn_results": knn_results,
            })

    # Now sweep thresholds
    conf_thresholds = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]
    geo_radii = [0, 200, 500, 1000]
    grid = []

    best_median = float("inf")
    best_params = {"confidence_threshold": 0.3, "geo_radius": 0}

    for ct in conf_thresholds:
        for gr in geo_radii:
            gated_dists = []
            n_knn = 0
            for rec in records:
                knn_info = rec["knn_results"][gr]
                used_knn = rec["confidence"] < ct and knn_info["n_nb"] >= 3
                if used_knn:
                    gated_dists.append(knn_info["dist"])
                    n_knn += 1
                else:
                    gated_dists.append(rec["top1_dist"])

            gated_arr = np.array(gated_dists)
            med = float(np.median(gated_arr))
            entry = {
                "confidence_threshold": ct,
                "geo_radius": gr,
                "median_km": med,
                "pct_25km": float((gated_arr < 25).mean() * 100),
                "pct_750km": float((gated_arr < 750).mean() * 100),
                "knn_rate": float(n_knn / max(len(records), 1) * 100),
            }
            grid.append(entry)

            if med < best_median:
                best_median = med
                best_params = {"confidence_threshold": ct, "geo_radius": gr}

    print(f"  Calibration complete: {len(records)} images, {len(grid)} combos tested")
    print(f"  Best: conf_thresh={best_params['confidence_threshold']}, "
          f"geo_radius={best_params['geo_radius']}km, median={best_median:.1f}km")

    return {
        "best_params": best_params,
        "best_median_km": best_median,
        "grid": grid,
        "num_images": len(records),
    }


# ── Dense Index Save/Load ────────────────────────────────────────────

def save_dense_faiss_index(
    dense_data: dict,
    output_dir: str,
):
    """Save dense FAISS index and per-image metadata."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    faiss.write_index(dense_data["index"], str(out / "faiss_dense.index"))
    np.save(out / "dense_gps.npy", dense_data["gps"])
    np.save(out / "dense_cell_ids.npy", dense_data["cell_ids"])

    config = {
        "type": "dense",
        "num_embeddings": dense_data["num_embeddings"],
        "embed_dim": dense_data["index"].d,
    }
    with open(out / "dense_config.json", "w") as f:
        json.dump(config, f, indent=2)

    print(f"  Saved dense FAISS index to {out} ({dense_data['num_embeddings']:,} vectors)")


def load_dense_faiss_index(index_dir: str) -> dict:
    """Load dense FAISS index and per-image metadata.

    Returns:
        dict with index, gps, cell_ids, num_embeddings
    """
    p = Path(index_dir)
    index = faiss.read_index(str(p / "faiss_dense.index"))
    gps = np.load(p / "dense_gps.npy")
    cell_ids = np.load(p / "dense_cell_ids.npy")

    print(f"  Loaded dense FAISS index: {index.ntotal:,} vectors")
    return {
        "index": index,
        "gps": gps,
        "cell_ids": cell_ids,
        "num_embeddings": index.ntotal,
    }
