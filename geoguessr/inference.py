"""
End-to-end geolocation inference pipeline.

Combines all stages:
  1. Load trained GeoLocator model
  2. Load FAISS index (dense v2 or legacy sparse) + geocell centroids
  3. Two-stage prediction: weighted coarse + gated dense k-NN

Usage:
    # V2 dense (default):
    python -m GEOPROJECT.geoguessr.inference \
        --image path/to/photo.jpg

    # Legacy sparse:
    python -m GEOPROJECT.geoguessr.inference \
        --image path/to/photo.jpg --mode sparse
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

from GEOPROJECT.geoguessr.data.geocells import haversine_km
from GEOPROJECT.geoguessr.model.faiss_refinement import (
    load_dense_faiss_index,
    load_faiss_index,
    predict_two_stage,
    predict_two_stage_v2,
    preprocess_image,
)
from GEOPROJECT.geoguessr.model.geolocator import GeoLocator


def load_inference_model(checkpoint_path: str, num_cells: int, device: torch.device) -> GeoLocator:
    """Load trained GeoLocator for inference."""
    model = GeoLocator(num_cells=num_cells).to(device)

    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    if "num_cells" in ckpt and ckpt["num_cells"] != num_cells:
        raise ValueError(
            f"Checkpoint num_cells ({ckpt['num_cells']}) != geocell config num_cells ({num_cells}). "
            f"Make sure the checkpoint matches the geocell directory."
        )
    missing, unexpected = model.load_state_dict(ckpt["model_state_dict"], strict=False)
    print(f"Loaded model from {checkpoint_path}")
    if missing:
        print(f"  Missing keys: {len(missing)} (expected for auxiliary heads if not saved)")
    if unexpected:
        print(f"  Unexpected keys: {len(unexpected)}")
    if "metrics" in ckpt and ckpt["metrics"]:
        m = ckpt["metrics"]
        print(f"  Training metrics: median={m.get('median_km', '?')}km, "
              f"<25km={m.get('pct_25km', '?')}%")

    model.eval()
    return model


def predict_v2(
    image_path: str,
    model: GeoLocator,
    dense_data: dict,
    centroids: np.ndarray,
    device: torch.device,
    top_k_cells: int = 50,
    confidence_threshold: float = 0.3,
    geo_radius: float = 0.0,
) -> dict:
    """Predict location from a single image using v2 pipeline."""
    image = preprocess_image(image_path).to(device)
    return predict_two_stage_v2(
        model, image, dense_data, centroids, device,
        top_k_cells=top_k_cells,
        confidence_threshold=confidence_threshold,
        geo_radius=geo_radius,
    )


def predict_legacy(
    image_path: str,
    model: GeoLocator,
    faiss_index,
    cluster_gps: np.ndarray,
    cluster_cell_ids: np.ndarray,
    centroids: np.ndarray,
    device: torch.device,
    top_k_cells: int = 50,
) -> dict:
    """Predict location using legacy sparse pipeline."""
    image = preprocess_image(image_path).to(device)
    return predict_two_stage(
        model, image, faiss_index, cluster_gps,
        cluster_cell_ids, centroids, device, top_k_cells,
    )


def main():
    if sys.platform == "win32":
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    parser = argparse.ArgumentParser(description="Geolocation inference")
    parser.add_argument("--image", type=str, required=True, help="Path to input image")
    parser.add_argument("--checkpoint", type=str, default="GEOPROJECT/checkpoints/geolocator_best.pt")
    parser.add_argument("--faiss-dir", type=str, default="GEOPROJECT/faiss_index")
    parser.add_argument("--geocell-dir", type=str, default="GEOPROJECT/data/osv5m_50k/semantic_cells")
    parser.add_argument("--mode", type=str, default="dense", choices=["dense", "sparse"],
                        help="Index mode: dense (v2) or sparse (legacy)")
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--confidence-threshold", type=float, default=0.3,
                        help="[dense only] Confidence threshold for k-NN gating")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load geocell config
    geocell_path = Path(args.geocell_dir)
    with open(geocell_path / "geocell_config.json") as f:
        geocell_config = json.load(f)
    num_cells = geocell_config["num_cells"]
    centroids = np.load(geocell_path / "cell_centroids.npy")

    # Load model
    model = load_inference_model(args.checkpoint, num_cells, device)

    if args.mode == "dense":
        # V2: Dense FAISS index
        dense_data = load_dense_faiss_index(args.faiss_dir)
        result = predict_v2(
            args.image, model, dense_data, centroids, device,
            top_k_cells=args.top_k,
            confidence_threshold=args.confidence_threshold,
        )

        # Display v2 results
        print(f"\nPrediction for: {args.image}")
        print(f"  Top-1 cell centroid:  ({result['top1_lat']:.4f}, {result['top1_lon']:.4f})")
        print(f"  Weighted coarse:      ({result['weighted_lat']:.4f}, {result['weighted_lon']:.4f})")
        print(f"  Dense k-NN:           ({result['knn_lat']:.4f}, {result['knn_lon']:.4f})")
        print(f"  Final (gated):        ({result['final_lat']:.4f}, {result['final_lon']:.4f})")
        print(f"  Confidence: {result['confidence']*100:.1f}%")
        print(f"  Used k-NN: {result['used_knn']} ({result['n_neighbors']} neighbors, "
              f"spread={result['nn_spread_km']:.0f}km)")

        # Distances between predictions
        top1_to_final = haversine_km(
            result['top1_lat'], result['top1_lon'],
            result['final_lat'], result['final_lon'],
        )
        print(f"  Top-1 -> Final refinement: {top1_to_final:.1f} km")

    else:
        # Legacy sparse
        faiss_index, cluster_gps, cluster_cell_ids = load_faiss_index(args.faiss_dir)
        result = predict_legacy(
            args.image, model, faiss_index, cluster_gps,
            cluster_cell_ids, centroids, device, args.top_k,
        )

        print(f"\nPrediction for: {args.image}")
        print(f"  Coarse (cell centroid): ({result['coarse_lat']:.4f}, {result['coarse_lon']:.4f})")
        print(f"  Fine (FAISS retrieval): ({result['fine_lat']:.4f}, {result['fine_lon']:.4f})")
        print(f"  Confidence: {result['confidence']*100:.1f}%")
        print(f"  Top cell: #{result['coarse_cell']}")

        dist = haversine_km(
            result['coarse_lat'], result['coarse_lon'],
            result['fine_lat'], result['fine_lon'],
        )
        print(f"  Coarse -> Fine refinement: {dist:.1f} km")


if __name__ == "__main__":
    main()
