"""
End-to-end geolocation inference pipeline.

Modes:
  - coarse: Top-1 cell centroid only (no FAISS needed)
  - dense:  Weighted coarse + gated dense k-NN (FAISS v2)
  - sparse: Legacy sparse FAISS pipeline

Usage:
    # Coarse only (works with any checkpoint + geocell config):
    python -m GEOPROJECT.geoguessr.inference \
        --image path/to/photo.jpg --mode coarse

    # Coarse with map output:
    python -m GEOPROJECT.geoguessr.inference \
        --image path/to/photo.jpg --mode coarse --map

    # Dense FAISS (default):
    python -m GEOPROJECT.geoguessr.inference \
        --image path/to/photo.jpg --mode dense
"""

import argparse
import html
import json
import platform
import subprocess
import sys
import webbrowser
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from GEOPROJECT.geoguessr.data.geocells import haversine_km
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


def predict_coarse(
    image_path: str,
    model: GeoLocator,
    centroids: np.ndarray,
    device: torch.device,
) -> dict:
    """Predict location using top-1 cell centroid only. No FAISS needed."""
    from GEOPROJECT.geoguessr.model.faiss_refinement import preprocess_image

    image = preprocess_image(image_path).to(device)

    with torch.no_grad(), torch.amp.autocast(device.type, dtype=torch.bfloat16):
        outputs = model(image)

    probs = F.softmax(outputs["geo"].float(), dim=-1).squeeze(0).cpu().numpy()
    top_cell = int(np.argmax(probs))
    confidence = float(probs[top_cell])

    lat, lon = float(centroids[top_cell, 0]), float(centroids[top_cell, 1])

    # Top-5 cells for context
    top5_idx = np.argsort(probs)[-5:][::-1]
    top5 = [(int(i), float(probs[i]), float(centroids[i, 0]), float(centroids[i, 1])) for i in top5_idx]

    return {
        "lat": lat,
        "lon": lon,
        "confidence": confidence,
        "cell": top_cell,
        "top5": top5,
    }


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
    from GEOPROJECT.geoguessr.model.faiss_refinement import preprocess_image, predict_two_stage_v2

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
    from GEOPROJECT.geoguessr.model.faiss_refinement import preprocess_image, predict_two_stage

    image = preprocess_image(image_path).to(device)
    return predict_two_stage(
        model, image, faiss_index, cluster_gps,
        cluster_cell_ids, centroids, device, top_k_cells,
    )


def generate_map(lat: float, lon: float, confidence: float, image_path: str, output_path: str):
    """Generate an interactive HTML map with a pin at the predicted location."""
    import folium

    m = folium.Map(
        location=[lat, lon],
        zoom_start=6,
        tiles="OpenStreetMap",
    )

    # Add satellite layer as an option
    folium.TileLayer(
        tiles="https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
        attr="Esri",
        name="Satellite",
    ).add_to(m)

    folium.LayerControl().add_to(m)

    # Pin with popup
    safe_name = html.escape(Path(image_path).name)
    popup_html = (
        f"<b>GeoLocator Prediction</b><br>"
        f"Lat: {lat:.4f}, Lon: {lon:.4f}<br>"
        f"Confidence: {confidence*100:.1f}%<br>"
        f"Image: {safe_name}"
    )
    folium.Marker(
        location=[lat, lon],
        popup=folium.Popup(popup_html, max_width=300),
        tooltip=f"{lat:.2f}, {lon:.2f} ({confidence*100:.0f}%)",
        icon=folium.Icon(color="red", icon="info-sign"),
    ).add_to(m)

    m.save(output_path)
    return output_path


def main():
    if sys.platform == "win32":
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    parser = argparse.ArgumentParser(description="Geolocation inference")
    parser.add_argument("--image", type=str, required=True, help="Path to input image")
    parser.add_argument("--checkpoint", type=str, default="GEOPROJECT/checkpoints/geolocator_best.pt")
    parser.add_argument("--faiss-dir", type=str, default="GEOPROJECT/faiss_index")
    parser.add_argument("--geocell-dir", type=str, default="GEOPROJECT/data/osv5m_50k/semantic_cells")
    parser.add_argument("--mode", type=str, default="coarse", choices=["coarse", "dense", "sparse"],
                        help="coarse = top-1 cell only, dense = FAISS v2, sparse = legacy FAISS")
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--confidence-threshold", type=float, default=0.3,
                        help="[dense only] Confidence threshold for k-NN gating")
    parser.add_argument("--map", action="store_true", help="Generate HTML map and open in browser")
    parser.add_argument("--map-output", type=str, default=None,
                        help="Output path for map HTML (default: prediction_map.html)")
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

    # Predict based on mode
    if args.mode == "coarse":
        result = predict_coarse(args.image, model, centroids, device)

        print(f"\nPrediction for: {args.image}")
        print(f"  Location: ({result['lat']:.4f}, {result['lon']:.4f})")
        print(f"  Confidence: {result['confidence']*100:.1f}%")
        print(f"  Cell: #{result['cell']}")
        print(f"\n  Top-5 cells:")
        for cell_id, prob, clat, clon in result["top5"]:
            print(f"    #{cell_id}: ({clat:.2f}, {clon:.2f}) — {prob*100:.1f}%")

        pred_lat, pred_lon, pred_conf = result["lat"], result["lon"], result["confidence"]

    elif args.mode == "dense":
        from GEOPROJECT.geoguessr.model.faiss_refinement import load_dense_faiss_index

        dense_data = load_dense_faiss_index(args.faiss_dir)
        result = predict_v2(
            args.image, model, dense_data, centroids, device,
            top_k_cells=args.top_k,
            confidence_threshold=args.confidence_threshold,
        )

        print(f"\nPrediction for: {args.image}")
        print(f"  Top-1 cell centroid:  ({result['top1_lat']:.4f}, {result['top1_lon']:.4f})")
        print(f"  Weighted coarse:      ({result['weighted_lat']:.4f}, {result['weighted_lon']:.4f})")
        print(f"  Dense k-NN:           ({result['knn_lat']:.4f}, {result['knn_lon']:.4f})")
        print(f"  Final (gated):        ({result['final_lat']:.4f}, {result['final_lon']:.4f})")
        print(f"  Confidence: {result['confidence']*100:.1f}%")
        print(f"  Used k-NN: {result['used_knn']} ({result['n_neighbors']} neighbors, "
              f"spread={result['nn_spread_km']:.0f}km)")

        top1_to_final = haversine_km(
            result['top1_lat'], result['top1_lon'],
            result['final_lat'], result['final_lon'],
        )
        print(f"  Top-1 -> Final refinement: {top1_to_final:.1f} km")

        pred_lat, pred_lon = result["final_lat"], result["final_lon"]
        pred_conf = result["confidence"]

    else:
        from GEOPROJECT.geoguessr.model.faiss_refinement import load_faiss_index

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

        pred_lat, pred_lon = result["fine_lat"], result["fine_lon"]
        pred_conf = result["confidence"]

    # Generate map if requested
    if args.map:
        map_path = args.map_output or "prediction_map.html"
        map_path = str(Path(map_path).resolve())
        generate_map(pred_lat, pred_lon, pred_conf, args.image, map_path)
        print(f"\n  Map saved to: {map_path}")

        # Open in browser — handle WSL where webbrowser can't resolve Linux paths
        is_wsl = "microsoft" in platform.uname().release.lower()
        if is_wsl:
            # Convert to Windows path and open with Windows browser
            try:
                win_path = subprocess.check_output(
                    ["wslpath", "-w", map_path], text=True
                ).strip()
                subprocess.Popen(["cmd.exe", "/c", "start", "", win_path],
                                 stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            except (FileNotFoundError, subprocess.CalledProcessError):
                print(f"  Could not open browser automatically. Open manually: {map_path}")
        else:
            webbrowser.open(f"file://{map_path}")


if __name__ == "__main__":
    main()
