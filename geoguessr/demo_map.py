"""
Generate demo map with predicted pin, actual pin, and distance line.

Usage:
    python -m GEOPROJECT.geoguessr.demo_map \
        --image path/to/image.jpg \
        --actual-lat 13.3525 --actual-lon 74.7934 \
        --output prediction_map.html
"""

import argparse
import json
import math
from pathlib import Path

import folium
import numpy as np
import torch
import torch.nn.functional as F


def haversine_km(lat1, lon1, lat2, lon2):
    R = 6371.0
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlam = math.radians(lon2 - lon1)
    a = math.sin(dphi/2)**2 + math.cos(phi1)*math.cos(phi2)*math.sin(dlam/2)**2
    return R * 2 * math.asin(math.sqrt(a))


def run_inference(image_path, checkpoint, geocell_dir, device):
    from GEOPROJECT.geoguessr.model.geolocator import GeoLocator
    from GEOPROJECT.geoguessr.model.faiss_refinement import preprocess_image

    geocell_path = Path(geocell_dir)
    with open(geocell_path / "geocell_config.json") as f:
        config = json.load(f)
    num_cells = config["num_cells"]
    centroids = np.load(geocell_path / "cell_centroids.npy")

    model = GeoLocator(num_cells=num_cells).to(device)
    ckpt = torch.load(checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"], strict=False)
    model.eval()

    image = preprocess_image(image_path).to(device)
    with torch.no_grad(), torch.amp.autocast(device.type, dtype=torch.bfloat16):
        outputs = model(image)

    probs = F.softmax(outputs["geo"].float(), dim=-1).squeeze(0).cpu().numpy()
    top_cell = int(np.argmax(probs))
    confidence = float(probs[top_cell])
    lat, lon = float(centroids[top_cell, 0]), float(centroids[top_cell, 1])

    return lat, lon, confidence


def generate_demo_map(pred_lat, pred_lon, pred_conf, actual_lat, actual_lon, image_name, output_path):
    dist_km = haversine_km(pred_lat, pred_lon, actual_lat, actual_lon)

    # Centre map between the two points
    centre_lat = (pred_lat + actual_lat) / 2
    centre_lon = (pred_lon + actual_lon) / 2

    m = folium.Map(location=[centre_lat, centre_lon], zoom_start=5, tiles="OpenStreetMap")

    folium.TileLayer(
        tiles="https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
        attr="Esri", name="Satellite",
    ).add_to(m)
    folium.LayerControl().add_to(m)

    # Predicted pin — red
    folium.Marker(
        location=[pred_lat, pred_lon],
        popup=folium.Popup(
            f"<b>🤖 Model Prediction</b><br>"
            f"({pred_lat:.4f}, {pred_lon:.4f})<br>"
            f"Confidence: {pred_conf*100:.1f}%<br>"
            f"Image: {image_name}",
            max_width=280,
        ),
        tooltip=f"Predicted: ({pred_lat:.2f}, {pred_lon:.2f})",
        icon=folium.Icon(color="red", icon="screenshot", prefix="glyphicon"),
    ).add_to(m)

    # Actual pin — green
    folium.Marker(
        location=[actual_lat, actual_lon],
        popup=folium.Popup(
            f"<b>📍 Actual Location</b><br>"
            f"({actual_lat:.4f}, {actual_lon:.4f})<br>"
            f"Manipal, Karnataka, India",
            max_width=280,
        ),
        tooltip=f"Actual: ({actual_lat:.2f}, {actual_lon:.2f})",
        icon=folium.Icon(color="green", icon="map-marker", prefix="glyphicon"),
    ).add_to(m)

    # Line between prediction and actual
    folium.PolyLine(
        locations=[[pred_lat, pred_lon], [actual_lat, actual_lon]],
        color="#FF4444",
        weight=2.5,
        dash_array="8 6",
        tooltip=f"Error: {dist_km:.0f} km",
    ).add_to(m)

    # Distance label at midpoint
    folium.Marker(
        location=[centre_lat, centre_lon],
        icon=folium.DivIcon(
            html=f'<div style="background:rgba(0,0,0,0.65);color:white;padding:4px 8px;'
                 f'border-radius:4px;font-size:13px;font-weight:bold;white-space:nowrap;">'
                 f'⟷ {dist_km:.0f} km error</div>',
            icon_size=(140, 30),
            icon_anchor=(70, 15),
        ),
    ).add_to(m)

    m.save(output_path)
    print(f"Map saved to: {output_path}")
    print(f"Predicted:    ({pred_lat:.4f}, {pred_lon:.4f})  conf={pred_conf*100:.1f}%")
    print(f"Actual:       ({actual_lat:.4f}, {actual_lon:.4f})  Manipal, India")
    print(f"Error:        {dist_km:.0f} km")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True)
    parser.add_argument("--actual-lat", type=float, default=13.3525)
    parser.add_argument("--actual-lon", type=float, default=74.7934)
    parser.add_argument("--checkpoint", default="GEOPROJECT/checkpoints/geolocator_best.pt")
    parser.add_argument("--geocell-dir", default="GEOPROJECT/data/osv5m_50k/semantic_cells")
    parser.add_argument("--output", default="bingbong_prediction.html")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pred_lat, pred_lon, pred_conf = run_inference(
        args.image, args.checkpoint, args.geocell_dir, device
    )

    generate_demo_map(
        pred_lat, pred_lon, pred_conf,
        args.actual_lat, args.actual_lon,
        Path(args.image).name,
        args.output,
    )


if __name__ == "__main__":
    main()
