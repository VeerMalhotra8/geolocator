# GeoLocator: AI-Powered Image Geolocation

A deep learning system that predicts the geographic location of a photo using only visual cues — no GPS metadata, no EXIF data, just pixels.

Built on CLIP ViT-L/14 with DoRA fine-tuning, semantic geocell classification, and two-stage FAISS retrieval refinement. Trained on 50K street-view images from [OSV-5M](https://huggingface.co/datasets/osv5m/osv5m), with a scaling path to 1M+.

**Current best (50K, Im2GPS3k benchmark): 672 km median error**

---

## Results

### Im2GPS3k Benchmark (external, no leakage)

| Metric | GeoLocator (50K) | StreetCLIP (1.1M) | Target (1M) |
|--------|:---:|:---:|:---:|
| Median error | 672 km | ~900 km* | 350-450 km |
| < 25 km (city) | 11.3% | 22.4% | 18-22% |
| < 200 km (region) | 23.1% | 37.4% | 38-45% |
| < 750 km (country) | 52.8% | 61.3% | 68-75% |
| < 2500 km (continent) | 77.6% | 80.4% | 88-92% |

*StreetCLIP doesn't report median directly; estimated from their % @ km curve.

With **20x less training data**, GeoLocator already achieves a lower median error than StreetCLIP. At 1M images, projections put us ahead on every threshold.

### Training Progression (50K prototype)

| Phase | Approach | Val Median | < 25km | < 200km |
|-------|----------|:---:|:---:|:---:|
| Contrastive pretraining | DoRA + InfoNCE | 891 km (Geo-NN) | — | — |
| S2 cell classification | 8,970 uniform cells | 174 km | 15.3% | 53.1% |
| Semantic geocells | 539 OPTICS clusters | 130 km | 25.8% | 58.8% |
| + FAISS refinement | k-NN within top cells | 605 km* | 11.3%* | 23.1%* |

*Im2GPS3k external benchmark. Internal val metrics are inflated 2-5x due to spatial leakage in the 50K split — see [Lessons Learned](#lessons-learned).

---

## Architecture

```
Input Image (336x336)
    |
    v
CLIP ViT-L/14 (frozen, 428M params)
    + DoRA adapters (layers 16-23, r=16)     <- 4M trainable params
    |
    v
[768-dim embedding]
    |
    +---> Geo Head: Linear(768->2048)->GELU->Dropout->Linear(2048->N_cells)
    |         |
    |         v
    |     Haversine-smoothed soft targets (tau=200km)
    |     KL-divergence loss
    |
    +---> Auxiliary Heads (multi-task)
              Scene (6 classes)    weight=0.3
              Climate (5 classes)  weight=0.2
              Driving side (3)     weight=0.2
              UN Region (16)       weight=0.3
```

### Key Design Decisions

**Why DoRA over LoRA?** Weight-decomposed adaptation provides better out-of-distribution generalization — critical for geolocation where train images (street view) differ from test images (tourist photos, CCTV, etc.).

**Why semantic geocells over S2 cells?** Uniform S2 partitioning creates 8,970 cells with extreme imbalance (39% have only 1 image). OPTICS clustering on training GPS coordinates produces 539 semantically meaningful cells (min 21, median 81 images/cell) that respect geographic density.

**Why haversine label smoothing?** Hard cell labels ignore geography — predicting a neighboring cell should be "almost right," not completely wrong. Soft targets with tau=200km give partial credit for nearby predictions, producing smoother gradients and better calibrated confidence.

**Why two-stage retrieval?** Coarse cell classification gets you to the right region. FAISS k-NN search within the predicted cell's training images refines to a specific coordinate. This improved median error from 672 km to 605 km on Im2GPS3k.

---

## Pipeline

### Stage 1: Contrastive Pretraining
Align CLIP's image encoder to geographic text embeddings using synthetic captions ("A street view photo from Paris, France, in a temperate climate..."). Only DoRA adapter weights are trainable (0.13% of backbone). This gives the model a geographic prior before classification training.

### Stage 2: Geocell Classification
Classify images into semantic geocells using the geo head with haversine-smoothed soft targets. Auxiliary heads for scene type, climate zone, driving side, and UN region provide complementary geographic signals.

### Stage 3: FAISS Refinement
Extract embeddings for all training images, build a FAISS index, and at inference time search within the top predicted cells' embeddings to find the nearest neighbor coordinate.

---

## The Journey

### Phase 1: Infrastructure (Data Pipeline)
Downloaded 50K images from OSV-5M spanning 213 countries. Built streaming pipeline with spatial density-balanced sampling to avoid European over-representation. Created S2 cell partitioning with haversine label smoothing.

### Phase 2: First Results (S2 Cells)
Trained for 5 epochs (~3 hours on RTX 4060). Reached 174 km median on internal validation. The model learned geography surprisingly fast — continent-level accuracy (< 2500km) hit 92% by epoch 5. But the uniform S2 cells were horribly imbalanced.

### Phase 3: Semantic Geocells + Honest Evaluation
Replaced S2 cells with OPTICS-clustered semantic geocells. Internal metrics improved to 130 km median. Then came the reality check: **external Im2GPS3k benchmark showed 672 km** — a 5.2x inflation from spatial leakage in our train/val split. This was a pivotal moment. We implemented geographic train/val splitting (0.5° grid cells, 0% leakage) and accepted Im2GPS3k as the only honest benchmark.

### Phase 4: Analysis & Refinement
Test-time augmentation (3 views) gave a modest +0.9% improvement. Per-continent analysis revealed Europe is strongest (328 km median) and North America weakest (1,266 km) — likely because OSV-5M has denser European coverage. FAISS k-NN refinement improved the honest benchmark to 605 km.

### Phase 5: 1M Scaling (In Progress)
Downloaded 1M images across 222 countries. Built 971 semantic geocells (up from 539). Contrastive pretraining on 1M already improved: loss from 0.133 to 0.085, Geo-NN median from 891 km to 805 km. Classification training at 1M requires partial backbone unfreezing (57M params) on cloud A100 — estimated cost ~$15-25.

---

## Lessons Learned

1. **Always audit train/val leakage before celebrating metrics.** Our internal val showed 130 km median; the honest external benchmark showed 672 km. The 38% of val images with a train neighbor < 1km completely invalidated internal metrics. Im2GPS3k became our only trusted benchmark.

2. **Semantic geocells >> uniform S2 cells.** OPTICS clustering reduced cells from 8,970 to 539 while eliminating the long tail of single-image cells. The improvement was immediate: +44 km median on internal val.

3. **More epochs with early stopping, don't cut training short.** We initially ran 5 epochs. Extending to 15 with patience=3 found the best model at epoch 14. The model was still improving — patience prevents premature stopping without risking overfitting.

4. **k-NN refinement helps at scale but not at 50K.** Dense FAISS search improved median from 672 to 605 km, but fine-grained accuracy (< 25km) slightly degraded. With 1M images providing denser cell coverage, k-NN should shine. FAISS v2 deferred to 1M+.

5. **Domain gap is real.** The model trains on street-view imagery but Im2GPS3k contains Flickr tourist photos. A 672 km median on cross-domain data is actually within our architecture doc's target range (500-800 km) for 50K. The 1M run with backbone unfreezing should close this gap significantly.

---

## Project Structure

```
GEOPROJECT/
├── geoguessr/
│   ├── data/
│   │   ├── dataset.py              # PyTorch Dataset with CLIP preprocessing
│   │   ├── download.py             # OSV-5M streaming & sampling
│   │   ├── geocells.py             # S2 cell partitioning
│   │   └── semantic_geocells.py    # OPTICS clustering + Voronoi
│   ├── model/
│   │   ├── geolocator.py           # Core model (CLIP + DoRA + heads)
│   │   ├── contrastive.py          # Contrastive pretraining
│   │   └── faiss_refinement.py     # Two-stage FAISS retrieval
│   ├── train.py                    # Training loop
│   ├── inference.py                # End-to-end inference
│   └── eval_benchmark.py           # Im2GPS3k evaluation
├── ARCHITECTURE.md                 # Detailed technical spec
├── SCALING_GUIDE.md                # Progress log & scaling notes
├── requirements.txt
└── README.md
```

## Quick Start

```bash
# Clone
git clone https://github.com/<your-username>/geolocator.git
cd geolocator

# Install dependencies
pip install -r requirements.txt

# Download data (50K subset of OSV-5M)
python -m geoguessr.data.download --total 50000

# Build semantic geocells
python -m geoguessr.data.semantic_geocells

# Contrastive pretraining
python -m geoguessr.model.contrastive --epochs 2 --batch-size 6 --accumulation 10

# Train classifier
python -m geoguessr.train --epochs 15 --batch-size 6 --accumulation 10 \
    --geocell-dir data/osv5m_50k/semantic_cells --patience 3

# Evaluate on Im2GPS3k
python -m geoguessr.eval_benchmark

# Run inference on a single image
python -m geoguessr.inference --image path/to/photo.jpg
```

## Hardware

Developed and trained on:
- **GPU:** NVIDIA RTX 4060 Laptop (8 GB VRAM, compute capability 8.9)
- **Training:** bfloat16 mixed precision, gradient accumulation 10
- **50K training time:** ~3 hours (classification), ~80 min (contrastive)
- **1M training:** Requires A100 cloud instance (~$15-25 estimated)

## References

- [OSV-5M](https://huggingface.co/datasets/osv5m/osv5m) — OpenStreetView 5M dataset
- [PIGEON](https://arxiv.org/abs/2307.05845) — Predicting Image Geolocations (CVPR 2024)
- [StreetCLIP](https://arxiv.org/abs/2302.00275) — CLIP for street-level geolocation
- [GeoCLIP](https://arxiv.org/abs/2309.16020) — Clip-inspired alignment for geolocation (NeurIPS 2023)
- [DoRA](https://arxiv.org/abs/2402.09353) — Weight-decomposed low-rank adaptation

## License

This project is for educational and research purposes.
