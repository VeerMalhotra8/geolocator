# GeoLocator — Image Geolocation with CLIP + DoRA

Predicts where a photo was taken from pixels alone — no GPS, no EXIF, just the image.

Uses CLIP ViT-L/14 with DoRA adapters and semantic geocell classification. Trained on 50K street-view images from [OSV-5M](https://huggingface.co/datasets/osv5m/osv5m), scaling to 1M next.

**50K Im2GPS3k benchmark: 672 km median error**

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

With 20x less training data, GeoLocator already beats StreetCLIP on median error. 1M should close the gap on the threshold metrics too.

### Training Progression (50K)

| Phase | Approach | Val Median | < 25km | < 200km |
|-------|----------|:---:|:---:|:---:|
| Contrastive pretraining | DoRA + InfoNCE | 891 km (Geo-NN) | — | — |
| S2 cell classification | 8,970 uniform cells | 174 km | 15.3% | 53.1% |
| Semantic geocells | 539 OPTICS clusters | 130 km | 25.8% | 58.8% |

Internal val metrics are inflated ~5x due to spatial leakage (38% of val images had a train neighbor <1km). Im2GPS3k is the honest benchmark — see [Challenges](#challenges).

### Ablation Study (50K, Im2GPS3k)

Tested modifications before committing to the 1M run:

**Geocell / tau ablations:**

| Config | Im2GPS3k Median | <25km | <200km | <750km | <2500km | Verdict |
|--------|:---:|:---:|:---:|:---:|:---:|:---:|
| Baseline (539 cells, tau=200) | 672 km | 11.3% | 23.2% | 52.5% | 77.6% | — |
| tau=100 only (539 cells) | 674 km | 12.2% | 24.0% | 52.2% | 77.0% | no change |
| **1424 cells + tau=100** | **617 km** | **13.1%** | **26.3%** | **55.2%** | **78.1%** | **-8.2%, best** |
| Hierarchical continent mask | 673 km | 11.3% | 23.2% | 52.5% | 77.6% | useless, dropped |
| MixUp (beta=0.1) | — | — | — | — | — | never ran |

More geocells + lower tau was the clear winner. Tau alone does nothing — they're coupled. Continent masking was a waste — the geo head already concentrates probability on the right continent. MixUp was planned but we moved on to 1M.

**FAISS inference methods (no retraining, post-hoc):**

| Method | Median km | <25km | <200km | <750km | <2500km |
|--------|:---:|:---:|:---:|:---:|:---:|
| Top-1 coarse (baseline) | 672 km | **11.3%** | **23.1%** | 52.8% | **77.6%** |
| Weighted coarse | 783 km | 0.7% | 17.2% | 48.7% | 75.7% |
| k-NN raw (gr=1000km) | **605 km** | 2.0% | 21.4% | **55.2%** | 75.0% |
| Gated (conf=0.6, gr=1000km) | 613 km | 3.0% | 21.8% | 54.9% | 75.2% |

k-NN improved median by 10% but destroyed city-level accuracy (<25km: 11.3% → 2.0%). Weighted coarse was strictly worse than top-1 everywhere. Fundamental tradeoff — neighbor averaging helps coarse, hurts fine. Deferred to 5M where density might actually help.

**TTA (test-time augmentation, 3 views):**

| Metric | Top-1 | TTA | Delta |
|--------|:---:|:---:|:---:|
| Median | 672 km | 666 km | -0.9% |
| <200km | 23.4% | 24.3% | +0.9pp |
| <750km | 52.8% | 53.3% | +0.5pp |

Marginal at 50K. Should scale better with more data.

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

### Why these choices

**DoRA over LoRA** — Weight-decomposed adaptation handles OOD better. Geolocation needs extreme OOD generalization (train on street view, test on tourist photos).

**Semantic geocells over S2 cells** — Uniform S2 gives 8,970 cells where 39% have a single image. OPTICS clustering produces 539 cells (min 21, median 81 images/cell) that actually respect where data is dense.

**Haversine label smoothing** — Predicting a neighboring cell should be "almost right," not completely wrong. Soft targets with tau=200km give partial credit for nearby predictions.

---

## Pipeline

**Stage 1: Contrastive Pretraining** — Align CLIP's image encoder to geographic text embeddings using synthetic captions from GPS metadata. Only DoRA weights trainable (0.13% of backbone). Gives the model a geographic prior before classification.

**Stage 2: Geocell Classification** — Classify into semantic geocells with haversine-smoothed soft targets. Auxiliary heads for scene, climate, driving side, and UN region provide extra geographic signal.

**Stage 3: FAISS Refinement (deferred)** — Dense k-NN search over all training embeddings. Improved median from 672 to 605 km on Im2GPS3k but killed fine precision (<25km: 11.3% → 2.0%). Neighbor averaging blurs precise predictions. Deferred until 5M scale where density should actually help.

---

## What happened along the way

**Phase 1-2 (Data + S2 cells):** Downloaded 50K from OSV-5M across 213 countries. Trained 5 epochs on 8,970 S2 cells, hit 174 km internal val. Continent-level accuracy reached 92% fast, but the S2 cells were horribly imbalanced.

**Phase 3 (Semantic geocells + reality check):** OPTICS clustering brought cells down to 539, internal val improved to 130 km. Then we ran Im2GPS3k and got 672 km — 5.2x inflation from spatial leakage in our train/val split. That was a wake-up call. Implemented geographic block holdout (0.5° grid cells, 0% leakage) and made Im2GPS3k the only benchmark that matters.

**Phase 4 (Analysis):** TTA (3 views) gave +0.9% — marginal. Per-continent breakdown showed Europe is strongest (328 km), North America is worst (1,266 km, only 1.8% <25km). Ran ablation study: more geocells (1424) + lower tau (100) = 617 km, best result at 50K.

**Phase 5 (1M scaling, in progress):** Downloaded 1M images (222 countries, density-balanced). Built 971 semantic geocells. Contrastive pretrain on 1M: loss dropped 36%. Classification training needs an A100 (~$15-25 for 12-18h). Code is ready, waiting on compute.

---

## Challenges

1. **Spatial leakage inflated everything.** Internal val said 130 km; Im2GPS3k said 672 km. 38% of val images had a train neighbor within 1 km. Switching to geographic block holdout and using Im2GPS3k as the only trusted metric was the single most important decision.

2. **S2 cells don't work at 50K.** 39% of cells had one image. OPTICS semantic geocells fixed this — fewer cells (539 vs 8,970) with way better balance.

3. **Training needs patience.** Initially ran 5 epochs. Going to 15 with early stopping found the best model at epoch 14. Still improving when we stopped.

4. **FAISS doesn't help at 50K scale.** k-NN refined median from 672 to 605 km but destroyed city-level accuracy. At 50K the neighbor density is too sparse — averaging just adds noise. Deferred to 5M.

5. **Domain gap is real.** Training on street view, evaluating on Flickr tourist photos. 672 km on cross-domain data is within our architecture target (500-800 km) for 50K.

---

## Project Structure

```
GEOPROJECT/
├── geoguessr/
│   ├── data/
│   │   ├── dataset.py              # PyTorch Dataset with CLIP preprocessing
│   │   ├── download.py             # OSV-5M streaming & density-balanced sampling
│   │   ├── geocells.py             # S2 cell partitioning
│   │   └── semantic_geocells.py    # OPTICS clustering + Voronoi
│   ├── model/
│   │   ├── geolocator.py           # Core model (CLIP + DoRA + heads)
│   │   ├── contrastive.py          # Contrastive pretraining
│   │   └── faiss_refinement.py     # Dense k-NN retrieval
│   ├── train.py                    # Training loop
│   ├── inference.py                # End-to-end inference
│   └── eval_benchmark.py           # Im2GPS3k evaluation
├── ARCHITECTURE.md                 # Full technical spec
├── SCALING_GUIDE.md                # Progress log & scaling notes
├── requirements.txt
└── README.md
```

## Quick Start

```bash
# Clone
git clone https://github.com/VeerMalhotra8/geolocator.git
cd geolocator

# Install
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

# Evaluate
python -m geoguessr.eval_benchmark

# Inference
python -m geoguessr.inference --image path/to/photo.jpg
```

## Hardware

- **GPU:** NVIDIA RTX 4060 Laptop (8 GB VRAM)
- **Training:** bfloat16, gradient accumulation 10
- **50K time:** ~3h classification, ~80 min contrastive
- **1M:** needs A100 (~$15-25)

## References

- [OSV-5M](https://huggingface.co/datasets/osv5m/osv5m) — OpenStreetView 5M (CVPR 2024)
- [PIGEON](https://arxiv.org/abs/2307.05845) — Predicting Image Geolocations (CVPR 2024)
- [StreetCLIP](https://arxiv.org/abs/2302.00275) — CLIP for street-level geolocation
- [GeoCLIP](https://arxiv.org/abs/2309.16020) — CLIP-inspired alignment for geolocation (NeurIPS 2023)
- [DoRA](https://arxiv.org/abs/2402.09353) — Weight-decomposed low-rank adaptation

## License

For educational and research purposes.
