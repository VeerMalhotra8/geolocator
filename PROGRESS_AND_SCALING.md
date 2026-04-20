# AI GeoGuessr — Progress Log & Scaling Guide

> Track what's been built, what shortcuts were taken for 50K prototyping,
> and exactly what needs to change when scaling to 1M/5M.

---

## Current State: 50K Prototype (Tier 1)

### Phase 1: Data & Infrastructure [COMPLETE]

**What was built:**
- `geoguessr/data/download.py` — Streams OSV-5M from HuggingFace, downloads N spread-out ZIP shards, stratified samples by country
- `geoguessr/data/dataset.py` — PyTorch Dataset/DataLoader with CLIP ViT-L/14 336px preprocessing, auxiliary label encoding (climate 5-class, scene 6-class, drive_side 3-class, region 15-class), haversine-smoothed soft targets
- `geoguessr/data/geocells.py` — S2 cell partitioning (level 8) + haversine label smoothing (tau=200km, sparse top-200)

**Data:**
- 50,000 images from 10 shards (0, 9, 18, 27, 36, 45, 54, 63, 72, 81)
- 213 countries, lat -51.7 to 78.7, lon -176.7 to 178.7
- 8,970 S2 cells (level 8), median 2 images/cell, max 457
- Train: 45,047 / Val: 4,953 (90/10 stratified by country)

**Scaling notes for 1M:**
| Component | Current (50K) | Change for 1M | Change for 5M |
|-----------|--------------|---------------|---------------|
| Download | 10 shards (~25 GB bandwidth) | All 98 shards (~250 GB) | Same |
| Stratified sample | 50K from 500K pool | 1M from 4.9M (just increase --total) | Use all 4.9M |
| S2 cells | Level 8 → 8,970 cells | Level 8 → ~40K cells (more data fills more cells) | ~80K+ cells |
| Smooth precompute | 9s for 50K | ~3 min for 1M | ~15 min for 5M |
| DataLoader workers | num_workers=2 | Same (16GB RAM limit) | Same |
| Phase 3 migration | S2 cells (uniform) | **REPLACE with OPTICS semantic geocells** | Same |

**Known limitations at 50K:**
- S2 cells have extreme imbalance (38.8% cells have only 1 image) — fixed by semantic geocells at scale
- 366 cells exist only in val split — fixed by building cells from train-only at scale
- Top-200 truncation captures ~85% median probability mass — sufficient but monitor at scale

---

### Phase 1.5: Contrastive Pretraining [COMPLETE]

**What was built:**
- `geoguessr/model/contrastive.py` — Contrastive pretraining with geographic captions (code-reviewed, PASS)
- Caption generation from metadata (country, region, city, climate, drive_side, land_cover)
- InfoNCE loss aligning image embeddings with text embeddings
- Only DoRA adapter weights trainable: 540,672 / 428M (0.13%)

**Training results:**
- Command: `python -m GEOPROJECT.geoguessr.model.contrastive --epochs 2 --batch-size 6 --accumulation 10 --num-workers 2`
- Duration: ~79 min total (39.1 + 39.5 min/epoch), 3.5 batch/s, GPU at 89-100%
- Loss: 0.1611 (epoch 1) → 0.1330 (epoch 2)
- Checkpoints: `contrastive_epoch1.pt`, `contrastive_epoch2.pt`, `contrastive_best.pt`

**Geo-NN evaluation (100 queries, k=10):**
| Metric | Before | After | Change |
|--------|--------|-------|--------|
| Median NN dist | 1197 km | 891 km | -25.6% |
| <100km | 10.5% | 11.8% | +1.3pp |
| <500km | 28.9% | 33.5% | +4.6pp |
| <1000km | 46.2% | 53.0% | +6.8pp |

**Scaling notes for 1M:**
- DoRA weights from 50K carry forward as starting checkpoint
- Run 1 more epoch on 1M to refine (not restart)
- More diverse captions from more cities/regions improves embedding quality

---

### Phase 2: Core Classification Model [COMPLETE — S2 CELLS]

**What was built:**
- `geoguessr/model/geolocator.py` — Full GeoLocator model (code-reviewed, PASS)
  - StreetCLIP backbone + DoRA adapters (layers 16-23, r=16, alpha=32)
  - Geo head: Linear(768→2048)→GELU→Dropout→Linear(2048→num_cells)
  - Auxiliary heads: scene(6), climate(5), driving(3), region(16)
  - `num_region=16` (15 UN regions + 1 unknown) — fixed during review
- `geoguessr/train.py` — Training loop (code-reviewed, PASS)
  - KL-div loss (upcast to float32) + weighted CE aux losses (1.0/0.3/0.2/0.2/0.3)
  - CosineWithWarmup scheduler (LR clamped to prevent negatives on resume)
  - validate() with try/finally for model.train() safety
  - Early stopping on val median_km, patience=3
  - Checkpoint saves trainable params via requires_grad filter

**Phase 2 Training results (S2 cells, 8970 cells, 5 epochs, ~2.9h):**
| Epoch | Val Median km | <25km | <200km | <750km | <2500km |
|-------|--------------|-------|--------|--------|---------|
| 1 | 522.6 | 5.7% | 29.5% | 58.1% | 83.9% |
| 2 | 265.5 | 10.4% | 43.2% | 73.0% | 89.7% |
| 3 | 229.9 | 12.3% | 46.8% | 76.1% | 91.2% |
| 4 | 184.4 | 14.8% | 51.4% | 79.1% | 91.4% |
| 5 | **174.4** | **15.3%** | **53.1%** | **80.0%** | **92.1%** |

**Scaling notes for 1M:**
- Resume from 50K checkpoint, train on 1M
- Batch size stays 6 (VRAM limited), accumulation stays 10
- Training time: ~55 hours for 1M vs ~5 hours for 50K
- Early stopping patience=3 likely triggers at epoch 4-6

---

### Phase 3: Semantic Geocells + FAISS [TRAINING COMPLETE — FAISS PENDING]

**What was built:**
- `geoguessr/data/semantic_geocells.py` — OPTICS clustering + Voronoi tessellation (code-reviewed, PASS)
  - `build_semantic_geocells_fast()`: vectorized remap + haversine noise assignment
  - Coords converted to radians for OPTICS haversine metric
  - Save/load config with centroids + cell indices
- `geoguessr/model/faiss_refinement.py` — Two-stage FAISS retrieval (code-reviewed, PASS)
  - `extract_embeddings()`: batch extraction with bf16 autocast (device-aware)
  - `cluster_within_cells()`: OPTICS on L2-normalized embeddings (euclidean ≈ cosine)
  - `build_faiss_index()`: IndexFlatL2 (L2 on normalized = cosine ranking)
  - `predict_two_stage()`: O(1) set membership, robust brute-force fallback
  - `evaluate_two_stage()`: same fixes, sample_idx tracking verified correct
  - Cluster truncation sorted by size (largest kept)
- `geoguessr/inference.py` — End-to-end inference pipeline (code-reviewed, PASS)
  - load_state_dict logging for missing/unexpected keys
  - Top-level imports for json/Path

**Semantic geocell build results (2026-03-20):**
- OPTICS: min_samples=20, max_eps=0.0087 rad (~50km), xi cluster method
- 539 semantic cells (down from 8,970 S2 cells)
- Images per cell: min=21, median=81, max=469, mean=93
- 53.8% noise points (26,891) assigned to nearest centroid via Voronoi
- Build time: 887s (OPTICS) + 2s (smooth targets)
- Haversine smoothing: tau=200km, top-200 (covers all 539 cells)

**Code changes for semantic geocell support (2026-03-20, code-reviewed PASS):**
- `train.py`: Added `--geocell-dir` CLI arg, generic centroids loading via np.load
- `train.py`: Added num_cells validation in load_checkpoint (prevents mismatched resume)
- `train.py`: Fixed hard-coded "cuda" autocast → device.type
- `inference.py`: Added num_cells validation in load_inference_model
- Removed unused `nn` import

**Training command (semantic cells):**
```bash
python -m GEOPROJECT.geoguessr.train --epochs 15 --batch-size 6 --accumulation 10 \
    --geocell-dir GEOPROJECT/data/osv5m_50k/semantic_cells --patience 3
```

**FAISS commands (run after semantic cell training):**
```bash
# Build FAISS index
python -m GEOPROJECT.geoguessr.model.faiss_refinement build \
    --checkpoint GEOPROJECT/checkpoints/geolocator_best.pt \
    --output GEOPROJECT/faiss_index

# Run inference
python -m GEOPROJECT.geoguessr.inference --image path/to/photo.jpg
```

**Semantic geocell training results (15 epochs, 2026-03-20):**
| Epoch | Val Median km | <25km | <200km | <750km | <2500km |
|-------|--------------|-------|--------|--------|---------|
| 1 | 497.3 | 10.4% | 30.4% | 58.4% | 84.1% |
| 5 | 179.8 | 21.3% | 52.4% | 79.0% | 92.4% |
| 8 | 151.4 | 23.4% | 56.1% | 81.6% | 93.0% |
| 11 | 137.2 | 25.1% | 58.2% | 82.0% | 93.0% |
| **14 (best)** | **130.4** | **25.8%** | **58.8%** | **82.4%** | **93.4%** |
| 15 | 132.7 | 25.5% | 58.6% | 82.3% | 93.4% |

**CRITICAL: Internal val metrics are inflated due to spatial leakage (2026-03-20):**
- Spatial audit: 38% of val images have a train neighbor <1km, 75% <5km
- Root cause: random 90/10 split within country, no geographic separation
- Geocells also built on all data (train+val) instead of train-only

**Im2GPS3k external benchmark (2026-03-20, honest numbers):**
| Metric | Internal Val (leaky) | Im2GPS3k (honest) | Inflation |
|--------|---------------------|-------------------|-----------|
| Median km | 130.4 | **672.0** | 5.2x |
| <25km | 25.8% | **11.3%** | 2.3x |
| <200km | 58.8% | **23.1%** | 2.5x |
| <750km | 82.4% | **52.8%** | 1.6x |
| <2500km | 93.4% | **77.6%** | 1.2x |

Note: Im2GPS3k contains Flickr tourist photos (not street view) — harder domain.
672km median is within architecture doc target (500-800km) for 50K prototype.

**Pipeline fixes needed before scaling:**
1. Geographic train/val split (by city cluster, min 25km separation)
2. Rebuild geocells from train-only data
3. Always evaluate on Im2GPS3k (eval_benchmark.py written)
4. Run FAISS two-stage refinement (highest ROI next action — no retraining needed)

**S2 → OPTICS semantic geocells impact:**
- S2 cells had extreme imbalance: 38.8% of cells had only 1 image
- Semantic cells: minimum 21 images/cell, much better for learning
- Fewer cells (539 vs 8970) = easier classification task + more data per cell
- Haversine smoothing covers all cells (top-200 ≥ 539), so full probability mass captured

**FAISS v2: Dense k-NN refinement pipeline (2026-03-24, code-reviewed PASS):**

Code changes:
- `faiss_refinement.py`: Added v2 functions — `weighted_centroid_spherical` (antimeridian-safe spherical averaging), `build_dense_faiss_index` (IndexFlatIP over all ~45K training embeddings), `predict_weighted_coarse` (probability-weighted cell centroid average), `predict_dense_knn` (cell-filtered k-NN with inverse-distance GPS weighting), `predict_two_stage_v2` (confidence-gated: weighted coarse when confident, k-NN when uncertain), `evaluate_two_stage_v2` (4-tier metrics), `calibrate_thresholds` (sweep confidence_threshold x geo_radius), `save/load_dense_faiss_index`
- `run_faiss_eval.py`: Rewritten with `--mode dense|sparse` and `--calibrate` flag. Dense mode builds IndexFlatIP over all embeddings (no OPTICS clustering), evaluates 4-tier metrics (top1, weighted, knn_raw, gated). Fixed critical bug: training DataLoader had shuffle=True causing embedding-GPS misalignment — now uses dedicated non-shuffled loader for extraction.
- `eval_benchmark.py`: Added `--two-stage` and `--faiss-dir` flags for dense FAISS evaluation. Default output now includes weighted coarse alongside top-1.
- `inference.py`: Updated to use v2 pipeline by default with `--mode dense|sparse`. Shows top-1, weighted, k-NN, and gated predictions with confidence and neighbor stats.
- `~/.claude/skills/ml-validate/SKILL.md`: Created ML validation skill with 5 check phases (data split integrity, metric sanity, pipeline correctness, label contamination, regression check).

Review findings fixed:
- Removed dead `distance_threshold` parameter from 3 functions (was threaded through but never used)
- Fixed antimeridian fallback in `weighted_centroid_spherical` (was using np.mean on lons)
- Fixed geo_radius filter to use spherical weighted center instead of fragile median
- Fixed DataLoader shuffle bug in `run_faiss_eval.py` (embeddings were in random order)
- Added `@torch.no_grad()` decorator to `calibrate_thresholds`
- Removed dead `labels_csv` parameter from `evaluate_two_stage_v2`

**Im2GPS3k dense FAISS evaluation results (2026-03-25):**

| Method | Median km | <25km | <200km | <750km | <2500km |
|--------|-----------|-------|--------|--------|---------|
| Top-1 coarse (baseline) | 672.0 | **11.3%** | **23.1%** | 52.8% | 77.6% |
| Weighted coarse | 783.0 | 0.7% | 17.2% | 48.7% | 75.7% |
| k-NN raw (gr=1000km) | **604.6** | 2.0% | 21.4% | **55.2%** | 75.0% |
| Gated (conf=0.6, gr=1000) | 613.1 | 3.0% | 21.8% | 54.9% | 75.2% |

Calibration: conf_threshold=0.6, geo_radius=1000km, kNN gate rate=95.7%.
Dense index: 45,047 vectors (all training embeddings), IndexFlatIP, ~132 MB.
Embedding extraction: ~1648s. Evaluation: ~164s per pass. Calibration: ~261s.

**Key findings:**
1. **k-NN improves median by 10%** (672 → 604.6km) and <750km by +2.4pp — the approach works
2. **k-NN hurts fine precision**: <25km drops from 11.3% to 2.0% because neighbor averaging blurs precise single-cell predictions
3. **Weighted coarse is worse than top-1 across all metrics** (783km vs 672km) — averaging dilutes accurate predictions. Gating fallback changed from weighted to top-1.
4. **Confidence gating partially recovers fine precision**: at conf=0.6, <25km goes from 2.0% (pure k-NN) to 3.0% (gated), while keeping median at 613km
5. **Trade-off is fundamental**: k-NN excels at reducing large errors (median, <750km) but hurts when the model is already right (<25km). The 4.3% of confident top-1 predictions contribute most of the <25km accuracy.

**Commands:**
```bash
# Build dense index + evaluate:
python -m GEOPROJECT.run_faiss_eval --mode dense --calibrate

# Eval with existing index:
python -m GEOPROJECT.geoguessr.eval_benchmark --benchmark im2gps3k --two-stage
```

---

### Phase 4: Explainability [IN PROGRESS — TTA + Error Analysis COMPLETE]

**Test-Time Augmentation (TTA) — Im2GPS3k (2026-03-25):**

Code: `eval_benchmark.py` — added `--tta` flag and `--error-analysis` flag.
3 views: center crop (baseline), horizontal flip, zoomed-out center crop (resize 448, crop 336).
Average softmax probabilities across views, then argmax.

| Metric | Top-1 | TTA (3 views) | Delta |
|--------|-------|---------------|-------|
| Median km | 671.7 | **665.8** | -5.9 (-0.9%) |
| <25km | 11.2% | 11.3% | +0.1pp |
| <200km | 23.4% | **24.3%** | +0.9pp |
| <750km | 52.8% | **53.3%** | +0.5pp |
| <2500km | 77.5% | 77.7% | +0.1pp |

TTA changed 18.4% of predictions. Improvement is modest at 50K scale — expected +1-3% at larger scale.
Time: 286s (~10.5 img/s), ~1.7x slower than batched baseline.

**Per-Continent Error Analysis — Im2GPS3k (2026-03-25):**

| Continent | N | % of Total | Median km | <25km | <200km | <750km | <2500km |
|-----------|---|-----------|-----------|-------|--------|--------|---------|
| **Europe** | 1102 | 36.8% | **327.6** | 14.6% | 36.1% | **70.3%** | 84.2% |
| **Asia** | 647 | 21.6% | 634.4 | **20.7%** | 31.2% | 53.0% | **86.7%** |
| **North America** | 947 | 31.6% | 1265.9 | 1.8% | 6.7% | 37.2% | 67.5% |
| South America | 114 | 3.8% | 1346.6 | 10.5% | 17.5% | 38.6% | 67.5% |
| Africa | 145 | 4.8% | 945.4 | 8.3% | 19.3% | 44.1% | 64.8% |
| Oceania | 42 | 1.4% | 1287.7 | 4.8% | 38.1% | 45.2% | 69.0% |

**Key insights:**
1. **North America is the biggest weakness**: 32% of benchmark, worst median (1266km), only 1.8% <25km. The 50K training set likely underrepresents US/Canada diversity. Scaling will help most here.
2. **Europe is strongest**: Best median (328km), 70% <750km. OSV-5M is European-heavy.
3. **Asia has bimodal accuracy**: Highest <25km (20.7%!) — major Asian cities are well-learned. But 634km median means many mid-range errors exist.
4. **Africa and South America underrepresented** in both training data and benchmark (<5% each).
5. **Worst predictions are ~19,000km errors**: Model predicts opposite hemisphere, mostly for Southeast Asian and Oceanian images. These are domain-gap failures (street view vs tourist photos).

**Scaling priorities based on error analysis:**
- Priority 1: More US/Canada/Mexico training images (biggest accuracy gap vs representation)
- Priority 2: More African + South American training images (underrepresented)
- Priority 3: Revisit FAISS k-NN at 5M scale (density analysis suggests it may help then)

**Commands:**
```bash
# TTA evaluation:
python -m GEOPROJECT.geoguessr.eval_benchmark --benchmark im2gps3k --tta --error-analysis

# Standard evaluation (for comparison):
python -m GEOPROJECT.geoguessr.eval_benchmark --benchmark im2gps3k
```

### Phase 4.5: Pre-1M Decision Gate [RESOLVED]

**Decision 1: DoRA-only vs Partial Backbone Unfreeze → DORA-ONLY**
- Rationale: all ablations (B, C, D, E) were DoRA-only. Switching to partial unfreeze at 1M changes two variables simultaneously — can't attribute gains to data scaling vs architecture.
- DoRA-only keeps the comparison clean. Partial unfreeze deferred to post-1M if needed.
- Trainable params: 4,015,161 (1.30%), adapters on layers 16-23.

**Decision 2: StreetView-domain evaluation → DEFERRED**
- eval_benchmark.py doesn't support OSV-5M test split (no held-out split downloaded).
- Setup cost non-trivial; insight is post-training interpretation, not blocking.
- Deferred to after 1M training.

**Ablation D: MixUp (2026-04-07) — COMPLETE, DROPPED**
- Config: 1424 cells + tau=100 (Ablation C, 617km baseline) + MixUp beta(0.1, 0.1)
- Code: `--mixup-alpha 0.1` added to train.py (code-reviewed PASS, 1 cycle)
- Training: 15 epochs, best at epoch 12, ~7.7h total on RTX 4060
- Internal val: 103km median, 30.6% <25km, 61.5% <200km (looked better than Ablation C)
- **Im2GPS3k (honest): 646km median — WORSE than Ablation C (617km), +29km regression**
- MixUp improved internal val (leaky) but hurt cross-domain generalization
- Verdict: **DROP MixUp**

**Updated ablation summary table:**

| Ablation | Config | Im2GPS3k Median | <25km | <200km | <750km | <2500km | Verdict |
|---|---|---|---|---|---|---|---|
| **Baseline** | 539 cells, tau=200 | 672km | 11.3% | 23.2% | 52.5% | 77.6% | -- |
| **B** (tau only) | 539 cells, tau=100 | 674km | 12.2% | 24.0% | 52.2% | 77.0% | **NO CHANGE** |
| **C** (cells+tau) | 1424 cells, tau=100 | **617km** | **13.1%** | **26.3%** | **55.2%** | **78.1%** | **-55km, SHIP IT** |
| **D** (MixUp) | 1424 cells, tau=100, mixup=0.1 | 646km | 13.0% | 25.8% | 53.8% | 78.6% | **WORSE, DROP** |
| **E** (hier. masking) | inference-only continent mask | 673km | 11.3% | 23.2% | 52.5% | 77.6% | **NO CHANGE, DROP** |

**Final 1M config: geocells + tau=100 (Ablation C), no MixUp. Ablation study complete.**

---

### Phase 5: 1M Scaling [IN PROGRESS — Code Complete, Download Running]

**Plan:** `/home/squishy33/.claude/plans/delightful-nibbling-piglet.md`

**Code changes completed (2026-03-25):**

**Step 0a: Geographic train/val split (dataset.py, code-reviewed PASS):**
- Replaced random 90/10 split with geographic block holdout (0.5° grid cells)
- Entire grid cells held out → no spatial leakage
- 50K verification: 44,568 train / 5,432 val (10.9%), 0% <1km leakage (was 38%), median separation 31.7km
- Removed `RandomHorizontalFlip` from training augmentation (corrupts driving side + text direction)

**Step 1: S2 cell density-balanced sampling (download.py, code-reviewed PASS, 2 review cycles):**
- New `spatial_sample()` using S2 level-4 cells (~510 equal-area cells) for even geographic distribution
- Cells with fewer images than target → take ALL (boosts underrepresented regions like Africa, Central Asia)
- Surplus redistributed proportionally to cells with headroom
- Within-cell country-proportional sampling with remainder top-up
- Selective shard extraction: extract ONLY selected images per shard, delete ZIP immediately
- `--sampling-strategy spatial|country` CLI arg (spatial default)
- Downloads from all 98 shards for full 4.9M pool coverage
- Peak disk: ~103GB (selected images) + 2.5GB (one shard) vs 490GB storing everything
- Fixed: NaN coordinate guard, HF cache preservation (uses temp dir), edge case when total < num_cells

**Step 2: OPTICS subsample + vectorized Voronoi (semantic_geocells.py, code-reviewed PASS):**
- New `build_semantic_geocells_subsampled()`: OPTICS on geographic subsample, Voronoi-assign all
- `voronoi_assign_batched()`: fully vectorized haversine with batched broadcasting, 0.7s for 50K
- Stratification uses O(n log n) argsort+searchsorted instead of O(n*cells) loop
- `--subsample N` CLI arg (recommended 200K for 1M+)
- OPTICS params for 1M: `--min-samples 50 --tau 100`
- Expected: 1,500-2,500 cells (vs 539 at 50K)

**Step 4: Partial backbone unfreeze (geolocator.py + train.py, code-reviewed PASS, 2 review cycles):**
- New `unfreeze_layers` param: fully unfreeze last N ViT encoder layers
- When `unfreeze_layers=4`: layers 20-23 fully trainable, DoRA on 16-19 only
- Three-tier parameter groups: DoRA (lr=1e-4) / heads (lr=5e-4) / backbone (lr=2e-5)
- Unfreezing happens AFTER PEFT wrapping (PEFT re-freezes everything)
- Bounds validation, checkpoint resume validation, empty group filtering
- Contrastive checkpoint loading guarded with `_has_dora` flag

| Config | Trainable Params | % of Total |
|--------|-----------------|------------|
| DoRA-only (default, 4060) | 4,015,161 | 1.30% |
| Partial unfreeze 4 layers (A100) | 57,123,310 | 18.37% |
| — DoRA adapters (layers 16-19) | 270,336 | |
| — Unfrozen backbone (layers 20-23) | 50,384,896 | |
| — Classification heads | 6,468,078 | |

**Torch upgraded to 2.6.0+cu124** (required by transformers 5.3.0 CVE fix).

**Download COMPLETE (2026-03-25):**
- 1,000,000 images from 98 shards, S2 level-4 density-balanced
- Output: `/mnt/d/GEOPROJECT/data/osv5m_1m/`
- 222 countries, lat -54.9 to 78.3, lon -176.8 to 178.5
- Geographic distribution: US 19.2%, RU 4.8%, AU 4.0%, BR 4.0%, JP 3.9%
- North America (US/CA/MX): 23.2%, South America: 8.9%, Africa: 6.1%

**Geographic split (2026-03-26):**
- Train: 895,157 / Val: 104,843 (10.5%)
- Leakage audit (n=300): 3.0% <1km, 20.7% <5km, median 10.9km
- Acceptable at this density — geographic block holdout working correctly

**Step 2 results: Semantic geocells COMPLETE (2026-03-26):**
- OPTICS on 200K subsample: min_samples=50, max_eps=0.0087 rad
- **971 cells** (below predicted 1,500-2,500, but proportionally correct for 895K images)
- Cell sizes: min=68, p10=209, median=623, p90=2,079, max=6,274, mean=922
- Inter-centroid distances: median 143km, p10=79km, p90=275km
- 52.7% noise points, Voronoi-assigned. Build time: 2,025s total
- tau=100km: top-1 prob mean=0.471, 8 cells/image >1% mass (proportionally matches PIGEOTTO's tau=65/2076 cells)
- Smooth targets: 895,157 × 200, all rows sum to 1.0000

**Step 3 results: Contrastive pretrain on 1M COMPLETE (2026-03-26):**
- 1 epoch from 50K checkpoint, batch=6, accumulation=10, 166K steps
- Loss: 0.1330 (50K) → **0.0847** (1M, 36% improvement)
- Checkpoint: `GEOPROJECT/checkpoints/contrastive_best.pt` (48 DoRA params)
- Geo-NN evaluation:

| Metric | 50K (baseline) | 1M | Change |
|--------|---------------|-----|--------|
| Median NN dist | 891 km | **805 km** | -9.7% |
| <500km | 29.2% | 35.5% | +6.3pp |
| <1000km | 49.0% | 55.2% | +6.2pp |
| 25th pct | 344 km | 342 km | -0.6% |

Note: Modest Geo-NN improvement is expected — contrastive sets the embedding floor, supervised classification is where the real gains happen.

**Pre-A100 readiness audit (2026-03-26):**
- 971 cells: GO (922 img/cell, proportional to PIGEOTTO's ratio)
- tau=100: GO (ratio to inter-centroid distance matches PIGEOTTO)
- Contrastive checkpoint: GO (healthy convergence)
- Partial unfreeze code: GO (57.1M params, PEFT ordering verified)
- Data split: GO (3% <1km leakage, acceptable at density)
- Smooth targets: GO (numerical integrity verified)

**Geocell rebuild history (2026-04-10):**

| Run | Subsample | min_samples | tau | Cells | Method | Status |
|-----|-----------|-------------|-----|-------|--------|--------|
| v1 (971) | 200K | 50 | 100 | 971 | global OPTICS | superseded |
| v2 (4279) | 200K | 15 | 65 | 4,279 | global OPTICS | superseded — too many cells for 1M |
| v3 (country) | 500K | 20 | 65 | TBD | **per-country OPTICS** | **RUNNING** (overnight, PID 93413) |

**v3 country-aware approach:**
- `semantic_geocells.py`: added `build_semantic_geocells_country_constrained()` + `--country-aware` CLI flag
- OPTICS runs per-country → clusters never span country borders (matches PIGEOTTO's admin boundary motivation)
- Countries with <20 subsample pts → single centroid fallback
- Voronoi assignment country-constrained (global fallback for countries absent from subsample)
- Code-reviewed (PASS, 1 cycle). CRITICAL fix: NaN lat/lon guard. WARNINGs fixed: large-country collapse logging, subsample-less OOM warning.
- Expected cells: 1,500-3,000 (US ~96K subsample pts will dominate)
- Log: `/tmp/rebuild_country.log`
- Output: `/mnt/d/GEOPROJECT/data/osv5m_1m/semantic_cells_country/`
- Command:
```bash
python -m GEOPROJECT.geoguessr.data.semantic_geocells \
    --metadata /mnt/d/GEOPROJECT/data/osv5m_1m/metadata_train.csv \
    --output /mnt/d/GEOPROJECT/data/osv5m_1m/semantic_cells_country \
    --subsample 500000 --min-samples 20 --tau 65 --country-aware --seed 42
```

**Fact-check from PIGEON paper (Table 5):**
- PIGEON: 100K images → 2,203 cells, tau=75
- PIGEOTTO: 4.5M images → 2,076 cells, tau=65
- NOTE: PIGEOTTO uses admin2 shapefiles (GADM), not pure OPTICS. Our approach is country-level constraint only.
- topK=40 for PIGEOTTO is refinement candidates, NOT cell count (earlier notes were wrong)

**v3 geocell build results (2026-04-11):**
- 6,794 cells, tau=65, min_samples=20, 500K subsample
- Cell stats: min=1, median=90, max=2,280 imgs/cell
- 195 countries used OPTICS, 27 single-centroid fallback
- Build time: 12,201s OPTICS + 26s Voronoi = ~3.4hr
- Output: `/mnt/d/GEOPROJECT/data/osv5m_1m/semantic_cells_country/`

---

## 1M Classification Training — FAILED (2026-04-12/13)

**Setup:**
- Instance: Vast.ai RTX 5090 (Ontario, $0.41/hr, instance 34632364)
- Data transfer: ~45GB images via tar/rsync from local NTFS (WSL /mnt/d/) → remote
- Transfer speed bottleneck: NTFS via WSL ~2MB/s (cannot avoid without native Linux storage)

**Training command:**
```bash
python -u -m GEOPROJECT.geoguessr.train \
    --epochs 15 --batch-size 6 --accumulation 10 \
    --data-dir /workspace/GEOPROJECT/data/osv5m_1m \
    --geocell-dir /workspace/GEOPROJECT/data/osv5m_1m/semantic_cells_country \
    --contrastive-checkpoint /workspace/GEOPROJECT/checkpoints/contrastive_best.pt \
    --patience 3 --num-workers 0
```

**Failure log:**

| # | Step died | Root cause | Fix applied |
|---|-----------|------------|-------------|
| 1 | Launch | `ModuleNotFoundError: s2sphere` | `pip install s2sphere` on remote |
| 2 | ~28,700 | Silent OOM kill — cgroup memory limit 174GB, workers fork + duplicate 1.4GB smooth targets | `--num-workers 2` → `--num-workers 0` |
| 3 | ~13,300 | Same OOM (num-workers 0 not yet applied) | Applied `--num-workers 0` |
| 4 | ~28,700 | `OSError: image file is truncated (4 bytes)` — corrupted image in dataset | Added `ImageFile.LOAD_TRUNCATED_IMAGES = True` to `dataset.py` line 16 |
| 5 | ~28,700 | `OSError: image file is truncated (12 bytes)` — LOAD_TRUNCATED_IMAGES insufficient for severely corrupt files | Need try/except around `Image.open()` to return black image fallback |

**Fixes still needed before next attempt:**
1. **Wrap image loading in try/except** in `dataset.py` line 238-239:
   ```python
   try:
       image = Image.open(img_path).convert("RGB")
   except Exception:
       image = Image.new("RGB", (336, 336), (0, 0, 0))
   ```
2. **Apply this to local dataset.py** (not just remote patch) so it persists
3. **Speed issue**: `--num-workers 0` dropped throughput from 14 b/s → 5 b/s. Each epoch takes ~4hr vs ~2hr. Consider `--num-workers 1` as a compromise — one prefetch worker, much less memory than 2.
4. **Pre-validate images** before training: scan all 1M images for truncated files, remove or replace. Prevents mid-epoch crashes.

**Budget spent:** ~$3.85 on failed attempts. $6.18 remaining.

**What worked well:**
- DoRA checkpoint loading: correct, 540K params loaded cleanly
- 6,794 cell setup: compatible with model, no dimension errors
- GPU: RTX 5090 32GB, stable when training ran
- Data integrity: all 1M images landed, geocells + smooth targets correct

**Remaining steps:**
1. Fix `dataset.py` image loading (try/except) — apply locally
2. Pre-scan images for corruption (optional but recommended)
3. Spin up new Vast.ai instance, re-upload, train with fixed code
4. FAISS refinement + top-K fusion + temperature softmax (post-training TODO)
5. Final evaluation on Im2GPS3k
6. Expected: 300-450km median on Im2GPS3k

---

## Quick Reference: How to Scale

```bash
# 1. Download more data (adjust --num-shards or --total)
python -m GEOPROJECT.geoguessr.data.download --total 1000000 --num-shards 98

# 2. Rebuild geocells (same command, new data)
python -m GEOPROJECT.geoguessr.data.geocells --metadata GEOPROJECT/data/osv5m_1m/metadata.csv

# 3. Resume training from checkpoint
python -m GEOPROJECT.geoguessr.train --resume checkpoint_50k.pt --data GEOPROJECT/data/osv5m_1m/

# 4. At 1M+: Replace S2 cells with semantic geocells (Phase 3)
#    - Run OPTICS clustering on training GPS coordinates
#    - Rebuild classification head with new cell count
#    - Retrain from contrastive checkpoint (not from scratch)
```
