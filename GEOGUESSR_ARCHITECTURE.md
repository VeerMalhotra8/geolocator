# AI GeoGuessr — Complete Architecture & Theory Bible

> **Permanent reference document.** This file contains the full architecture, theory,
> and implementation plan for the AI GeoGuessr project. Refer back to this at any phase.

---

# SECTION 0: HARDWARE & REVIEW CORRECTIONS

## Training Hardware: Local RTX 4060 Laptop

| Spec | Value |
|------|-------|
| GPU | NVIDIA RTX 4060 Laptop (Ada Lovelace) |
| VRAM | 8 GB GDDR6 |
| Compute Capability | **8.9** → supports bfloat16 |
| FP16 Tensor Core TFLOPS | ~194 (~3x faster than Kaggle T4) |
| System RAM | 16 GB |
| Storage | Local SSD (unlimited for our purposes) |
| Session Limits | **None** — train continuously |

**Why local 4060 > Kaggle T4:**
- **3x faster** compute (194 vs 65 TFLOPS FP16)
- **bfloat16 supported** → better numerical stability, no GradScaler needed
- **No session limits** → train uninterrupted, no checkpoint/resume headaches
- **Unlimited local storage** → can store full datasets, no 20GB cap
- **Faster iteration** → no upload/download, no queue waiting
- **Tradeoff:** 8GB VRAM (vs T4's 16GB) → smaller batch sizes, compensated by gradient accumulation

**VRAM budget (8GB, bfloat16):**
```
CLIP ViT-L/14 frozen weights (bf16):      ~600 MB
DoRA parameters:                           ~3 MB
Classification + aux heads:                ~40 MB
Forward activations (batch=6, bf16):       ~2.5 GB
Backward gradients (DoRA + heads):         ~1.2 GB
Optimizer states (AdamW):                  ~200 MB
CUDA/PyTorch overhead:                     ~2 GB
──────────────────────────────────────────────────
TOTAL:                                     ~6.5 GB ✓ (fits in 8GB)
Safe batch size: 6 (with accumulation_steps=10 → effective batch=60)
```

## Review Corrections (integrated throughout)

| Issue | Fix |
|-------|-----|
| **Missing contrastive pretraining** | Added Phase 1.5: lightweight contrastive pretraining with geographic captions before cell classification |
| **LoRA OOD weakness** | Switched to **DoRA** (Weight-Decomposed LoRA) — better OOD generalization |
| **No success criteria** | Added concrete baselines and targets |
| **OPTICS max_eps units** | Fixed: scikit-learn haversine metric expects **radians**, not degrees |
| **Haversine smoothing compute cost** | Added sparse precomputation strategy (top-K nearest cells only) |
| **No validation split / early stopping** | Added proper train/val split with early stopping |
| **Auxiliary label noise** | Added confidence weighting for driving-side labels on non-road images |

---

# SECTION 1: CLIP — The Foundation

## 1.1 What Problem Does CLIP Solve?

Before CLIP (2021), CV models were trained on fixed label sets. CLIP trains on
**image-caption pairs** (400M from the internet). The model learns to match
images with text descriptions → understands virtually any visual concept.

## 1.2 How CLIP Works Internally

Two encoders trained together via contrastive learning:

```
IMAGE ENCODER (ViT-L/14)              TEXT ENCODER (Transformer)
  Photo of street                       "A street in Bangkok, Thailand"
       │                                         │
       ▼                                         ▼
  Vision Transformer                    Text Transformer
  (24 layers, 1024-dim)                (12 layers)
       │                                         │
       ▼                                         ▼
  768-dim embedding                    768-dim embedding
       │                                         │
       └──────── SHOULD BE SIMILAR ──────────────┘
```

**InfoNCE loss:** In a batch of N pairs, create N×N similarity matrix. Diagonal
(correct pairs) should be high, off-diagonal (wrong pairs) should be low.

```
L = -1/N × Σᵢ log( exp(sim(imgᵢ,txtᵢ)/τ) / Σⱼ exp(sim(imgᵢ,txtⱼ)/τ) )

sim(a,b) = cosine_similarity = (a·b) / (||a||×||b||)
τ = learned temperature
```

## 1.3 Vision Transformer (ViT-L/14) Step-by-Step

```
Input: 336×336×3 RGB

Step 1: PATCH EMBEDDING
  Split into 14×14 patches → 24×24 = 576 patches
  Linear project each (14×14×3 = 588) → 1024 dims

Step 2: PREPEND [CLS] + POSITIONAL EMBEDDINGS
  577 tokens × 1024 dims
  Positions encode spatial layout (without them, model can't distinguish
  top-left from bottom-right)

Step 3: 24 TRANSFORMER LAYERS
  Each layer:
    a) Multi-Head Self-Attention (16 heads × 64 dims)
       Q = token × W_Q  ("what am I looking for?")
       K = token × W_K  ("what do I contain?")
       V = token × W_V  ("what information do I carry?")
       Attention = softmax(QK^T / √64) × V
       → Every patch attends to every other patch
       + Residual + LayerNorm

    b) Feed-Forward Network
       Linear(1024→4096) → GELU → Linear(4096→1024)
       + Residual + LayerNorm

Step 4: EXTRACT [CLS] TOKEN → project to 768 dims → L2 normalize
```

**Layer specialization:**
- Layers 1-6: Edges, textures, colors
- Layers 7-12: Object parts, texture patterns
- Layers 13-18: Objects, scene composition
- Layers 19-24: High-level abstractions, task-specific features

## 1.4 Why CLIP? (Decision Justification)

| Option | Why rejected |
|--------|-------------|
| ResNet-50 (ImageNet) | Only 1000 classes. No geographic visual-semantic features from 400M web images |
| DINOv2 (Meta) | No text encoder → can't use geographic captions for pretraining |
| CLIP ViT-B/16 | 512-dim embeddings, +5-8% worse on fine-grained tasks |
| CLIP ViT-L/14 (224px) | 336px needed to resolve road signs and small text |
| Train from scratch | ~$10M equivalent compute. Infeasible |

**Paper basis:** PIGEON (CVPR 2024), StreetCLIP (2023), GeoCLIP (NeurIPS 2023)

## 1.5 Fine-Tuning Strategy

### Our Approach: DoRA (Weight-Decomposed Low-Rank Adaptation)

```
DoRA = Weight-Decomposed Low-Rank Adaptation

Standard LoRA: W' = W + AB  (adds low-rank update)
DoRA:          W' = m × (W + AB) / ||W + AB||  (decomposes into magnitude + direction)

WHY DoRA over LoRA?
- LoRA creates "intruder dimensions" that hurt OOD generalization
  (Shuttleworth et al., NeurIPS 2024: "LoRA vs Full Fine-tuning: An Illusion of Equivalence")
- DoRA's magnitude-direction decomposition more closely approximates full fine-tuning
- Geolocation REQUIRES extreme OOD generalization (225 countries, wildly varying conditions)
- DoRA adds negligible overhead vs. LoRA (~5% more parameters)
- Paper: "DoRA: Weight-Decomposed Low-Rank Adaptation" (Liu et al., 2024)

Config:
  rank = 16
  target_modules = ["q_proj", "v_proj"]  (attention Q and V in last 8 layers)
  trainable params: ~800K (0.26% of backbone)
  VRAM: ~6.5 GB on RTX 4060 with bfloat16 → fits comfortably in 8GB
```

### Why not partial unfreezing or full fine-tuning?
- Partial unfreeze (last 4 layers): ~50M params, needs ~18-22GB VRAM → doesn't fit on 4060 (8GB)
- Full fine-tune: 304M params, needs 30-40GB VRAM → way too much
- DoRA with rank=16 is the optimal strategy for our 8GB VRAM budget
- If results plateau, can rent an A100 on vast.ai ($1/hr) for a partial-unfreeze run as a stretch goal

---

# SECTION 2: GEOCELL CREATION

## 2.1 Why Classification, Not Regression?

Direct (lat, lon) regression is extremely hard:
- 2D continuous output → highly non-convex loss landscape
- Models converge to the global average (~Atlantic Ocean)
- All top methods use classification

**Solution (PlaNet, 2016):** Divide Earth into cells → predict which cell.

## 2.2 Semantic Geocells (PIGEON approach)

**Why over uniform S2 cells?**
S2 cells have extreme class imbalance (Manhattan: millions of images, Sahara: 50).
PIGEON's semantic geocells create data-driven cells (+112.6 km improvement).

**Creation process:**

```
Step 1: OPTICS CLUSTERING on training GPS coordinates
  OPTICS finds clusters of VARYING density (cities=small dense, rural=large sparse)

  CORRECTED parameters:
    min_samples = 50
    max_eps = 0.0087  (≈0.5 degrees in RADIANS — scikit-learn haversine
                       expects radians, NOT degrees)
    metric = 'haversine'
    Input coordinates must be in RADIANS: np.radians([lat, lon])

  Output: ~5,000-15,000 clusters

Step 2: ASSIGN NOISE POINTS to nearest cluster centroid

Step 3: VORONOI TESSELLATION
  Each point on Earth → assigned to nearest cluster centroid
  Creates contiguous, non-overlapping cells covering the globe

  WHY VORONOI?
  - Complete coverage (no gaps)
  - No overlaps
  - Natural: nearest centroid assignment
  - Efficient at inference

Step 4: VERIFY BALANCE → target 100-5,000 images per cell
```

**Implementation plan:** Start with S2 cells (level 6-8) for Phase 1 prototyping,
switch to semantic geocells in Phase 3.

## 2.3 Haversine Label Smoothing

**Problem:** Hard labels penalize predicting a 5km-away cell identically to a 10,000km-away cell.

**Fix:** Soft labels based on haversine distance to each cell centroid:

```
p_i = exp(-d_i / τ) / Σ_j exp(-d_j / τ)

d_i = haversine distance from true GPS to cell i's centroid (km)
τ = 200 km (temperature — PIGEON ablation: tested {50,100,200,500,1000}, 200 optimal)

Example:
  Cell:      4719    4720    4721    4722    4723
  Dist(km):  15      5       0       8       22
  Target:   [0.05,   0.22,   0.43,   0.16,   0.03]
```

**OPTIMIZATION (from review):** Computing distances to ALL ~10K cells per sample is expensive.
Precompute sparse targets: for each training image, store only top-50 nearest cells' distances.
All other cells get probability ≈ 0. This reduces computation by ~200x.

```python
# Precompute once during data preparation:
for img_idx in range(num_images):
    dists = haversine_all_centroids(gps[img_idx])
    top50_idx = np.argpartition(dists, 50)[:50]
    top50_dists = dists[top50_idx]
    # Store sparse: {img_idx: (top50_idx, top50_dists)}
    # At training time, construct sparse soft target from this
```

**Loss:** KL-divergence (proper divergence for soft distributions, not cross-entropy
which assumes one-hot targets).

---

# SECTION 3: TRAINING PIPELINE

## 3.1 Dataset: OSV-5M

**Full OSV-5M:** 5.1M street view images, 225 countries, CVPR 2024 benchmark.

**Local storage advantage:** No Kaggle 20GB cap. We can download the full dataset
to local SSD (~250GB). However, training on 5M images would take ~12 days on the 4060.

**Tiered training strategy — fast iteration, long runs only when confident:**

```
TIER 0: SMOKE TEST — 1K-5K images (~1-2 minutes)
  Purpose: Verify the pipeline runs end-to-end without crashing
  - Check DataLoader, loss computation, backward pass, checkpointing
  - Confirm VRAM fits, bf16 works, gradients flow through DoRA
  - Run after ANY code change before committing to a longer run
  - No accuracy expectations — just "does it not crash?"

TIER 1: DEBUG — 10K-50K images (~10-30 minutes)
  Purpose: Verify learning signal and debug architecture
  - Loss should decrease meaningfully within 1-2 epochs
  - Check gradient norms, learning rates, auxiliary head balance
  - Inspect a few predictions qualitatively (are they on the right continent?)
  - Iterate multiple times per day — this is where most development happens

TIER 2: HYPERPARAMETER TUNING — 100K-200K images (~2-4 hours)
  Purpose: Tune hyperparameters and validate design decisions
  - Compare DoRA ranks (8 vs 16 vs 32)
  - Test geocell counts, label smoothing temperatures
  - Try auxiliary head weight ratios
  - Run 3-5 epochs, track validation metrics on Im2GPS3k
  - Can run 2-3 experiments per day

TIER 3: SERIOUS RUN — 1M stratified subset (~55 hours, ~2.3 days)
  Purpose: Full training for final model evaluation
  - Only run AFTER Tier 2 confirms hyperparameters are good
  - Geographically stratified sampling preserves global coverage
  - Minimum 100 images per country (oversample small countries)
  - Cap at 50K per country (undersample USA/Europe)
  - ~50GB on disk
  - Set it and forget it — checkpoint every 5K steps
  - Early stopping (patience=3) may finish in 4-6 epochs (~30-40 hours)

TIER 4 (OPTIONAL): FULL SCALE — 2M-5M images (~5-12 days)
  Purpose: Squeeze out last few % if 1M results leave headroom
  - Already downloaded locally — just expand the sampling
  - More data helps most for underrepresented regions (Africa, Central Asia)
  - Same code, same hyperparameters — just more data
  - Only attempt if Tier 3 model is close to but not meeting success criteria

RULE: 95% of development time is spent in Tiers 0-2 (minutes to hours).
      Tiers 3-4 are "fire and forget" runs you do 1-2 times total.
```

**Evaluation benchmarks:**
- Im2GPS3k (3,000 images) — classic benchmark
- YFCC4k (4,000 images) — diverse
- OSV-5M test split — modern benchmark with leaderboard

## 3.2 Phase 1.5: Contrastive Pretraining [NEW — from review]

**What was missing:** PIGEON doesn't go straight to classification. It first does
contrastive pretraining with geographic captions. This teaches the backbone to
encode geographic features into the embedding space — critical for Stage 3
(FAISS retrieval) to work.

```
WHY THIS MATTERS:
  Stage 3 refinement assumes that embedding similarity ≈ geographic proximity.
  Without contrastive pretraining, the CLIP embeddings encode generic visual
  similarity (two red cars are "close") not geographic similarity (two Bangkok
  streets are "close"). The two-stage pipeline BREAKS without this.

LIGHTWEIGHT VERSION (on RTX 4060):
  1. Take StreetCLIP as starting point (already partially geo-adapted)
  2. Generate synthetic captions from GPS metadata:
     Template: "A street view photo in {city}, {region}, {country}.
               Climate: {climate}. Driving: {driving_side}."
  3. Contrastive fine-tuning for 1-2 epochs:
     - Use CLIP's own contrastive loss (InfoNCE)
     - Match image embeddings to geographic text embeddings
     - Only update DoRA parameters (not full backbone)
  4. This takes ~1-2 hours on RTX 4060 for 1M images (bf16)
     - Prototype first on Tier 1 (10K-50K) in minutes to verify loss decreases
  5. After this, the embedding space is geographically structured

Paper basis: PIGEON Stage 1 + StreetCLIP methodology
```

## 3.3 Model Architecture

```python
class GeoLocator(nn.Module):
    def __init__(self, num_cells):
        super().__init__()

        # Backbone: StreetCLIP (CLIP ViT-L/14 fine-tuned for geo)
        self.clip = CLIPModel.from_pretrained("geolocal/StreetCLIP")
        self.vision_encoder = self.clip.vision_model
        self.visual_projection = self.clip.visual_projection  # 1024→768

        # Freeze all backbone params
        for param in self.vision_encoder.parameters():
            param.requires_grad = False

        # Apply DoRA to last 8 layers (Q, V projections)
        # DoRA = magnitude × direction decomposition of LoRA
        from peft import LoraConfig, get_peft_model
        dora_config = LoraConfig(
            r=16, lora_alpha=32,
            target_modules=["q_proj", "v_proj"],
            use_dora=True,  # Enable DoRA
            layers_to_transform=list(range(16, 24)),
        )
        self.vision_encoder = get_peft_model(self.vision_encoder, dora_config)

        # Classification head (main task)
        self.geo_head = nn.Sequential(
            nn.Linear(768, 2048),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(2048, num_cells)
        )

        # Auxiliary heads (multi-task regularization + explainability)
        self.scene_head = nn.Sequential(nn.Linear(768, 256), nn.ReLU(), nn.Linear(256, 6))
        self.climate_head = nn.Sequential(nn.Linear(768, 256), nn.ReLU(), nn.Linear(256, 5))
        self.driving_head = nn.Sequential(nn.Linear(768, 256), nn.ReLU(), nn.Linear(256, 3))
        self.region_head = nn.Sequential(nn.Linear(768, 256), nn.ReLU(), nn.Linear(256, 15))

    def forward(self, images):
        vision_out = self.vision_encoder(images)
        cls_token = vision_out.last_hidden_state[:, 0, :]
        embedding = self.visual_projection(cls_token)
        embedding = F.normalize(embedding, dim=-1)

        return {
            'geo': self.geo_head(embedding),
            'scene': self.scene_head(embedding),
            'climate': self.climate_head(embedding),
            'driving': self.driving_head(embedding),
            'region': self.region_head(embedding),
            'embedding': embedding,
        }
```

## 3.4 Training Loop

```python
# bfloat16 — RTX 4060 (compute capability 8.9) supports it natively
# bf16 advantage over fp16: wider dynamic range → no GradScaler needed
# This simplifies the training loop significantly

for batch_idx, (images, smooth_targets, aux_labels) in enumerate(dataloader):
    with torch.cuda.amp.autocast(dtype=torch.bfloat16):
        outputs = model(images)
        loss = compute_loss(outputs, smooth_targets, aux_labels)
        loss = loss / accumulation_steps

    loss.backward()  # No GradScaler needed with bf16!

    if (batch_idx + 1) % accumulation_steps == 0:
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        optimizer.zero_grad()
        scheduler.step()

    # Checkpoint every 5000 steps (safety — no session limits but good practice)
    if (batch_idx + 1) % 5000 == 0:
        save_checkpoint(model, optimizer, scheduler, batch_idx, epoch)
```

**Training config:**
```
batch_size = 6 (safe for 8GB VRAM with bf16)
accumulation_steps = 10 (effective batch = 60)
optimizer = AdamW(differential LR: DoRA=1e-4, heads=5e-4)
weight_decay = 0.01
scheduler = cosine with 1000-step warmup
epochs = 8-10 (early stopping patience=3 likely triggers at 4-6)

TRAINING TIME BY TIER (RTX 4060, bf16):
  Time per step: ~0.15 seconds (3x faster than T4)

  Tier 0 (1K-5K):     ~1-2 minutes    ← run after every code change
  Tier 1 (10K-50K):   ~10-30 minutes  ← main development loop
  Tier 2 (100K-200K): ~2-4 hours      ← hyperparameter tuning
  Tier 3 (1M):        ~30-55 hours    ← 1-2 final runs total
  Tier 4 (5M):        ~6-12 days      ← only if needed

  → 95% of dev time is Tier 0-2 (minutes to hours)
  → Tier 3+ are "set and forget" — you sleep, it trains
```

**Memory budget (RTX 4060, 8GB VRAM, bfloat16):**
```
CLIP ViT-L/14 weights (frozen, bf16):     ~600 MB
DoRA parameters:                           ~3 MB
Classification + aux heads:                ~40 MB
Forward activations (batch=6, bf16):       ~2.5 GB
Backward gradients (DoRA + heads):         ~1.2 GB
Optimizer states (AdamW):                  ~200 MB
CUDA/PyTorch overhead:                     ~2 GB
──────────────────────────────────────────────────
TOTAL:                                     ~6.5 GB ✓
Headroom:                                  ~1.5 GB
```

**Note on 16GB system RAM:** 1M images won't fit in RAM simultaneously.
DataLoader handles this via lazy loading (reads from disk per batch).
Use `num_workers=2` (not more — 16GB RAM is shared with the system).
Precomputed haversine sparse targets stored as memory-mapped numpy arrays.

## 3.5 Loss Function

```python
def compute_loss(outputs, smooth_geo_targets, aux_labels):
    geo_loss = F.kl_div(
        F.log_softmax(outputs['geo'], dim=-1),
        smooth_geo_targets,
        reduction='batchmean'
    )
    scene_loss = F.cross_entropy(outputs['scene'], aux_labels['scene'])
    climate_loss = F.cross_entropy(outputs['climate'], aux_labels['climate'])
    driving_loss = F.cross_entropy(outputs['driving'], aux_labels['driving'])
    region_loss = F.cross_entropy(outputs['region'], aux_labels['region'])

    total = (1.0 * geo_loss
           + 0.3 * scene_loss
           + 0.2 * climate_loss
           + 0.2 * driving_loss
           + 0.3 * region_loss)
    return total
```

## 3.6 Validation & Early Stopping [NEW — from review]

```
Train/Val split: 90/10 from the 1M subset (stratified by country)
Validation metric: median haversine distance (km) on val set
Early stopping: patience = 3 epochs (stop if val metric doesn't improve)
Log to W&B: loss curves, distance histograms, per-continent accuracy
```

---

# SECTION 4: TWO-STAGE REFINEMENT + FAISS

## 4.1 Why Two Stages?

Cell classification gives ~50-200km precision. PIGEON's insight: the CLIP
embedding encodes finer geographic information than the cell label. Two images
from the same neighborhood have more similar embeddings than two from different
parts of the same city. Exploit this with nearest-neighbor retrieval.

## 4.2 Pre-Computing Location Clusters

```
After training completes:

Step 1: Extract embeddings for all training images (768-dim each)
  Process in batches, save to disk (~3 GB for 1M images × 768 × 4 bytes)

Step 2: Within each geocell, cluster embeddings with OPTICS
  clusters = OPTICS(cell_embeddings, min_samples=10, metric='euclidean')
  For each cluster: store (mean_embedding, median_GPS, cell_id)
  Result: ~20,000-80,000 location clusters

Step 3: Build FAISS IndexFlatL2 over all cluster centroid embeddings
  IndexFlatL2: exact brute-force search
  At 80K vectors × 768 dims: ~230 MB, search in ~20ms
  No approximation error → no accuracy loss from retrieval
```

## 4.3 Inference Pipeline

```
Input: query image

STAGE 2 (Coarse):
  image → CLIP → 768-dim embedding → geo_head → cell probabilities
  top_K = top-50 cells by probability (PIGEON uses K=50)

STAGE 3 (Fine):
  Retrieve all clusters from top-50 cells (~500-3000 centroids)
  Find nearest centroid by L2 distance (≡ max cosine similarity for L2-normalized vectors)
  Nearest centroid's GPS = final prediction

Result: ~1-25km precision (vs. ~50-200km without refinement)
```

**Why PIGEON's two-stage over alternatives:**
- Cell-only: 50-200km precision. Not enough.
- Direct regression: converges to mean. Doesn't work.
- GeoCLIP (contrastive only): must search entire database. Slower, less accurate without cell shortlist.
- GeoToken (autoregressive): more complex to implement for marginal gains.
- GeoRanker (learning-to-rank): best results but most complex. Future enhancement.

---

# SECTION 5: AUXILIARY HEADS & EXPLAINABILITY

## 5.1 Multi-Task Learning

**Paper basis:** ISNs (2018) — scene classification as auxiliary task improves
geolocation by 3-5% through gradient sharing. Forces backbone to learn
discriminative geographic features.

**Labels derived from GPS (no manual annotation):**
- Scene type: from OpenStreetMap land use polygons or Places365 pretrained classifier
- Climate: Köppen classification raster lookup
- Driving side: country-level lookup table
- Region: GPS → country → UN geographic region

**Confidence weighting for noisy labels (from review):**
Driving side labels are meaningless for non-road images (forests, interiors).
Weight driving loss by P(road_visible) estimated from scene classifier.

## 5.2 Grad-CAM Explainability

Single backward pass → attention heatmap showing which regions drove the prediction.
Combine with auxiliary head outputs for structured explanations:
```
"Prediction: Bangkok, Thailand (78% confidence)
 Clues: Urban scene (92%), Tropical climate (88%),
        Left-hand traffic (76%), Southeast Asia (85%)
 [Heatmap highlights: Thai script on sign, temple roof, tuk-tuk]"
```

---

# SECTION 6: SUCCESS CRITERIA [NEW — from review]

## Baselines (measure first, before any training)
1. StreetCLIP zero-shot on Im2GPS3k (just run inference, no training)
2. OSV-5M paper's published baselines
3. GeoCLIP published numbers

## Targets
| Metric | Baseline (StreetCLIP ZS) | Our Target | Stretch (PIGEON-level) |
|--------|--------------------------|------------|----------------------|
| 25km (city) | ~15% | 25-30% | 35-40% |
| 200km (region) | ~35% | 50-55% | 55-65% |
| 750km (country) | ~55% | 70-75% | 75-85% |
| 2500km (continent) | ~80% | 88-92% | 92-95% |
| Median error (km) | ~1500 | ~500-800 | ~300-500 |

## Definition of done
- Model beats StreetCLIP zero-shot by >10% at 25km threshold
- Live Gradio demo on HuggingFace Spaces
- Clean GitHub repo with reproducible training pipeline
- Results table vs. baselines

---

# SECTION 7: PHASED IMPLEMENTATION PLAN

## Phase 1: Data & Infrastructure (Week 1)
- Download OSV-5M 1M stratified subset (or stream via HF datasets)
- Implement S2 cell partitioning (level 6-8, for prototyping)
- Generate auxiliary labels from GPS metadata
- Build Dataset/DataLoader with sparse haversine smoothing
- Set up W&B tracking + checkpoint/resume infrastructure
- **GATE:** DataLoader returns (image, smooth_target, aux_labels) correctly

## Phase 1.5: Contrastive Pretraining (Week 1-2)
- Generate synthetic geographic captions from GPS metadata
- Contrastive fine-tuning of StreetCLIP with DoRA (1-2 epochs, ~5 hours)
- Verify: embedding similarity correlates with geographic proximity
- **GATE:** Nearest-neighbor retrieval on embeddings returns geographically close images

## Phase 2: Core Classification Model (Week 2-3)
- Add classification head + auxiliary heads
- Implement haversine label smoothing with sparse precomputation
- Training loop: bfloat16 AMP + gradient accumulation + checkpointing
- **Iterate fast:** Tier 0 (smoke test) → Tier 1 (debug) → Tier 2 (tune hyperparams)
- **One Tier 3 run** (1M, ~2 days) only after Tier 2 confirms good hyperparameters
- Evaluate on Im2GPS3k: must beat StreetCLIP zero-shot
- **GATE:** >60% country-level accuracy on Im2GPS3k

## Phase 3: Semantic Geocells + Refinement (Week 3-4)
- Replace S2 cells with OPTICS semantic geocells
- Retrain model with better cells
- Extract embeddings → OPTICS clustering within cells → FAISS index
- Implement two-stage inference
- Evaluate city-level improvement
- **GATE:** >20% at 25km threshold

## Phase 4: Explainability + Error Analysis (Week 4-5)
- Grad-CAM visualization
- Structured explanation pipeline
- Per-continent, per-scene, urban/rural accuracy breakdown
- Failure mode categorization
- Test-Time Augmentation (TTA): average predictions over 3 crops → free +1-3%

## Phase 5: Demo + Deployment (Week 5-6)
- Gradio app: upload image → map prediction + explanation + confidence heatmap
- "Play against the AI" mode
- Benchmark comparison table
- Deploy to HuggingFace Spaces (free)
- Record 2-minute demo video, clean GitHub repo, write blog post

---

# SECTION 8: DECISION SUMMARY

| Component | Our Choice | Paper Basis | Key Alternative | Why Rejected |
|-----------|-----------|-------------|-----------------|-------------|
| Backbone | CLIP ViT-L/14 336px | PIGEON, StreetCLIP | DINOv2, ResNet | No text encoder; weaker geo features |
| Fine-tuning | DoRA (rank=16) | DoRA (Liu 2024) | LoRA | LoRA has OOD issues (intruder dimensions) |
| Starting point | StreetCLIP | StreetCLIP paper | Raw CLIP | Already geo-adapted |
| Contrastive pretrain | Yes (lightweight, 1-2 epochs) | PIGEON Stage 1 | Skip it | Embedding space needs geographic structure for Stage 3 |
| Cell creation | Semantic geocells (OPTICS+Voronoi) | PIGEON | S2 uniform | +112.6 km (PIGEON ablation) |
| Label smoothing | Haversine (τ=200, sparse top-50) | PIGEON | Hard labels | Geography-aware gradients |
| Loss | KL-div (main) + CE (aux) | Standard | Pure CE | KL-div for soft distributions |
| Multi-task | Scene, climate, driving, region | ISNs | Single-task | +3-5% accuracy + explainability |
| Optimizer | AdamW (differential LR) | Standard | SGD | More stable for fine-tuning |
| Precision | bfloat16 | — | float16 | RTX 4060 supports bf16; better numerical stability, no GradScaler needed |
| Refinement | Two-stage (cell→FAISS retrieval) | PIGEON | Single-stage | 2-3x city-level improvement |
| Retrieval | FAISS IndexFlatL2 | Standard | HNSW, IVF | Exact; fast at our scale |
| Explainability | Grad-CAM + aux heads | Standard CV | SHAP | Fast, prediction-specific |
| Training data | OSV-5M (1M stratified subset) | OSV-5M (CVPR 2024) | Full 5M | Storage/compute constraints |
| Compute | Local RTX 4060 (8GB, bf16) | — | Kaggle T4, vast.ai | Local is 3x faster, no session limits, bf16 support, unlimited storage |
| Demo | Gradio on HF Spaces | Standard | Streamlit | Free hosting |

---

# SECTION 9: RISK REGISTER

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| 8GB VRAM too tight for batch_size=6 | Low | Medium | Reduce to batch=4 with accumulation=15. Enable gradient checkpointing if needed |
| DoRA still underperforms partial unfreeze | Moderate | Medium | Rent A100 on vast.ai ($10-15) for a partial-unfreeze comparison run |
| Embedding space not geographically structured | High | High | Phase 1.5 contrastive pretraining addresses this |
| 1M images insufficient for rare regions (Africa, Central Asia) | High | Medium | Oversample underrepresented regions; scale to 2-3M if needed (local storage allows it) |
| Haversine smoothing becomes data loading bottleneck | Moderate | Low | Sparse precomputation (top-50 cells only) |
| 16GB system RAM limits DataLoader workers | Moderate | Low | Use num_workers=2, memory-mapped numpy arrays for precomputed targets |
| Laptop thermal throttling during long training runs | Moderate | Medium | Use a cooling pad; monitor GPU temp; reduce batch size if throttling occurs |
| StreetCLIP license (CC BY-NC 4.0) blocks commercial use | Low | Low | Portfolio project, not commercial. If needed, start from raw CLIP |

---

# SECTION 10: WHAT VISUAL FEATURES THE MODEL LEARNS

**Strong signals:**
- Text/script on signs (Thai, Arabic, Cyrillic, Hangul → narrows to few countries)
- Road marking styles (US yellow center, UK double yellow edge, Japan paint)
- Driving side (left = UK, Japan, Australia, India, etc.)
- License plate shapes
- Vegetation biome (tropical, boreal, Mediterranean, desert)

**Medium signals:**
- Architecture style (colonial, Soviet, Scandinavian, Japanese)
- Road surface quality and width
- Utility pole / power line styles
- Vehicle types (tuk-tuks, rickshaws)
- Sun angle / shadow direction

**Weak but useful:**
- Soil color (red laterite = tropical, sandy = arid)
- Cloud patterns, sky color
- Urban density, building heights
- Fence/wall construction styles
