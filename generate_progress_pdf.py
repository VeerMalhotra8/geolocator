"""Generate a detailed progress report PDF for the AI GeoGuessr project."""

from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import mm
from reportlab.lib.colors import HexColor
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak,
    HRFlowable,
)
from reportlab.lib.enums import TA_LEFT, TA_CENTER
from datetime import datetime

def build_pdf():
    doc = SimpleDocTemplate(
        "GEOPROJECT/PROGRESS_REPORT_2026-03-20.pdf",
        pagesize=A4,
        topMargin=20*mm, bottomMargin=20*mm,
        leftMargin=18*mm, rightMargin=18*mm,
    )

    styles = getSampleStyleSheet()
    title_style = ParagraphStyle("Title2", parent=styles["Title"], fontSize=20, spaceAfter=6)
    h1 = ParagraphStyle("H1", parent=styles["Heading1"], fontSize=15, spaceBefore=14, spaceAfter=6,
                         textColor=HexColor("#1a1a2e"))
    h2 = ParagraphStyle("H2", parent=styles["Heading2"], fontSize=12, spaceBefore=10, spaceAfter=4,
                         textColor=HexColor("#16213e"))
    body = ParagraphStyle("Body2", parent=styles["BodyText"], fontSize=9.5, leading=13, spaceAfter=4)
    bold_body = ParagraphStyle("BoldBody", parent=body, fontName="Helvetica-Bold")
    small = ParagraphStyle("Small", parent=body, fontSize=8.5, leading=11)
    red_body = ParagraphStyle("RedBody", parent=bold_body, textColor=HexColor("#cc0000"))
    green_body = ParagraphStyle("GreenBody", parent=bold_body, textColor=HexColor("#006600"))

    elements = []

    # ── TITLE ──
    elements.append(Paragraph("AI GeoGuessr — Progress Report", title_style))
    elements.append(Paragraph(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M')} | Phase 3 Complete | Next: Pipeline Fixes + Phase 4", body))
    elements.append(HRFlowable(width="100%", thickness=1, color=HexColor("#333333")))
    elements.append(Spacer(1, 6))

    # ── SECTION 1: CURRENT STATE SUMMARY ──
    elements.append(Paragraph("1. Current State Summary", h1))
    elements.append(Paragraph(
        "Model: StreetCLIP ViT-L/14 + DoRA adapters (0.13% params trainable) → 539 semantic geocell classifier + 4 auxiliary heads. "
        "Trained on 50K images from OSV-5M (100 countries × 500 imgs, stratified). 15 epochs of semantic geocell training completed.", body))

    state_data = [
        ["Component", "Status", "Notes"],
        ["Phase 1: Data & Infra", "COMPLETE", "50K images, DataLoader, S2 cells"],
        ["Phase 1.5: Contrastive Pretrain", "COMPLETE", "2 epochs, NN dist -25.6%"],
        ["Phase 2: Core Classification", "COMPLETE", "5 epochs S2 cells, 174km median"],
        ["Phase 3: Semantic Geocells", "COMPLETE (training)", "15 epochs, 130km median (internal)"],
        ["Phase 3: FAISS Refinement", "NOT STARTED", "Code written, needs checkpoint"],
        ["Phase 4: Explainability", "NOT STARTED", "Grad-CAM, error analysis"],
        ["Phase 5: Demo", "NOT STARTED", "Gradio app"],
    ]
    t = Table(state_data, colWidths=[140, 100, 230])
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), HexColor("#1a1a2e")),
        ("TEXTCOLOR", (0, 0), (-1, 0), HexColor("#ffffff")),
        ("FONTSIZE", (0, 0), (-1, -1), 8.5),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("GRID", (0, 0), (-1, -1), 0.5, HexColor("#cccccc")),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [HexColor("#f5f5f5"), HexColor("#ffffff")]),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("LEFTPADDING", (0, 0), (-1, -1), 6),
    ]))
    elements.append(t)
    elements.append(Spacer(1, 8))

    # ── SECTION 2: TRAINING RESULTS ──
    elements.append(Paragraph("2. Training Results — Semantic Geocells (Phase 3)", h1))
    elements.append(Paragraph(
        "539 OPTICS semantic geocells (down from 8,970 S2 cells). Min 21 imgs/cell, median 81, max 469. "
        "15 epochs, batch 6, accumulation 10, CosineWithWarmup, patience=3 (never triggered).", body))

    train_data = [
        ["Epoch", "Val Median km", "<25km", "<200km", "<750km", "<2500km", "Train Loss"],
        ["1", "497.3", "10.4%", "30.4%", "58.4%", "84.1%", "3.765"],
        ["3", "217.2", "19.3%", "48.4%", "76.1%", "91.4%", "1.415"],
        ["5", "179.8", "21.3%", "52.4%", "79.0%", "92.4%", "1.152"],
        ["8", "151.4", "23.4%", "56.1%", "81.6%", "93.0%", "0.943"],
        ["11", "137.2", "25.1%", "58.2%", "82.0%", "93.0%", "0.827"],
        ["14 (best)", "130.4", "25.8%", "58.8%", "82.4%", "93.4%", "0.773"],
        ["15", "132.7", "25.5%", "58.6%", "82.3%", "93.4%", "0.772"],
    ]
    t = Table(train_data, colWidths=[60, 75, 50, 50, 50, 55, 60])
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), HexColor("#16213e")),
        ("TEXTCOLOR", (0, 0), (-1, 0), HexColor("#ffffff")),
        ("FONTSIZE", (0, 0), (-1, -1), 8),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("GRID", (0, 0), (-1, -1), 0.5, HexColor("#cccccc")),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [HexColor("#f5f5f5"), HexColor("#ffffff")]),
        ("BACKGROUND", (0, 6), (-1, 6), HexColor("#e6ffe6")),
        ("ALIGN", (1, 0), (-1, -1), "CENTER"),
    ]))
    elements.append(t)
    elements.append(Spacer(1, 6))

    elements.append(Paragraph("S2 Cells vs Semantic Geocells (best epoch):", h2))
    comp_data = [
        ["Metric", "S2 Cells (8970)", "Semantic (539)", "Change"],
        ["Median km", "174.4", "130.4", "-25.2%"],
        ["<25km", "15.3%", "25.8%", "+10.5pp"],
        ["<200km", "53.1%", "58.8%", "+5.7pp"],
        ["<750km", "80.0%", "82.4%", "+2.4pp"],
    ]
    t = Table(comp_data, colWidths=[80, 100, 100, 80])
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), HexColor("#16213e")),
        ("TEXTCOLOR", (0, 0), (-1, 0), HexColor("#ffffff")),
        ("FONTSIZE", (0, 0), (-1, -1), 8.5),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("GRID", (0, 0), (-1, -1), 0.5, HexColor("#cccccc")),
        ("ALIGN", (1, 0), (-1, -1), "CENTER"),
    ]))
    elements.append(t)

    # ── SECTION 3: CRITICAL FINDING — METRIC INFLATION ──
    elements.append(Spacer(1, 10))
    elements.append(Paragraph("3. CRITICAL FINDING: Validation Metric Inflation", h1))
    elements.append(Paragraph(
        "The internal val metrics (130km median) are significantly inflated due to spatial leakage in the train/val split. "
        "External benchmark evaluation on Im2GPS3k (3,000 Flickr images, zero overlap with training) reveals the true performance.", red_body))

    elements.append(Paragraph("Spatial Proximity Audit Results:", h2))
    elements.append(Paragraph(
        "38% of val images have a training image within 1km. 75% within 5km. 96% within 25km. "
        "Median nearest-train-neighbor distance: 1.6km. The model memorizes locations rather than learning generalizable features.", body))

    audit_data = [
        ["Threshold", "Val imgs with train neighbor closer", "Implication"],
        ["< 0.1 km", "3.1% (152 images)", "Near-duplicate leakage"],
        ["< 1.0 km", "38.0% (1,880 images)", "Same street/block"],
        ["< 5.0 km", "74.6% (3,697 images)", "Same neighborhood"],
        ["< 25 km", "95.9% (4,749 images)", "Same city"],
    ]
    t = Table(audit_data, colWidths=[70, 170, 160])
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), HexColor("#cc0000")),
        ("TEXTCOLOR", (0, 0), (-1, 0), HexColor("#ffffff")),
        ("FONTSIZE", (0, 0), (-1, -1), 8.5),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("GRID", (0, 0), (-1, -1), 0.5, HexColor("#cccccc")),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [HexColor("#fff0f0"), HexColor("#ffffff")]),
    ]))
    elements.append(t)
    elements.append(Spacer(1, 6))

    elements.append(Paragraph("Im2GPS3k Benchmark Results (THE REAL NUMBERS):", h2))
    real_data = [
        ["Metric", "Internal Val (leaky)", "Im2GPS3k (honest)", "Inflation Factor"],
        ["Median km", "130.4", "672.0", "5.2x"],
        ["<25km", "25.8%", "11.3%", "2.3x"],
        ["<200km", "58.8%", "23.1%", "2.5x"],
        ["<750km", "82.4%", "52.8%", "1.6x"],
        ["<2500km", "93.4%", "77.6%", "1.2x"],
    ]
    t = Table(real_data, colWidths=[80, 110, 110, 90])
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), HexColor("#cc0000")),
        ("TEXTCOLOR", (0, 0), (-1, 0), HexColor("#ffffff")),
        ("FONTSIZE", (0, 0), (-1, -1), 8.5),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("GRID", (0, 0), (-1, -1), 0.5, HexColor("#cccccc")),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [HexColor("#fff0f0"), HexColor("#ffe0e0")]),
        ("ALIGN", (1, 0), (-1, -1), "CENTER"),
    ]))
    elements.append(t)
    elements.append(Spacer(1, 4))
    elements.append(Paragraph(
        "Note: 672km median on Im2GPS3k with only 50K training images is still within the architecture doc's target range (500-800km). "
        "The model IS learning — the internal metrics were just misleadingly optimistic. "
        "Im2GPS3k is also harder than OSV-5M (Flickr photos vs street view — more varied angles, indoor shots, etc.).", body))

    elements.append(Paragraph("Root Causes:", h2))
    causes = [
        ["Issue", "Severity", "Impact", "Fix"],
        ["Random 90/10 train/val split\n(no geographic separation)", "CRITICAL",
         "38% val images have train neighbor <1km.\nModel memorizes locations.", "Geographic split by city cluster\nor use external benchmarks only"],
        ["Geocells built on all data\n(train+val)", "MODERATE",
         "Cell centroids influenced by val GPS.\nTarget space contaminated.", "Rebuild geocells from train-only\n(architecture doc already specifies this)"],
        ["No external benchmark eval\n(Im2GPS3k, YFCC4k)", "HIGH",
         "No honest performance signal.\nAll optimization against leaky metric.", "FIXED: eval_benchmark.py written\nand Im2GPS3k eval complete"],
    ]
    t = Table(causes, colWidths=[120, 60, 150, 150])
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), HexColor("#333333")),
        ("TEXTCOLOR", (0, 0), (-1, 0), HexColor("#ffffff")),
        ("FONTSIZE", (0, 0), (-1, -1), 7.5),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("GRID", (0, 0), (-1, -1), 0.5, HexColor("#cccccc")),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
    ]))
    elements.append(t)

    # ── SECTION 4: WHAT'S BEEN BUILT ──
    elements.append(PageBreak())
    elements.append(Paragraph("4. Complete Inventory — What's Been Built", h1))

    code_data = [
        ["File", "Purpose", "Status"],
        ["geoguessr/data/download.py", "Stream OSV-5M from HuggingFace, stratified sample by country", "COMPLETE"],
        ["geoguessr/data/dataset.py", "PyTorch Dataset/DataLoader, CLIP preprocessing, aux labels, haversine smoothing", "COMPLETE"],
        ["geoguessr/data/geocells.py", "S2 cell partitioning + haversine label smoothing", "COMPLETE (superseded)"],
        ["geoguessr/data/semantic_geocells.py", "OPTICS clustering + Voronoi + fast haversine smoothing", "COMPLETE"],
        ["geoguessr/model/contrastive.py", "Contrastive pretraining with geographic captions", "COMPLETE"],
        ["geoguessr/model/geolocator.py", "GeoLocator: StreetCLIP + DoRA + geo head + aux heads", "COMPLETE"],
        ["geoguessr/model/faiss_refinement.py", "Two-stage FAISS retrieval (build index + predict)", "COMPLETE (code)"],
        ["geoguessr/train.py", "Training loop: KL-div + aux CE, early stopping, checkpointing", "COMPLETE"],
        ["geoguessr/inference.py", "End-to-end inference pipeline", "COMPLETE"],
        ["geoguessr/eval_benchmark.py", "External benchmark evaluation (Im2GPS3k, YFCC4k)", "NEW — just built"],
    ]
    t = Table(code_data, colWidths=[140, 220, 90])
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), HexColor("#1a1a2e")),
        ("TEXTCOLOR", (0, 0), (-1, 0), HexColor("#ffffff")),
        ("FONTSIZE", (0, 0), (-1, -1), 7.5),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("GRID", (0, 0), (-1, -1), 0.5, HexColor("#cccccc")),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [HexColor("#f5f5f5"), HexColor("#ffffff")]),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
    ]))
    elements.append(t)
    elements.append(Spacer(1, 6))

    elements.append(Paragraph("Checkpoints on disk:", h2))
    ckpt_data = [
        ["File", "Description"],
        ["contrastive_best.pt", "Contrastive pretraining (2 epochs) — DoRA adapter weights"],
        ["geolocator_best.pt", "Best semantic geocell model (epoch 14, 130.4km internal / 672km Im2GPS3k)"],
        ["geolocator_epoch1-15.pt", "All epoch checkpoints for semantic geocell training"],
        ["step_1000-8000.pt", "Mid-training step checkpoints"],
        ["training_log.json", "Full training metrics for all 15 epochs"],
        ["im2gps3k_results.json", "Im2GPS3k benchmark results"],
    ]
    t = Table(ckpt_data, colWidths=[140, 330])
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), HexColor("#16213e")),
        ("TEXTCOLOR", (0, 0), (-1, 0), HexColor("#ffffff")),
        ("FONTSIZE", (0, 0), (-1, -1), 8),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("GRID", (0, 0), (-1, -1), 0.5, HexColor("#cccccc")),
    ]))
    elements.append(t)

    # ── SECTION 5: HARDWARE & SCALING ──
    elements.append(Spacer(1, 10))
    elements.append(Paragraph("5. Hardware & Scaling Analysis", h1))
    elements.append(Paragraph(
        "Current: RTX 4060 Laptop (8GB VRAM), batch 6, accumulation 10. "
        "50K training: ~33 min/epoch. GPU runs at 89-94C, 94% utilization.", body))

    scale_data = [
        ["", "50K (current)", "1M (Tier 3)", "5M (Tier 4)"],
        ["Data size", "~25 GB", "~50 GB", "~250 GB"],
        ["Steps/epoch", "833", "16,667", "83,333"],
        ["Time/epoch (4060)", "33 min", "~11 hours", "~55 hours"],
        ["5 epochs total", "2.75 hours", "~55 hours", "~275 hours"],
        ["Recommended GPU", "Local 4060", "Local 4060 (2 days)", "Cloud H200 SXM (~23h, ~₹3,200)"],
    ]
    t = Table(scale_data, colWidths=[100, 100, 110, 160])
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), HexColor("#16213e")),
        ("TEXTCOLOR", (0, 0), (-1, 0), HexColor("#ffffff")),
        ("FONTSIZE", (0, 0), (-1, -1), 8),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTNAME", (0, 1), (0, -1), "Helvetica-Bold"),
        ("GRID", (0, 0), (-1, -1), 0.5, HexColor("#cccccc")),
        ("ALIGN", (1, 0), (-1, -1), "CENTER"),
    ]))
    elements.append(t)
    elements.append(Spacer(1, 4))
    elements.append(Paragraph(
        "Best cloud option for 5M run: H200 SXM 1x at ₹140/hr (~$1.67). ~23 hours, ~₹3,200 total. "
        "HuggingFace streaming eliminates need to download 250GB — stream images on-the-fly during training.", body))

    # ── SECTION 6: FUTURE PLAN ──
    elements.append(Paragraph("6. Future Plan — Prioritized Next Steps", h1))

    elements.append(Paragraph("PRIORITY 1: Fix Pipeline (before spending money on 5M)", h2))
    p1_data = [
        ["Task", "What to do", "Why"],
        ["A. Geographic val split", "Split train/val by city cluster or OPTICS region.\nNo val image within 25km+ of any train image.", "Eliminates spatial leakage.\nInternal metrics become trustworthy."],
        ["B. Geocells from train-only", "Rebuild OPTICS geocells using only train split.\nArchitecture doc already specifies this.", "Removes target space contamination.\nAligns with PIGEON methodology."],
        ["C. Always eval on Im2GPS3k", "Use eval_benchmark.py after every training run.\nThis is the honest metric.", "Zero leakage. Standard benchmark.\nComparable to published results."],
        ["D. FAISS two-stage refinement", "Build FAISS index from training embeddings.\nRun two-stage inference on Im2GPS3k.", "Phase 3 completion. Expected to\nimprove city-level (<25km) by 2-3x."],
    ]
    t = Table(p1_data, colWidths=[110, 185, 170])
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), HexColor("#006600")),
        ("TEXTCOLOR", (0, 0), (-1, 0), HexColor("#ffffff")),
        ("FONTSIZE", (0, 0), (-1, -1), 7.5),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("GRID", (0, 0), (-1, -1), 0.5, HexColor("#cccccc")),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
    ]))
    elements.append(t)
    elements.append(Spacer(1, 6))

    elements.append(Paragraph("PRIORITY 2: Phase 4 — Explainability & Error Analysis", h2))
    elements.append(Paragraph(
        "Grad-CAM visualization. Per-continent/per-country accuracy breakdown on Im2GPS3k. "
        "Failure mode categorization (indoor vs outdoor, urban vs rural, text-heavy vs featureless). "
        "Test-Time Augmentation (TTA): average predictions over 3 crops.", body))

    elements.append(Paragraph("PRIORITY 3: Scale to 1M on Local 4060 (Tier 3)", h2))
    elements.append(Paragraph(
        "After pipeline is bulletproof: download 1M stratified subset from OSV-5M. "
        "Rebuild semantic geocells (expect ~2,000-5,000 cells at 1M). "
        "Train from contrastive checkpoint, ~55 hours (2.3 days). "
        "Evaluate on Im2GPS3k — target: 400-500km median.", body))

    elements.append(Paragraph("PRIORITY 4: Scale to 5M on Cloud GPU (Tier 4)", h2))
    elements.append(Paragraph(
        "Only after 1M results confirm pipeline works. H200 SXM 1x at ₹140/hr. "
        "Stream from HuggingFace (no 250GB download). ~23 hours, ~₹3,200. "
        "Need: streaming dataset code, cloud setup script, robust checkpointing for cloud interrupts. "
        "Target: 250-350km median on Im2GPS3k.", body))

    elements.append(Paragraph("PRIORITY 5: Phase 5 — Demo & Deployment", h2))
    elements.append(Paragraph(
        "Gradio app on HuggingFace Spaces. Upload image → map prediction + Grad-CAM heatmap + "
        "confidence + clue explanation (scene, climate, driving side, region). "
        "'Play against the AI' mode. Blog post + GitHub repo.", body))

    # ── SECTION 7: COMPARISON TO TARGETS ──
    elements.append(Spacer(1, 10))
    elements.append(Paragraph("7. Where We Stand vs Architecture Doc Targets", h1))

    target_data = [
        ["Metric", "StreetCLIP ZS\n(baseline)", "Our Model\n(Im2GPS3k)", "Target\n(arch doc)", "Stretch\n(PIGEON)"],
        ["<25km (city)", "~15%", "11.3%", "25-30%", "35-40%"],
        ["<200km (region)", "~35%", "23.1%", "50-55%", "55-65%"],
        ["<750km (country)", "~55%", "52.8%", "70-75%", "75-85%"],
        ["<2500km (cont.)", "~80%", "77.6%", "88-92%", "92-95%"],
        ["Median km", "~1500", "672", "500-800", "300-500"],
    ]
    t = Table(target_data, colWidths=[80, 80, 80, 80, 80])
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), HexColor("#1a1a2e")),
        ("TEXTCOLOR", (0, 0), (-1, 0), HexColor("#ffffff")),
        ("FONTSIZE", (0, 0), (-1, -1), 8),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTNAME", (0, 1), (0, -1), "Helvetica-Bold"),
        ("GRID", (0, 0), (-1, -1), 0.5, HexColor("#cccccc")),
        ("ALIGN", (1, 0), (-1, -1), "CENTER"),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [HexColor("#f5f5f5"), HexColor("#ffffff")]),
    ]))
    elements.append(t)
    elements.append(Spacer(1, 4))
    elements.append(Paragraph(
        "Median km (672) is within target range (500-800). But threshold metrics are below target — "
        "this is expected with only 50K training images and no FAISS refinement yet. "
        "The <25km metric (11.3%) is below StreetCLIP baseline (15%), likely because our 539 cells are too coarse "
        "for city-level precision. FAISS refinement should recover this. Scaling to 1M+ will push all metrics up.", body))

    # ── SECTION 8: STRATEGIC ANALYSIS ──
    elements.append(Spacer(1, 10))
    elements.append(Paragraph("8. Strategic Analysis — What the Im2GPS3k Results Tell Us", h1))

    elements.append(Paragraph("Key Insight: We're WORSE than StreetCLIP zero-shot at city-level (<25km).", red_body))
    elements.append(Paragraph(
        "Our model gets 11.3% at <25km vs StreetCLIP's ~15% baseline. This means the classification head (539 cells) "
        "is too coarse for city-level precision — predicting a cell centroid often lands 50-100km away from the true location. "
        "However, we're competitive at country/continent level (52.8% at <750km, 77.6% at <2500km), showing the model "
        "DOES learn broad geographic features. The problem is purely resolution.", body))

    strat_data = [
        ["Finding", "Diagnosis", "Strategy"],
        ["11.3% <25km\n(below 15% baseline)", "539 cells too coarse.\nCell centroids are ~100km apart.\nClassification alone can't hit 25km.", "FAISS refinement is CRITICAL.\nTwo-stage inference (cell → NN retrieval)\nshould recover 20-30% at <25km."],
        ["672km median\n(within 500-800 target)", "Broad geography working.\nModel distinguishes continents/regions.\nBut many wrong-country errors.", "More training data (1M) will fill gaps.\nCurrent 50K has only 500/country.\nRare regions (Central Asia, Africa) starved."],
        ["52.8% <750km\n(≈ StreetCLIP baseline)", "Minimal improvement over\nzero-shot at country level.\n50K not enough to beat a model\ntrained on 400M images.", "Scale is the lever here.\n1M images should push to 65-70%.\n5M should hit 75%+."],
        ["Domain gap:\nIm2GPS3k = Flickr photos\nTrain data = Street View", "Im2GPS3k has indoor shots,\ntourist photos, unusual angles.\nOSV-5M is all dashcam/street.", "Accept: Im2GPS3k is HARDER\nthan street-view benchmarks.\nAlso eval on OSV-5M test split\nfor fair comparison."],
    ]
    t = Table(strat_data, colWidths=[100, 150, 200])
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), HexColor("#1a1a2e")),
        ("TEXTCOLOR", (0, 0), (-1, 0), HexColor("#ffffff")),
        ("FONTSIZE", (0, 0), (-1, -1), 7.5),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("GRID", (0, 0), (-1, -1), 0.5, HexColor("#cccccc")),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [HexColor("#f5f5f5"), HexColor("#ffffff")]),
    ]))
    elements.append(t)
    elements.append(Spacer(1, 6))

    elements.append(Paragraph("Revised Success Milestones:", h2))
    mile_data = [
        ["Milestone", "Expected Result", "What It Proves"],
        ["FAISS refinement on\ncurrent 50K model", "<25km: 15-20%\nMedian: 500-600km", "Two-stage pipeline works.\nRecovers city-level precision."],
        ["Pipeline fixes\n(geo split + train-only cells)", "Honest internal metrics\naligned with Im2GPS3k", "Trustworthy optimization signal.\nNo more false progress."],
        ["1M training\n(local 4060, ~2 days)", "<25km: 20-25%\nMedian: 350-450km", "Scale improves performance.\nJustifies investment in 5M run."],
        ["5M training\n(cloud H200, ~1 day)", "<25km: 28-35%\nMedian: 250-350km", "Competitive with published\nstate-of-art on Im2GPS3k."],
        ["FAISS on 5M model", "<25km: 35-45%\nMedian: 150-250km", "Full pipeline, all data.\nPortfolio-ready result."],
    ]
    t = Table(mile_data, colWidths=[110, 130, 180])
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), HexColor("#006600")),
        ("TEXTCOLOR", (0, 0), (-1, 0), HexColor("#ffffff")),
        ("FONTSIZE", (0, 0), (-1, -1), 7.5),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("GRID", (0, 0), (-1, -1), 0.5, HexColor("#cccccc")),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [HexColor("#f0fff0"), HexColor("#ffffff")]),
    ]))
    elements.append(t)
    elements.append(Spacer(1, 6))

    elements.append(Paragraph("Highest-ROI Next Action: Run FAISS refinement on current model.", green_body))
    elements.append(Paragraph(
        "This is zero-cost (no retraining needed) and should dramatically improve <25km accuracy. "
        "The FAISS code is already written (faiss_refinement.py). Just need to: "
        "(1) extract embeddings from training images, (2) cluster within cells, (3) build FAISS index, "
        "(4) re-evaluate on Im2GPS3k with two-stage inference. If <25km jumps to 15-20%, the architecture is validated "
        "and we can scale with confidence.", body))

    # ── SECTION 9: WSL & INDIA MODEL (FUTURE) ──
    elements.append(Spacer(1, 10))
    elements.append(Paragraph("9. Deferred Items", h1))
    elements.append(Paragraph("WSL Migration:", h2))
    elements.append(Paragraph(
        "WSL2 Ubuntu 24.04 already installed (just stopped). nvidia-smi works in WSL. "
        "Plan: install CUDA toolkit, set up conda env, copy project (~30GB), verify GPU, benchmark speed. "
        "Expected 10-20% training speed improvement. Do this before any multi-day training run.", body))
    elements.append(Paragraph("India-Only Model:", h2))
    elements.append(Paragraph(
        "Download 500K-1M India street images via Mapillary API v4 (free). "
        "OSV-5M only has 50K India images with South India bias. "
        "Build India-specific semantic geocells (~300-800 cells), swap region head for state classification (31 classes). "
        "Fine-tune from global contrastive checkpoint. 'Where in India?' demo. DEFERRED — revisit after scaling.", body))

    # ── FOOTER ──
    elements.append(Spacer(1, 20))
    elements.append(HRFlowable(width="100%", thickness=0.5, color=HexColor("#999999")))
    elements.append(Paragraph(
        f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')} | "
        "Key files: GEOGUESSR_ARCHITECTURE.md (design), PROGRESS_AND_SCALING.md (detailed log), "
        "checkpoints/training_log.json (epoch metrics), checkpoints/im2gps3k_results.json (benchmark)",
        small))

    doc.build(elements)
    print("PDF generated: GEOPROJECT/PROGRESS_REPORT_2026-03-20.pdf")


if __name__ == "__main__":
    build_pdf()
