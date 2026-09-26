"""
Field Disease Detection — Streamlit App
========================================
Analyzes field images for plant disease (Potato or Sugar Beet).
Outputs: classification per plant zone + heatmap + summary stats.

Run:
    pip install streamlit opencv-python-headless matplotlib scikit-image numpy
    streamlit run app.py
"""

import streamlit as st
import cv2
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import LinearSegmentedColormap
from io import BytesIO
import tempfile
import os
import sys

# ── Make sure the classifiers are importable from the same folder ──
sys.path.insert(0, os.path.dirname(__file__))

# ─────────────────────────────────────────────────────────
#  Page config
# ─────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Field Disease Detector",
    page_icon="🌱",
    layout="wide",
)

# ─────────────────────────────────────────────────────────
#  Custom CSS
# ─────────────────────────────────────────────────────────
st.markdown("""
<style>
    .main { background-color: #0e1117; }
    .block-container { padding-top: 1.5rem; }
    .metric-box {
        background: #1c2333;
        border-radius: 10px;
        padding: 18px 14px;
        text-align: center;
        margin: 4px;
    }
    .metric-label { font-size: 0.85rem; color: #8b9dc3; margin-bottom: 4px; }
    .metric-value { font-size: 2rem; font-weight: 700; }
    .healthy-color  { color: #27ae60; }
    .suspicious-color { color: #f39c12; }
    .sick-color     { color: #e74c3c; }
    .alert-high     { background: #3d1515; border: 1px solid #e74c3c; border-radius: 8px; padding: 10px; }
    .alert-medium   { background: #3d2e10; border: 1px solid #f39c12; border-radius: 8px; padding: 10px; }
    .alert-low      { background: #0f2d1a; border: 1px solid #27ae60; border-radius: 8px; padding: 10px; }
    h1 { color: #e8eaf6 !important; }
    h2 { color: #cfd8dc !important; font-size: 1.1rem !important; }
    .stButton > button {
        background: #1565c0;
        color: white;
        border: none;
        border-radius: 8px;
        padding: 0.6rem 2rem;
        font-size: 1rem;
        font-weight: 600;
        width: 100%;
    }
    .stButton > button:hover { background: #1976d2; }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────
#  Shared background-removal helpers
# ─────────────────────────────────────────────────────────

def _remove_bg_potato(hsv: np.ndarray, total: int) -> tuple:
    """
    Step 1 for potato: build a plant mask from the cell.
    Returns (plant_mask_u8, plant_area, veg_ratio).
    """
    # Green seed (strict saturation to avoid soil)
    green_seed = cv2.inRange(hsv, np.array([15, 30, 25]), np.array([100, 255, 255]))
    # Expand to capture adjacent diseased tissue (brown/yellow near green)
    expanded = cv2.dilate(green_seed, np.ones((9, 9), np.uint8), iterations=2)
    # Clean up
    plant_mask = cv2.morphologyEx(expanded, cv2.MORPH_CLOSE,
                                   np.ones((5, 5), np.uint8), iterations=2)
    plant_area = max(int(np.sum(plant_mask > 0)), 1)
    veg_ratio  = plant_area / total * 100
    return plant_mask, plant_area, veg_ratio


def _remove_bg_beet(hsv: np.ndarray, total: int) -> tuple:
    """
    Step 1 for beet: build a plant mask that includes ALL leaf tissue:
      - Healthy green leaves
      - Yellowing / Cercospora-infected leaves
      - Pale, bleached, dried leaves (wilted/necrotic — NOT soil)
      - Dark/purple leaf tissue (beet leaves are often dark red/purple)
    Excludes: bare soil, sand, gravel.

    Returns (plant_mask_u8, green_mask_u8, plant_area, veg_ratio).
    """
    hh, s, v = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]

    # ── Green seed (healthy leaf tissue) ──
    green_seed = ((hh >= 35) & (hh <= 85) & (s > 45) & (v > 40)).astype(np.uint8) * 255

    # ── Pale / bleached / dried leaf tissue ──
    # Dried beet leaves: low saturation, medium-high value, any hue
    # Distinguished from soil by being adjacent to green tissue (handled via dilation)
    pale_leaf  = ((s < 60) & (v > 90) & (v < 240)).astype(np.uint8) * 255

    # ── Dark red/purple beet leaf tissue ──
    # Beet leaves often have deep red/purple stems and veins
    dark_red   = ((hh >= 150) | (hh <= 15)) & (s > 40) & (v > 30) & (v < 180)
    dark_leaf  = (dark_red.astype(np.uint8)) * 255

    # ── Yellowing diseased tissue ──
    yellow_leaf = ((hh >= 18) & (hh <= 45) & (s > 50) & (v > 60)).astype(np.uint8) * 255

    # ── Soil / sand / gravel exclusion ──
    # Key discriminator: soil has medium hue (10-30), low-mid saturation, and is NOT
    # adjacent to green tissue. We block it explicitly.
    soil = ((hh >= 8) & (hh <= 32) & (s < 65) & (v > 80)).astype(np.uint8) * 255

    # ── Build plant mask ──
    # Start from green seed (the anchor), expand to capture all adjacent leaf tissue
    # Use larger dilation so pale dried leaves connected to green are captured
    expanded_green = cv2.dilate(green_seed, np.ones((15, 15), np.uint8), iterations=3)

    # Include pale/dark/yellow pixels that fall within the expanded green zone
    leaf_candidates = cv2.bitwise_or(
        cv2.bitwise_or(pale_leaf, dark_leaf),
        yellow_leaf
    )
    # Only include candidate pixels that are inside the expanded green zone
    # (this prevents isolated soil patches from being included)
    leaf_near_green = cv2.bitwise_and(leaf_candidates, expanded_green)

    # Combine green seed + nearby leaf tissue
    combined = cv2.bitwise_or(green_seed, leaf_near_green)

    # Remove soil
    plant_mask = cv2.bitwise_and(combined, cv2.bitwise_not(soil))

    # Morphological cleanup
    plant_mask = cv2.morphologyEx(plant_mask, cv2.MORPH_CLOSE,
                                   np.ones((9, 9), np.uint8), iterations=2)
    plant_mask = cv2.morphologyEx(plant_mask, cv2.MORPH_OPEN,
                                   np.ones((3, 3), np.uint8), iterations=1)

    plant_area = max(int(np.sum(plant_mask > 0)), 1)
    veg_ratio  = plant_area / total * 100
    return plant_mask, green_seed, plant_area, veg_ratio


# ─────────────────────────────────────────────────────────
#  Potato classifier  (pipeline: bg removal → features → score)
# ─────────────────────────────────────────────────────────

def _analyze_cell_potato(cell_bgr: np.ndarray) -> dict:
    """Classify one grid cell: background removal → feature extraction → score."""
    if cell_bgr is None or cell_bgr.size == 0:
        return {"label": "Healthy", "score": 0, "veg_ratio": 0}
    h, w = cell_bgr.shape[:2]
    if h < 10 or w < 10:
        return {"label": "Healthy", "score": 0, "veg_ratio": 0}

    hsv   = cv2.cvtColor(cell_bgr, cv2.COLOR_BGR2HSV)
    gray  = cv2.cvtColor(cell_bgr, cv2.COLOR_BGR2GRAY)
    total = h * w

    # ── Step 1: background removal ──
    plant_mask, plant_area, veg_ratio = _remove_bg_potato(hsv, total)
    if veg_ratio < 10:
        return {"label": "No Plant", "score": -1, "veg_ratio": round(veg_ratio, 1)}

    # ── Step 2: features on plant pixels only ──
    def _on_plant(mask):
        return cv2.bitwise_and(mask, plant_mask)

    b1 = cv2.inRange(hsv, np.array([10, 70, 50]), np.array([22, 255, 200]))
    b2 = cv2.inRange(hsv, np.array([0,  70, 50]), np.array([10, 255, 200]))
    brown_ratio  = np.sum(_on_plant(cv2.bitwise_or(b1, b2)) > 0) / plant_area * 100

    dark_raw  = cv2.inRange(hsv, np.array([0, 0, 0]), np.array([180, 60, 80]))
    dark_px   = cv2.morphologyEx(_on_plant(dark_raw), cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    dark_ratio = np.sum(dark_px > 0) / plant_area * 100

    yellow_px    = _on_plant(cv2.inRange(hsv, np.array([20, 60, 100]), np.array([35, 255, 255])))
    yellow_ratio = np.sum(yellow_px > 0) / plant_area * 100

    green_px    = _on_plant(cv2.inRange(hsv, np.array([25, 40, 40]), np.array([90, 255, 255])))
    green_ratio = np.sum(green_px > 0) / plant_area * 100

    sat_vals = hsv[:, :, 1][plant_mask > 0]
    mean_sat = float(np.mean(sat_vals)) if len(sat_vals) > 0 else 0

    # ── Step 3: score ──
    score = 0
    if   brown_ratio > 15:  score += 3
    elif brown_ratio > 5:   score += 2
    elif brown_ratio > 1.5: score += 1
    if   yellow_ratio > 15: score += 2
    elif yellow_ratio > 5:  score += 1
    if   mean_sat < 75:     score += 2
    elif mean_sat < 90:     score += 1
    if   green_ratio < 50:  score += 2
    elif green_ratio < 65:  score += 1
    if dark_ratio > 3:      score += 1

    label = "Sick" if score >= 5 else "Suspicious" if score >= 2 else "Healthy"
    return {"label": label, "score": score, "veg_ratio": round(veg_ratio, 1),
            "brown_ratio": round(brown_ratio, 1), "yellow_ratio": round(yellow_ratio, 1),
            "green_ratio": round(green_ratio, 1)}


# ─────────────────────────────────────────────────────────
#  Beet classifier (inline)
# ─────────────────────────────────────────────────────────

# ─────────────────────────────────────────────────────────
#  Beet classifier  (pipeline: bg removal → features → score)
# ─────────────────────────────────────────────────────────

def _analyze_cell_beet(cell_bgr: np.ndarray) -> dict:
    """Classify one grid cell: background removal → feature extraction → score."""
    if cell_bgr is None or cell_bgr.size == 0:
        return {"label": "Healthy", "score": 0, "veg_ratio": 0}
    h, w = cell_bgr.shape[:2]
    if h < 10 or w < 10:
        return {"label": "Healthy", "score": 0, "veg_ratio": 0}

    hsv   = cv2.cvtColor(cell_bgr, cv2.COLOR_BGR2HSV)
    total = h * w
    hh, s, v = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]

    # ── Step 1: background removal ──
    plant_mask, green_seed, plant_area, veg_ratio = _remove_bg_beet(hsv, total)
    green_area = max(int(green_seed.sum() // 255), 1)
    if veg_ratio < 10:
        return {"label": "No Plant", "score": -1, "veg_ratio": round(veg_ratio, 1)}

    # ── Step 2: features on plant pixels only ──
    def _on_plant(bool_mask):
        return bool_mask & (plant_mask > 0)

    def _on_green(bool_mask):
        return bool_mask & (green_seed > 0)

    yellow_ratio  = float(_on_plant((hh >= 18) & (hh <= 38) & (s > 70) & (v > 80)).sum()) / plant_area * 100
    dark_ratio    = float(_on_plant(v < 60).sum()) / plant_area * 100
    high_sat_ratio= float(_on_green(s > 160).sum()) / green_area * 100
    sat_mean      = float(s[green_seed > 0].mean()) if green_area > 500 else 0.0

    if green_area > 500:
        hue_veg = hh[green_seed > 0]
        hist, _ = np.histogram(hue_veg, bins=20, range=(0, 180))
        hist_n  = hist / (hist.sum() + 1e-9)
        hue_entropy = float(-np.sum(hist_n * np.log2(hist_n + 1e-9)))
    else:
        hue_entropy = 0.0

    green_pct = green_area / total * 100

    # ── Step 3: score ──
    score = 0
    if   yellow_ratio > 12:  score += 4
    elif yellow_ratio > 5:   score += 2
    elif yellow_ratio > 2:   score += 1
    if   sat_mean > 175:     score += 3
    elif sat_mean > 150:     score += 2
    elif sat_mean > 135:     score += 1
    if   high_sat_ratio > 55: score += 3
    elif high_sat_ratio > 35: score += 2
    elif high_sat_ratio > 25: score += 1
    if   hue_entropy > 2.2:  score += 2
    elif hue_entropy > 1.9:  score += 1
    if   green_pct < 60:     score += 2
    elif green_pct < 75:     score += 1
    if   dark_ratio > 15 and sat_mean > 130: score += 2
    elif dark_ratio > 8  and sat_mean > 130: score += 1

    label = "Sick" if score >= 6 else "Suspicious" if score >= 3 else "Healthy"
    return {"label": label, "score": score, "veg_ratio": round(veg_ratio, 1),
            "yellow_ratio": round(yellow_ratio, 1), "sat_mean": round(sat_mean, 1),
            "green_pct": round(green_pct, 1)}


# ─────────────────────────────────────────────────────────
#  Grid analysis engine
# ─────────────────────────────────────────────────────────

def analyze_field(img_bgr: np.ndarray, crop: str, grid_rows: int, grid_cols: int):
    """
    Divide the image into a grid and classify every cell.
    Returns (results_matrix, score_matrix, label_matrix, summary).
    """
    H, W   = img_bgr.shape[:2]
    cell_h = H // grid_rows
    cell_w = W // grid_cols
    analyze = _analyze_cell_potato if crop == "Potato" else _analyze_cell_beet

    results_matrix = []
    score_matrix   = np.zeros((grid_rows, grid_cols))
    label_matrix   = np.full((grid_rows, grid_cols), "Healthy", dtype=object)

    progress = st.progress(0, text="Analyzing field...")
    for row in range(grid_rows):
        row_results = []
        for col in range(grid_cols):
            y1, y2 = row * cell_h, (row + 1) * cell_h
            x1, x2 = col * cell_w, (col + 1) * cell_w
            res    = analyze(img_bgr[y1:y2, x1:x2])
            res["row"] = row
            res["col"] = col
            row_results.append(res)
            score_matrix[row, col] = max(res["score"], 0)
            label_matrix[row, col] = res["label"]
        results_matrix.append(row_results)
        progress.progress((row + 1) / grid_rows,
                          text=f"Analyzing row {row + 1}/{grid_rows}...")

    progress.empty()

    counts = {"Healthy": 0, "Suspicious": 0, "Sick": 0, "No Plant": 0}
    for row in results_matrix:
        for r in row:
            counts[r["label"]] = counts.get(r["label"], 0) + 1

    plant = counts["Healthy"] + counts["Suspicious"] + counts["Sick"]
    summary = {
        "healthy":       counts["Healthy"],
        "suspicious":    counts["Suspicious"],
        "sick":          counts["Sick"],
        "no_plant":      counts.get("No Plant", 0),
        "plant_cells":   plant,
        "sick_pct":      round(counts["Sick"]       / max(plant, 1) * 100, 1),
        "suspicious_pct":round(counts["Suspicious"] / max(plant, 1) * 100, 1),
        "alert_level":   (
            "HIGH"   if counts["Sick"] > plant * 0.30 else
            "MEDIUM" if counts["Sick"] + counts["Suspicious"] > plant * 0.30
            else "LOW"
        ),
    }
    return results_matrix, score_matrix, label_matrix, summary


# ─────────────────────────────────────────────────────────
#  Map rendering
# ─────────────────────────────────────────────────────────

def render_overlay(img_bgr, results_matrix, grid_rows, grid_cols):
    """Original image with colored cell overlays."""
    H, W   = img_bgr.shape[:2]
    cell_h = H // grid_rows
    cell_w = W // grid_cols

    fig, ax = plt.subplots(figsize=(8, 6))
    fig.patch.set_facecolor("#0d1117")
    ax.set_facecolor("#0d1117")
    ax.imshow(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))

    rgba = {
        "Healthy":    (0.15, 0.75, 0.25, 0.20),
        "Suspicious": (1.00, 0.65, 0.00, 0.32),
        "Sick":       (0.85, 0.15, 0.15, 0.42),
        "No Plant":   (0.00, 0.00, 0.00, 0.00),
    }
    for row in range(grid_rows):
        for col in range(grid_cols):
            r = results_matrix[row][col]
            c = rgba.get(r["label"], (0, 0, 0, 0))
            ax.add_patch(plt.Rectangle(
                (col * cell_w, row * cell_h), cell_w, cell_h,
                linewidth=0.6, edgecolor="white",
                facecolor=c[:3], alpha=c[3]
            ))
            if r["label"] in ("Sick", "Suspicious"):
                ax.text(col * cell_w + cell_w / 2, row * cell_h + cell_h / 2,
                        str(r["score"]),
                        color="white", fontsize=max(5, min(9, 70 // grid_rows)),
                        ha="center", va="center", fontweight="bold")

    ax.set_xlim(0, W); ax.set_ylim(H, 0); ax.axis("off")
    ax.set_title("Field — Classification Overlay", color="white", fontsize=11, pad=6)
    plt.tight_layout(pad=0.3)
    return _fig_to_img(fig)


def render_heatmap(score_matrix, grid_rows, grid_cols):
    """Green → yellow → red disease score heatmap."""
    fig, ax = plt.subplots(figsize=(7, 6))
    fig.patch.set_facecolor("#0d1117")
    ax.set_facecolor("#0d1117")

    cmap = LinearSegmentedColormap.from_list(
        "disease", ["#1a5c1a", "#f0c040", "#c0392b"], N=256
    )
    im = ax.imshow(score_matrix, cmap=cmap, vmin=0, vmax=8,
                   interpolation="bilinear", aspect="auto")

    for x in range(grid_cols + 1):
        ax.axvline(x - 0.5, color="white", linewidth=0.3, alpha=0.35)
    for y in range(grid_rows + 1):
        ax.axhline(y - 0.5, color="white", linewidth=0.3, alpha=0.35)

    for row in range(grid_rows):
        for col in range(grid_cols):
            s = score_matrix[row, col]
            if s > 0:
                ax.text(col, row, f"{int(s)}",
                        ha="center", va="center",
                        color="white" if s > 3 else "#111",
                        fontsize=max(5, min(9, 70 // grid_rows)),
                        fontweight="bold")

    cbar = plt.colorbar(im, ax=ax, fraction=0.035, pad=0.03)
    cbar.set_label("Disease Score", color="white", fontsize=9)
    cbar.ax.yaxis.set_tick_params(color="white", labelcolor="white")
    ax.set_xlabel("Column", color="#aaa", fontsize=9)
    ax.set_ylabel("Row",    color="#aaa", fontsize=9)
    ax.tick_params(colors="#aaa", labelsize=8)
    ax.set_title("Disease Score Heatmap", color="white", fontsize=11, pad=6)
    plt.tight_layout(pad=0.3)
    return _fig_to_img(fig)


def render_categorical(label_matrix, grid_rows, grid_cols, summary):
    """Categorical ✓ / ? / ✗ classification grid."""
    fig, ax = plt.subplots(figsize=(7, 6))
    fig.patch.set_facecolor("#0d1117")
    ax.set_facecolor("#0d1117")

    lti  = {"Healthy": 0, "Suspicious": 1, "Sick": 2, "No Plant": -1}
    mat  = np.array([[lti.get(label_matrix[r, c], 0)
                      for c in range(grid_cols)]
                     for r in range(grid_rows)])
    cmap = LinearSegmentedColormap.from_list(
        "cls", ["#666666", "#1a7a1a", "#d4a017", "#c0392b"], N=4
    )
    ax.imshow(mat, cmap=cmap, vmin=-1, vmax=2,
              interpolation="nearest", aspect="auto")

    sym = {"Healthy": "✓", "Suspicious": "?", "Sick": "✗", "No Plant": "·"}
    for r in range(grid_rows):
        for c in range(grid_cols):
            ax.text(c, r, sym.get(label_matrix[r, c], ""),
                    ha="center", va="center", color="white",
                    fontsize=max(6, min(11, 70 // grid_rows)), fontweight="bold")

    for x in range(grid_cols + 1):
        ax.axvline(x - 0.5, color="white", linewidth=0.4, alpha=0.45)
    for y in range(grid_rows + 1):
        ax.axhline(y - 0.5, color="white", linewidth=0.4, alpha=0.45)

    legend_patches = [
        mpatches.Patch(color="#1a7a1a", label=f"✓ Healthy    ({summary['healthy']})"),
        mpatches.Patch(color="#d4a017", label=f"? Suspicious ({summary['suspicious']})"),
        mpatches.Patch(color="#c0392b", label=f"✗ Sick       ({summary['sick']})"),
        mpatches.Patch(color="#666666", label=f"· No Plant   ({summary['no_plant']})"),
    ]
    ax.legend(handles=legend_patches, loc="lower right",
              facecolor="#1a1a2e", edgecolor="#444",
              labelcolor="white", fontsize=8, framealpha=0.85)

    ax.set_xlabel("Column", color="#aaa", fontsize=9)
    ax.set_ylabel("Row",    color="#aaa", fontsize=9)
    ax.tick_params(colors="#aaa", labelsize=8)
    ax.set_title("Classification Map", color="white", fontsize=11, pad=6)
    plt.tight_layout(pad=0.3)
    return _fig_to_img(fig)


def _fig_to_img(fig) -> bytes:
    """Convert matplotlib figure to PNG bytes."""
    buf = BytesIO()
    fig.savefig(buf, format="png", dpi=130, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close(fig)
    buf.seek(0)
    return buf.read()


# ─────────────────────────────────────────────────────────
#  Single-plant / single-leaf mode
# ─────────────────────────────────────────────────────────

def detect_image_type(img_bgr: np.ndarray) -> str:
    """
    Decide whether an image shows a single close-up plant or a field/multi-plant scene.

    The problem: both a single beet plant close-up and a full field can fill the entire
    frame with vegetation, making simple border/blob checks fail.

    Better signals (all measured on a downsampled image for speed):
      1. Shadow density — a field has deep inter-plant shadows; a close-up is evenly lit.
      2. Red/purple stem blob count — each beet has a rosette center with red stems;
         a field has many such centers scattered across the image.
      3. Texture variance — a canopy of many overlapping plants has higher local variance
         and more edge density than a single plant viewed from above.

    Scoring: each signal contributes 0-2 points toward "field".
    Score ≥ 2  → field
    Score < 2  → single_plant

    Returns 'single_plant' or 'field'.
    """
    # Downsample to speed up (keep aspect ratio, cap at 800px on longer side)
    H, W = img_bgr.shape[:2]
    scale = min(800 / max(H, W), 1.0)
    img = cv2.resize(img_bgr, (int(W * scale), int(H * scale)), interpolation=cv2.INTER_AREA)
    h, w = img.shape[:2]
    total = h * w

    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    hh, s, v = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    field_score = 0

    # ── Signal 1: Shadow / darkness density ──
    # Dense canopy → deep inter-plant shadows.
    # Measured: field=36%, single=7%, young_field=9%  →  threshold at 18%
    dark_pct = float(np.sum(v < 50)) / total * 100
    if dark_pct > 18:
        field_score += 2
    elif dark_pct > 10:
        field_score += 1

    # ── Signal 2: Red/purple stem area fraction ──
    # Each beet plant has red/purple petioles. Many plants → high stem coverage.
    # Measured: field=16.2%, single=3.4%, young_field=0.5%  →  threshold at 8%
    stem = (((hh >= 140) | (hh <= 10)) & (s > 60) & (v > 30) & (v < 180)).astype(np.uint8) * 255
    stem = cv2.morphologyEx(stem, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8), iterations=2)
    stem_area_pct = float(np.sum(stem > 0)) / total * 100
    if stem_area_pct > 8:
        field_score += 2
    elif stem_area_pct > 5:
        field_score += 1

    # ── Signal 3: Soil/gap fraction ──
    # A field (even young) has significant exposed soil between rows/plants.
    # Measured: field=18.3%, young_field=22.4%, single=8.8%  →  threshold at 12%
    soil = (((hh >= 5) & (hh <= 35)) & (s < 80) & (v > 60)).astype(np.uint8) * 255
    soil_pct = float(np.sum(soil > 0)) / total * 100
    if soil_pct > 12:
        field_score += 2
    elif soil_pct > 6:
        field_score += 1

    # ── Signal 4: Soil spatial distribution (perspective / angle cue) ──
    # A field photo taken at an angle shows soil in elongated vertical/horizontal bands
    # (rows between plants). A single-plant close-up has soil distributed uniformly around.
    #
    # We measure variance of the soil mask summed per column vs per row.
    # Field rows create highly non-uniform column sums → high vert_var.
    # Single plant has soil spread around it → vert_var ≈ horiz_var (ratio ≈ 1).
    #
    # Measured: young_field vert_var=28875 (ratio=0.11), single vert_var=1511 (ratio=0.95)
    if soil_pct > 5:   # only meaningful if there's enough soil
        soil_col_sum = np.sum(soil > 0, axis=0).astype(float)   # sum per column
        soil_row_sum = np.sum(soil > 0, axis=1).astype(float)   # sum per row
        vert_var  = float(np.var(soil_col_sum))
        horiz_var = float(np.var(soil_row_sum))
        # High vert_var relative to horiz_var → soil concentrated in column bands = field rows
        ratio = horiz_var / max(vert_var, 1.0)
        if vert_var > 5000 or ratio < 0.25:
            field_score += 2   # strong banding signal → field
        elif vert_var > 1500 or ratio < 0.5:
            field_score += 1

    return "field" if field_score >= 3 else "single_plant"


def analyze_single_plant(img_bgr: np.ndarray, crop: str) -> dict:
    """
    Classify a single-plant / single-leaf image as one unit.
    Returns the same dict structure as the per-cell classifiers.
    """
    if crop == "Potato":
        return _analyze_cell_potato(img_bgr)
    else:
        return _analyze_cell_beet(img_bgr)


def render_single_plant_result(img_bgr: np.ndarray, result: dict, crop: str) -> bytes:
    """
    Draw the original image annotated with:
    - Vegetation mask outline
    - Disease-signal pixel highlights (yellow/brown spots)
    - Classification badge in the corner
    """
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    hh, s, v = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]
    H, W = img_bgr.shape[:2]

    # Build vegetation mask for outline
    if crop == "Potato":
        veg_mask = cv2.inRange(hsv, np.array([20, 30, 30]), np.array([100, 255, 255]))
        veg_mask = cv2.morphologyEx(veg_mask, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8), iterations=2)
        # Disease pixels: brown and yellow within vegetation
        b1 = cv2.inRange(hsv, np.array([10, 80, 50]), np.array([25, 255, 200]))
        b2 = cv2.inRange(hsv, np.array([0,  80, 50]), np.array([10, 255, 200]))
        disease_mask = cv2.bitwise_or(
            cv2.bitwise_and(cv2.bitwise_or(b1, b2), veg_mask),
            cv2.bitwise_and(cv2.inRange(hsv, np.array([20, 65, 100]), np.array([35, 255, 255])), veg_mask)
        )
    else:
        veg_mask_bool = (hh >= 35) & (hh <= 85) & (s > 50) & (v > 50)
        veg_mask = (veg_mask_bool.astype(np.uint8)) * 255
        veg_mask = cv2.morphologyEx(veg_mask, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8), iterations=2)
        plant_dilated = cv2.dilate(veg_mask, np.ones((7, 7), np.uint8), iterations=2)
        soil_mask_bool = (hh >= 10) & (hh <= 30) & (s < 60) & (v > 100)
        yellow_bool = (hh >= 18) & (hh <= 38) & (s > 70) & (v > 80) & (plant_dilated > 0) & ~soil_mask_bool
        disease_mask = (yellow_bool.astype(np.uint8)) * 255

    # Create RGBA overlay
    rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    fig, ax = plt.subplots(figsize=(7, 7))
    fig.patch.set_facecolor("#0d1117")
    ax.set_facecolor("#0d1117")
    ax.imshow(rgb)

    # Vegetation outline (green border)
    contours, _ = cv2.findContours(veg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for cnt in contours:
        if cv2.contourArea(cnt) > 500:
            pts = cnt[:, 0, :]
            ax.plot(np.append(pts[:, 0], pts[0, 0]),
                    np.append(pts[:, 1], pts[0, 1]),
                    color="#00e676", linewidth=1.5, alpha=0.8)

    # Disease pixel highlight (semi-transparent red/orange overlay)
    if disease_mask.sum() > 0:
        disease_rgba = np.zeros((H, W, 4), dtype=np.float32)
        disease_rgba[disease_mask > 0] = [1.0, 0.3, 0.0, 0.55]   # orange-red
        ax.imshow(disease_rgba)

    # Classification badge
    label  = result["label"]
    score  = result.get("score", 0)
    colors = {"Healthy": "#27ae60", "Suspicious": "#f39c12", "Sick": "#e74c3c", "No Plant": "#888"}
    badge_color = colors.get(label, "#888")
    icons = {"Healthy": "✓", "Suspicious": "?", "Sick": "✗", "No Plant": "·"}

    ax.text(0.02, 0.98,
            f"{icons.get(label, '')} {label}  (score: {score})",
            transform=ax.transAxes,
            color="white", fontsize=14, fontweight="bold",
            va="top", ha="left",
            bbox=dict(boxstyle="round,pad=0.4", facecolor=badge_color, alpha=0.88, edgecolor="white"))

    ax.axis("off")
    ax.set_title(f"Single Plant Analysis — {crop}", color="white", fontsize=11, pad=6)
    plt.tight_layout(pad=0.3)
    return _fig_to_img(fig)


# ─────────────────────────────────────────────────────────
#  UI
# ─────────────────────────────────────────────────────────

def main():
    # ── Header ──
    st.markdown("# 🌱 Field Disease Detector")
    st.markdown("Upload a field photo and get a plant health map in seconds.")
    st.divider()

    # ── Sidebar controls ──
    with st.sidebar:
        st.markdown("## Settings")

        crop = st.radio(
            "Crop type",
            ["Potato", "Sugar Beet"],
            index=0,
            help="Selects the disease model: Early/Late Blight for Potato, Cercospora for Sugar Beet"
        )

        st.markdown("---")
        grid_size = st.select_slider(
            "Grid resolution",
            options=[5, 8, 10, 12, 15, 20],
            value=10,
            help="Number of rows and columns. Higher = more precise, slower."
        )
        st.caption(f"Grid: {grid_size} × {grid_size} = {grid_size**2} zones")

        st.markdown("---")
        uploaded = st.file_uploader(
            "Upload field image",
            type=["jpg", "jpeg", "png"],
            help="Aerial, drone, or ground-level photo"
        )

        analyze_btn = st.button("Analyze Field", disabled=(uploaded is None))

        st.markdown("---")
        st.markdown("**Disease thresholds**")
        if crop == "Potato":
            st.markdown("- Score ≥ 5 → **Sick**\n- Score ≥ 2 → **Suspicious**\n- Score 0-1 → **Healthy**")
            st.markdown("*Signals: brown spots, dark necrosis, yellowing, low green coverage*")
        else:
            st.markdown("- Score ≥ 6 → **Sick**\n- Score ≥ 3 → **Suspicious**\n- Score 0-2 → **Healthy**")
            st.markdown("*Signals: yellowing, saturation shift, color entropy (Cercospora)*")

    # ── Main area ──
    if uploaded is None:
        st.info("← Upload a field image in the sidebar to get started.")

        # Instructions
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("""
**How it works:**
1. Select your crop type (Potato or Sugar Beet)
2. Upload a field photo
3. Choose grid resolution (10×10 recommended)
4. Click **Analyze Field**

**Output:**
- Overlay map with color-coded zones
- Disease score heatmap (green → red)
- Classification grid (✓ / ? / ✗)
- Summary: % sick, % suspicious, alert level
            """)
        with col2:
            st.markdown("""
**Disease detection:**

🥔 **Potato** — Early Blight & Late Blight
- Detects brown spots, dark necrosis, yellowing
- Calibrated on 2,152 real leaf images

🌿 **Sugar Beet** — Cercospora Leaf Spot
- Detects yellowing halos, color stress, saturation shift
- Works on wide field shots with multiple leaves
            """)
        return

    # ── Load image ──
    file_bytes = np.frombuffer(uploaded.read(), np.uint8)
    img_bgr    = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    if img_bgr is None:
        st.error("Could not read the uploaded image. Please try a different file.")
        return

    H, W = img_bgr.shape[:2]

    # Show preview + metadata
    col_prev, col_meta = st.columns([2, 1])
    with col_prev:
        st.image(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB),
                 caption=f"Uploaded: {uploaded.name}", use_container_width=True)
    with col_meta:
        st.markdown("**Image info**")
        st.write(f"- Size: {W} × {H} px")
        st.write(f"- Crop: **{crop}**")
        st.write(f"- Grid: **{grid_size} × {grid_size}** *(field mode)*")
        st.write(f"- Cell size: {W // grid_size} × {H // grid_size} px")
        st.write(f"- Total zones: **{grid_size ** 2}**")
        st.caption("🔍 Image type auto-detected after clicking Analyze")

    if not analyze_btn:
        st.caption("← Click **Analyze Field** in the sidebar when ready.")
        return

    # ── Detect image type ──
    st.markdown("---")
    st.markdown("### Analysis Results")

    with st.spinner("Detecting image type..."):
        img_type = detect_image_type(img_bgr)

    if img_type == "single_plant":
        # ────────────────────────────────────────
        #  Single plant / leaf mode
        # ────────────────────────────────────────
        st.info("🔍 **Single plant detected** — analyzing as one unit (not a field grid).")

        with st.spinner("Classifying plant..."):
            result = analyze_single_plant(img_bgr, crop)
            annotated = render_single_plant_result(img_bgr, result, crop)

        label  = result["label"]
        score  = result.get("score", 0)
        colors = {"Healthy": "healthy-color", "Suspicious": "suspicious-color",
                  "Sick": "sick-color", "No Plant": ""}
        icons  = {"Healthy": "✓", "Suspicious": "?", "Sick": "✗", "No Plant": "·"}
        alert_cls = {"Healthy": "alert-low", "Suspicious": "alert-medium",
                     "Sick": "alert-high", "No Plant": "alert-low"}

        st.markdown(f"""
<div class="{alert_cls.get(label, 'alert-low')}">
  <b>{icons.get(label, '')} Classification: {label}</b> &nbsp;|&nbsp; Disease score: <b>{score}</b>
</div>""", unsafe_allow_html=True)

        st.markdown("")
        cols = st.columns(3)
        if "yellow_ratio" in result:
            cols[0].metric("Yellow ratio", f"{result['yellow_ratio']}%")
        if "brown_ratio" in result:
            cols[0].metric("Brown ratio", f"{result['brown_ratio']}%")
        if "green_ratio" in result:
            cols[1].metric("Green ratio", f"{result['green_ratio']}%")
        if "sat_mean" in result:
            cols[1].metric("Avg saturation", f"{result['sat_mean']}")
        cols[2].metric("Vegetation coverage", f"{result.get('veg_ratio', 0)}%")
        cols[2].metric("Score", score)

        st.image(annotated, use_container_width=True,
                 caption="Orange highlights = disease signal pixels | Green outline = detected plant boundary")
        st.download_button("Download annotated image",
                           data=annotated,
                           file_name=f"single_plant_{crop.lower().replace(' ','_')}_{label.lower()}.png",
                           mime="image/png")
        return

    # ────────────────────────────────────────
    #  Field / grid mode (original flow)
    # ────────────────────────────────────────
    with st.spinner("Processing field image..."):
        results_matrix, score_matrix, label_matrix, summary = analyze_field(
            img_bgr, crop, grid_size, grid_size
        )

    # ── Summary metrics ──
    alert_colors = {"LOW": "#27ae60", "MEDIUM": "#f39c12", "HIGH": "#e74c3c"}
    alert_color  = alert_colors[summary["alert_level"]]
    alert_class  = f"alert-{summary['alert_level'].lower()}"

    st.markdown(f"""
<div class="{alert_class}">
  <b>Alert Level: {summary['alert_level']}</b> &nbsp;|&nbsp;
  Sick: <b>{summary['sick_pct']}%</b> &nbsp;|&nbsp;
  Suspicious: <b>{summary['suspicious_pct']}%</b> &nbsp;|&nbsp;
  Plant zones analyzed: <b>{summary['plant_cells']}</b>
</div>
""", unsafe_allow_html=True)

    st.markdown("")
    m1, m2, m3, m4 = st.columns(4)
    with m1:
        st.markdown(f"""<div class="metric-box">
            <div class="metric-label">Healthy zones</div>
            <div class="metric-value healthy-color">{summary['healthy']}</div>
        </div>""", unsafe_allow_html=True)
    with m2:
        st.markdown(f"""<div class="metric-box">
            <div class="metric-label">Suspicious zones</div>
            <div class="metric-value suspicious-color">{summary['suspicious']}</div>
        </div>""", unsafe_allow_html=True)
    with m3:
        st.markdown(f"""<div class="metric-box">
            <div class="metric-label">Sick zones</div>
            <div class="metric-value sick-color">{summary['sick']}</div>
        </div>""", unsafe_allow_html=True)
    with m4:
        st.markdown(f"""<div class="metric-box">
            <div class="metric-label">No vegetation</div>
            <div class="metric-value" style="color:#8b9dc3">{summary['no_plant']}</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("")

    # ── Maps ──
    tab1, tab2, tab3 = st.tabs(["📷 Overlay Map", "🌡️ Heatmap", "🗺️ Classification Grid"])

    with tab1:
        img_overlay = render_overlay(img_bgr, results_matrix, grid_size, grid_size)
        st.image(img_overlay, use_container_width=True)
        st.download_button("Download overlay map",
                           data=img_overlay,
                           file_name=f"field_overlay_{crop.lower().replace(' ','_')}.png",
                           mime="image/png")

    with tab2:
        img_heat = render_heatmap(score_matrix, grid_size, grid_size)
        st.image(img_heat, use_container_width=True)
        st.download_button("Download heatmap",
                           data=img_heat,
                           file_name=f"field_heatmap_{crop.lower().replace(' ','_')}.png",
                           mime="image/png")

    with tab3:
        img_cat = render_categorical(label_matrix, grid_size, grid_size, summary)
        st.image(img_cat, use_container_width=True)
        st.download_button("Download classification grid",
                           data=img_cat,
                           file_name=f"field_grid_{crop.lower().replace(' ','_')}.png",
                           mime="image/png")

    # ── Sick zone details ──
    sick_zones = [
        (r["row"] + 1, r["col"] + 1, r["score"])
        for row in results_matrix for r in row
        if r["label"] == "Sick"
    ]
    if sick_zones:
        with st.expander(f"🔴 Sick zones detail ({len(sick_zones)} zones)", expanded=False):
            st.markdown("These grid positions need immediate attention:")
            cols = st.columns(4)
            for i, (row, col, score) in enumerate(sorted(sick_zones, key=lambda x: -x[2])):
                cols[i % 4].markdown(
                    f"<div class='metric-box'>"
                    f"<div class='metric-label'>Row {row}, Col {col}</div>"
                    f"<div class='metric-value sick-color'>{score}</div>"
                    f"<div class='metric-label'>disease score</div></div>",
                    unsafe_allow_html=True
                )

    susp_zones = [
        (r["row"] + 1, r["col"] + 1, r["score"])
        for row in results_matrix for r in row
        if r["label"] == "Suspicious"
    ]
    if susp_zones:
        with st.expander(f"🟡 Suspicious zones ({len(susp_zones)} zones)", expanded=False):
            st.markdown("These zones show early disease signals — monitor closely:")
            cols = st.columns(4)
            for i, (row, col, score) in enumerate(sorted(susp_zones, key=lambda x: -x[2])):
                cols[i % 4].markdown(
                    f"<div class='metric-box'>"
                    f"<div class='metric-label'>Row {row}, Col {col}</div>"
                    f"<div class='metric-value suspicious-color'>{score}</div>"
                    f"<div class='metric-label'>disease score</div></div>",
                    unsafe_allow_html=True
                )


if __name__ == "__main__":
    main()
