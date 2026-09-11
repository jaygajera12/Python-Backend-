"""
Saree Micro Design Detector
----------------------------
Detects small repeating micro-patterns / motifs (buttis, dots, small
florals, zari specks, etc.) on saree fabric using classical computer
vision techniques — no training data required.

Pipeline:
  1. Preprocessing        -> grayscale, denoise, contrast enhancement (CLAHE)
  2. Contour detection     -> finds small, repeated shapes (candidate motifs)
  3. FFT periodicity check -> confirms the fabric has a repeating pattern
  4. LBP texture analysis  -> separates "micro-patterned" regions from plain fabric
  5. Visualization         -> draws boxes on detected micro-design regions

Usage:
    python saree_micro_design_detector.py path/to/saree.jpg

Output:
    - <name>_detected.png   : image with micro-design regions boxed
    - <name>_fft.png        : frequency spectrum (shows periodicity)
    - <name>_texture_map.png: heatmap of texture density (micro-pattern likelihood)
    - Console summary with counts / stats
"""

import sys
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage.feature import local_binary_pattern


# ----------------------------- CONFIG -----------------------------
MIN_MOTIF_AREA = 8          # smallest contour area (px) to count as a micro-motif
MAX_MOTIF_AREA = 800         # largest contour area (px) — bigger = not "micro"
LBP_RADIUS = 3
LBP_POINTS = 8 * LBP_RADIUS
TEXTURE_GRID = 16            # grid size (px) for texture-density heatmap
TEXTURE_VAR_THRESHOLD = 25   # variance threshold to flag a cell as "patterned"
# --------------------------------------------------------------------


def load_image(path):
    img = cv2.imread(path)
    if img is None:
        raise FileNotFoundError(f"Could not read image: {path}")
    return img


def preprocess(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)
    denoised = cv2.bilateralFilter(enhanced, d=5, sigmaColor=50, sigmaSpace=50)
    return denoised


def detect_motif_contours(gray):
    """Find small repeated shapes = candidate micro-motifs."""
    edges = cv2.Canny(gray, 40, 120)
    edges = cv2.dilate(edges, np.ones((2, 2), np.uint8), iterations=1)
    contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    motifs = []
    for c in contours:
        area = cv2.contourArea(c)
        if MIN_MOTIF_AREA <= area <= MAX_MOTIF_AREA:
            x, y, w, h = cv2.boundingRect(c)
            # filter out very elongated shapes (likely fabric folds/borders, not motifs)
            aspect = w / float(h + 1e-5)
            if 0.3 < aspect < 3.0:
                motifs.append((x, y, w, h, area))
    return motifs, edges


def fft_periodicity(gray):
    """Check if the fabric has a repeating (periodic) micro-pattern."""
    f = np.fft.fft2(gray)
    fshift = np.fft.fftshift(f)
    magnitude = 20 * np.log(np.abs(fshift) + 1)

    # exclude the DC component (center) and measure energy in mid-frequency ring
    h, w = magnitude.shape
    cy, cx = h // 2, w // 2
    Y, X = np.ogrid[:h, :w]
    dist = np.sqrt((X - cx) ** 2 + (Y - cy) ** 2)

    low_mask = dist < 5
    mid_mask = (dist >= 5) & (dist < min(h, w) * 0.25)

    mid_energy = magnitude[mid_mask].mean()
    low_energy = magnitude[low_mask].mean()
    periodicity_score = mid_energy / (low_energy + 1e-5)

    return magnitude, periodicity_score


def texture_density_map(gray):
    """LBP-based texture variance map — highlights micro-patterned regions."""
    lbp = local_binary_pattern(gray, LBP_POINTS, LBP_RADIUS, method="uniform")
    h, w = gray.shape
    heatmap = np.zeros((h // TEXTURE_GRID, w // TEXTURE_GRID))

    for i in range(heatmap.shape[0]):
        for j in range(heatmap.shape[1]):
            cell = lbp[i * TEXTURE_GRID:(i + 1) * TEXTURE_GRID,
                       j * TEXTURE_GRID:(j + 1) * TEXTURE_GRID]
            heatmap[i, j] = cell.var()

    patterned_ratio = (heatmap > TEXTURE_VAR_THRESHOLD).sum() / heatmap.size
    return heatmap, patterned_ratio


def draw_results(img, motifs):
    out = img.copy()
    for (x, y, w, h, area) in motifs:
        cv2.rectangle(out, (x, y), (x + w, y + h), (0, 255, 0), 1)
    return out


def main(image_path):
    name = os.path.splitext(os.path.basename(image_path))[0]
    out_dir = os.path.dirname(image_path) or "."

    img = load_image(image_path)
    gray = preprocess(img)

    # 1. Contour-based motif detection
    motifs, edges = detect_motif_contours(gray)

    # 2. FFT periodicity check
    magnitude, periodicity_score = fft_periodicity(gray)

    # 3. Texture density (LBP)
    heatmap, patterned_ratio = texture_density_map(gray)

    # ---- Verdict ----
    has_micro_design = (
        len(motifs) > 15 and periodicity_score > 1.05 and patterned_ratio > 0.15
    )

    # ---- Save outputs ----
    detected_img = draw_results(img, motifs)
    cv2.imwrite(f"{out_dir}/{name}_detected.png", detected_img)

    plt.figure(figsize=(6, 6))
    plt.imshow(magnitude, cmap="inferno")
    plt.title("Frequency Spectrum (FFT)")
    plt.axis("off")
    plt.savefig(f"{out_dir}/{name}_fft.png", bbox_inches="tight")
    plt.close()

    plt.figure(figsize=(6, 6))
    plt.imshow(heatmap, cmap="viridis")
    plt.title("Texture Density Map (micro-pattern likelihood)")
    plt.colorbar(label="LBP variance")
    plt.axis("off")
    plt.savefig(f"{out_dir}/{name}_texture_map.png", bbox_inches="tight")
    plt.close()

    # ---- Report ----
    print("=" * 50)
    print(f"Saree Micro Design Detection Report: {name}")
    print("=" * 50)
    print(f"Candidate micro-motifs detected : {len(motifs)}")
    print(f"FFT periodicity score           : {periodicity_score:.3f}  (>1.05 suggests repeating pattern)")
    print(f"Textured area ratio (LBP)        : {patterned_ratio:.2%}")
    print("-" * 50)
    print(f"VERDICT: {'Micro design pattern DETECTED' if has_micro_design else 'No significant micro design detected'}")
    print("=" * 50)
    print(f"Saved: {name}_detected.png, {name}_fft.png, {name}_texture_map.png")

    return {
        "motif_count": len(motifs),
        "periodicity_score": periodicity_score,
        "patterned_ratio": patterned_ratio,
        "has_micro_design": has_micro_design,
    }


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python saree_micro_design_detector.py <image_path>")
        sys.exit(1)
    main(sys.argv[1])
