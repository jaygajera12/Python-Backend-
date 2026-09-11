import os
import uuid
import json
import base64
import time
from concurrent.futures import ThreadPoolExecutor
import shutil
from pathlib import Path
from typing import Optional, List
import cv2
import numpy as np
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from dotenv import load_dotenv
import os
load_dotenv()
try:
    from google import genai
    from google.genai import types
except ImportError:
    genai = None
GEMINI_API_KEY = os.getenv('GEMINI_API_KEY')
GEMINI_IMAGE_MODEL = os.getenv('GEMINI_IMAGE_MODEL', 'gemini-3.1-flash-image')
UPLOAD_DIR = 'uploads'
GENERATED_DIR = 'generated'
ANALYSIS_DIR = 'design_maps'
PREVIEW_DIR = 'design_previews'
BACKGROUND_DIR = 'backgrounds'
CRYSTAL_DIR = 'crystal_references'
for folder in [UPLOAD_DIR, GENERATED_DIR, ANALYSIS_DIR, PREVIEW_DIR, BACKGROUND_DIR, CRYSTAL_DIR]:
    os.makedirs(folder, exist_ok=True)
app = FastAPI(
    title='SareeViz AI - Micro Lace + Crystal/Swarovski-Type Design Lock',
    version='7.0.0',
    docs_url=None,
    redoc_url=None,
    openapi_url=None
)
app.add_middleware(CORSMiddleware, allow_origins=['*'], allow_credentials=True, allow_methods=['*'], allow_headers=['*'])
app.mount('/uploads', StaticFiles(directory=UPLOAD_DIR), name='uploads')
app.mount('/generated', StaticFiles(directory=GENERATED_DIR), name='generated')
app.mount('/design-previews', StaticFiles(directory=PREVIEW_DIR), name='design-previews')
app.mount('/design-maps', StaticFiles(directory=ANALYSIS_DIR), name='design-maps')
app.mount('/backgrounds', StaticFiles(directory=BACKGROUND_DIR), name='backgrounds')
app.mount('/crystal-references', StaticFiles(directory=CRYSTAL_DIR), name='crystal-references')
gemini_client = None
if GEMINI_API_KEY and genai:
    try:
        gemini_client = genai.Client(api_key=GEMINI_API_KEY)
    except Exception:
        gemini_client = None
ALLOWED_TYPES = {'image/jpeg': '.jpg', 'image/png': '.png', 'image/webp': '.webp'}

def validate_image(file: UploadFile) -> bool:
    return file.content_type in ALLOWED_TYPES

async def save_uploaded_image(file: UploadFile):
    image_bytes = await file.read()
    if not image_bytes:
        raise ValueError('Uploaded image is empty.')
    extension = ALLOWED_TYPES.get(file.content_type, '.jpg')
    filename = f'{uuid.uuid4().hex}{extension}'
    path = os.path.join(UPLOAD_DIR, filename)
    with open(path, 'wb') as f:
        f.write(image_bytes)
    return (image_bytes, filename, path)

def read_image(path: str):
    image = cv2.imread(path, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError('Could not read uploaded image.')
    return image

def resize_keep_ratio(image, max_side=1800):
    h, w = image.shape[:2]
    scale = min(1.0, max_side / max(h, w))
    if scale >= 1.0:
        return image
    return cv2.resize(image, (max(1, int(w * scale)), max(1, int(h * scale))), interpolation=cv2.INTER_AREA)

def enhance_for_detail(image):
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.2, tileGridSize=(8, 8))
    l = clahe.apply(l)
    enhanced = cv2.cvtColor(cv2.merge([l, a, b]), cv2.COLOR_LAB2BGR)
    blur = cv2.GaussianBlur(enhanced, (0, 0), 1.0)
    sharp = cv2.addWeighted(enhanced, 1.25, blur, -0.25, 0)
    return sharp

def dominant_colors(image, k=8):
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    pixels = rgb.reshape(-1, 3).astype(np.float32)
    keep = np.max(pixels, axis=1) > 25
    pixels = pixels[keep]
    if len(pixels) < k:
        return []
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)
    _, labels, centers = cv2.kmeans(pixels, k, None, criteria, 10, cv2.KMEANS_PP_CENTERS)
    counts = np.bincount(labels.flatten())
    order = np.argsort(counts)[::-1]
    total = len(pixels)
    result = []
    for idx in order:
        rgb_value = centers[idx].astype(int).tolist()
        result.append({'rgb': rgb_value, 'hex': '#{:02X}{:02X}{:02X}'.format(*rgb_value), 'percentage': round(float(counts[idx]) / total * 100, 2)})
    return result

def create_fine_detail_map(image):
    enhanced = enhance_for_detail(image)
    gray = cv2.cvtColor(enhanced, cv2.COLOR_BGR2GRAY)
    edges_small = cv2.Canny(gray, 35, 100)
    edges_medium = cv2.Canny(gray, 60, 150)
    edges_large = cv2.Canny(gray, 90, 190)
    combined = cv2.bitwise_or(edges_small, edges_medium)
    combined = cv2.bitwise_or(combined, edges_large)
    kernel = np.ones((2, 2), np.uint8)
    combined = cv2.morphologyEx(combined, cv2.MORPH_CLOSE, kernel, iterations=1)
    return combined

def detect_bright_detail_candidates(image):
    """
    Detect small bright/reflection candidates.

    These may correspond to:
    - crystals
    - stones
    - zari highlights
    - white embroidery
    - reflective highlights

    They are NOT automatically classified
    as Swarovski/Siroki.
    """
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    _, saturation, value = cv2.split(hsv)
    bright_mask = cv2.inRange(value, 180, 255)
    moderate_saturation = cv2.inRange(saturation, 0, 210)
    mask = cv2.bitwise_and(bright_mask, moderate_saturation)
    kernel = np.ones((2, 2), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask)
    candidates = []
    h_img, w_img = image.shape[:2]
    min_area = max(2, int(w_img * h_img * 1e-06))
    max_area = max(80, int(w_img * h_img * 0.0005))
    for i in range(1, num_labels):
        area = int(stats[i, cv2.CC_STAT_AREA])
        if not min_area <= area <= max_area:
            continue
        x = int(stats[i, cv2.CC_STAT_LEFT])
        y = int(stats[i, cv2.CC_STAT_TOP])
        width = int(stats[i, cv2.CC_STAT_WIDTH])
        height = int(stats[i, cv2.CC_STAT_HEIGHT])
        cx, cy = centroids[i]
        candidates.append({'x': round(float(cx), 2), 'y': round(float(cy), 2), 'width': width, 'height': height, 'area': area, 'relative_x': round(float(cx) / w_img, 5), 'relative_y': round(float(cy) / h_img, 5), 'type': 'bright_detail_candidate'})
    return (candidates, mask)

def detect_horizontal_design_bands(image):
    """
    Finds strong horizontal decorative
    regions. These can indicate borders,
    pallu bands or other horizontal design
    structures.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150)
    horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (max(15, image.shape[1] // 30), 3))
    horizontal = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, horizontal_kernel)
    projection = np.sum(horizontal > 0, axis=1)
    threshold = max(10, int(image.shape[1] * 0.08))
    bands = []
    in_band = False
    start = 0
    for y, value in enumerate(projection):
        if value >= threshold and (not in_band):
            start = y
            in_band = True
        elif value < threshold and in_band:
            end = y
            if end - start >= 3:
                bands.append({'y_start': start, 'y_end': end, 'relative_start': round(start / image.shape[0], 4), 'relative_end': round(end / image.shape[0], 4)})
            in_band = False
    if in_band:
        end = len(projection) - 1
        if end - start >= 3:
            bands.append({'y_start': start, 'y_end': end, 'relative_start': round(start / image.shape[0], 4), 'relative_end': round(end / image.shape[0], 4)})
    merged = []
    for band in bands:
        if not merged or band['y_start'] - merged[-1]['y_end'] > 8:
            merged.append(band)
        else:
            merged[-1]['y_end'] = band['y_end']
            merged[-1]['relative_end'] = round(band['y_end'] / image.shape[0], 4)
    return merged

def detect_side_borders(image):
    """
    Estimates strong repeated vertical/side
    border candidates near the left/right
    portions of the reference image.
    """
    h, w = image.shape[:2]
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150)
    projection = np.mean(edges > 0, axis=0)
    strong = projection > max(0.035, np.percentile(projection, 80))
    runs = []
    start = None
    for x, value in enumerate(strong):
        if value and start is None:
            start = x
        elif not value and start is not None:
            if x - start >= max(3, int(w * 0.005)):
                runs.append((start, x))
            start = None
    if start is not None and w - start >= max(3, int(w * 0.005)):
        runs.append((start, w))
    edge_runs = []
    for x1, x2 in runs:
        center = (x1 + x2) / 2
        if center < w * 0.25 or center > w * 0.75:
            edge_runs.append({'x_start': x1, 'x_end': x2, 'relative_start': round(x1 / w, 4), 'relative_end': round(x2 / w, 4)})
    return edge_runs

def detect_texture_regions(image):
    """
    Produces a texture/contrast map that is
    useful for locating repeated motifs and
    decorative regions.

    OpenCV does not name a motif by itself;
    this is a visual-region detector.
    """
    enhanced = enhance_for_detail(image)
    gray = cv2.cvtColor(enhanced, cv2.COLOR_BGR2GRAY)
    lap = cv2.Laplacian(gray, cv2.CV_64F)
    magnitude = np.uint8(np.clip(np.abs(lap), 0, 255))
    threshold = cv2.threshold(magnitude, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    kernel = np.ones((3, 3), np.uint8)
    threshold = cv2.morphologyEx(threshold, cv2.MORPH_OPEN, kernel, iterations=1)
    contours, _ = cv2.findContours(threshold, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    regions = []
    h, w = gray.shape
    min_area = max(12, int(h * w * 1e-05))
    max_area = int(h * w * 0.08)
    for contour in contours:
        area = cv2.contourArea(contour)
        if area < min_area or area > max_area:
            continue
        x, y, bw, bh = cv2.boundingRect(contour)
        regions.append({'x': x, 'y': y, 'width': bw, 'height': bh, 'area': round(float(area), 2), 'relative_x': round(x / w, 5), 'relative_y': round(y / h, 5), 'relative_width': round(bw / w, 5), 'relative_height': round(bh / h, 5), 'type': 'texture_region'})
    regions.sort(key=lambda x: x['area'], reverse=True)
    return (regions[:1500], threshold)

def create_visual_design_map(image, detail_map, bright_mask, texture_mask, output_path):
    """
    Creates one diagnostic image.

    White/gray = original saree
    Red = fine-detail/edge map
    Green = bright reflective candidates
    Blue = texture/motif-like regions
    """
    base = image.copy()
    red = np.zeros_like(base)
    red[:, :, 2] = detail_map
    green = np.zeros_like(base)
    green[:, :, 1] = bright_mask
    blue = np.zeros_like(base)
    blue[:, :, 0] = texture_mask
    preview = cv2.addWeighted(base, 0.68, red, 0.24, 0)
    preview = cv2.addWeighted(preview, 0.88, green, 0.18, 0)
    preview = cv2.addWeighted(preview, 0.88, blue, 0.18, 0)
    cv2.imwrite(output_path, preview, [cv2.IMWRITE_JPEG_QUALITY, 94])

def analyze_saree_opencv(image_path: str, max_side: int=1200):
    original = read_image(image_path)
    image = resize_keep_ratio(original, max_side=max(600, min(int(max_side), 1800)))
    detail_map = create_fine_detail_map(image)
    bright_candidates, bright_mask = detect_bright_detail_candidates(image)
    horizontal_bands = detect_horizontal_design_bands(image)
    side_borders = detect_side_borders(image)
    colors = dominant_colors(image)
    texture_regions, texture_mask = detect_texture_regions(image)
    edge_density = float(np.count_nonzero(detail_map) / detail_map.size)
    if edge_density < 0.03:
        pattern_level = 'low'
    elif edge_density < 0.08:
        pattern_level = 'medium'
    else:
        pattern_level = 'high'
    analysis = {'detector': 'OpenCV', 'image': {'width': int(image.shape[1]), 'height': int(image.shape[0])}, 'colors': colors, 'pattern': {'edge_density_percent': round(edge_density * 100, 3), 'level': pattern_level}, 'fine_detail': {'candidate_count': len(bright_candidates), 'candidates': bright_candidates[:5000], 'note': 'Candidates are visual bright/reflective details. They are not automatically classified as Swarovski or Siroki.'}, 'texture_regions': {'count': len(texture_regions), 'regions': texture_regions}, 'horizontal_design_bands': horizontal_bands, 'side_border_candidates': side_borders, 'design_lock': {'preserve_colors': True, 'preserve_pattern': True, 'preserve_motifs': True, 'preserve_border': True, 'preserve_pallu': True, 'preserve_embroidery': True, 'preserve_reflective_details': True, 'do_not_redesign': True, 'reference_priority': 'highest'}}
    return (analysis, image, detail_map, bright_mask, texture_mask)
def build_design_lock(opencv_analysis, semantic_analysis):
    semantic = {}
    if semantic_analysis.get('available') and isinstance(semantic_analysis.get('result'), dict):
        semantic = semantic_analysis['result']
    return {'version': '1.0', 'reference_priority': 'highest', 'opencv': opencv_analysis, 'semantic_ai': semantic, 'preservation_rules': {'keep_original_color': True, 'keep_original_motifs': True, 'keep_motif_arrangement': True, 'keep_original_border': True, 'keep_original_pallu': True, 'keep_embroidery': True, 'keep_zari': True, 'keep_visible_crystals_or_stones': True, 'keep_pattern_density': True, 'do_not_recolor': True, 'do_not_redesign': True, 'do_not_invent_new_motifs': True, 'do_not_remove_visible_details': True}, 'generation_instruction': 'Use the uploaded original saree image as the primary garment reference. Transform only the presentation/drape. Preserve the original saree design as closely as possible.'}

class StatusResponse(BaseModel):
    success: bool
    message: str

class UploadRequestResponse(BaseModel):
    success: bool
    request_id: str
    status: str
    uploaded_files: list
    message: str

class AnalyzeRequestResponse(BaseModel):
    success: bool
    request_id: str
    status: str
    analysis: dict
    files: dict
    message: str

class LockRequestResponse(BaseModel):
    success: bool
    request_id: str
    status: str
    design_lock: dict
    file: str
    message: str

class DesignLockResponse(BaseModel):
    success: bool
    request_id: str
    design_lock: dict

class DetectionResponse(BaseModel):
    success: bool
    request_id: str
    detection: dict

class HealthResponse(BaseModel):
    success: bool
    opencv: bool
    gemini: bool
    image_generation: bool
    image_model: Optional[str] = None

class ImageGenerationResponse(BaseModel):
    success: bool
    request_id: str
    status: str
    generated_image: dict
    generation_spec: dict
    message: str

def get_request_dir(request_id: str) -> str:
    return os.path.join(UPLOAD_DIR, request_id)

def list_request_images(request_id: str):
    folder = get_request_dir(request_id)
    if not os.path.isdir(folder):
        return []
    images = []
    for filename in sorted(os.listdir(folder)):
        path = os.path.join(folder, filename)
        if not os.path.isfile(path):
            continue
        if Path(filename).suffix.lower() in {'.jpg', '.jpeg', '.png', '.webp'}:
            images.append(path)
    return images

def save_request_metadata(request_id: str, data: dict):
    folder = get_request_dir(request_id)
    os.makedirs(folder, exist_ok=True)
    with open(os.path.join(folder, 'request.json'), 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

def build_request_design_lock(request_id: str, combined_analysis: dict):
    return {'version': '1.0', 'request_id': request_id, 'status': 'design_locked', 'reference_images': combined_analysis.get('reference_images', 0), 'analysis': combined_analysis, 'preservation_rules': {'keep_original_color': True, 'keep_original_motifs': True, 'keep_motif_arrangement': True, 'keep_original_border': True, 'keep_original_pallu': True, 'keep_embroidery': True, 'keep_zari': True, 'keep_visible_crystals_or_stones': True, 'keep_pattern_density': True, 'do_not_recolor': True, 'do_not_redesign': True, 'do_not_invent_new_motifs': True, 'do_not_remove_visible_details': True, 'do_not_replace_reference_saree': True}, 'generation_instruction': 'Use the uploaded original saree image(s) as the primary garment reference. Only transform draping, pose and presentation. Preserve the original saree colors, motifs, border, pallu, embroidery, zari and visible decorative details as closely as possible.', 'generation_prompt_status': 'not_generated'}
def detect_lace_strip(image_path: str, request_id: str) -> dict:
    """
    Detect a likely lace/border strip from the saree image using OpenCV.
    Saves a high-resolution crop as a separate lace reference image.
    """
    image = cv2.imread(image_path)
    if image is None:
        raise RuntimeError(f'Could not read image: {image_path}')
    height, width = image.shape[:2]
    scale = min(1.0, 1400.0 / max(height, width))
    if scale < 1.0:
        small = cv2.resize(image, (max(1, int(width * scale)), max(1, int(height * scale))), interpolation=cv2.INTER_AREA)
    else:
        small = image.copy()
    hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)
    lower_white = np.array([0, 0, 120], dtype=np.uint8)
    upper_white = np.array([180, 150, 255], dtype=np.uint8)
    white_mask = cv2.inRange(hsv, lower_white, upper_white)
    red1 = cv2.inRange(hsv, np.array([0, 70, 60], dtype=np.uint8), np.array([12, 255, 255], dtype=np.uint8))
    red2 = cv2.inRange(hsv, np.array([165, 70, 60], dtype=np.uint8), np.array([180, 255, 255], dtype=np.uint8))
    red_mask = cv2.bitwise_or(red1, red2)
    combined = cv2.bitwise_or(white_mask, red_mask)
    kernel_long_v = cv2.getStructuringElement(cv2.MORPH_RECT, (max(3, small.shape[1] // 120), max(15, small.shape[0] // 18)))
    kernel_long_h = cv2.getStructuringElement(cv2.MORPH_RECT, (max(15, small.shape[1] // 18), max(3, small.shape[0] // 120)))
    vertical = cv2.morphologyEx(combined, cv2.MORPH_CLOSE, kernel_long_v)
    horizontal = cv2.morphologyEx(combined, cv2.MORPH_CLOSE, kernel_long_h)
    merged = cv2.bitwise_or(vertical, horizontal)
    contours, _ = cv2.findContours(merged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    candidates = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        area = w * h
        if area < small.shape[0] * small.shape[1] * 0.005:
            continue
        vertical_score = h / max(1, w)
        horizontal_score = w / max(1, h)
        if vertical_score >= 2.0 or horizontal_score >= 2.0:
            candidates.append({'x': x, 'y': y, 'w': w, 'h': h, 'area': area, 'vertical_score': vertical_score, 'horizontal_score': horizontal_score})
    if not candidates:
        gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
        col_score = np.mean(gray > 150, axis=0)
        window = max(8, small.shape[1] // 12)
        best_x = 0
        best_score = -1.0
        for x in range(0, max(1, small.shape[1] - window + 1), max(1, window // 4)):
            score = float(np.mean(col_score[x:x + window]))
            if score > best_score:
                best_score = score
                best_x = x
        x, y, w, h = (best_x, 0, window, small.shape[0])
        candidates.append({'x': x, 'y': y, 'w': w, 'h': h, 'area': w * h, 'vertical_score': h / max(1, w), 'horizontal_score': 0})

    def candidate_score(c):
        elongation = max(c['vertical_score'], c['horizontal_score'])
        return c['area'] * min(elongation, 20.0)
    best = max(candidates, key=candidate_score)
    sx = 1.0 / scale
    x1 = int(best['x'] * sx)
    y1 = int(best['y'] * sx)
    x2 = int((best['x'] + best['w']) * sx)
    y2 = int((best['y'] + best['h']) * sx)
    pad_x = max(12, int((x2 - x1) * 0.45))
    pad_y = max(12, int((y2 - y1) * 0.03))
    x1 = max(0, x1 - pad_x)
    y1 = max(0, y1 - pad_y)
    x2 = min(width, x2 + pad_x)
    y2 = min(height, y2 + pad_y)
    lace_crop = image[y1:y2, x1:x2]
    if lace_crop.size == 0:
        raise RuntimeError('Lace detection produced an empty crop.')
    lace_name = f'{request_id}_lace_reference.png'
    lace_path = os.path.abspath(os.path.join(ANALYSIS_DIR, lace_name))
    cv2.imwrite(lace_path, lace_crop, [cv2.IMWRITE_PNG_COMPRESSION, 1])
    return {'success': True, 'request_id': request_id, 'file_name': lace_name, 'local_path': lace_path, 'url': f'/design-maps/{lace_name}', 'bbox': {'x': x1, 'y': y1, 'width': x2 - x1, 'height': y2 - y1}}

def get_image_dimensions(image_path: str):
    image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
    if image is None:
        return (None, None)
    height, width = image.shape[:2]
    return (width, height)

def build_micro_lace_scale_instruction(saree_image_path: str, lace_image_path: Optional[str]) -> str:
    """Build a strict scale guide for the micro-lace."""
    if not lace_image_path or not os.path.exists(lace_image_path):
        return 'MICRO-LACE SCALE LOCK:\n- Keep the lace extremely narrow and micro-scale exactly as shown.\n- Do not enlarge or thicken it.\n'
    saree_w, _ = get_image_dimensions(saree_image_path)
    lace_w, _ = get_image_dimensions(lace_image_path)
    if not saree_w or not lace_w:
        return 'MICRO-LACE SCALE LOCK:\n- Keep the lace extremely narrow and micro-scale exactly as shown.\n- Do not enlarge or thicken it.\n'
    width_ratio = lace_w / float(saree_w)
    return f'MICRO-LACE SCALE LOCK — STRICT:\n- The provided lace image is a micro-detail reference.\n- Preserve the lace as a SMALL, NARROW, FINE border element.\n- Do NOT enlarge, thicken, widen, magnify, or turn it into a broad decorative strip.\n- Maintain the same relative visual scale seen in the source.\n- Reference crop width ratio to source image width is approximately {width_ratio:.4f}; use this as a scale guide.\n- Keep repeated micro motifs tiny, dense, and closely spaced.\n- Keep lace line thickness, stone size, motif spacing and border width proportionally consistent with the reference.\n- Never compensate for higher output resolution by making lace motifs larger.\n'

def _make_lace_alpha(lace_bgr):
    """Build an alpha mask for the lace without changing the lace RGB pixels."""
    if lace_bgr is None or lace_bgr.size == 0:
        return None
    h, w = lace_bgr.shape[:2]
    # Estimate the photographed/scanned background from the four corners.
    patch = max(2, min(h, w) // 10)
    corners = np.concatenate([
        lace_bgr[:patch, :patch].reshape(-1, 3),
        lace_bgr[:patch, -patch:].reshape(-1, 3),
        lace_bgr[-patch:, :patch].reshape(-1, 3),
        lace_bgr[-patch:, -patch:].reshape(-1, 3),
    ], axis=0).astype(np.float32)
    bg = np.median(corners, axis=0)
    dist = np.linalg.norm(lace_bgr.astype(np.float32) - bg[None, None, :], axis=2)
    # Soft mask: only background-like pixels become transparent.
    alpha = np.clip((dist - 10.0) * 10.0, 0, 255).astype(np.uint8)
    alpha = cv2.GaussianBlur(alpha, (3, 3), 0)
    return alpha


def _paste_original_lace(base_bgr, lace_bgr, x, y, target_w, target_h):
    """Paste original lace pixels after generation; AI cannot redraw this layer."""
    if lace_bgr is None or lace_bgr.size == 0:
        return base_bgr
    target_w = max(2, int(target_w))
    target_h = max(2, int(target_h))
    lace = cv2.resize(lace_bgr, (target_w, target_h), interpolation=cv2.INTER_LANCZOS4)
    alpha = _make_lace_alpha(lace)
    if alpha is None:
        return base_bgr

    out = base_bgr.copy()
    H, W = out.shape[:2]
    x1 = max(0, int(x))
    y1 = max(0, int(y))
    x2 = min(W, x1 + target_w)
    y2 = min(H, y1 + target_h)
    if x2 <= x1 or y2 <= y1:
        return out

    crop = out[y1:y2, x1:x2].astype(np.float32)
    lace_crop = lace[:y2-y1, :x2-x1].astype(np.float32)
    a = (alpha[:y2-y1, :x2-x1].astype(np.float32) / 255.0)[..., None]
    out[y1:y2, x1:x2] = np.clip(crop * (1.0 - a) + lace_crop * a, 0, 255).astype(np.uint8)
    return out


def preserve_lace_after_generation(generated_path, lace_reference_path, orientation='horizontal'):
    """
    Deterministic lace preservation.

    Gemini generates the model/saree once. After that, the original uploaded
    lace pixels are composited locally. Gemini is never called again.

    NOTE: because a single generated image does not expose a machine-readable
    garment mask, placement is a conservative micro-border heuristic. The lace
    artwork itself is taken from the uploaded reference, not regenerated.
    """
    if os.getenv('EXACT_LACE_PIXEL_PRESERVATION', 'true').lower() not in {'1', 'true', 'yes', 'on'}:
        return False
    if not os.path.exists(generated_path) or not os.path.exists(lace_reference_path):
        return False

    base = cv2.imread(generated_path, cv2.IMREAD_COLOR)
    lace = cv2.imread(lace_reference_path, cv2.IMREAD_COLOR)
    if base is None or lace is None:
        return False

    H, W = base.shape[:2]
    lh, lw = lace.shape[:2]
    is_vertical = str(orientation).lower() == 'vertical'

    # Keep the reference aspect ratio. Do not stretch the lace.
    if is_vertical:
        target_h = max(120, int(H * 0.72))
        target_w = max(2, int(round(lw * target_h / max(1, lh))))
        max_w = max(4, int(min(H, W) * 0.035))
        if target_w > max_w:
            target_w = max_w
            target_h = max(120, int(round(lh * target_w / max(1, lw))))
        x = int(W * 0.82)
        y = max(0, (H - target_h) // 2)
    else:
        target_w = max(160, int(W * 0.72))
        target_h = max(2, int(round(lh * target_w / max(1, lw))))
        max_h = max(4, int(min(H, W) * 0.035))
        if target_h > max_h:
            target_h = max_h
            target_w = max(160, int(round(lw * target_h / max(1, lh))))
        x = max(0, (W - target_w) // 2)
        y = int(H * 0.86)

    result = _paste_original_lace(base, lace, x, y, target_w, target_h)
    cv2.imwrite(generated_path, result, [cv2.IMWRITE_PNG_COMPRESSION, 1])
    return True

def extract_generated_image_bytes(response):
    """
    Extract the first generated image as raw bytes + MIME type
    from the Gemini image-generation response.
    """
    if not response:
        return (None, None)
    parts = getattr(response, 'parts', None) or []
    for part in parts:
        inline_data = getattr(part, 'inline_data', None)
        if inline_data is None:
            continue
        data = getattr(inline_data, 'data', None)
        if data:
            if isinstance(data, str):
                try:
                    data = base64.b64decode(data)
                except Exception:
                    continue
            mime_type = getattr(inline_data, 'mime_type', None) or 'image/png'
            return (data, mime_type)
    candidates = getattr(response, 'candidates', None) or []
    for candidate in candidates:
        content = getattr(candidate, 'content', None)
        content_parts = getattr(content, 'parts', None) or []
        for part in content_parts:
            inline_data = getattr(part, 'inline_data', None)
            if inline_data is None:
                continue
            data = getattr(inline_data, 'data', None)
            if data:
                if isinstance(data, str):
                    try:
                        data = base64.b64decode(data)
                    except Exception:
                        continue
                mime_type = getattr(inline_data, 'mime_type', None) or 'image/png'
                return (data, mime_type)
    return (None, None)

def generate_image_with_gemini(reference_image_path: str, generation_prompt: str, image_size: str='2K', background_image_path: Optional[str]=None, lace_image_path: Optional[str]=None, crystal_image_path: Optional[str]=None, crystal_lock_instruction: str=''):
    """ONE AND ONLY ONE Gemini call: image references + final prompt."""
    if not gemini_client:
        raise RuntimeError('Gemini client is not configured. Set GEMINI_API_KEY in .env.')

    image_size = str(image_size).upper()
    if image_size not in {'1K', '2K', '4K'}:
        image_size = '2K'

    def make_part(path):
        with open(path, 'rb') as f:
            data = f.read()
        ext = Path(path).suffix.lower()
        mime = {'.jpg':'image/jpeg', '.jpeg':'image/jpeg', '.png':'image/png', '.webp':'image/webp'}.get(ext, 'image/png')
        return {'inline_data': {'mime_type': mime, 'data': base64.b64encode(data).decode('utf-8')}}

    # The final request contains the actual references so the image model can
    # visually preserve the saree and especially the uploaded lace.
    content_parts = [make_part(reference_image_path)]
    if lace_image_path and os.path.exists(lace_image_path):
        content_parts.append(make_part(lace_image_path))
    if crystal_image_path and os.path.exists(crystal_image_path):
        content_parts.append(make_part(crystal_image_path))
    if background_image_path and os.path.exists(background_image_path):
        content_parts.append(make_part(background_image_path))
    content_parts.append(generation_prompt)

    response = gemini_client.models.generate_content(
        model=GEMINI_IMAGE_MODEL,
        contents=content_parts,
        config=types.GenerateContentConfig(
            response_modalities=['IMAGE'],
            image_config=types.ImageConfig(aspect_ratio='3:4', image_size=image_size),
            automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True)
        )
    )

    image_bytes, generated_mime = extract_generated_image_bytes(response)
    if not image_bytes:
        text_parts = []
        for part in getattr(response, 'parts', None) or []:
            part_text = getattr(part, 'text', None)
            if part_text:
                text_parts.append(part_text)
        raise RuntimeError(
            'Gemini image generation returned no image. Check GEMINI_IMAGE_MODEL, API access/billing, and SDK version.'
            + (' Gemini text response: ' + ' '.join(text_parts) if text_parts else '')
        )
    return image_bytes, generated_mime or 'image/png'

@app.post('/api/saree/generate-all', summary='Single API for complete SareeViz generation', description='ONE API ONLY. Upload saree, exact micro-lace, optional Swarovski-type/crystal reference, and optional background. OpenCV performs local analysis/lace detection, then ONE Gemini 3.1 Flash Image call uses the uploaded reference images and final prompt. Swarovski authenticity is NOT verified; only the visible crystal design is matched.')
async def generate_all(saree_image: UploadFile=File(..., description='Choose File: saree design/reference image.'), lace_image: UploadFile=File(..., description='Choose File: exact micro-lace/patta reference image.'), crystal_reference_image: Optional[UploadFile]=File(None, description='Optional Choose File: Swarovski-type / crystal / rhinestone reference. Visual design matching only; authenticity is NOT verified.'), background_image: Optional[UploadFile]=File(None, description='Optional Choose File: background/scene reference image.'), custom_prompt: str=Form('', description='Optional custom prompt. Example: front-facing pose, full body, luxury fashion catalogue, studio lighting.'), image_size: str=Form('2K', description='Gemini image output size: 1K, 2K or 4K.'), gemini_analysis: bool=Form(False, description='False = faster OpenCV analysis only. True = also run Gemini 3.7 Flash semantic analysis.'), strict_lace_lock: bool=True, strict_crystal_lock: bool=True):
    started_at = time.perf_counter()
    micro_lace_lock_instruction = 'ABSOLUTE MICRO-LACE LOCK: Use the uploaded lace reference as an exact visual source. Keep the lace MICRO-SCALE, extremely narrow and fine. Preserve the exact repeating motif sequence, motif geometry, motif density, spacing, width, thickness, edge/piping, thread/mesh structure, white/stone/crystal-like micro details, teal/green base and red edge appearance as visible in the reference. DO NOT enlarge, thicken, widen, magnify, simplify, stylize, redesign, recolor, reinterpret, invent, merge, remove, blur or hide any lace detail. Never turn it into a broad/oversized border or panel. Higher output resolution MUST NOT increase the physical size of the lace motifs. Apply this lock everywhere the lace appears on the saree.'
    allowed_extensions = {'.jpg', '.jpeg', '.png', '.webp'}
    saree_ext = Path(saree_image.filename or '').suffix.lower()
    lace_ext = Path(lace_image.filename or '').suffix.lower()
    if saree_ext not in allowed_extensions:
        return {'success': False, 'status': 'invalid_saree_format', 'message': 'Saree image must be JPG, JPEG, PNG or WEBP.'}
    if lace_ext not in allowed_extensions:
        return {'success': False, 'status': 'invalid_lace_format', 'message': 'Lace image must be JPG, JPEG, PNG or WEBP.'}
    if crystal_reference_image is not None:
        crystal_ext = Path(crystal_reference_image.filename or '').suffix.lower()
        if crystal_ext not in allowed_extensions:
            return {'success': False, 'status': 'invalid_crystal_reference_format', 'message': 'Crystal/Swarovski-type reference must be JPG, JPEG, PNG or WEBP.'}
    else:
        crystal_ext = ''
    if background_image is not None:
        bg_ext = Path(background_image.filename or '').suffix.lower()
        if bg_ext not in allowed_extensions:
            return {'success': False, 'status': 'invalid_background_format', 'message': 'Background image must be JPG, JPEG, PNG or WEBP.'}
    else:
        bg_ext = ''
    saree_bytes = await saree_image.read()
    lace_bytes = await lace_image.read()
    crystal_reference_bytes = await crystal_reference_image.read() if crystal_reference_image is not None else b''
    background_bytes = await background_image.read() if background_image is not None else b''
    if not saree_bytes:
        return {'success': False, 'status': 'empty_saree_file', 'message': 'Saree image is empty.'}
    if not lace_bytes:
        return {'success': False, 'status': 'empty_lace_file', 'message': 'Lace image is empty.'}
    if crystal_reference_image is not None and not crystal_reference_bytes:
        return {'success': False, 'status': 'empty_crystal_reference_file', 'message': 'Crystal/Swarovski-type reference image is empty.'}
    request_id = f'req_{uuid.uuid4().hex[:12]}'
    request_dir = os.path.abspath(os.path.join(UPLOAD_DIR, request_id))
    os.makedirs(request_dir, exist_ok=True)
    saree_name = f'{request_id}_saree{saree_ext}'
    saree_path = os.path.join(request_dir, saree_name)
    with open(saree_path, 'wb') as f:
        f.write(saree_bytes)
    lace_input_name = f'{request_id}_lace_input{lace_ext}'
    lace_input_path = os.path.join(ANALYSIS_DIR, lace_input_name)
    with open(lace_input_path, 'wb') as f:
        f.write(lace_bytes)
    crystal_reference_name = None
    crystal_reference_path = None
    if crystal_reference_image is not None and crystal_reference_bytes:
        crystal_reference_name = f'{request_id}_swarovski_type_reference{crystal_ext}'
        crystal_reference_path = os.path.abspath(os.path.join(CRYSTAL_DIR, crystal_reference_name))
        with open(crystal_reference_path, 'wb') as f:
            f.write(crystal_reference_bytes)
    background_name = None
    background_path = None
    if background_image is not None and background_bytes:
        background_name = f'{request_id}_background{bg_ext}'
        background_path = os.path.abspath(os.path.join(BACKGROUND_DIR, background_name))
        with open(background_path, 'wb') as f:
            f.write(background_bytes)
    analysis_started = time.perf_counter()
    opencv_analysis, analyzed_image, detail_map, bright_mask, texture_mask = analyze_saree_opencv(saree_path, max_side=1200)
    detection_name = f'{request_id}_{Path(saree_path).stem}_detection.json'
    detection_path = os.path.join(ANALYSIS_DIR, detection_name)
    with open(detection_path, 'w', encoding='utf-8') as f:
        json.dump(opencv_analysis, f, indent=2, ensure_ascii=False)
    analysis_seconds = round(time.perf_counter() - analysis_started, 3)
    # LOW-COST MODE: no Gemini semantic-analysis call.
    semantic_analysis = {'available': False, 'enabled': False, 'message': 'Disabled for low-cost mode; OpenCV only.'}
    design_lock = build_design_lock(opencv_analysis, semantic_analysis)
    lock_name = f'{request_id}_design_lock.json'
    lock_path = os.path.join(ANALYSIS_DIR, lock_name)
    with open(lock_path, 'w', encoding='utf-8') as f:
        json.dump(design_lock, f, indent=2, ensure_ascii=False)
    # LOW-COST MODE: lace detection is local OpenCV; no Gemini analysis call.
    local_lace = detect_lace_strip(lace_input_path, request_id)
    local_img = cv2.imread(lace_input_path, cv2.IMREAD_UNCHANGED)
    local_h, local_w = local_img.shape[:2] if local_img is not None else (1, 1)
    local_bbox = local_lace.get('bbox', {}) if isinstance(local_lace, dict) else {}
    lace_detection = {
        'available': bool(local_lace.get('success')),
        'found': bool(local_lace.get('success')),
        'confidence': 1.0 if local_lace.get('success') else 0.0,
        'orientation': 'vertical' if local_bbox.get('height', 0) >= local_bbox.get('width', 0) else 'horizontal',
        'bbox': {
            'x': round(local_bbox.get('x', 0) / max(1, local_w), 6),
            'y': round(local_bbox.get('y', 0) / max(1, local_h), 6),
            'width': round(local_bbox.get('width', 0) / max(1, local_w), 6),
            'height': round(local_bbox.get('height', 0) / max(1, local_h), 6)
        }
    }
    # LOW-COST MODE: no separate Gemini crystal-analysis call.
    crystal_read = {
        'available': False,
        'enabled': False,
        'found': bool(crystal_reference_path and crystal_reference_bytes),
        'message': 'Crystal analysis disabled for low-cost mode.'
    }

    lace_reference = None
    lace_path = None
    if lace_detection.get('available') and lace_detection.get('found'):
        lace_cv = cv2.imdecode(np.frombuffer(lace_bytes, dtype=np.uint8), cv2.IMREAD_UNCHANGED)
        if lace_cv is not None:
            lace_h, lace_w = lace_cv.shape[:2]
            bbox = lace_detection.get('bbox', {})
            x = max(0.0, min(1.0, float(bbox.get('x', 0.0))))
            y = max(0.0, min(1.0, float(bbox.get('y', 0.0))))
            w = max(0.0, min(1.0 - x, float(bbox.get('width', 0.0))))
            h = max(0.0, min(1.0 - y, float(bbox.get('height', 0.0))))
            x1 = int(x * lace_w)
            y1 = int(y * lace_h)
            x2 = int((x + w) * lace_w)
            y2 = int((y + h) * lace_h)
            pad_x = max(4, int((x2 - x1) * 0.08))
            pad_y = max(4, int((y2 - y1) * 0.02))
            x1 = max(0, x1 - pad_x)
            y1 = max(0, y1 - pad_y)
            x2 = min(lace_w, x2 + pad_x)
            y2 = min(lace_h, y2 + pad_y)
            lace_crop = lace_cv[y1:y2, x1:x2]
            if lace_crop.size:
                lace_name = f'{request_id}_lace_reference.png'
                lace_path = os.path.abspath(os.path.join(ANALYSIS_DIR, lace_name))
                cv2.imwrite(lace_path, lace_crop, [cv2.IMWRITE_PNG_COMPRESSION, 1])
                lace_reference = {'file_name': lace_name, 'local_path': lace_path, 'url': f'/design-maps/{lace_name}', 'width': x2 - x1, 'height': y2 - y1, 'micro_lace_locked': True, 'confidence': lace_detection.get('confidence'), 'orientation': lace_detection.get('orientation'), 'bbox_original_pixels': {'x': x1, 'y': y1, 'width': x2 - x1, 'height': y2 - y1}}
    if not lace_reference:
        try:
            os.remove(lace_input_path)
        except OSError:
            pass
        return {'success': False, 'request_id': request_id, 'status': 'micro_lace_not_detected', 'message': 'Local OpenCV could not detect the micro-lace reference. Final image generation was stopped to avoid changing the lace.'}
    try:
        os.remove(lace_input_path)
    except OSError:
        pass
    crystal_read['enabled'] = bool(crystal_reference_path and crystal_reference_bytes)
    crystal_lock_instruction = ''
    if crystal_read.get('found'):
        crystal_lock_instruction = json.dumps(crystal_read.get('design', {}), ensure_ascii=False)
    elif crystal_reference_path:
        crystal_lock_instruction = 'Use the uploaded crystal reference image directly as the visual authority. Do not guess missing crystal details.'
    # FAST MODE: build the generation specification locally instead of making
    # another Gemini request. This removes one full model round-trip.
    generation_prompt_local = (
        'Create a photorealistic adult Indian female fashion catalogue image. '
        'Use the uploaded saree image as the highest-priority garment reference. '
        'Preserve original color, body pattern, motifs, motif arrangement, border, pallu, '
        'embroidery, zari and all visible decorative details. Only change presentation, pose, '
        'drape, camera/composition and background when requested. Do not redesign, recolor, '
        'replace, simplify or invent garment details. The uploaded lace image is an exact locked '
        'micro-lace reference: preserve exact geometry, motif sequence, density, spacing, width, '
        'thickness, edge shape, thread/mesh, beads/rhinestones, colors and placement. Keep it '
        'at the same tiny physical scale. Never enlarge, thicken, widen, magnify, redesign, '
        'recolor or invent lace details. Return one highly realistic full-body image.'
    )
    if custom_prompt.strip():
        generation_prompt_local += '\n\nUSER CUSTOM PROMPT (presentation only):\n' + custom_prompt.strip()
    generation_spec = {'available': True, 'model': 'local-fast-prompt', 'input_type': 'design_lock_json', 'result': {
        'generation_prompt': generation_prompt_local,
        'negative_prompt': 'oversized lace, enlarged lace motifs, thick lace, wide lace, redesigned lace, recolored lace, invented motifs, missing lace details, generic saree, altered border, altered pallu',
        'model_requirements': {'photorealistic': True, 'full_body': True},
        'preservation_lock': design_lock.get('preservation_rules', {})
    }}
    try:
        if isinstance(generation_spec, dict):
            generation_spec.setdefault('micro_lace_lock', {})
            generation_spec['micro_lace_lock'].update({'locked': bool(strict_lace_lock), 'scale': 'micro', 'allow_enlarge': False, 'allow_thicken': False, 'allow_widen': False, 'allow_magnify': False, 'allow_redesign': False, 'allow_recolor': False, 'allow_invent': False, 'instruction': micro_lace_lock_instruction})
    except Exception:
        pass
    try:
        if isinstance(generation_spec, dict):
            generation_spec.setdefault('crystal_design_lock', {})
            generation_spec['crystal_design_lock'].update({'locked': bool(strict_crystal_lock and crystal_reference_path), 'reference_type': 'swarovski_type_crystal_reference', 'authenticity_verified': False, 'visual_reference_only': True, 'stone_size_locked': True, 'shape_locked': True, 'faceting_locked': True, 'spacing_locked': True, 'density_locked': True, 'pattern_locked': True, 'placement_locked': True, 'shine_character_locked': True, 'allow_enlarge': False, 'allow_invent': False, 'allow_redesign': False, 'allow_recolor': False, 'analysis': crystal_read.get('design', {}), 'instruction': crystal_lock_instruction})
    except Exception:
        pass
    if not generation_spec.get('available'):
        return {'success': False, 'request_id': request_id, 'status': 'generation_prompt_failed', 'analysis': opencv_analysis, 'design_lock': design_lock, 'micro_lace': lace_reference, 'message': generation_spec.get('message', generation_spec.get('error', 'Gemini generation prompt failed.'))}
    generation_result = generation_spec.get('result', {})
    generated_prompt = str(generation_result.get('generation_prompt', '')).strip()
    if custom_prompt.strip():
        generated_prompt += '\n\nUSER CUSTOM PROMPT:\n' + custom_prompt.strip()
    generation_name = f'{request_id}_generation_prompt.json'
    generation_path = os.path.join(ANALYSIS_DIR, generation_name)
    saved_generation = {'request_id': request_id, 'status': 'generation_prompt_created', 'source': 'single_api', 'custom_prompt': custom_prompt, 'generation_spec': {**generation_spec, 'result': {**generation_result, 'generation_prompt': generated_prompt}}}
    with open(generation_path, 'w', encoding='utf-8') as f:
        json.dump(saved_generation, f, indent=2, ensure_ascii=False)
    image_size = image_size.upper()
    if image_size not in {'1K', '2K', '4K'}:
        image_size = '2K'
    # Final request is a SINGLE Gemini image-generation call.
    # The actual saree/lace/crystal/background references are included in that one call.
    design_context = json.dumps({
        'saree_design_analysis': opencv_analysis,
        'micro_lace_detection': lace_detection,
        'micro_lace_lock': {
            'locked': True,
            'scale': 'micro',
            'exact_visual_reference_required': True,
            'preserve_geometry': True,
            'preserve_motif_sequence': True,
            'preserve_density': True,
            'preserve_spacing': True,
            'preserve_width': True,
            'preserve_thickness': True,
            'do_not_enlarge': True,
            'do_not_redesign': True,
            'do_not_recolor': True
        },
        'crystal_design_analysis': crystal_read.get('design', {}) if isinstance(crystal_read, dict) else {}
    }, ensure_ascii=False, separators=(',', ':'))
    generated_prompt = (
        generated_prompt
        + '\n\nSTRICT MICRO-LACE DESIGN LOCK:\n'
        + micro_lace_lock_instruction
        + '\n\nSTRUCTURED DESIGN DATA (TEXT ONLY — FOLLOW EXACTLY):\n'
        + design_context
            )
    image_bytes_out, generated_mime = generate_image_with_gemini(saree_path, generated_prompt, image_size=image_size, background_image_path=background_path, lace_image_path=lace_path, crystal_image_path=crystal_reference_path, crystal_lock_instruction=crystal_lock_instruction)
    output_ext = '.png'
    if generated_mime == 'image/jpeg':
        output_ext = '.jpg'
    elif generated_mime == 'image/webp':
        output_ext = '.webp'
    output_name = f'{request_id}_final{output_ext}'
    output_path = os.path.abspath(os.path.join(GENERATED_DIR, output_name))
    with open(output_path, 'wb') as f:
        f.write(image_bytes_out)
    # FINAL DETERMINISTIC LACE LAYER: the uploaded lace pixels are applied after
    # Gemini generation so the model cannot redraw/reinterpret the lace itself.
    try:
        preserve_lace_after_generation(
            output_path,
            lace_path,
            orientation=str(lace_reference.get('orientation') or lace_detection.get('orientation') or 'horizontal')
        )
    except Exception as lace_post_error:
        print(f'WARNING: exact lace pixel preservation failed: {lace_post_error}')
    total_seconds = round(time.perf_counter() - started_at, 3)
    metadata = {'request_id': request_id, 'status': 'complete_success', 'model': GEMINI_IMAGE_MODEL, 'analysis_model': 'NONE (OpenCV only)', 'image_size': image_size, 'saree_reference': saree_name, 'lace_reference': lace_reference.get('file_name'), 'background_reference': background_name, 'crystal_reference': crystal_reference_name, 'crystal_design_read': crystal_read, 'micro_lace_locked': True, 'lace_design_lock': {'locked': True, 'scale': 'micro', 'exact_visual_reference': True, 'pattern_locked': True, 'allow_enlarge': False, 'allow_thicken': False, 'allow_redesign': False, 'allow_recolor': False}, 'custom_prompt': custom_prompt, 'generation_prompt_file': generation_name, 'local_path': output_path, 'url': f'/generated/{output_name}', 'performance': {'analysis_seconds': analysis_seconds, 'total_seconds': total_seconds}}
    complete_metadata_name = f'{request_id}_complete_generation.json'
    complete_metadata_path = os.path.join(ANALYSIS_DIR, complete_metadata_name)
    with open(complete_metadata_path, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)
    save_request_metadata(request_id, {'request_id': request_id, 'status': 'complete_success', 'saree_reference': saree_name, 'lace_reference': lace_reference.get('file_name'), 'crystal_reference': crystal_reference_name, 'background_reference': background_name, 'design_lock': lock_name, 'generation_prompt': generation_name, 'generated_image': f'/generated/{output_name}'})
    return {'success': True, 'request_id': request_id, 'status': 'complete_success', 'workflow': ['saree_uploaded', 'request_id_created', 'design_analyzed', 'design_lock_created', 'micro_lace_detected', 'swarovski_type_crystal_read', 'crystal_design_locked', 'generation_prompt_created', 'final_image_generated'], 'references': {'saree': saree_name, 'micro_lace': lace_reference, 'crystal_reference': {'file_name': crystal_reference_name, 'url': f'/crystal-references/{crystal_reference_name}', 'read': crystal_read} if crystal_reference_name else None, 'background': background_name}, 'lace_design_lock': {'locked': True, 'scale': 'micro', 'exact_visual_reference': True, 'pattern_locked': True, 'motif_sequence_locked': True, 'motif_density_locked': True, 'spacing_locked': True, 'width_locked': True, 'edge_shape_locked': True, 'stone_size_locked': True, 'allow_enlarge': False, 'allow_thicken': False, 'allow_redesign': False, 'allow_recolor': False, 'allow_invent': False}, 'files': {'design_lock': f'/design-maps/{lock_name}', 'generation_prompt': f'/design-maps/{generation_name}', 'final_image': f'/generated/{output_name}'}, 'generated_image': {'file_name': output_name, 'local_path': output_path, 'url': f'/generated/{output_name}', 'mime_type': generated_mime, 'model': GEMINI_IMAGE_MODEL, 'image_size': image_size}, 'custom_prompt': custom_prompt, 'performance': {'analysis_seconds': analysis_seconds, 'total_seconds': total_seconds}, 'message': 'LOW-COST mode: OpenCV local analysis -> local lace detection -> ONE Gemini 3.1 Flash Image generation call with reference images.'}
if __name__ == '__main__':
    import uvicorn
    uvicorn.run('main:app', host='0.0.0.0', port=8000)
