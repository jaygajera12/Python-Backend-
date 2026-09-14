import os
import uuid
import json
import base64
import time
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
app = FastAPI(title='SareeViz AI - Micro Lace + Crystal/Swarovski-Type Design Lock', description='Upload a saree image in Swagger. Python/OpenCV performs the complete local visual analysis: colors, fine details, texture regions, decorative bands, borders, brightness and pattern density. A detailed Design Lock JSON and visual detection map are returned.', version='7.0.0')
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
    """Detailed local saree analysis. No Gemini analysis call is used."""
    original = read_image(image_path)
    image = resize_keep_ratio(original, max_side=max(600, min(int(max_side), 1800)))
    detail_map = create_fine_detail_map(image)
    bright_candidates, bright_mask = detect_bright_detail_candidates(image)
    horizontal_bands = detect_horizontal_design_bands(image)
    side_borders = detect_side_borders(image)
    colors = dominant_colors(image)
    texture_regions, texture_mask = detect_texture_regions(image)

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    edge_density = float(np.count_nonzero(detail_map) / max(1, detail_map.size))
    brightness_mean = float(np.mean(gray))
    saturation_mean = float(np.mean(hsv[:, :, 1]))
    texture_strength = float(np.std(gray))

    if edge_density < 0.03:
        pattern_level = 'low'
    elif edge_density < 0.08:
        pattern_level = 'medium'
    else:
        pattern_level = 'high'

    if brightness_mean < 70:
        brightness_level = 'dark'
    elif brightness_mean < 170:
        brightness_level = 'medium'
    else:
        brightness_level = 'bright'

    if saturation_mean < 45:
        saturation_level = 'low'
    elif saturation_mean < 110:
        saturation_level = 'medium'
    else:
        saturation_level = 'high'

    if texture_strength < 25:
        texture_level = 'smooth'
    elif texture_strength < 55:
        texture_level = 'medium'
    else:
        texture_level = 'high'

    analysis = {
        'detector': 'Python/OpenCV',
        'analysis_type': 'detailed_local_visual_analysis',
        'image': {
            'width': int(image.shape[1]),
            'height': int(image.shape[0]),
            'channels': int(image.shape[2]),
        },
        'colors': colors,
        'visual_statistics': {
            'mean_brightness': round(brightness_mean, 3),
            'brightness_level': brightness_level,
            'mean_saturation': round(saturation_mean, 3),
            'saturation_level': saturation_level,
            'texture_strength_std': round(texture_strength, 3),
            'texture_level': texture_level,
        },
        'pattern': {
            'edge_density_percent': round(edge_density * 100, 3),
            'level': pattern_level,
        },
        'fine_detail': {
            'candidate_count': len(bright_candidates),
            'candidates': bright_candidates[:5000],
            'note': 'OpenCV bright/reflective detail candidates. These are visual candidates and are not automatically classified as Swarovski or Siroki.'
        },
        'texture_regions': {
            'count': len(texture_regions),
            'regions': texture_regions,
        },
        'horizontal_design_bands': horizontal_bands,
        'side_border_candidates': side_borders,
        'detection_summary': {
            'decorative_band_count': len(horizontal_bands),
            'side_border_count': len(side_borders),
            'texture_region_count': len(texture_regions),
            'bright_detail_candidate_count': len(bright_candidates),
            'analysis_complete': True,
        },
        'design_lock': {
            'preserve_colors': True,
            'preserve_pattern': True,
            'preserve_motifs': True,
            'preserve_border': True,
            'preserve_pallu': True,
            'preserve_embroidery': True,
            'preserve_reflective_details': True,
            'do_not_redesign': True,
            'analysis_source': 'Python/OpenCV only',
        },
    }
    return (analysis, image, detail_map, bright_mask, texture_mask)

def build_design_lock(opencv_analysis):
    return {'version': '1.0', 'reference_priority': 'highest', 'analysis_engine': 'Python/OpenCV', 'opencv': opencv_analysis, 'preservation_rules': {'keep_original_color': True, 'keep_original_motifs': True, 'keep_motif_arrangement': True, 'keep_original_border': True, 'keep_original_pallu': True, 'keep_embroidery': True, 'keep_zari': True, 'keep_visible_crystals_or_stones': True, 'keep_pattern_density': True, 'do_not_recolor': True, 'do_not_redesign': True, 'do_not_invent_new_motifs': True, 'do_not_remove_visible_details': True}, 'generation_instruction': 'Use the uploaded original saree image as the primary garment reference. Transform only the presentation/drape. Preserve the original saree design as closely as possible.'}

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
GENERATION_SPEC_SYSTEM_PROMPT = '\nYou are a professional Indian fashion / textile prompt engineer.\n\nYou will receive a DESIGN LOCK JSON created from OpenCV detection and\nPython/OpenCV detailed visual analysis of an uploaded saree reference.\n\nYour job is NOT to redesign the saree and NOT to generate an image.\nYour job is to convert the JSON into a precise image-generation\nspecification that a separate virtual try-on / image-generation model\ncan follow.\n\nCRITICAL PRESERVATION RULES:\n1. Preserve the original saree color.\n2. Preserve the original body pattern and motif arrangement.\n3. Preserve the border design, width and placement as supported by the data.\n4. Preserve the pallu design and visible placement.\n5. Preserve embroidery, zari and visible crystal/stone-like details.\n6. Do not invent missing details.\n7. Do not replace the saree with a generic saree.\n8. Do not recolor or redesign the garment.\n9. Only change presentation variables such as model, pose, drape,\n   camera/composition and background when requested.\n10. If a detail is uncertain or not supported by the JSON, say that it\n    must follow the uploaded reference image rather than guessing.\n\nReturn ONLY valid JSON with exactly these top-level keys:\n- generation_prompt\n- negative_prompt\n- model_requirements\n- preservation_lock\n\nThe generation_prompt must be a single detailed prompt for a separate\nimage-generation / virtual-try-on model.\n'

POSE_PRESETS = {
    'Classic Catalog Standing Pose': 'Classic catalog standing pose: adult Indian female fashion model standing upright in a graceful, balanced full-body pose, front-facing or slightly turned, elegant catalogue posture, saree fully visible.',
    'Hand on Waist': 'Hand on waist pose: adult Indian female fashion model standing confidently with one hand naturally resting on the waist and the other arm relaxed, keeping saree drape, pallu, border and lace visible.',
    'Saree Standing Pose on Stairs': 'Saree standing pose on stairs: adult Indian female fashion model standing elegantly on a refined staircase, full-body composition, one foot naturally positioned on a stair step, saree and pallu clearly visible.',
    'Elegant Staircase Photoshoot Pose': 'Elegant staircase photoshoot pose: adult Indian female fashion model posing gracefully on a staircase in premium Indian fashion catalogue style, full-body view, natural elegant stance, saree drape and pallu clearly visible.'
}

def normalize_pose_prompt(pose: str) -> str:
    pose = (pose or '').strip()
    return POSE_PRESETS.get(pose, pose) if pose else ''

def generate_image_with_gemini(reference_image_path: str, generation_prompt: str, image_size: str='2K', background_image_path: Optional[str]=None, lace_image_path: Optional[str]=None, crystal_image_path: Optional[str]=None, blouse_image_path: Optional[str]=None, crystal_lock_instruction: str=''):
    """
    Generate an image with Gemini 3.1 Flash Image and return:
    raw image bytes, MIME type, and the model response.
    """
    if not gemini_client:
        raise RuntimeError('Gemini client is not configured. Set GEMINI_API_KEY in .env.')
    image_size = image_size.upper()
    if image_size not in {'1K', '2K', '4K'}:
        image_size = '2K'
    with open(reference_image_path, 'rb') as f:
        reference_bytes = f.read()
    extension = Path(reference_image_path).suffix.lower()
    mime_type = {'.jpg': 'image/jpeg', '.jpeg': 'image/jpeg', '.png': 'image/png', '.webp': 'image/webp'}.get(extension, 'image/jpeg')
    lace_instruction = ''
    if lace_image_path and os.path.exists(lace_image_path):
        lace_instruction = '\nMICRO-LACE DESIGN LOCK — ABSOLUTE / HIGHEST PRIORITY:\n- The uploaded lace-patta image is a LOCKED garment-design reference.\n- The lace must remain at the SAME MICRO SCALE as shown in the reference.\n- Preserve the exact lace geometry, motif sequence, motif density, spacing,\n  width, thickness, edge shape, thread/mesh structure, beads/rhinestones,\n  stone size and placement.\n- Do NOT enlarge, thicken, widen, magnify, stylize, simplify, redesign,\n  replace, recolor, reinterpret, or invent the lace.\n- Do NOT remove, merge, blur, hide, or crop away visible lace details.\n- Do NOT turn micro-lace into a large applique, broad panel, oversized border,\n  or decorative band.\n- Tiny motifs must remain tiny and dense in the final saree.\n- The lace is an exact visual + scale reference, not a general style reference.\n'
    micro_lace_scale_instruction = build_micro_lace_scale_instruction(reference_image_path, lace_image_path)
    crystal_instruction = ''
    if blouse_image_path and os.path.exists(blouse_image_path):
        with open(blouse_image_path, 'rb') as f:
            blouse_bytes = f.read()
        blouse_extension = Path(blouse_image_path).suffix.lower()
        blouse_mime_type = {'.jpg': 'image/jpeg', '.jpeg': 'image/jpeg', '.png': 'image/png', '.webp': 'image/webp'}.get(blouse_extension, 'image/png')
        content_parts.append({'inline_data': {'mime_type': blouse_mime_type, 'data': base64.b64encode(blouse_bytes).decode('utf-8')}})
    if crystal_image_path and os.path.exists(crystal_image_path):
        crystal_instruction = '\nSWAROVSKI-TYPE / CRYSTAL DESIGN LOCK — VISUAL REFERENCE ONLY:\n- Use the uploaded crystal reference as the exact visual reference for the visible crystal/rhinestone design.\n- Preserve crystal micro-size, cut/shape, faceting, spacing, density, repeat pattern, alignment, placement and reflective character.\n- Preserve visible white/clear stone appearance and its relationship to the lace/garment.\n- DO NOT enlarge, exaggerate, simplify, redesign, recolor or invent the crystal arrangement.\n- DO NOT claim or imply authenticity; match only the visible design shown in the reference.\n- Higher output resolution MUST NOT increase the physical size of individual stones.\n'
        if crystal_lock_instruction:
            crystal_instruction += '\nCRYSTAL ANALYSIS LOCK:\n' + crystal_lock_instruction + '\n'
    blouse_instruction = ''
    if blouse_image_path and os.path.exists(blouse_image_path):
        blouse_instruction = '''
BLOUSE DESIGN LOCK — VISUAL REFERENCE:
- The uploaded blouse image is the exact blouse reference.
- Preserve blouse color, fabric appearance, neckline, sleeve design, embroidery,
  embellishment, silhouette and visible construction details.
- Do NOT redesign, recolor, simplify, replace or invent blouse details.
- Keep the blouse visually consistent with the uploaded reference.
'''
    background_instruction = ''
    if background_image_path and os.path.exists(background_image_path):
        background_instruction = '\nBACKGROUND REFERENCE:\n- A second uploaded image is provided as the background reference.\n- Use that image as the background/scene reference.\n- Preserve the background scene, environment, perspective, lighting direction,\n  overall composition and visual character as closely as possible.\n- Do not use a white background when a background reference is supplied.\n- Do not replace the background with an unrelated scene.\n'
    final_prompt = f"\nUse the uploaded saree reference image as the highest-priority garment reference.\n\n{generation_prompt}\n\n{background_instruction}\n\n{blouse_instruction}\n\n{lace_instruction}\n\n{micro_lace_scale_instruction}\n\nIMPORTANT:\n- The uploaded saree is the exact garment reference.\n- Preserve garment proportions from the reference; do not scale small lace\n  details into large decorative graphics.\n- Preserve the saree's original color, motifs, motif arrangement, border,\n  pallu, embroidery, zari, visible crystal/stone-like details, AND ALL LACE DETAILS.\n- The lace is a critical MICRO garment feature and must remain micro-scale.\n- Match the lace visual size relative to the saree border and saree body in the source; it must not become a large applique, panel, or oversized border.\n- If a lace reference image is supplied, use it as an exact pattern-and-scale\n  reference, not merely a style suggestion.\n- Preserve the exact lace width, physical scale, motif density, motif spacing,\n  edge shape, thread thickness, mesh, dots/beads, holes, trims and decorative\n  edging visible in the reference.\n- Do NOT enlarge, thicken, widen, magnify, stylize, simplify, merge, replace,\n  or redesign the lace.\n- The repeated lace motifs must remain small and dense exactly as in the\n  reference, with no oversized symbols or oversized gaps.\n- Keep the lace sharp without making its elements larger than the source.\n- Preserve the lace along every visible saree edge where it exists.\n- Do not redesign, recolor, replace, simplify, or invent the saree.\n- Keep the saree visually recognizable as the uploaded reference.\n- Generate a highly realistic adult Indian female fashion model.\n- Full-body standing composition.\n- Premium Indian fashion catalogue photography.\n- Natural realistic fabric drape.\n- The lace must remain physically and proportionally small, fine and narrow in the final garment exactly as shown by the reference.\n- Do not make the lace larger merely because output resolution is higher.\n- Return the generated image as the image output.\n"
    content_parts = [{'inline_data': {'mime_type': mime_type, 'data': base64.b64encode(reference_bytes).decode('utf-8')}}]
    if background_image_path and os.path.exists(background_image_path):
        with open(background_image_path, 'rb') as f:
            background_bytes = f.read()
        background_extension = Path(background_image_path).suffix.lower()
        background_mime_type = {'.jpg': 'image/jpeg', '.jpeg': 'image/jpeg', '.png': 'image/png', '.webp': 'image/webp'}.get(background_extension, 'image/jpeg')
        content_parts.append({'inline_data': {'mime_type': background_mime_type, 'data': base64.b64encode(background_bytes).decode('utf-8')}})
    if lace_image_path and os.path.exists(lace_image_path):
        with open(lace_image_path, 'rb') as f:
            lace_bytes = f.read()
        lace_extension = Path(lace_image_path).suffix.lower()
        lace_mime_type = {'.jpg': 'image/jpeg', '.jpeg': 'image/jpeg', '.png': 'image/png', '.webp': 'image/png'}.get(lace_extension, 'image/png')
        content_parts.append({'inline_data': {'mime_type': lace_mime_type, 'data': base64.b64encode(lace_bytes).decode('utf-8')}})
    if crystal_image_path and os.path.exists(crystal_image_path):
        with open(crystal_image_path, 'rb') as f:
            crystal_bytes = f.read()
        crystal_extension = Path(crystal_image_path).suffix.lower()
        crystal_mime_type = {'.jpg': 'image/jpeg', '.jpeg': 'image/jpeg', '.png': 'image/png', '.webp': 'image/webp'}.get(crystal_extension, 'image/png')
        content_parts.append({'inline_data': {'mime_type': crystal_mime_type, 'data': base64.b64encode(crystal_bytes).decode('utf-8')}})
    content_parts.append(final_prompt)
    try:
        from google.genai import types
    except ImportError:
        raise RuntimeError('google-genai is required. Run: python -m pip install -U google-genai')
    response = gemini_client.models.generate_content(model=GEMINI_IMAGE_MODEL, contents=content_parts, config=types.GenerateContentConfig(response_modalities=['IMAGE'], image_config=types.ImageConfig(aspect_ratio='3:4', image_size=image_size)))
    image_bytes, generated_mime = extract_generated_image_bytes(response)
    if not image_bytes:
        text_parts = []
        for part in getattr(response, 'parts', None) or []:
            part_text = getattr(part, 'text', None)
            if part_text:
                text_parts.append(part_text)
        raise RuntimeError('Gemini image generation returned no image. Check GEMINI_IMAGE_MODEL, API access/billing, and the SDK version. ' + (' Gemini text response: ' + ' '.join(text_parts) if text_parts else ' No image data found in the response.'))
    return (image_bytes, generated_mime or 'image/png')

@app.post('/api/saree/generate-all', summary='Single API for complete SareeViz generation', description='ONE API ONLY. Upload saree, exact micro-lace, optional Swarovski-type/crystal reference, and optional background. Uses local OpenCV analysis and direct lace/blouse/crystal references; uses Python/OpenCV analysis only; Gemini 3.1 Flash Image generates the final image. Swarovski authenticity is NOT verified; only the visible crystal design is matched.')
async def generate_all(saree_image: UploadFile=File(..., description='Choose File: saree design/reference image.'), lace_image: UploadFile=File(..., description='Choose File: exact micro-lace/patta reference image.'), blouse_image: UploadFile=File(..., description='Choose File: exact blouse reference image.'), crystal_reference_image: Optional[UploadFile]=File(None, description='Optional Choose File: Swarovski-type / crystal / rhinestone reference. Visual design matching only; authenticity is NOT verified.'), background_image: Optional[UploadFile]=File(None, description='Optional Choose File: background/scene reference image.'), custom_prompt: str=Form('', description='Optional custom prompt. Example: front-facing pose, full body, luxury fashion catalogue, studio lighting.'), pose: str=Form('', description='Optional pose preset: Classic Catalog Standing Pose, Hand on Waist, Saree Standing Pose on Stairs, or Elegant Staircase Photoshoot Pose.'), image_size: str=Form('2K', description='Gemini image output size: 1K, 2K or 4K.'), strict_lace_lock: bool=True, strict_crystal_lock: bool=True):
    started_at = time.perf_counter()
    micro_lace_lock_instruction = 'ABSOLUTE MICRO-LACE LOCK: Use the uploaded lace reference as an exact visual source. Keep the lace MICRO-SCALE, extremely narrow and fine. Preserve the exact repeating motif sequence, motif geometry, motif density, spacing, width, thickness, edge/piping, thread/mesh structure, white/stone/crystal-like micro details, teal/green base and red edge appearance as visible in the reference. DO NOT enlarge, thicken, widen, magnify, simplify, stylize, redesign, recolor, reinterpret, invent, merge, remove, blur or hide any lace detail. Never turn it into a broad/oversized border or panel. Higher output resolution MUST NOT increase the physical size of the lace motifs. Apply this lock everywhere the lace appears on the saree.'
    allowed_extensions = {'.jpg', '.jpeg', '.png', '.webp'}
    saree_ext = Path(saree_image.filename or '').suffix.lower()
    lace_ext = Path(lace_image.filename or '').suffix.lower()
    blouse_ext = Path(blouse_image.filename or '').suffix.lower()
    if saree_ext not in allowed_extensions:
        return {'success': False, 'status': 'invalid_saree_format', 'message': 'Saree image must be JPG, JPEG, PNG or WEBP.'}
    if lace_ext not in allowed_extensions:
        return {'success': False, 'status': 'invalid_lace_format', 'message': 'Lace image must be JPG, JPEG, PNG or WEBP.'}
    if blouse_ext not in allowed_extensions:
        return {'success': False, 'status': 'invalid_blouse_format', 'message': 'Blouse image must be JPG, JPEG, PNG or WEBP.'}
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
    blouse_bytes = await blouse_image.read()
    crystal_reference_bytes = await crystal_reference_image.read() if crystal_reference_image is not None else b''
    background_bytes = await background_image.read() if background_image is not None else b''
    if not saree_bytes:
        return {'success': False, 'status': 'empty_saree_file', 'message': 'Saree image is empty.'}
    if not lace_bytes:
        return {'success': False, 'status': 'empty_lace_file', 'message': 'Lace image is empty.'}
    if not blouse_bytes:
        return {'success': False, 'status': 'empty_blouse_file', 'message': 'Blouse image is empty.'}
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
    blouse_name = f'{request_id}_blouse{blouse_ext}'
    blouse_path = os.path.join(request_dir, blouse_name)
    with open(blouse_path, 'wb') as f:
        f.write(blouse_bytes)
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
    # ============================================================
    # PYTHON / OPENCV ANALYSIS ONLY
    # ============================================================
    # No Gemini analysis model is used anywhere in this flow.
    # The detailed OpenCV result above is the complete analysis source.
    design_lock = build_design_lock(opencv_analysis)

    # No Gemini call for lace detection.
    # The user-uploaded lace image itself is the exact visual reference.
    lace_reference = {
        'file_name': f'{request_id}_lace_input{lace_ext}',
        'local_path': lace_input_path,
        'url': f'/design-maps/{os.path.basename(lace_input_path)}',
        'micro_lace_locked': True,
        'reference_direct': True,
        'detection_model': None,
        'message': 'Direct uploaded lace reference; no separate Gemini analysis call.'
    }

    # No separate Gemini crystal analysis call.
    # The uploaded crystal image is passed directly to the image model.
    crystal_read = {
        'available': False,
        'enabled': False,
        'found': bool(crystal_reference_path),
        'reference_direct': bool(crystal_reference_path),
        'message': (
            'Crystal reference passed directly to image generation; '
            'no separate Gemini crystal-analysis call.'
            if crystal_reference_path
            else 'No crystal reference supplied.'
        ),
        'design': {}
    }

    crystal_lock_instruction = (
        'Use the uploaded crystal reference image directly as the visual authority. '
        'Match only visible crystal/rhinestone design. Do not guess missing details.'
        if crystal_reference_path else ''
    )

    # No second Gemini call for JSON prompt conversion.
    # Build the JSON generation specification locally from the existing
    # Design Lock, preserving the existing JSON-based generation flow.
    generation_result = {
        'generation_prompt': '''
Generate a highly realistic adult Indian female fashion catalogue image.
Use the uploaded saree reference as the highest-priority garment reference.
Use the uploaded blouse reference as the exact blouse reference.
Use the uploaded lace reference as the exact micro-lace reference.
If a crystal reference is supplied, use it as the exact visible crystal-design reference.

Preserve the original saree color, body pattern, motifs, motif arrangement,
border, pallu, embroidery, zari and all visible decorative details.
Preserve the lace as a very small, narrow, fine micro-scale element.
Preserve the blouse design from its uploaded reference.
Only change presentation variables such as model, pose, drape, camera,
composition, lighting and background when requested.

Do not redesign, recolor, replace, simplify, enlarge or invent garment details.
Return a photorealistic premium Indian fashion catalogue image.
'''.strip(),
        'negative_prompt': '''
redesigned saree, changed saree color, changed motifs, invented embroidery,
oversized lace, thick lace, widened lace, enlarged stones, invented crystals,
different blouse, redesigned blouse, generic saree, missing pallu, missing border,
blurred lace, simplified lace, distorted garment, cropped full body
'''.strip(),
        'model_requirements': {
            'adult_indian_female_model': True,
            'full_body': True,
            'photorealistic': True
        },
        'preservation_lock': design_lock.get('preservation_rules', {})
    }

    generation_spec = {
        'available': True,
        'model': 'local_json_spec_no_extra_gemini_call',
        'input_type': 'design_lock_json',
        'result': generation_result
    }

    # Keep strict lock metadata without calling Gemini again.
    generation_spec['micro_lace_lock'] = {
        'locked': bool(strict_lace_lock),
        'scale': 'micro',
        'allow_enlarge': False,
        'allow_thicken': False,
        'allow_widen': False,
        'allow_magnify': False,
        'allow_redesign': False,
        'allow_recolor': False,
        'allow_invent': False,
        'instruction': micro_lace_lock_instruction
    }

    generation_spec['crystal_design_lock'] = {
        'locked': bool(strict_crystal_lock and crystal_reference_path),
        'reference_type': 'swarovski_type_crystal_reference',
        'authenticity_verified': False,
        'visual_reference_only': True,
        'stone_size_locked': True,
        'shape_locked': True,
        'faceting_locked': True,
        'spacing_locked': True,
        'density_locked': True,
        'pattern_locked': True,
        'placement_locked': True,
        'shine_character_locked': True,
        'allow_enlarge': False,
        'allow_invent': False,
        'allow_redesign': False,
        'allow_recolor': False,
        'analysis': {},
        'instruction': crystal_lock_instruction
    }

    generated_prompt = str(
        generation_result.get('generation_prompt', '')
    ).strip()

    if custom_prompt.strip():
        generated_prompt += (
            '\n\nUSER CUSTOM PROMPT:\n' +
            custom_prompt.strip()
        )

    generation_name = f'{request_id}_generation_prompt.json'
    generation_path = os.path.join(ANALYSIS_DIR, generation_name)

    saved_generation = {
        'request_id': request_id,
        'status': 'generation_prompt_created',
        'source': 'single_api_cost_optimized',
        'custom_prompt': custom_prompt,
        'generation_spec': generation_spec
    }

    with open(generation_path, 'w', encoding='utf-8') as f:
        json.dump(saved_generation, f, indent=2, ensure_ascii=False)

    image_size = image_size.upper()
    if image_size not in {'1K', '2K', '4K'}:
        image_size = '2K'
    generated_prompt = generated_prompt + '\n\nSTRICT MICRO-LACE DESIGN LOCK:\n' + micro_lace_lock_instruction
    image_bytes_out, generated_mime = generate_image_with_gemini(saree_path, generated_prompt, image_size=image_size, background_image_path=background_path, lace_image_path=lace_input_path, crystal_image_path=crystal_reference_path, blouse_image_path=blouse_path, crystal_lock_instruction=crystal_lock_instruction)
    output_ext = '.png'
    if generated_mime == 'image/jpeg':
        output_ext = '.jpg'
    elif generated_mime == 'image/webp':
        output_ext = '.webp'
    output_name = f'{request_id}_final{output_ext}'
    output_path = os.path.abspath(os.path.join(GENERATED_DIR, output_name))
    with open(output_path, 'wb') as f:
        f.write(image_bytes_out)
    total_seconds = round(time.perf_counter() - started_at, 3)
    metadata = {'request_id': request_id, 'status': 'complete_success', 'model': GEMINI_IMAGE_MODEL, 'image_size': image_size, 'saree_reference': saree_name, 'lace_reference': lace_reference.get('file_name'), 'blouse_reference': blouse_name, 'background_reference': background_name, 'crystal_reference': crystal_reference_name, 'crystal_design_read': crystal_read, 'micro_lace_locked': True, 'lace_design_lock': {'locked': True, 'scale': 'micro', 'exact_visual_reference': True, 'pattern_locked': True, 'allow_enlarge': False, 'allow_thicken': False, 'allow_redesign': False, 'allow_recolor': False}, 'custom_prompt': custom_prompt, 'pose': pose, 'generation_prompt_file': generation_name, 'local_path': output_path, 'url': f'/generated/{output_name}', 'performance': {'analysis_seconds': analysis_seconds, 'total_seconds': total_seconds}}
    complete_metadata_name = f'{request_id}_complete_generation.json'
    complete_metadata_path = os.path.join(ANALYSIS_DIR, complete_metadata_name)
    with open(complete_metadata_path, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)
    save_request_metadata(request_id, {'request_id': request_id, 'status': 'complete_success', 'saree_reference': saree_name, 'lace_reference': lace_reference.get('file_name'), 'blouse_reference': blouse_name, 'crystal_reference': crystal_reference_name, 'background_reference': background_name, 'design_lock': lock_name, 'generation_prompt': generation_name, 'generated_image': f'/generated/{output_name}'})
    return {'success': True, 'request_id': request_id, 'status': 'complete_success', 'workflow': ['saree_uploaded', 'request_id_created', 'design_analyzed', 'design_lock_created', 'micro_lace_reference_locked', 'crystal_reference_locked', 'generation_prompt_created', 'final_image_generated'], 'references': {'saree': saree_name, 'micro_lace': lace_reference, 'blouse': {'file_name': blouse_name, 'local_path': blouse_path}, 'crystal_reference': {'file_name': crystal_reference_name, 'url': f'/crystal-references/{crystal_reference_name}', 'read': crystal_read} if crystal_reference_name else None, 'background': background_name}, 'lace_design_lock': {'locked': True, 'scale': 'micro', 'exact_visual_reference': True, 'pattern_locked': True, 'motif_sequence_locked': True, 'motif_density_locked': True, 'spacing_locked': True, 'width_locked': True, 'edge_shape_locked': True, 'stone_size_locked': True, 'allow_enlarge': False, 'allow_thicken': False, 'allow_redesign': False, 'allow_recolor': False, 'allow_invent': False}, 'files': {'design_lock': f'/design-maps/{lock_name}', 'generation_prompt': f'/design-maps/{generation_name}', 'final_image': f'/generated/{output_name}'}, 'generated_image': {'file_name': output_name, 'local_path': output_path, 'url': f'/generated/{output_name}', 'mime_type': generated_mime, 'model': GEMINI_IMAGE_MODEL, 'image_size': image_size}, 'custom_prompt': custom_prompt, 'pose': pose, 'available_poses': list(POSE_PRESETS.keys()), 'performance': {'analysis_seconds': analysis_seconds, 'total_seconds': total_seconds}, 'message': 'Cost-optimized flow completed: upload -> local analysis -> Design Lock -> direct reference locks -> local JSON generation spec -> final Gemini image.'}
if __name__ == '__main__':
    import uvicorn
    uvicorn.run('main:app', host='0.0.0.0', port=8000, reload=True)
