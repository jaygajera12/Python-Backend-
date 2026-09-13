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
GEMINI_ANALYSIS_MODEL = os.getenv('GEMINI_ANALYSIS_MODEL', 'gemini-3.7-flash')
GEMINI_IMAGE_MODEL = os.getenv('GEMINI_IMAGE_MODEL', 'gemini-3.1-flash-image')
UPLOAD_DIR = 'uploads'
GENERATED_DIR = 'generated'
ANALYSIS_DIR = 'design_maps'
PREVIEW_DIR = 'design_previews'
BACKGROUND_DIR = 'backgrounds'
CRYSTAL_DIR = 'crystal_references'
for folder in [UPLOAD_DIR, GENERATED_DIR, ANALYSIS_DIR, PREVIEW_DIR, BACKGROUND_DIR, CRYSTAL_DIR]:
    os.makedirs(folder, exist_ok=True)
app = FastAPI(title='SareeViz AI - Micro Lace + Crystal/Swarovski-Type Design Lock', description='Upload a saree image in Swagger. OpenCV detects colors, fine details, motif/texture regions, decorative bands and border candidates. Optional Gemini semantic analysis can identify visible textile details. A Design Lock JSON and visual detection map are returned.', version='7.0.0')
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
SEMANTIC_PROMPT = '\nAnalyze the uploaded saree image as a professional Indian\ntextile/fashion designer.\n\nReturn ONLY valid JSON.\n\nOnly describe details that are visibly supported.\n\nIdentify:\n- saree type, if visible\n- dominant color\n- secondary colors\n- fabric appearance\n- body design\n- visible motif types\n- motif colors\n- motif arrangement\n- border description\n- border width\n- pallu description\n- pallu colors\n- embroidery\n- zari\n- crystal/stone-like decorative details\n- pattern density\n- any clearly visible special details\n\nDo not invent hidden details.\n\nThis is a DESIGN LOCK for a later image generator.\n\nTherefore explicitly preserve:\n- original colors\n- body pattern\n- visible motifs\n- motif arrangement\n- border\n- pallu\n- embroidery\n- zari\n- visible stones/crystals\n- overall visual identity\n\nDo NOT redesign the saree.\n'

def semantic_ai_analysis(image_bytes, mime_type):
    if not gemini_client:
        return {'available': False, 'message': 'Gemini semantic analysis is disabled. OpenCV detection is still active. Set GEMINI_API_KEY to enable semantic analysis.'}
    try:
        response = gemini_client.models.generate_content(model=GEMINI_ANALYSIS_MODEL, contents=[{'inline_data': {'mime_type': mime_type, 'data': base64.b64encode(image_bytes).decode('utf-8')}}, SEMANTIC_PROMPT], config={'response_mime_type': 'application/json'})
        text = response.text or '{}'
        try:
            parsed = json.loads(text)
        except json.JSONDecodeError:
            parsed = {'raw': text, 'parse_error': True}
        return {'available': True, 'model': GEMINI_ANALYSIS_MODEL, 'result': parsed}
    except Exception as e:
        return {'available': False, 'error': str(e)}

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
GENERATION_SPEC_SYSTEM_PROMPT = '\nYou are a professional Indian fashion / textile prompt engineer.\n\nYou will receive a DESIGN LOCK JSON created from OpenCV detection and\nGemini visual analysis of an uploaded saree reference.\n\nYour job is NOT to redesign the saree and NOT to generate an image.\nYour job is to convert the JSON into a precise image-generation\nspecification that a separate virtual try-on / image-generation model\ncan follow.\n\nCRITICAL PRESERVATION RULES:\n1. Preserve the original saree color.\n2. Preserve the original body pattern and motif arrangement.\n3. Preserve the border design, width and placement as supported by the data.\n4. Preserve the pallu design and visible placement.\n5. Preserve embroidery, zari and visible crystal/stone-like details.\n6. Do not invent missing details.\n7. Do not replace the saree with a generic saree.\n8. Do not recolor or redesign the garment.\n9. Only change presentation variables such as model, pose, drape,\n   camera/composition and background when requested.\n10. If a detail is uncertain or not supported by the JSON, say that it\n    must follow the uploaded reference image rather than guessing.\n\nReturn ONLY valid JSON with exactly these top-level keys:\n- generation_prompt\n- negative_prompt\n- model_requirements\n- preservation_lock\n\nThe generation_prompt must be a single detailed prompt for a separate\nimage-generation / virtual-try-on model.\n'

def build_json_wise_gemini_prompt(design_lock: dict, custom_prompt: str='') -> str:
    design_lock_json = json.dumps(design_lock, ensure_ascii=False, indent=2)
    custom_section = ''
    if custom_prompt.strip():
        custom_section = f'\nUSER CUSTOM PROMPT:\n{custom_prompt.strip()}\n\nUse this custom prompt only for non-garment presentation variables such as\npose, camera angle, composition, background, lighting, mood, or styling.\nThe DESIGN LOCK and uploaded reference image always have higher priority.\nNever recolor, redesign, replace, remove, simplify, or invent saree details.\n'
    return f'\nConvert this DESIGN LOCK JSON into a precise image-generation specification.\n\nDESIGN LOCK JSON:\n```json\n{design_lock_json}\n```\n\n{custom_section}\n\nThe generated model must be an adult Indian female fashion model.\nDefault presentation:\n- full body\n- elegant standing pose\n- premium Indian fashion catalogue photography\n- clean pure white background\n- realistic natural fabric drape\n- saree remains the primary visual subject\n- high detail and photorealistic finish\n\nRemember: do not create new saree details. The uploaded reference image and\nDESIGN LOCK have highest priority.\n'

def gemini_json_wise_prompt(design_lock: dict, custom_prompt: str=''):
    """Send the Design Lock JSON to Gemini and return structured JSON."""
    if not gemini_client:
        return {'available': False, 'message': 'Gemini generation-spec conversion is disabled. Set GEMINI_API_KEY to enable it.'}
    try:
        prompt = build_json_wise_gemini_prompt(design_lock, custom_prompt)
        response = gemini_client.models.generate_content(model=GEMINI_ANALYSIS_MODEL, contents=[GENERATION_SPEC_SYSTEM_PROMPT, prompt], config={'response_mime_type': 'application/json'})
        text = response.text or '{}'
        try:
            parsed = json.loads(text)
        except json.JSONDecodeError:
            parsed = {'raw': text, 'parse_error': True}
        return {'available': True, 'model': GEMINI_ANALYSIS_MODEL, 'input_type': 'design_lock_json', 'result': parsed}
    except Exception as e:
        return {'available': False, 'error': str(e)}
MICRO_LACE_DETECTION_PROMPT = '\nYou are an expert textile/fashion visual inspection system.\n\nAnalyze the uploaded saree/lace image at very high visual detail.\n\nTASK:\nLocate the exact visible MICRO LACE-PATTA / narrow decorative lace strip.\n\nImportant:\n- Detect the actual narrow lace/border strip, not the whole saree.\n- The lace may be a very thin vertical or horizontal strip.\n- Look for tiny repeated rhinestone/stone/thread motifs, micro geometry,\n  edging, piping and repeated pattern structure.\n- Do not confuse the large body print, pallu, floral print, or model with lace.\n- Return ONE best bounding box covering the complete useful lace strip.\n- Keep the box tight around the lace while including its full width and\n  enough surrounding fabric to understand its edge relationship.\n- Use normalized coordinates from 0 to 1.\n- x/y are the top-left corner.\n- width/height are normalized.\n- Also return orientation: "vertical", "horizontal", or "unknown".\n- Return JSON only.\n\nJSON schema:\n{\n  "found": true,\n  "confidence": 0.0,\n  "orientation": "vertical",\n  "bbox": {\n    "x": 0.0,\n    "y": 0.0,\n    "width": 0.0,\n    "height": 0.0\n  },\n  "description": "brief description of the micro lace"\n}\n\nIf no lace is visible:\n{\n  "found": false,\n  "confidence": 0.0,\n  "orientation": "unknown",\n  "bbox": {\n    "x": 0.0,\n    "y": 0.0,\n    "width": 0.0,\n    "height": 0.0\n  },\n  "description": ""\n}\n'

def gemini_micro_lace_bbox(image_bytes: bytes, mime_type: str) -> dict:
    """
    Use Gemini 3.7 Flash to locate the exact micro-lace region.
    Gemini returns normalized bbox coordinates; the original image is then
    cropped locally without resizing.
    """
    if not gemini_client:
        return {'available': False, 'found': False, 'message': 'Gemini client is not configured.'}
    try:
        response = gemini_client.models.generate_content(model=GEMINI_ANALYSIS_MODEL, contents=[{'inline_data': {'mime_type': mime_type, 'data': base64.b64encode(image_bytes).decode('utf-8')}}, MICRO_LACE_DETECTION_PROMPT], config={'response_mime_type': 'application/json'})
        raw_text = (response.text or '{}').strip()
        try:
            parsed = json.loads(raw_text)
        except json.JSONDecodeError:
            start = raw_text.find('{')
            end = raw_text.rfind('}')
            if start >= 0 and end > start:
                parsed = json.loads(raw_text[start:end + 1])
            else:
                raise RuntimeError('Gemini returned invalid JSON for micro-lace detection.')
        return {'available': True, **parsed}
    except Exception as e:
        return {'available': False, 'found': False, 'error': str(e)}


CRYSTAL_DETECTION_PROMPT = """
You are an expert textile embellishment and luxury-fashion visual inspection system.

Analyze the uploaded reference image specifically for visible crystal/rhinestone/stone embellishment.
This may be a Swarovski-type crystal reference, but you MUST NOT claim authenticity or brand origin from the image.
Classify it only as a visual crystal/stone design reference.

TASK:
1. Determine whether a visible crystal/rhinestone/stone design is present.
2. Describe the visual crystal design precisely enough to preserve it in an image-generation workflow.
3. Identify visible cut/shape, approximate micro-size, spacing, density, repeat pattern, alignment, placement style,
   reflective/shine behavior, surrounding thread/mesh/piping, and relationship to the lace.
4. Look for tiny white/clear stones, faceted points, round stones, teardrops, hearts, flowers, geometric clusters,
   borders, rows, repeated motifs, or other visible crystal geometry.
5. Do NOT invent crystals that are not visible.
6. Do NOT infer authenticity, manufacturer, or genuine Swarovski status.
7. Return JSON only.

JSON schema:
{
  "found": true,
  "confidence": 0.0,
  "reference_type": "swarovski_type_crystal_reference",
  "design": {
    "stone_shape": "",
    "stone_scale": "micro",
    "stone_size_description": "",
    "cut_or_faceting": "",
    "spacing": "",
    "density": "",
    "repeat_pattern": "",
    "alignment": "",
    "placement": "",
    "shine_character": "",
    "surrounding_structure": "",
    "color_appearance": "",
    "relationship_to_lace": ""
  },
  "description": "brief visual description"
}

If no visible crystal/stone design is present, return:
{
  "found": false,
  "confidence": 0.0,
  "reference_type": "swarovski_type_crystal_reference",
  "design": {},
  "description": ""
}
"""

def gemini_crystal_design_read(image_bytes: bytes, mime_type: str) -> dict:
    """Read visible crystal/rhinestone design from a Swarovski-type reference image."""
    if not gemini_client:
        return {'available': False, 'found': False, 'message': 'Gemini client is not configured.'}
    try:
        response = gemini_client.models.generate_content(
            model=GEMINI_ANALYSIS_MODEL,
            contents=[
                {'inline_data': {'mime_type': mime_type, 'data': base64.b64encode(image_bytes).decode('utf-8')}},
                CRYSTAL_DETECTION_PROMPT,
            ],
            config={'response_mime_type': 'application/json'}
        )
        raw_text = (response.text or '{}').strip()
        try:
            parsed = json.loads(raw_text)
        except json.JSONDecodeError:
            start = raw_text.find('{')
            end = raw_text.rfind('}')
            if start >= 0 and end > start:
                parsed = json.loads(raw_text[start:end + 1])
            else:
                raise RuntimeError('Gemini returned invalid JSON for crystal/Swarovski-type design reading.')
        return {'available': True, 'model': GEMINI_ANALYSIS_MODEL, **parsed}
    except Exception as e:
        return {'available': False, 'found': False, 'error': str(e)}

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
    if crystal_image_path and os.path.exists(crystal_image_path):
        crystal_instruction = '\nSWAROVSKI-TYPE / CRYSTAL DESIGN LOCK — VISUAL REFERENCE ONLY:\n- Use the uploaded crystal reference as the exact visual reference for the visible crystal/rhinestone design.\n- Preserve crystal micro-size, cut/shape, faceting, spacing, density, repeat pattern, alignment, placement and reflective character.\n- Preserve visible white/clear stone appearance and its relationship to the lace/garment.\n- DO NOT enlarge, exaggerate, simplify, redesign, recolor or invent the crystal arrangement.\n- DO NOT claim or imply authenticity; match only the visible design shown in the reference.\n- Higher output resolution MUST NOT increase the physical size of individual stones.\n'
        if crystal_lock_instruction:
            crystal_instruction += '\nCRYSTAL ANALYSIS LOCK:\n' + crystal_lock_instruction + '\n'
    background_instruction = ''
    if background_image_path and os.path.exists(background_image_path):
        background_instruction = '\nBACKGROUND REFERENCE:\n- A second uploaded image is provided as the background reference.\n- Use that image as the background/scene reference.\n- Preserve the background scene, environment, perspective, lighting direction,\n  overall composition and visual character as closely as possible.\n- Do not use a white background when a background reference is supplied.\n- Do not replace the background with an unrelated scene.\n'
    final_prompt = f"\nUse the uploaded saree reference image as the highest-priority garment reference.\n\n{generation_prompt}\n\n{background_instruction}\n\n{lace_instruction}\n\n{micro_lace_scale_instruction}\n\nIMPORTANT:\n- The uploaded saree is the exact garment reference.\n- Preserve garment proportions from the reference; do not scale small lace\n  details into large decorative graphics.\n- Preserve the saree's original color, motifs, motif arrangement, border,\n  pallu, embroidery, zari, visible crystal/stone-like details, AND ALL LACE DETAILS.\n- The lace is a critical MICRO garment feature and must remain micro-scale.\n- Match the lace visual size relative to the saree border and saree body in the source; it must not become a large applique, panel, or oversized border.\n- If a lace reference image is supplied, use it as an exact pattern-and-scale\n  reference, not merely a style suggestion.\n- Preserve the exact lace width, physical scale, motif density, motif spacing,\n  edge shape, thread thickness, mesh, dots/beads, holes, trims and decorative\n  edging visible in the reference.\n- Do NOT enlarge, thicken, widen, magnify, stylize, simplify, merge, replace,\n  or redesign the lace.\n- The repeated lace motifs must remain small and dense exactly as in the\n  reference, with no oversized symbols or oversized gaps.\n- Keep the lace sharp without making its elements larger than the source.\n- Preserve the lace along every visible saree edge where it exists.\n- Do not redesign, recolor, replace, simplify, or invent the saree.\n- Keep the saree visually recognizable as the uploaded reference.\n- Generate a highly realistic adult Indian female fashion model.\n- Full-body standing composition.\n- Premium Indian fashion catalogue photography.\n- Natural realistic fabric drape.\n- The lace must remain physically and proportionally small, fine and narrow in the final garment exactly as shown by the reference.\n- Do not make the lace larger merely because output resolution is higher.\n- Return the generated image as the image output.\n"
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

@app.post('/api/saree/generate-all', summary='Single API for complete SareeViz generation', description='ONE API ONLY. Upload saree, exact micro-lace, optional Swarovski-type/crystal reference, and optional background. Gemini 3.7 Flash reads the micro-lace and crystal design, applies strict locks, creates the JSON-wise generation prompt, and Gemini 3.1 Flash Image generates the final image. Swarovski authenticity is NOT verified; only the visible crystal design is matched.')
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
    semantic_analysis = {'available': False, 'enabled': False, 'message': 'Gemini semantic analysis skipped for speed.'}
    if gemini_analysis:
        semantic_analysis = semantic_ai_analysis(saree_bytes, {'.jpg': 'image/jpeg', '.jpeg': 'image/jpeg', '.png': 'image/png', '.webp': 'image/webp'}.get(saree_ext, 'image/jpeg'))
        semantic_analysis['enabled'] = True
    design_lock = build_design_lock(opencv_analysis, semantic_analysis)
    lock_name = f'{request_id}_design_lock.json'
    lock_path = os.path.join(ANALYSIS_DIR, lock_name)
    with open(lock_path, 'w', encoding='utf-8') as f:
        json.dump(design_lock, f, indent=2, ensure_ascii=False)
    lace_detection = gemini_micro_lace_bbox(lace_bytes, {'.jpg': 'image/jpeg', '.jpeg': 'image/jpeg', '.png': 'image/png', '.webp': 'image/webp'}.get(lace_ext, 'image/jpeg'))
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
                lace_reference = {'file_name': lace_name, 'local_path': lace_path, 'url': f'/design-maps/{lace_name}', 'width': x2 - x1, 'height': y2 - y1, 'micro_lace_locked': True, 'gemini_model': GEMINI_ANALYSIS_MODEL, 'confidence': lace_detection.get('confidence'), 'orientation': lace_detection.get('orientation'), 'bbox_original_pixels': {'x': x1, 'y': y1, 'width': x2 - x1, 'height': y2 - y1}}
    if not lace_reference:
        try:
            os.remove(lace_input_path)
        except OSError:
            pass
        return {'success': False, 'request_id': request_id, 'status': 'micro_lace_not_detected', 'message': 'Gemini 3.7 Flash could not detect the micro-lace reference. Final image generation was stopped to avoid changing the lace.'}
    try:
        os.remove(lace_input_path)
    except OSError:
        pass
    crystal_read = {'available': False, 'enabled': False, 'found': False, 'message': 'No Swarovski-type / crystal reference supplied.'}
    if crystal_reference_path and crystal_reference_bytes:
        crystal_read = gemini_crystal_design_read(crystal_reference_bytes, {'.jpg': 'image/jpeg', '.jpeg': 'image/jpeg', '.png': 'image/png', '.webp': 'image/webp'}.get(crystal_ext, 'image/jpeg'))
        crystal_read['enabled'] = True
    crystal_lock_instruction = ''
    if crystal_read.get('found'):
        crystal_lock_instruction = json.dumps(crystal_read.get('design', {}), ensure_ascii=False)
    elif crystal_reference_path:
        crystal_lock_instruction = 'Use the uploaded crystal reference image directly as the visual authority. Do not guess missing crystal details.'
    generation_spec = gemini_json_wise_prompt(design_lock)
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
    generated_prompt = generated_prompt + '\n\nSTRICT MICRO-LACE DESIGN LOCK:\n' + micro_lace_lock_instruction
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
    total_seconds = round(time.perf_counter() - started_at, 3)
    metadata = {'request_id': request_id, 'status': 'complete_success', 'model': GEMINI_IMAGE_MODEL, 'analysis_model': GEMINI_ANALYSIS_MODEL, 'image_size': image_size, 'saree_reference': saree_name, 'lace_reference': lace_reference.get('file_name'), 'background_reference': background_name, 'crystal_reference': crystal_reference_name, 'crystal_design_read': crystal_read, 'micro_lace_locked': True, 'lace_design_lock': {'locked': True, 'scale': 'micro', 'exact_visual_reference': True, 'pattern_locked': True, 'allow_enlarge': False, 'allow_thicken': False, 'allow_redesign': False, 'allow_recolor': False}, 'custom_prompt': custom_prompt, 'generation_prompt_file': generation_name, 'local_path': output_path, 'url': f'/generated/{output_name}', 'performance': {'analysis_seconds': analysis_seconds, 'total_seconds': total_seconds}}
    complete_metadata_name = f'{request_id}_complete_generation.json'
    complete_metadata_path = os.path.join(ANALYSIS_DIR, complete_metadata_name)
    with open(complete_metadata_path, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)
    save_request_metadata(request_id, {'request_id': request_id, 'status': 'complete_success', 'saree_reference': saree_name, 'lace_reference': lace_reference.get('file_name'), 'crystal_reference': crystal_reference_name, 'background_reference': background_name, 'design_lock': lock_name, 'generation_prompt': generation_name, 'generated_image': f'/generated/{output_name}'})
    return {'success': True, 'request_id': request_id, 'status': 'complete_success', 'workflow': ['saree_uploaded', 'request_id_created', 'design_analyzed', 'design_lock_created', 'micro_lace_detected', 'swarovski_type_crystal_read', 'crystal_design_locked', 'generation_prompt_created', 'final_image_generated'], 'references': {'saree': saree_name, 'micro_lace': lace_reference, 'crystal_reference': {'file_name': crystal_reference_name, 'url': f'/crystal-references/{crystal_reference_name}', 'read': crystal_read} if crystal_reference_name else None, 'background': background_name}, 'lace_design_lock': {'locked': True, 'scale': 'micro', 'exact_visual_reference': True, 'pattern_locked': True, 'motif_sequence_locked': True, 'motif_density_locked': True, 'spacing_locked': True, 'width_locked': True, 'edge_shape_locked': True, 'stone_size_locked': True, 'allow_enlarge': False, 'allow_thicken': False, 'allow_redesign': False, 'allow_recolor': False, 'allow_invent': False}, 'files': {'design_lock': f'/design-maps/{lock_name}', 'generation_prompt': f'/design-maps/{generation_name}', 'final_image': f'/generated/{output_name}'}, 'generated_image': {'file_name': output_name, 'local_path': output_path, 'url': f'/generated/{output_name}', 'mime_type': generated_mime, 'model': GEMINI_IMAGE_MODEL, 'image_size': image_size}, 'custom_prompt': custom_prompt, 'performance': {'analysis_seconds': analysis_seconds, 'total_seconds': total_seconds}, 'message': 'All steps completed in one API call: upload -> request ID -> analysis -> Design Lock -> micro-lace -> generation prompt -> final Gemini image.'}
if __name__ == '__main__':
    import uvicorn
    uvicorn.run('main:app', host='0.0.0.0', port=8000, reload=True)
