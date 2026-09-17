import os
import json
import uuid
from pathlib import Path
from datetime import datetime

import cv2
import numpy as np
from dotenv import load_dotenv

from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse

from google import genai
from google.genai import types


# ============================================================
# CONFIG
# ============================================================

load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

if not GEMINI_API_KEY:
    raise RuntimeError("GEMINI_API_KEY is missing in .env")

MODEL = "gemini-3.1-flash-image"

BASE_DIR = Path(__file__).resolve().parent

OUTPUT_DIR = BASE_DIR / "outputs"
DEBUG_DIR = BASE_DIR / "debug"
REFERENCE_DIR = DEBUG_DIR / "references"
MASK_DIR = DEBUG_DIR / "masks"
VERIFY_DIR = DEBUG_DIR / "verification"

for directory in [
    OUTPUT_DIR,
    DEBUG_DIR,
    REFERENCE_DIR,
    MASK_DIR,
    VERIFY_DIR,
]:
    directory.mkdir(parents=True, exist_ok=True)


client = genai.Client(api_key=GEMINI_API_KEY)


# ============================================================
# FASTAPI
# ============================================================

app = FastAPI(
    title="SareeViz - AI Saree Design Lock",
    version="2.0.0",
)


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================
# BASIC HELPERS
# ============================================================

def decode_image(data: bytes):
    array = np.frombuffer(data, dtype=np.uint8)

    image = cv2.imdecode(array, cv2.IMREAD_COLOR)

    if image is None:
        raise ValueError("Invalid image")

    return image


def encode_png(image):
    success, encoded = cv2.imencode(
        ".png",
        image,
        [cv2.IMWRITE_PNG_COMPRESSION, 3],
    )

    if not success:
        raise ValueError("Could not encode image")

    return encoded.tobytes()


def clamp(value, minimum, maximum):
    return max(minimum, min(value, maximum))


def resize_for_analysis(image, max_size=1400):

    h, w = image.shape[:2]

    scale = min(1.0, max_size / max(h, w))

    if scale == 1.0:
        return image.copy(), 1.0

    resized = cv2.resize(
        image,
        (
            int(w * scale),
            int(h * scale),
        ),
        interpolation=cv2.INTER_AREA,
    )

    return resized, scale


def crop_region(image, x, y, w, h):

    height, width = image.shape[:2]

    x = clamp(int(x), 0, width - 1)
    y = clamp(int(y), 0, height - 1)

    w = clamp(int(w), 1, width - x)
    h = clamp(int(h), 1, height - y)

    return image[y:y + h, x:x + w]


# ============================================================
# REFLECTOR DETECTION
# ============================================================

def detect_reflector(image):

    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    lower = np.array([0, 0, 185])
    upper = np.array([180, 75, 255])

    mask = cv2.inRange(
        hsv,
        lower,
        upper,
    )

    kernel = np.ones((5, 5), np.uint8)

    mask = cv2.morphologyEx(
        mask,
        cv2.MORPH_OPEN,
        kernel,
    )

    mask = cv2.morphologyEx(
        mask,
        cv2.MORPH_CLOSE,
        kernel,
    )

    total_pixels = mask.shape[0] * mask.shape[1]

    bright_ratio = float(
        cv2.countNonZero(mask) / max(total_pixels, 1)
    )

    return {
        "detected": bright_ratio > 0.01,
        "bright_ratio": round(bright_ratio, 5),
        "note": (
            "Bright low-saturation regions detected. "
            "Treat these as reflector/lighting artifacts, "
            "not as saree design."
        ),
    }


# ============================================================
# LACE DETECTION
# ============================================================

def detect_lace(image):

    h, w = image.shape[:2]

    gray = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2GRAY,
    )

    edges = cv2.Canny(
        gray,
        50,
        150,
    )

    vertical_kernel = cv2.getStructuringElement(
        cv2.MORPH_RECT,
        (3, 15),
    )

    vertical_edges = cv2.morphologyEx(
        edges,
        cv2.MORPH_CLOSE,
        vertical_kernel,
    )

    left_width = max(1, int(w * 0.15))
    right_start = int(w * 0.85)

    left = vertical_edges[:, :left_width]
    right = vertical_edges[:, right_start:]

    left_score = float(
        np.count_nonzero(left) /
        max(left.size, 1)
    )

    right_score = float(
        np.count_nonzero(right) /
        max(right.size, 1)
    )

    detected = (
        left_score > 0.025 or
        right_score > 0.025
    )

    return {
        "detected": detected,

        "left_score": round(
            left_score,
            5,
        ),

        "right_score": round(
            right_score,
            5,
        ),

        "estimated_left_width_ratio": 0.15,

        "estimated_right_width_ratio": 0.15,

        "preserve": True,

        "instruction": (
            "Preserve lace exactly. "
            "Do not increase lace height, width, "
            "thickness, spacing or motif size."
        ),
    }


# ============================================================
# MICRO DESIGN DETECTION
# ============================================================

def detect_micro_design(image):

    gray = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2GRAY,
    )

    blurred = cv2.GaussianBlur(
        gray,
        (5, 5),
        0,
    )

    difference = cv2.absdiff(
        gray,
        blurred,
    )

    threshold = cv2.adaptiveThreshold(
        difference,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        31,
        5,
    )

    kernel = np.ones(
        (2, 2),
        np.uint8,
    )

    threshold = cv2.morphologyEx(
        threshold,
        cv2.MORPH_OPEN,
        kernel,
    )

    contours, _ = cv2.findContours(
        threshold,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE,
    )

    h, w = image.shape[:2]

    regions = []

    for contour in contours:

        x, y, cw, ch = cv2.boundingRect(
            contour
        )

        area = cw * ch

        if area < 8:
            continue

        if area > (w * h * 0.03):
            continue

        regions.append(
            {
                "x": int(x),
                "y": int(y),
                "width": int(cw),
                "height": int(ch),
                "area": int(area),
            }
        )

    regions = sorted(
        regions,
        key=lambda item: item["area"],
        reverse=True,
    )

    regions = regions[:40]

    return {
        "detected": len(regions) > 0,
        "count": len(regions),
        "regions": regions,
        "instruction": (
            "Preserve micro-design, crystals, "
            "small motifs, embroidery details, "
            "texture and motif spacing."
        ),
    }


# ============================================================
# BORDER DETECTION
# ============================================================

def detect_border(image):

    h, w = image.shape[:2]

    gray = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2GRAY,
    )

    edges = cv2.Canny(
        gray,
        50,
        150,
    )

    top = edges[
        :max(1, int(h * 0.20)),
        :
    ]

    bottom = edges[
        int(h * 0.80):,
        :
    ]

    top_density = float(
        np.count_nonzero(top) /
        max(top.size, 1)
    )

    bottom_density = float(
        np.count_nonzero(bottom) /
        max(bottom.size, 1)
    )

    return {
        "top_edge_density": round(
            top_density,
            5,
        ),

        "bottom_edge_density": round(
            bottom_density,
            5,
        ),

        "instruction": (
            "Preserve border exactly. "
            "Do not redesign or resize the border."
        ),
    }


# ============================================================
# PALLU CANDIDATE DETECTION
# ============================================================

def detect_pallu(image):

    h, w = image.shape[:2]

    left_width = int(w * 0.35)

    right_start = int(w * 0.65)

    return {
        "semantic_detection": False,

        "candidates": [
            {
                "name": "left",
                "x": 0,
                "y": 0,
                "width": left_width,
                "height": h,
            },

            {
                "name": "right",
                "x": right_start,
                "y": 0,
                "width": w - right_start,
                "height": h,
            },
        ],

        "instruction": (
            "Pallu candidate areas are protected "
            "from redesign. Preserve original pallu "
            "pattern, embroidery and border."
        ),
    }


# ============================================================
# ANALYZE SAREE
# ============================================================

def analyze_saree(image):

    original_h, original_w = image.shape[:2]

    analysis_image, scale = resize_for_analysis(
        image
    )

    lace = detect_lace(
        analysis_image
    )

    micro = detect_micro_design(
        analysis_image
    )

    border = detect_border(
        analysis_image
    )

    pallu = detect_pallu(
        analysis_image
    )

    reflector = detect_reflector(
        analysis_image
    )

    if scale != 1.0:

        for region in micro["regions"]:

            region["x"] = int(
                region["x"] / scale
            )

            region["y"] = int(
                region["y"] / scale
            )

            region["width"] = int(
                region["width"] / scale
            )

            region["height"] = int(
                region["height"] / scale
            )

        for candidate in pallu["candidates"]:

            candidate["x"] = int(
                candidate["x"] / scale
            )

            candidate["y"] = int(
                candidate["y"] / scale
            )

            candidate["width"] = int(
                candidate["width"] / scale
            )

            candidate["height"] = int(
                candidate["height"] / scale
            )

    return {
        "image": {
            "width": original_w,
            "height": original_h,
        },

        "opencv": {
            "lace": lace,
            "micro_design": micro,
            "border": border,
            "pallu": pallu,
            "reflector": reflector,
        },

        "semantic_analysis": False,

        "design_lock": {
            "enabled": True,

            "preserve": [
                "exact saree colors",
                "exact print",
                "exact embroidery",
                "exact micro-design",
                "exact crystals",
                "exact lace",
                "exact border",
                "exact pallu",
                "exact texture",
                "exact motif spacing",
                "exact motif scale",
                "exact motif orientation",
            ],

            "allowed_changes": [
                "model",
                "pose",
                "draping",
                "camera framing",
                "background",
                "lighting",
            ],
        },

        "post_generation_verification": True,
    }


# ============================================================
# PROTECTED MASK
# ============================================================

def create_protected_mask(
    image,
    analysis,
):

    h, w = image.shape[:2]

    mask = np.zeros(
        (h, w),
        dtype=np.uint8,
    )

    # ----------------------------
    # Micro design
    # ----------------------------

    for region in analysis[
        "opencv"
    ]["micro_design"]["regions"]:

        x = region["x"]
        y = region["y"]
        rw = region["width"]
        rh = region["height"]

        x1 = clamp(
            x - 8,
            0,
            w - 1,
        )

        y1 = clamp(
            y - 8,
            0,
            h - 1,
        )

        x2 = clamp(
            x + rw + 8,
            0,
            w,
        )

        y2 = clamp(
            y + rh + 8,
            0,
            h,
        )

        mask[y1:y2, x1:x2] = 255

    # ----------------------------
    # Left/right lace areas
    # ----------------------------

    lace_width = int(w * 0.15)

    mask[
        :,
        :lace_width
    ] = 255

    mask[
        :,
        w - lace_width:
    ] = 255

    # ----------------------------
    # Border
    # ----------------------------

    border_height = int(h * 0.20)

    mask[
        :border_height,
        :
    ] = 255

    mask[
        h - border_height:,
        :
    ] = 255

    # ----------------------------
    # Pallu candidates
    # ----------------------------

    for candidate in analysis[
        "opencv"
    ]["pallu"]["candidates"]:

        x = candidate["x"]
        y = candidate["y"]

        cw = candidate["width"]
        ch = candidate["height"]

        x1 = clamp(
            x,
            0,
            w - 1,
        )

        y1 = clamp(
            y,
            0,
            h - 1,
        )

        x2 = clamp(
            x + cw,
            0,
            w,
        )

        y2 = clamp(
            y + ch,
            0,
            h,
        )

        mask[y1:y2, x1:x2] = 255

    # ----------------------------
    # Dilate protection
    # ----------------------------

    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (9, 9),
    )

    mask = cv2.dilate(
        mask,
        kernel,
        iterations=1,
    )

    return mask


# ============================================================
# REFERENCE CROPS
# ============================================================

def create_reference_crops(
    image,
    analysis,
    job_id,
):

    job_dir = REFERENCE_DIR / job_id

    job_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    references = []

    # ----------------------------
    # Lace crops
    # ----------------------------

    h, w = image.shape[:2]

    lace_width = int(w * 0.15)

    left_lace = crop_region(
        image,
        0,
        0,
        lace_width,
        h,
    )

    right_lace = crop_region(
        image,
        w - lace_width,
        0,
        lace_width,
        h,
    )

    left_path = job_dir / "lace_left.png"
    right_path = job_dir / "lace_right.png"

    cv2.imwrite(
        str(left_path),
        left_lace,
    )

    cv2.imwrite(
        str(right_path),
        right_lace,
    )

    references.append(
        str(left_path)
    )

    references.append(
        str(right_path)
    )

    # ----------------------------
    # Micro design crops
    # ----------------------------

    micro_regions = analysis[
        "opencv"
    ]["micro_design"]["regions"]

    for index, region in enumerate(
        micro_regions[:8]
    ):

        crop = crop_region(
            image,
            region["x"],
            region["y"],
            region["width"],
            region["height"],
        )

        path = (
            job_dir /
            f"micro_{index + 1}.png"
        )

        cv2.imwrite(
            str(path),
            crop,
        )

        references.append(
            str(path)
        )

    return references


# ============================================================
# DEBUG IMAGE
# ============================================================

def create_detection_debug(
    image,
    analysis,
):

    debug = image.copy()

    h, w = image.shape[:2]

    # Micro
    for region in analysis[
        "opencv"
    ]["micro_design"]["regions"]:

        x = region["x"]
        y = region["y"]

        rw = region["width"]
        rh = region["height"]

        cv2.rectangle(
            debug,
            (x, y),
            (x + rw, y + rh),
            (0, 255, 0),
            2,
        )

    # Lace
    lace_width = int(w * 0.15)

    cv2.rectangle(
        debug,
        (0, 0),
        (lace_width, h),
        (255, 0, 0),
        3,
    )

    cv2.rectangle(
        debug,
        (w - lace_width, 0),
        (w, h),
        (255, 0, 0),
        3,
    )

    # Border
    border_h = int(h * 0.20)

    cv2.rectangle(
        debug,
        (0, 0),
        (w, border_h),
        (0, 255, 255),
        3,
    )

    cv2.rectangle(
        debug,
        (0, h - border_h),
        (w, h),
        (0, 255, 255),
        3,
    )

    return debug


# ============================================================
# DYNAMIC PROMPT
# ============================================================

def build_dynamic_prompt(
    analysis,
    pose,
    background,
    has_lace_reference,
):

    lace = analysis[
        "opencv"
    ]["lace"]

    micro = analysis[
        "opencv"
    ]["micro_design"]

    border = analysis[
        "opencv"
    ]["border"]

    pallu = analysis[
        "opencv"
    ]["pallu"]

    reflector = analysis[
        "opencv"
    ]["reflector"]

    prompt = f"""
You are generating a premium Indian fashion catalogue photograph.

THIS IS A STRICT SAREE DESIGN PRESERVATION TASK.

The uploaded saree image is the MASTER DESIGN REFERENCE.

Do NOT redesign the saree.

Do NOT reinterpret the saree.

Do NOT invent a new saree.

Do NOT replace the saree pattern.

Do NOT simplify the saree.

Do NOT recolor the saree.

Do NOT change the original fabric texture.

DO NOT change:
- saree color
- print
- embroidery
- motifs
- micro motifs
- crystals
- Swarovski-like details
- border
- lace
- pallu
- motif spacing
- motif density
- motif scale
- motif orientation
- fabric texture
- embroidery thickness
- lace thickness
- lace width
- lace height

OpenCV detected the following:

LACE:
Detected = {lace["detected"]}
Left score = {lace["left_score"]}
Right score = {lace["right_score"]}

MICRO DESIGN:
Detected = {micro["detected"]}
Detected regions = {micro["count"]}

BORDER:
Top density = {border["top_edge_density"]}
Bottom density = {border["bottom_edge_density"]}

PALLU:
Candidate protection enabled = True
Semantic pallu detection = {pallu["semantic_detection"]}

REFLECTOR:
Bright artifact detected = {reflector["detected"]}

IMPORTANT:

Bright reflector regions are lighting artifacts.

Do NOT interpret reflector glare as a new saree design.

Preserve the original design underneath the lighting artifact.

LACE RULE:

{"A separate lace reference image is provided. Use it as an additional exact visual reference for the lace." if has_lace_reference else "Use the lace visible in the main saree reference."}

The lace must remain visually identical to the reference.

Do not make lace elements larger.

Do not make lace thicker.

Do not change lace height.

Do not change lace width.

Do not change lace motif spacing.

Do not replace lace with another lace.

MICRO-DESIGN RULE:

Preserve every visible small motif and micro-detail.

Small crystals and decorative details must remain small.

Do not merge small motifs into large motifs.

Do not create new motifs.

Do not remove motifs.

BORDER RULE:

The border must retain its original width, design, embroidery and proportions.

PALLU RULE:

Preserve the pallu's original pattern, border and decorative details.

DESIGN LOCK:

The saree itself must remain the same physical product.

Only these elements may change:

1. Human model
2. Pose
3. Saree draping arrangement required by the pose
4. Camera framing
5. Background
6. Natural studio lighting

Requested pose:
{pose}

Requested background:
{background}

The final result must look like a realistic professional Indian fashion catalogue photograph.

The model should look natural.

The saree should be realistically worn.

The original saree should remain clearly recognizable.

HIGH RESOLUTION.

Photorealistic.

Sharp fabric details.

Natural skin.

Professional catalogue photography.

NO TEXT.

NO LOGO.

NO WATERMARK.

FINAL PRIORITY:

SAREE DESIGN PRESERVATION HAS HIGHER PRIORITY THAN
CREATIVE STYLIZATION.

If there is any conflict between pose/background and saree design,
preserve the saree design.
"""

    return prompt.strip()


# ============================================================
# IMAGE GENERATION
# ============================================================

def generate_image(
    saree_bytes,
    lace_bytes,
    model_bytes,
    background_bytes,
    reference_paths,
    prompt_text,
    image_size,
):

    contents = []

    # Prompt
    contents.append(
        prompt_text
    )

    # Main saree
    contents.append(
        types.Part.from_bytes(
            data=saree_bytes,
            mime_type="image/png",
        )
    )

    # Separate lace
    if lace_bytes:

        contents.append(
            "The next image is the separate lace reference. "
            "Preserve this lace exactly."
        )

        contents.append(
            types.Part.from_bytes(
                data=lace_bytes,
                mime_type="image/png",
            )
        )

    # OpenCV reference crops
    for path in reference_paths:

        try:

            with open(
                path,
                "rb",
            ) as file:

                data = file.read()

            contents.append(
                types.Part.from_bytes(
                    data=data,
                    mime_type="image/png",
                )
            )

        except Exception:
            pass

    # Model reference
    if model_bytes:

        contents.append(
            "The next image is the optional model reference. "
            "Use the person as a visual reference only. "
            "Do not copy clothing from this image."
        )

        contents.append(
            types.Part.from_bytes(
                data=model_bytes,
                mime_type="image/png",
            )
        )

    # Background reference
    if background_bytes:

        contents.append(
            "The next image is the optional background reference. "
            "Use it only for the environment/background."
        )

        contents.append(
            types.Part.from_bytes(
                data=background_bytes,
                mime_type="image/png",
            )
        )

    # ========================================================
    # EXACTLY ONE GEMINI GENERATION CALL
    # ========================================================

    response = client.models.generate_content(
        model=MODEL,
        contents=contents,
        config=types.GenerateContentConfig(
            response_modalities=[
                "IMAGE"
            ],

            image_config=types.ImageConfig(
                image_size=image_size
            ),

            automatic_function_calling=types.AutomaticFunctionCallingConfig(
                disable=True
            ),
        ),
    )

    # Extract image
    if not response.candidates:

        raise RuntimeError(
            "Gemini returned no candidates"
        )

    for candidate in response.candidates:

        if not candidate.content:
            continue

        for part in candidate.content.parts:

            if getattr(
                part,
                "inline_data",
                None,
            ):

                image_data = (
                    part.inline_data.data
                )

                if image_data:
                    return image_data

    raise RuntimeError(
        "Gemini did not return an image"
    )


# ============================================================
# DESIGN VERIFICATION
# ============================================================

def hsv_histogram(image):

    hsv = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2HSV,
    )

    hist = cv2.calcHist(
        [hsv],
        [0, 1],
        None,
        [32, 32],
        [0, 180, 0, 256],
    )

    cv2.normalize(
        hist,
        hist,
        0,
        1,
        cv2.NORM_MINMAX,
    )

    return hist


def texture_descriptor(image):

    gray = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2GRAY,
    )

    gx = cv2.Sobel(
        gray,
        cv2.CV_32F,
        1,
        0,
        ksize=3,
    )

    gy = cv2.Sobel(
        gray,
        cv2.CV_32F,
        0,
        1,
        ksize=3,
    )

    magnitude = cv2.magnitude(
        gx,
        gy,
    )

    magnitude = cv2.resize(
        magnitude,
        (256, 256),
    )

    magnitude = cv2.normalize(
        magnitude,
        None,
        0,
        1,
        cv2.NORM_MINMAX,
    )

    return magnitude


def compare_texture(
    original,
    generated,
):

    original_texture = (
        texture_descriptor(
            original
        )
    )

    generated_texture = (
        texture_descriptor(
            generated
        )
    )

    result = cv2.matchTemplate(
        generated_texture,
        original_texture,
        cv2.TM_CCOEFF_NORMED,
    )

    score = float(
        result.max()
    )

    return max(
        0.0,
        min(1.0, score)
    )


def compare_edges(
    original,
    generated,
):

    original_gray = cv2.cvtColor(
        original,
        cv2.COLOR_BGR2GRAY,
    )

    generated_gray = cv2.cvtColor(
        generated,
        cv2.COLOR_BGR2GRAY,
    )

    original_edges = cv2.Canny(
        original_gray,
        50,
        150,
    )

    generated_edges = cv2.Canny(
        generated_gray,
        50,
        150,
    )

    original_edges = cv2.resize(
        original_edges,
        (256, 256),
    )

    generated_edges = cv2.resize(
        generated_edges,
        (256, 256),
    )

    result = cv2.matchTemplate(
        generated_edges,
        original_edges,
        cv2.TM_CCOEFF_NORMED,
    )

    score = float(
        result.max()
    )

    return max(
        0.0,
        min(1.0, score)
    )


def verify_design(
    original,
    generated,
):

    original_resized = cv2.resize(
        original,
        (512, 512),
    )

    generated_resized = cv2.resize(
        generated,
        (512, 512),
    )

    # Color
    original_hist = hsv_histogram(
        original_resized
    )

    generated_hist = hsv_histogram(
        generated_resized
    )

    color_score = cv2.compareHist(
        original_hist,
        generated_hist,
        cv2.HISTCMP_CORREL,
    )

    color_score = max(
        0.0,
        min(1.0, float(color_score))
    )

    # Texture
    texture_score = compare_texture(
        original_resized,
        generated_resized,
    )

    # Edges
    edge_score = compare_edges(
        original_resized,
        generated_resized,
    )

    overall = (
        0.45 * color_score +
        0.35 * texture_score +
        0.20 * edge_score
    )

    status = (
        "DESIGN_OK"
        if overall >= 0.55
        else "DESIGN_CHANGED"
    )

    return {
        "status": status,

        "overall_score": round(
            float(overall),
            4,
        ),

        "color_score": round(
            float(color_score),
            4,
        ),

        "texture_score": round(
            float(texture_score),
            4,
        ),

        "edge_score": round(
            float(edge_score),
            4,
        ),

        "threshold": 0.55,

        "note": (
            "This is a heuristic OpenCV verification. "
            "It is not pixel-level identity verification."
        ),
    }


# ============================================================
# SAVE DEBUG FILES
# ============================================================

def save_debug_files(
    job_id,
    analysis,
    prompt,
    detection_debug,
    protected_mask,
):

    analysis_path = (
        DEBUG_DIR /
        f"{job_id}_analysis.json"
    )

    prompt_path = (
        DEBUG_DIR /
        f"{job_id}_prompt.txt"
    )

    detection_path = (
        DEBUG_DIR /
        f"{job_id}_detections.png"
    )

    mask_path = (
        MASK_DIR /
        f"{job_id}_protected_mask.png"
    )

    with open(
        analysis_path,
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            analysis,
            file,
            indent=2,
        )

    with open(
        prompt_path,
        "w",
        encoding="utf-8",
    ) as file:

        file.write(prompt)

    cv2.imwrite(
        str(detection_path),
        detection_debug,
    )

    cv2.imwrite(
        str(mask_path),
        protected_mask,
    )

    return {
        "analysis": str(
            analysis_path
        ),

        "prompt": str(
            prompt_path
        ),

        "detections": str(
            detection_path
        ),

        "protected_mask": str(
            mask_path
        ),
    }


# ============================================================
# ROOT
# ============================================================

@app.get("/")
async def root():

    return {
        "success": True,

        "service": "SareeViz",

        "version": "2.0.0",

        "model": MODEL,

        "gemini_calls_per_generation": 1,

        "semantic_gemini_analysis": False,

        "opencv_analysis": True,

        "automatic_function_calling": False,

        "separate_lace_reference": True,

        "design_lock": True,
    }


# ============================================================
# GENERATE API
# ============================================================

@app.post("/generate")
async def generate(
    saree: UploadFile = File(...),

    lace: UploadFile | None = File(
        default=None
    ),

    model: UploadFile | None = File(
        default=None
    ),

    background: UploadFile | None = File(
        default=None
    ),

    pose: str = Form(
        default="Classic Catalog Standing Pose"
    ),

    background_name: str = Form(
        default="Premium white studio background"
    ),

    image_size: str = Form(
        default="2K"
    ),
):

    job_id = (
        datetime.now().strftime(
            "%Y%m%d_%H%M%S"
        )
        + "_"
        + uuid.uuid4().hex[:8]
    )

    # ========================================================
    # VALIDATE SIZE
    # ========================================================

    allowed_sizes = [
        "512",
        "1K",
        "2K",
        "4K",
    ]

    if image_size not in allowed_sizes:

        raise HTTPException(
            status_code=400,
            detail=(
                "image_size must be "
                "512, 1K, 2K or 4K"
            ),
        )

    # ========================================================
    # READ FILES
    # ========================================================

    saree_bytes = await saree.read()

    if not saree_bytes:

        raise HTTPException(
            status_code=400,
            detail="Saree image is empty",
        )

    lace_bytes = None
    model_bytes = None
    background_bytes = None

    if lace:
        lace_bytes = await lace.read()

    if model:
        model_bytes = await model.read()

    if background:
        background_bytes = await background.read()

    # ========================================================
    # DECODE SAREE
    # ========================================================

    try:

        saree_image = decode_image(
            saree_bytes
        )

    except Exception as error:

        raise HTTPException(
            status_code=400,
            detail=f"Invalid saree image: {error}",
        )

    # ========================================================
    # OPENCV ANALYSIS
    # ========================================================

    analysis = analyze_saree(
        saree_image
    )

    # ========================================================
    # PROTECTED MASK
    # ========================================================

    protected_mask = create_protected_mask(
        saree_image,
        analysis,
    )

    # ========================================================
    # REFERENCE CROPS
    # ========================================================

    reference_paths = (
        create_reference_crops(
            saree_image,
            analysis,
            job_id,
        )
    )

    # ========================================================
    # DEBUG
    # ========================================================

    detection_debug = (
        create_detection_debug(
            saree_image,
            analysis,
        )
    )

    # ========================================================
    # DYNAMIC PROMPT
    # ========================================================

    prompt = build_dynamic_prompt(
        analysis=analysis,
        pose=pose,
        background=background_name,
        has_lace_reference=(
            lace_bytes is not None
        ),
    )

    debug_files = save_debug_files(
        job_id=job_id,
        analysis=analysis,
        prompt=prompt,
        detection_debug=detection_debug,
        protected_mask=protected_mask,
    )

    # ========================================================
    # ONE GEMINI CALL
    # ========================================================

    try:

        generated_bytes = generate_image(
            saree_bytes=saree_bytes,
            lace_bytes=lace_bytes,
            model_bytes=model_bytes,
            background_bytes=background_bytes,
            reference_paths=reference_paths,
            prompt_text=prompt,
            image_size=image_size,
        )

    except Exception as error:

        raise HTTPException(
            status_code=500,
            detail=(
                f"Gemini image generation failed: "
                f"{str(error)}"
            ),
        )

    # ========================================================
    # DECODE GENERATED IMAGE
    # ========================================================

    try:

        generated_image = decode_image(
            generated_bytes
        )

    except Exception as error:

        raise HTTPException(
            status_code=500,
            detail=(
                f"Generated image could not be decoded: "
                f"{error}"
            ),
        )

    # ========================================================
    # VERIFY DESIGN
    # ========================================================

    verification = verify_design(
        saree_image,
        generated_image,
    )

    verification_path = (
        VERIFY_DIR /
        f"{job_id}_verification.json"
    )

    with open(
        verification_path,
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            verification,
            file,
            indent=2,
        )

    # ========================================================
    # SAVE OUTPUT
    # ========================================================

    output_filename = (
        f"{job_id}.png"
    )

    output_path = (
        OUTPUT_DIR /
        output_filename
    )

    with open(
        output_path,
        "wb",
    ) as file:

        file.write(
            generated_bytes
        )

    # ========================================================
    # RESPONSE
    # ========================================================

    return {
        "success": True,

        "job_id": job_id,

        "model": MODEL,

        "gemini_calls": 1,

        "semantic_analysis": False,

        "opencv_analysis": True,

        "automatic_function_calling": False,

        "separate_lace_reference": (
            lace_bytes is not None
        ),

        "design_lock": True,

        "pose": pose,

        "background": background_name,

        "image_size": image_size,

        "generated_image": (
            f"/outputs/{output_filename}"
        ),

        "design_verification": verification,

        "verification_file": (
            f"/debug/verification/"
            f"{job_id}_verification.json"
        ),

        "analysis": analysis,

        "debug": {
            "analysis": (
                f"/debug/"
                f"{job_id}_analysis.json"
            ),

            "prompt": (
                f"/debug/"
                f"{job_id}_prompt.txt"
            ),

            "detections": (
                f"/debug/"
                f"{job_id}_detections.png"
            ),

            "protected_mask": (
                f"/debug/masks/"
                f"{job_id}_protected_mask.png"
            ),
        },

        "message": (
            "OpenCV analysis completed and "
            "exactly one Gemini image-generation "
            "call was used."
        ),
    }


# ============================================================
# OUTPUT FILE
# ============================================================

@app.get("/outputs/{filename}")
async def get_output(filename: str):

    path = OUTPUT_DIR / filename

    if not path.exists():

        raise HTTPException(
            status_code=404,
            detail="Output image not found",
        )

    return FileResponse(
        path,
        media_type="image/png",
    )


# ============================================================
# DEBUG FILE
# ============================================================

@app.get("/debug/{filename}")
async def get_debug(filename: str):

    path = DEBUG_DIR / filename

    if not path.exists():

        raise HTTPException(
            status_code=404,
            detail="Debug file not found",
        )

    return FileResponse(
        path
    )


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":

    import uvicorn

    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
    )
