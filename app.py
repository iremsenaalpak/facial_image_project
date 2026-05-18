import json
import os
import shutil
import uuid
from typing import List

import cv2
import numpy as np
from fastapi import FastAPI, File, Form, UploadFile
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.requests import Request

from modules.face_landmark import FaceLandmarkDetector
from modules.preprocessing import run_preprocessing_pipeline
from modules.visualization import visualize_face_data
from modules.warping import FaceWarper
from modules.aging import aging_deaging_pipeline
from modules.export_utils import export_metrics_to_csv, export_metrics_to_pdf
from modules.frequency_analysis import run_frequency_analysis
from modules.evaluation import run_evaluation


app = FastAPI(title="FaceLab Studio")
templates = Jinja2Templates(directory="templates")

UPLOAD_DIR = "uploads"
RESULT_DIR = "results"
GLASSES_DIR = "assets/glasses"
STICKERS_DIR = "assets/stickers"

os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(RESULT_DIR, exist_ok=True)
os.makedirs(GLASSES_DIR, exist_ok=True)
os.makedirs(STICKERS_DIR, exist_ok=True)

# Kendi PNG gözlüklerini kullanıyorsan bu satır kapalı kalmalı.
# FaceWarper.ensure_glasses_assets(GLASSES_DIR)

app.mount("/uploads", StaticFiles(directory=UPLOAD_DIR), name="uploads")
app.mount("/results", StaticFiles(directory=RESULT_DIR), name="results")
app.mount("/assets", StaticFiles(directory="assets"), name="assets")
app.mount("/static", StaticFiles(directory="static"), name="static")

ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png"}


def draw_bbox(image, bbox, color=(0, 255, 0), thickness=2):
    image_copy = image.copy()
    x, y, w, h = bbox
    cv2.rectangle(
        image_copy,
        (x, y),
        (x + w, y + h),
        color,
        thickness
    )
    return image_copy


def save_image(image, path):
    cv2.imwrite(path, image)


def overlay_png(background, overlay, x, y, overlay_size=None):
    bg = background.copy()

    if overlay is None:
        return bg

    if overlay_size is not None:
        overlay = cv2.resize(
            overlay,
            overlay_size,
            interpolation=cv2.INTER_AREA
        )

    if len(overlay.shape) < 3 or overlay.shape[2] < 4:
        return bg

    if x < 0:
        overlay = overlay[:, abs(x):]
        x = 0

    if y < 0:
        overlay = overlay[abs(y):, :]
        y = 0

    h, w = overlay.shape[:2]

    if x >= bg.shape[1] or y >= bg.shape[0]:
        return bg

    if x + w > bg.shape[1]:
        overlay = overlay[:, :bg.shape[1] - x]
        w = overlay.shape[1]

    if y + h > bg.shape[0]:
        overlay = overlay[:bg.shape[0] - y, :]
        h = overlay.shape[0]

    overlay_img = overlay[..., :3]
    alpha = overlay[..., 3:] / 255.0

    bg_section = bg[y:y + h, x:x + w]

    blended = (
        bg_section * (1 - alpha)
        + overlay_img * alpha
    ).astype(np.uint8)

    bg[y:y + h, x:x + w] = blended

    return bg


def _crop_to_alpha(sticker, threshold=10):
    """Trim transparent padding so placement matches the visible content."""
    if sticker is None or sticker.shape[2] < 4:
        return sticker
    alpha = sticker[..., 3]
    ys, xs = np.where(alpha > threshold)
    if len(xs) == 0:
        return sticker
    return sticker[ys.min():ys.max() + 1, xs.min():xs.max() + 1]


def _split_at_horizontal_gap(sticker, threshold=10):
    """
    Split a sticker into (top, bottom) parts at the largest empty horizontal
    band found in the middle 60% of its height. Returns each piece already
    cropped to its alpha bounds. If no clear gap exists, returns (sticker, None).
    """
    if sticker is None or sticker.shape[2] < 4:
        return sticker, None

    alpha = sticker[..., 3]
    row_has_content = (alpha > threshold).any(axis=1)
    h = len(row_has_content)
    if h == 0:
        return sticker, None

    # Look only in the middle 60% for a gap
    lo, hi = int(h * 0.20), int(h * 0.80)
    gap_start = gap_end = None
    best_len = 0
    cur_start = None
    for i in range(lo, hi):
        if not row_has_content[i]:
            if cur_start is None:
                cur_start = i
        else:
            if cur_start is not None:
                length = i - cur_start
                if length > best_len:
                    best_len = length
                    gap_start, gap_end = cur_start, i
                cur_start = None
    if cur_start is not None:
        length = hi - cur_start
        if length > best_len:
            best_len = length
            gap_start, gap_end = cur_start, hi

    # Need a meaningful gap (at least 5% of sticker height)
    if best_len < int(h * 0.05) or gap_start is None:
        return _crop_to_alpha(sticker), None

    top = _crop_to_alpha(sticker[:gap_start])
    bottom = _crop_to_alpha(sticker[gap_end:])
    return top, bottom


def _alpha_centroid(sticker, threshold=10):
    """Return (cx_norm, cy_norm) in [0,1] — normalized centroid of visible pixels."""
    if sticker is None or sticker.shape[2] < 4:
        return 0.5, 0.5
    alpha = sticker[..., 3]
    ys, xs = np.where(alpha > threshold)
    if len(xs) == 0:
        return 0.5, 0.5
    h, w = alpha.shape
    return float(xs.mean()) / w, float(ys.mean()) / h


def _rotate_rgba(sticker, angle_deg):
    """Rotate an RGBA sticker around its center, expanding the canvas to fit."""
    if abs(angle_deg) < 0.5 or sticker is None:
        return sticker
    h, w = sticker.shape[:2]
    cx, cy = w / 2, h / 2
    M = cv2.getRotationMatrix2D((cx, cy), angle_deg, 1.0)
    cos, sin = abs(M[0, 0]), abs(M[0, 1])
    new_w = int(h * sin + w * cos)
    new_h = int(h * cos + w * sin)
    M[0, 2] += new_w / 2 - cx
    M[1, 2] += new_h / 2 - cy
    return cv2.warpAffine(
        sticker, M, (new_w, new_h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(0, 0, 0, 0),
    )


def _place_sticker(img, sticker, anchor_xy, target_width,
                    vy_frac=0.5, vx_frac=0.5, use_centroid=False,
                    rotation_deg=0.0):
    """
    Place a sticker on the image with aspect ratio preserved.

    target_width refers to the *unrotated* sticker. Rotation is applied
    after sizing, then the rotated canvas is centered on the same anchor.
    """
    if sticker is None:
        return img
    cropped = _crop_to_alpha(sticker)
    src_h, src_w = cropped.shape[:2]
    if src_w == 0 or src_h == 0:
        return img
    target_w = max(int(target_width), 1)
    target_h = max(int(target_w * src_h / src_w), 1)

    if use_centroid:
        cx_norm, cy_norm = _alpha_centroid(cropped)
    else:
        cx_norm, cy_norm = vx_frac, vy_frac

    # Resize first
    resized = cv2.resize(cropped, (target_w, target_h), interpolation=cv2.INTER_AREA)

    # The visual anchor point on the unrotated sticker
    anchor_local = np.array([target_w * cx_norm, target_h * cy_norm])

    if abs(rotation_deg) >= 0.5:
        rotated = _rotate_rgba(resized, rotation_deg)
        # Rotate the local anchor around the unrotated center to find its
        # new position within the rotated (expanded) canvas
        center = np.array([target_w / 2, target_h / 2])
        theta = np.deg2rad(-rotation_deg)  # cv2 rotates CCW for positive angle
        cos, sin = np.cos(theta), np.sin(theta)
        offset = anchor_local - center
        rotated_offset = np.array([
            offset[0] * cos - offset[1] * sin,
            offset[0] * sin + offset[1] * cos,
        ])
        rh, rw = rotated.shape[:2]
        anchor_in_rotated = np.array([rw / 2, rh / 2]) + rotated_offset
        x = int(anchor_xy[0] - anchor_in_rotated[0])
        y = int(anchor_xy[1] - anchor_in_rotated[1])
        return overlay_png(img, rotated, x, y, None)

    x = int(anchor_xy[0] - anchor_local[0])
    y = int(anchor_xy[1] - anchor_local[1])
    return overlay_png(img, resized, x, y, None)


def _face_roll_deg(landmarks):
    """Roll angle of the face (positive = head tilted to subject's right)."""
    if not landmarks or len(landmarks) <= max(_LM_LEFT_EYE_OUTER, _LM_RIGHT_EYE_OUTER):
        return 0.0
    lx, ly = landmarks[_LM_LEFT_EYE_OUTER]
    rx, ry = landmarks[_LM_RIGHT_EYE_OUTER]
    return float(np.degrees(np.arctan2(ry - ly, rx - lx)))


# Landmark anchor indices (MediaPipe FaceMesh refine_landmarks=True)
_LM_FOREHEAD = 10        # topmost forehead point (hairline area)
_LM_CHIN = 152
_LM_NOSE_TIP = 4         # canonical pronasale (the actual tip)
_LM_NOSE_BRIDGE = 6      # between the eyes
_LM_LEFT_CHEEKBONE = 234
_LM_RIGHT_CHEEKBONE = 454
_LM_RIGHT_CHEEK_APPLE = 280
_LM_RIGHT_TEMPLE = 356
_LM_LEFT_EYE_OUTER = 33
_LM_RIGHT_EYE_OUTER = 263


def _anchor(landmarks, idx, fallback):
    if landmarks and idx < len(landmarks):
        return landmarks[idx]
    return fallback


def apply_sticker_effect(image, effect, landmarks=None):
    img = image.copy()
    h, w = img.shape[:2]

    if effect == "none":
        return img

    # Compute scale and anchors from specific landmarks instead of the
    # FaceMesh bbox. Cheekbone-to-cheekbone is a robust face-width unit
    # that doesn't change with hair, neck, or background.
    if landmarks and len(landmarks) > max(_LM_RIGHT_CHEEKBONE, _LM_CHIN, _LM_FOREHEAD):
        cheek_l = landmarks[_LM_LEFT_CHEEKBONE]
        cheek_r = landmarks[_LM_RIGHT_CHEEKBONE]
        forehead = landmarks[_LM_FOREHEAD]
        chin = landmarks[_LM_CHIN]
        face_scale = max(abs(cheek_r[0] - cheek_l[0]), 1)
        face_cx = (cheek_l[0] + cheek_r[0]) // 2
        face_cy = (forehead[1] + chin[1]) // 2
    else:
        face_scale = max(min(w, h) // 3, 1)
        face_cx, face_cy = w // 2, h // 2
        forehead = (face_cx, face_cy - face_scale)
        chin = (face_cx, face_cy + face_scale)

    sticker_path = {
        "crown":       "crown.png",
        "cat_ears":    "cat_ears.png",
        "sparkles":    "sparkles.png",
        "freckles":    "freckles.png",
    }.get(effect)
    if sticker_path is None:
        return img

    sticker = cv2.imread(
        os.path.join(STICKERS_DIR, sticker_path),
        cv2.IMREAD_UNCHANGED
    )
    if sticker is None:
        return img

    roll = _face_roll_deg(landmarks)

    if effect == "crown":
        anchor = (face_cx, forehead[1])
        img = _place_sticker(img, sticker, anchor,
                              face_scale * 1.50, vy_frac=0.85,
                              rotation_deg=roll)

    elif effect == "cat_ears":
        ears, whiskers = _split_at_horizontal_gap(sticker)
        nose = _anchor(landmarks, _LM_NOSE_TIP, (face_cx, face_cy))

        # Ears anchored to the forehead landmark with mostly-upward placement.
        ear_anchor = (face_cx, forehead[1])
        ear_width = face_scale * 1.10

        if whiskers is not None:
            img = _place_sticker(
                img, ears, ear_anchor,
                ear_width, vy_frac=0.95,
                rotation_deg=roll
            )
            img = _place_sticker(
                img, whiskers, nose,
                face_scale * 1.05, vy_frac=0.5,
                rotation_deg=roll
            )
        else:
            img = _place_sticker(img, ears, nose,
                                  face_scale * 1.05, vy_frac=0.5,
                                  rotation_deg=roll)

    elif effect == "sparkles":
        img = _place_sticker(
            img, sticker, (face_cx, face_cy),
            face_scale * 2.20, use_centroid=True,
            rotation_deg=roll
        )

    elif effect == "freckles":
        # Freckles must follow face tilt — they sit on the nose bridge
        # and cheeks, which rotate together with the face.
        bridge = _anchor(landmarks, _LM_NOSE_BRIDGE,
                         (face_cx, face_cy - face_scale * 0.1))
        # Offset DOWN along the face's local-Y axis so tilt doesn't break it
        theta = np.deg2rad(roll + 90)  # face-down direction
        offset_len = face_scale * 0.20
        anchor = (
            bridge[0] + offset_len * np.cos(theta),
            bridge[1] + offset_len * np.sin(theta),
        )
        img = _place_sticker(img, sticker, anchor,
                              face_scale * 0.95, use_centroid=True,
                              rotation_deg=roll)

    return img

def render_home_page(
    request,
    message="",
    images=None,
    metadata=None,
    face_bbox=None,
    success=None,
    landmark_success=None,
    landmark_count=None,
    grouped_landmark_keys=None,
    landmark_message=None,
    landmark_json=None,
    face_id=None
):
    return templates.TemplateResponse(
        request=request, name="index.html", context={
            "request": request,
            "message": message,
            "images": images,
            "metadata": metadata,
            "face_bbox": face_bbox,
            "success": success,
            "landmark_success": landmark_success,
            "landmark_count": landmark_count,
            "grouped_landmark_keys": grouped_landmark_keys,
            "landmark_message": landmark_message,
            "landmark_json": landmark_json,
            "face_id": face_id
        }
    )


@app.get("/", response_class=HTMLResponse)
def home(request: Request):
    return render_home_page(request=request)


@app.post("/upload", response_class=HTMLResponse)
async def upload_image(
    request: Request,
    file: UploadFile = File(None),
    face_id: str = Form("")
):
    if not face_id and (file is None or not file.filename):
        return render_home_page(
            request=request,
            message="No file selected.",
            success=False
        )

    if file and file.filename:
        file_ext = os.path.splitext(file.filename)[1].lower()

        if file_ext not in ALLOWED_EXTENSIONS:
            return render_home_page(
                request=request,
                message="Unsupported file format. Please upload JPG, JPEG, or PNG.",
                success=False
            )

        unique_id = uuid.uuid4().hex
        uploaded_filename = f"{unique_id}{file_ext}"
        uploaded_path = os.path.join(UPLOAD_DIR, uploaded_filename)

        with open(uploaded_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
    else:
        unique_id = face_id
        uploaded_path = None
        for ext in ALLOWED_EXTENSIONS:
            p = os.path.join(UPLOAD_DIR, f"{face_id}{ext}")
            if os.path.exists(p):
                uploaded_path = p
                break
        
        if not uploaded_path:
            fallback = os.path.join(UPLOAD_DIR, f"{face_id}_face.jpg")
            if os.path.exists(fallback):
                uploaded_path = fallback
            else:
                return render_home_page(
                    request=request,
                    message="Previous image not found. Please upload again.",
                    success=False
                )

    result = run_preprocessing_pipeline(uploaded_path)

    if not result["success"]:
        return render_home_page(
            request=request,
            message=result["message"],
            metadata=result["metadata"],
            face_bbox=result["face_bbox"],
            success=False
        )

    original_image = result["original_image"]
    face_bbox = result["face_bbox"]
    cropped_face = result["cropped_face"]
    resized_face = result["resized_face"]
    grayscale_face = result["grayscale_face"]

    bbox_image = draw_bbox(original_image, face_bbox)

    face_image = (
        resized_face
        if resized_face is not None
        else cropped_face
    )

    landmark_success = False
    landmark_count = 0
    grouped_landmark_keys = []
    landmark_message = "Landmark detection was not run."
    landmark_json_text = "No landmark data."

    landmark_full_image = (
        face_image.copy()
        if face_image is not None
        else original_image.copy()
    )

    landmark_grouped_image = (
        face_image.copy()
        if face_image is not None
        else original_image.copy()
    )

    if face_image is not None:

        if len(face_image.shape) == 2:
            face_image = cv2.cvtColor(
                face_image,
                cv2.COLOR_GRAY2BGR
            )

        detector = FaceLandmarkDetector()

        landmark_result = detector.detect(face_image)

        landmark_success = landmark_result["success"]

        landmark_count = len(
            landmark_result["landmarks"]
        )

        grouped_landmark_keys = list(
            landmark_result["grouped_landmarks"].keys()
        )

        landmark_message = landmark_result["message"]

        if landmark_result["success"]:

            landmark_full_image = visualize_face_data(
                image=face_image,
                bbox=landmark_result["bbox"],
                landmarks=landmark_result["landmarks"],
                grouped_landmarks=landmark_result["grouped_landmarks"],
                show_bbox=True,
                show_landmarks=True,
                use_grouped=False
            )

            landmark_grouped_image = visualize_face_data(
                image=face_image,
                bbox=landmark_result["bbox"],
                landmarks=landmark_result["landmarks"],
                grouped_landmarks=landmark_result["grouped_landmarks"],
                show_bbox=True,
                show_landmarks=True,
                use_grouped=True
            )

            preview_data = {
                "bbox": landmark_result["bbox"],
                "landmark_count": landmark_count,
                "grouped_landmark_keys": grouped_landmark_keys,
                "image_size": landmark_result["image_size"],
                "message": landmark_result["message"]
            }

            landmark_json_text = json.dumps(
                preview_data,
                indent=4,
                ensure_ascii=False
            )

        detector.close()

    original_result_path = os.path.join(
        RESULT_DIR,
        f"{unique_id}_original.jpg"
    )

    bbox_result_path = os.path.join(
        RESULT_DIR,
        f"{unique_id}_bbox.jpg"
    )

    cropped_result_path = os.path.join(
        RESULT_DIR,
        f"{unique_id}_cropped.jpg"
    )

    resized_result_path = os.path.join(
        RESULT_DIR,
        f"{unique_id}_resized.jpg"
    )

    grayscale_result_path = os.path.join(
        RESULT_DIR,
        f"{unique_id}_grayscale.jpg"
    )

    landmark_full_result_path = os.path.join(
        RESULT_DIR,
        f"{unique_id}_landmark_full.jpg"
    )

    landmark_grouped_result_path = os.path.join(
        RESULT_DIR,
        f"{unique_id}_landmark_grouped.jpg"
    )

    save_image(original_image, original_result_path)
    save_image(bbox_image, bbox_result_path)
    save_image(cropped_face, cropped_result_path)
    save_image(resized_face, resized_result_path)
    save_image(grayscale_face, grayscale_result_path)
    save_image(landmark_full_image, landmark_full_result_path)
    save_image(landmark_grouped_image, landmark_grouped_result_path)

    if face_image is not None:
        save_image(
            face_image,
            os.path.join(
                UPLOAD_DIR,
                f"{unique_id}_face.jpg"
            )
        )

    image_urls = {
        "original": f"/results/{unique_id}_original.jpg",
        "bbox": f"/results/{unique_id}_bbox.jpg",
        "cropped": f"/results/{unique_id}_cropped.jpg",
        "resized": f"/results/{unique_id}_resized.jpg",
        "grayscale": f"/results/{unique_id}_grayscale.jpg",
        "landmark_full": f"/results/{unique_id}_landmark_full.jpg",
        "landmark_grouped": f"/results/{unique_id}_landmark_grouped.jpg"
    }

    return render_home_page(
        request=request,
        message=result["message"],
        images=image_urls,
        metadata=result["metadata"],
        face_bbox=result["face_bbox"],
        success=True,
        landmark_success=landmark_success,
        landmark_count=landmark_count,
        grouped_landmark_keys=grouped_landmark_keys,
        landmark_message=landmark_message,
        landmark_json=landmark_json_text,
        face_id=unique_id
    )


@app.get("/api/test")
def api_test():
    return {
        "message": "API is running"
    }


# ---------------------------------------------------------------------------
# Transform page helpers
# ---------------------------------------------------------------------------

def _hex_to_bgr(hex_color: str):

    hex_color = hex_color.lstrip("#")

    if len(hex_color) != 6:
        return (40, 80, 40)

    r = int(hex_color[0:2], 16)
    g = int(hex_color[2:4], 16)
    b = int(hex_color[4:6], 16)

    return (b, g, r)


EXPRESSION_MODES = {
    "none": "No Expression Change",
    "smile": "Smile Enhancement",
    "eyebrow_raise": "Eyebrow Raise",
    "lip_widen": "Lip Widening",
    "face_slim": "Face Slimming",
    "aging": "Aging",
    "deaging": "De-aging",
}


CREATIVE_OPTIONS = {
    "lip_color": "Lip Coloring",
    "hair_color": "Hair Coloring",
    "eye_color": "Eye Color",
    "frame_photo": "Photo Frame",
}

@app.get("/transform", response_class=HTMLResponse)
def transform_home(request: Request, image_id: str = None):
    return templates.TemplateResponse(
        request=request, name="transform.html", context={
            "request": request,
            "image_id": image_id
        }
    )


@app.post("/transform", response_class=HTMLResponse)
async def transform_image(
    request: Request,
    file: UploadFile = File(None),
    mode: str = Form("none"),
    intensity: float = Form(1.0),
    creative: List[str] = Form(default=[]),
    glasses_style: str = Form("none"),
    hair_color_hex: str = Form("#4a2c0a"),
    lip_color_hex: str = Form("#c0392b"),
    eye_color_hex: str = Form("#1e6db5"),
    frame_style: str = Form("classic"),
    face_id: str = Form(""),
    sticker_effect: str = Form("none"),
):
    if file and file.filename:
        file_ext = os.path.splitext(file.filename)[1].lower()
        if file_ext not in ALLOWED_EXTENSIONS:
            return templates.TemplateResponse(request=request, name="transform.html", context={"request": request, "error_message": "Unsupported format."})
        face_id = uuid.uuid4().hex
        uploaded_path = os.path.join(UPLOAD_DIR, f"{face_id}{file_ext}")
        with open(uploaded_path, "wb") as buf:
            import shutil
            shutil.copyfileobj(file.file, buf)
    else:
        if not face_id:
            return templates.TemplateResponse(request=request, name="transform.html", context={"request": request, "error_message": "No file selected."})
        uploaded_path = None
        for ext in ALLOWED_EXTENSIONS:
            p = os.path.join(UPLOAD_DIR, f"{face_id}{ext}")
            if os.path.exists(p):
                uploaded_path = p
                break
        if not uploaded_path:
            return templates.TemplateResponse(request=request, name="transform.html", context={"request": request, "error_message": "Previous image not found."})

    result = run_preprocessing_pipeline(uploaded_path)
    if not result["success"]:
        return templates.TemplateResponse(request=request, name="transform.html", context={"request": request, "error_message": result["message"]})

    original_image = result["original_image"]
    face_bbox = result["face_bbox"]
    face_image = result.get("resized_face")
    if face_image is None:
        face_image = result.get("cropped_face")

    if face_image is None:
        return templates.TemplateResponse(request=request, name="transform.html", context={"request": request, "error_message": "Could not extract face region."})

    if len(face_image.shape) == 2:
        face_image = cv2.cvtColor(face_image, cv2.COLOR_GRAY2BGR)

    detector = FaceLandmarkDetector()
    lm_result = detector.detect(face_image)
    detector.close()

    if not lm_result["success"]:
        return templates.TemplateResponse(
            request=request, name="transform.html", context={
                "request": request,
                "error_message": f"Landmark detection failed: {lm_result['message']}"
            }
        )

    landmarks = lm_result["landmarks"]

    warper = FaceWarper()

    transformed = face_image.copy()

    # ---------------------------------------------------------
    # Expression Warp / Aging-Deaging
    # ---------------------------------------------------------
    if mode in ["smile", "eyebrow_raise", "lip_widen", "face_slim"]:

        transformed = warper.warp_expression(
            transformed,
            landmarks,
            mode,
            intensity=max(0.1, min(intensity, 2.0))
        )

    elif mode in ["aging", "deaging"]:

        transformed = aging_deaging_pipeline(
            transformed,
            mode=mode,
            intensity=max(0.0, min(intensity, 2.0)) / 2.0,
            landmarks=landmarks
        )

    # ---------------------------------------------------------
    # Creative Overlays
    # ---------------------------------------------------------
    needs_creative = (
        bool(creative)
        or glasses_style != "none"
        or sticker_effect != "none"
    )

    if needs_creative:
        detector2 = FaceLandmarkDetector()
        lm2 = detector2.detect(transformed)
        detector2.close()

        lm2_pts = lm2["landmarks"] if lm2["success"] else landmarks

        if "lip_color" in creative:
            lip_bgr = _hex_to_bgr(lip_color_hex)
            transformed = warper.apply_lip_color(
                transformed,
                lm2_pts,
                color=lip_bgr
            )

        if glasses_style != "none":
            transformed = warper.apply_glasses(
                transformed,
                lm2_pts,
                style=glasses_style,
                assets_dir=GLASSES_DIR
            )

        # hair_color is applied AFTER paste-back (see below) so the
        # segmenter sees the full head, not just the cropped face band.

        if "eye_color" in creative:
            eye_bgr = _hex_to_bgr(eye_color_hex)
            transformed = warper.apply_eye_color(
                transformed,
                lm2_pts,
                color=eye_bgr
            )



    result_uid = uuid.uuid4().hex

    x, y, w, h = face_bbox
    final_output = original_image.copy()

    nothing_selected = (
        mode == "none"
        and not creative
        and glasses_style == "none"
        and sticker_effect == "none"
    )

    if not nothing_selected:
        transformed_resized = cv2.resize(transformed, (w, h))
        final_output[y:y+h, x:x+w] = transformed_resized

    # Effects that need the WHOLE image (not just the cropped face) run
    # after paste-back. Hair recolor needs to see the full hair to segment
    # it. Stickers need canvas above the head for crowns/ears.
    needs_full_image = (
        "hair_color" in creative
        or sticker_effect != "none"
    )
    if needs_full_image:
        cropped_h, cropped_w = face_image.shape[:2]
        translated_lms = [
            (
                x + int(lx * w / max(cropped_w, 1)),
                y + int(ly * h / max(cropped_h, 1))
            )
            for (lx, ly) in (lm2_pts if needs_creative else landmarks)
        ]

        if "hair_color" in creative:
            hair_bgr = _hex_to_bgr(hair_color_hex)
            final_output = warper.apply_hair_color(
                final_output,
                translated_lms,
                color=hair_bgr
            )

        if sticker_effect != "none":
            final_output = apply_sticker_effect(
                final_output,
                sticker_effect,
                translated_lms
            )

    # Capture the face region from the full final image BEFORE the frame
    # is applied, so analysis includes hair color / stickers / etc. The
    # frame is a decorative border around the whole image and would skew
    # the metrics, so it is excluded from analysis.
    if nothing_selected:
        analysis_transformed = face_image
    else:
        face_region_final = final_output[y:y+h, x:x+w]
        analysis_transformed = cv2.resize(
            face_region_final,
            (face_image.shape[1], face_image.shape[0])
        )

    if "frame_photo" in creative:
        final_output = warper.apply_frame_photo(
            final_output,
            None,
            style=frame_style
        )

    save_image(
        original_image,
        os.path.join(RESULT_DIR, f"{result_uid}_orig.jpg")
    )

    save_image(
        final_output,
        os.path.join(RESULT_DIR, f"{result_uid}_out.jpg")
    )

    freq_result = run_frequency_analysis(
        face_image,
        analysis_transformed
    )

    save_image(
        freq_result["original_spectrum"],
        os.path.join(RESULT_DIR, f"{result_uid}_orig_spectrum.jpg")
    )

    save_image(
        freq_result["processed_spectrum"],
        os.path.join(RESULT_DIR, f"{result_uid}_proc_spectrum.jpg")
    )

    frequency_table_html = freq_result["html_table"]

    evaluation_metrics = run_evaluation(
        face_image,
        analysis_transformed
    )

    evaluation_table_html = f"""
    <table>
        <tr>
            <th>Metric</th>
            <th>Value</th>
            <th>Meaning</th>
        </tr>
        <tr>
            <td>MSE</td>
            <td>{evaluation_metrics['mse']:.4f}</td>
            <td>Pixel-level difference between original and processed image</td>
        </tr>
        <tr>
            <td>PSNR</td>
            <td>{evaluation_metrics['psnr']:.4f} dB</td>
            <td>Signal quality based on MSE</td>
        </tr>
        <tr>
            <td>SSIM</td>
            <td>{evaluation_metrics['ssim']:.4f}</td>
            <td>Structural similarity between images</td>
        </tr>
    </table>
    """

    export_data = {
        "mse": evaluation_metrics["mse"],
        "psnr": evaluation_metrics["psnr"],
        "ssim": evaluation_metrics["ssim"],
        "total_energy_original": freq_result["original_energy"]["total_energy"],
        "total_energy_processed": freq_result["processed_energy"]["total_energy"],
        "low_frequency_original": freq_result["original_energy"]["low_frequency_energy"],
        "low_frequency_processed": freq_result["processed_energy"]["low_frequency_energy"],
        "high_frequency_original": freq_result["original_energy"]["high_frequency_energy"],
        "high_frequency_processed": freq_result["processed_energy"]["high_frequency_energy"],
        "high_low_ratio_original": freq_result["original_energy"]["high_low_ratio"],
        "high_low_ratio_processed": freq_result["processed_energy"]["high_low_ratio"],
    }

    csv_path = os.path.join(
        RESULT_DIR,
        f"{result_uid}_report.csv"
    )

    pdf_path = os.path.join(
        RESULT_DIR,
        f"{result_uid}_report.pdf"
    )

    export_metrics_to_csv(
        export_data,
        csv_path
    )

    export_metrics_to_pdf(
        export_data,
        pdf_path
    )

    return templates.TemplateResponse(
        request=request, name="transform.html", context={
            "request": request,

            "original_image":
                f"/results/{result_uid}_orig.jpg",

            "transformed_image":
                f"/results/{result_uid}_out.jpg",

            "orig_spectrum":
                f"/results/{result_uid}_orig_spectrum.jpg",

            "proc_spectrum":
                f"/results/{result_uid}_proc_spectrum.jpg",

            "frequency_table":
                frequency_table_html,

            "evaluation_table":
                evaluation_table_html,

            "mse":
                round(evaluation_metrics["mse"], 4),

            "psnr":
                round(evaluation_metrics["psnr"], 4),

            "ssim":
                round(evaluation_metrics["ssim"], 4),

            "ratio":
                "{:.4f} (Δ {:+.4f})".format(
                    freq_result["processed_energy"]["high_low_ratio"],
                    freq_result["difference"]["ratio_difference"]
                ),

            "applied_mode":
                mode,

            "applied_intensity":
                round(intensity, 2),

            "csv_report":
                f"/results/{result_uid}_report.csv",

            "pdf_report":
                f"/results/{result_uid}_report.pdf",

            "image_id":
                face_id
        }
    )