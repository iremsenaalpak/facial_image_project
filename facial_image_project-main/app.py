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


def apply_sticker_effect(image, effect, landmarks=None):
    img = image.copy()
    h, w = img.shape[:2]

    if landmarks:
        xs = [int(p[0]) for p in landmarks]
        ys = [int(p[1]) for p in landmarks]

        face_x1 = max(min(xs), 0)
        face_y1 = max(min(ys), 0)
        face_x2 = min(max(xs), w)
        face_y2 = min(max(ys), h)

        face_w = max(face_x2 - face_x1, 1)
        face_h = max(face_y2 - face_y1, 1)
        face_cx = face_x1 + face_w // 2

    else:
        face_x1, face_y1 = 0, 0
        face_w, face_h = w, h
        face_cx = w // 2

    if effect == "none":
        return img

    elif effect == "emoji_heart":
        sticker = cv2.imread(
            os.path.join(STICKERS_DIR, "heart.png"),
            cv2.IMREAD_UNCHANGED
        )

        sticker_w = int(face_w * 0.55)
        sticker_h = int(face_h * 0.30)

        x = int(face_cx + face_w * 0.12)
        y = int(face_y1 - sticker_h * 0.15)

        img = overlay_png(
            img,
            sticker,
            x,
            y,
            (sticker_w, sticker_h)
        )

    elif effect == "emoji_star":
        sticker = cv2.imread(
            os.path.join(STICKERS_DIR, "star.png"),
            cv2.IMREAD_UNCHANGED
        )

        sticker_w = int(face_w * 0.45)
        sticker_h = int(face_h * 0.25)

        x = int(face_cx + face_w * 0.18)
        y = int(face_y1 - sticker_h * 0.05)

        img = overlay_png(
            img,
            sticker,
            x,
            y,
            (sticker_w, sticker_h)
        )

    elif effect == "crown":
        sticker = cv2.imread(
            os.path.join(STICKERS_DIR, "crown.png"),
            cv2.IMREAD_UNCHANGED
        )

        sticker_w = int(face_w * 0.75)
        sticker_h = int(face_h * 0.28)

        x = int(face_cx - sticker_w / 2)
        y = int(face_y1 - sticker_h * 0.70)

        img = overlay_png(
            img,
            sticker,
            x,
            y,
            (sticker_w, sticker_h)
        )

    elif effect == "cat_ears":
        sticker = cv2.imread(
            os.path.join(STICKERS_DIR, "cat_ears.png"),
            cv2.IMREAD_UNCHANGED
        )

        sticker_w = int(face_w * 1.05)
        sticker_h = int(face_h * 0.38)

        x = int(face_cx - sticker_w / 2)
        y = int(face_y1 - sticker_h * 0.55)

        img = overlay_png(
            img,
            sticker,
            x,
            y,
            (sticker_w, sticker_h)
        )

    elif effect == "sparkles":
        sticker = cv2.imread(
            os.path.join(STICKERS_DIR, "sparkles.png"),
            cv2.IMREAD_UNCHANGED
        )

        sticker_w = int(face_w * 1.15)
        sticker_h = int(face_h * 1.00)

        x = int(face_cx - sticker_w / 2)
        y = int(face_y1 + face_h * 0.02)

        img = overlay_png(
            img,
            sticker,
            x,
            y,
            (sticker_w, sticker_h)
        )

    elif effect == "freckles":
        sticker = cv2.imread(
            os.path.join(STICKERS_DIR, "freckles.png"),
            cv2.IMREAD_UNCHANGED
        )

        sticker_w = int(face_w * 0.62)
        sticker_h = int(face_h * 0.22)

        x = int(face_cx - sticker_w / 2)
        y = int(face_y1 + face_h * 0.42)

        img = overlay_png(
            img,
            sticker,
            x,
            y,
            (sticker_w, sticker_h)
        )

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
        "index.html",
        {
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
    file: UploadFile = File(...)
):
    if not file.filename:
        return render_home_page(
            request=request,
            message="No file selected.",
            success=False
        )

    file_ext = os.path.splitext(file.filename)[1].lower()

    if file_ext not in ALLOWED_EXTENSIONS:
        return render_home_page(
            request=request,
            message="Unsupported file format. Please upload JPG, JPEG, or PNG.",
            success=False
        )

    unique_id = uuid.uuid4().hex

    uploaded_filename = f"{unique_id}{file_ext}"

    uploaded_path = os.path.join(
        UPLOAD_DIR,
        uploaded_filename
    )

    with open(uploaded_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

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
        "transform.html",
        {
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
    face_stored_path = os.path.join(
        UPLOAD_DIR,
        f"{face_id}_face.jpg"
    ) if face_id else ""

    if face_id and not os.path.exists(face_stored_path):
        fallback_path = os.path.join(
            RESULT_DIR,
            f"{face_id}_resized.jpg"
        )

        if not os.path.exists(fallback_path):
            fallback_path = os.path.join(
                RESULT_DIR,
                f"{face_id}_cropped.jpg"
            )

        if os.path.exists(fallback_path):
            face_stored_path = fallback_path

    if face_id and os.path.exists(face_stored_path):
        face_image = cv2.imread(face_stored_path)

        if face_image is None:
            face_id = ""
    else:
        face_id = ""

    if not face_id:

        if file is None or not file.filename:
            return templates.TemplateResponse(
                "transform.html",
                {
                    "request": request,
                    "error_message": "No file selected."
                }
            )

        file_ext = os.path.splitext(file.filename)[1].lower()

        if file_ext not in ALLOWED_EXTENSIONS:
            return templates.TemplateResponse(
                "transform.html",
                {
                    "request": request,
                    "error_message": "Unsupported format. Use JPG or PNG."
                }
            )

        face_id = uuid.uuid4().hex

        uploaded_path = os.path.join(
            UPLOAD_DIR,
            f"{face_id}{file_ext}"
        )

        with open(uploaded_path, "wb") as buf:
            shutil.copyfileobj(file.file, buf)

        result = run_preprocessing_pipeline(uploaded_path)

        if not result["success"]:
            return templates.TemplateResponse(
                "transform.html",
                {
                    "request": request,
                    "error_message": result["message"]
                }
            )

        face_image = result.get("resized_face")

        if face_image is None:
            face_image = result.get("cropped_face")

        if face_image is None:
            return templates.TemplateResponse(
                "transform.html",
                {
                    "request": request,
                    "error_message": "Could not extract face region."
                }
            )

        if len(face_image.shape) == 2:
            face_image = cv2.cvtColor(
                face_image,
                cv2.COLOR_GRAY2BGR
            )

        face_stored_path = os.path.join(
            UPLOAD_DIR,
            f"{face_id}_face.jpg"
        )

        save_image(face_image, face_stored_path)

    detector = FaceLandmarkDetector()
    lm_result = detector.detect(face_image)
    detector.close()

    if not lm_result["success"]:
        return templates.TemplateResponse(
            "transform.html",
            {
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
            intensity=min(intensity, 1.0)
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

        if "hair_color" in creative:
            hair_bgr = _hex_to_bgr(hair_color_hex)
            transformed = warper.apply_hair_color(
                transformed,
                lm2_pts,
                color=hair_bgr
            )

        if "eye_color" in creative:
            eye_bgr = _hex_to_bgr(eye_color_hex)
            transformed = warper.apply_eye_color(
                transformed,
                lm2_pts,
                color=eye_bgr
            )

        if "frame_photo" in creative:
            transformed = warper.apply_frame_photo(
                transformed,
                lm2_pts,
                style=frame_style
            )

    transformed = apply_sticker_effect(
        transformed,
        sticker_effect,
        landmarks
    )

    result_uid = uuid.uuid4().hex

    save_image(
        face_image,
        os.path.join(RESULT_DIR, f"{result_uid}_orig.jpg")
    )

    save_image(
        transformed,
        os.path.join(RESULT_DIR, f"{result_uid}_out.jpg")
    )

    freq_result = run_frequency_analysis(
        face_image,
        transformed
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
        transformed
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
        "transform.html",
        {
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
                round(
                    freq_result["processed_energy"]["high_low_ratio"],
                    4
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