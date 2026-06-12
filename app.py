import json
import os
import shutil
import uuid
import base64
from typing import List

import cv2
import numpy as np
from fastapi import FastAPI, File, Form, UploadFile
from fastapi.responses import HTMLResponse, FileResponse, RedirectResponse, StreamingResponse, JSONResponse
from realtime_effect import process_frame, set_effect
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


app = FastAPI(title="BeautyDSP Studio")
templates = Jinja2Templates(directory="templates")

UPLOAD_DIR = "uploads"
RESULT_DIR = "results"
GLASSES_DIR = "assets/glasses"
STICKERS_DIR = "assets/stickers"

os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(RESULT_DIR, exist_ok=True)
os.makedirs(GLASSES_DIR, exist_ok=True)
os.makedirs(STICKERS_DIR, exist_ok=True)

app.mount("/uploads", StaticFiles(directory=UPLOAD_DIR), name="uploads")
app.mount("/results", StaticFiles(directory=RESULT_DIR), name="results")
app.mount("/assets", StaticFiles(directory="assets"), name="assets")
app.mount("/static", StaticFiles(directory="static"), name="static")

camera = None


def generate_camera_frames():
    global camera

    camera = cv2.VideoCapture(0)

    while True:
        success, frame = camera.read()

        if not success:
            break

        frame = cv2.flip(frame, 1)

        frame = process_frame(frame)

        ret, buffer = cv2.imencode(".jpg", frame)

        if not ret:
            continue

        yield (
            b"--frame\r\n"
            b"Content-Type: image/jpeg\r\n\r\n" + buffer.tobytes() + b"\r\n"
        )

    camera.release()
    camera = None


@app.get("/camera_feed")
def camera_feed():
    return StreamingResponse(
        generate_camera_frames(),
        media_type="multipart/x-mixed-replace; boundary=frame"
    )


@app.post("/set_effect")
async def set_camera_effect(
    mode: str = Form(...),
    index: int = Form(...),
    enabled: bool = Form(True)
):
    set_effect(mode, index, enabled)
    return JSONResponse({"status": "ok"})

ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png"}


# CRITICAL FIX: NameError hatasını önlemek için fonksiyon en üstte tanımlandı
def _hex_to_bgr(hex_color: str):
    hex_color = hex_color.lstrip("#")
    if len(hex_color) != 6: return (40, 80, 40)
    return int(hex_color[4:6], 16), int(hex_color[2:4], 16), int(hex_color[0:2], 16)


def draw_bbox(image, bbox, color=(0, 255, 0), thickness=2):
    image_copy = image.copy()
    x, y, w, h = bbox
    cv2.rectangle(image_copy, (x, y), (x + w, y + h), color, thickness)
    return image_copy


def save_image(image, path):
    cv2.imwrite(path, image)


def overlay_png(background, overlay, x, y, overlay_size=None):
    bg = background.copy()
    if overlay is None: return bg
    if overlay_size is not None:
        overlay = cv2.resize(overlay, overlay_size, interpolation=cv2.INTER_AREA)
    if len(overlay.shape) < 3 or overlay.shape[2] < 4: return bg
    if x < 0:
        overlay = overlay[:, abs(x):]
        x = 0
    if y < 0:
        overlay = overlay[abs(y):, :]
        y = 0
    h, w = overlay.shape[:2]
    if x >= bg.shape[1] or y >= bg.shape[0]: return bg
    if x + w > bg.shape[1]:
        overlay = overlay[:, :bg.shape[1] - x]
        w = overlay.shape[1]
    if y + h > bg.shape[0]:
        overlay = overlay[:bg.shape[0] - y, :]
        h = overlay.shape[0]
    overlay_img = overlay[..., :3]
    alpha = overlay[..., 3:] / 255.0
    bg_section = bg[y:y + h, x:x + w]
    blended = (bg_section * (1 - alpha) + overlay_img * alpha).astype(np.uint8)
    bg[y:y + h, x:x + w] = blended
    return bg


def _crop_to_alpha(sticker, threshold=10):
    if sticker is None or sticker.shape[2] < 4: return sticker
    alpha = sticker[..., 3]
    ys, xs = np.where(alpha > threshold)
    if len(xs) == 0: return sticker
    return sticker[ys.min():ys.max() + 1, xs.min():xs.max() + 1]


def _split_at_horizontal_gap(sticker, threshold=10):
    if sticker is None or sticker.shape[2] < 4: return sticker, None
    alpha = sticker[..., 3]
    row_has_content = (alpha > threshold).any(axis=1)
    h = len(row_has_content)
    if h == 0: return sticker, None
    lo, hi = int(h * 0.20), int(h * 0.80)
    gap_start = gap_end = None
    best_len = 0
    cur_start = None
    for i in range(lo, hi):
        if not row_has_content[i]:
            if cur_start is None: cur_start = i
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
    if best_len < int(h * 0.05) or gap_start is None:
        return _crop_to_alpha(sticker), None
    top = _crop_to_alpha(sticker[:gap_start])
    bottom = _crop_to_alpha(sticker[gap_end:])
    return top, bottom


def _alpha_centroid(sticker, threshold=10):
    if sticker is None or sticker.shape[2] < 4: return 0.5, 0.5
    alpha = sticker[..., 3]
    ys, xs = np.where(alpha > threshold)
    if len(xs) == 0: return 0.5, 0.5
    h, w = alpha.shape
    return float(xs.mean()) / w, float(ys.mean()) / h


def _rotate_rgba(sticker, angle_deg):
    if abs(angle_deg) < 0.5 or sticker is None: return sticker
    h, w = sticker.shape[:2]
    cx, cy = w / 2, h / 2
    M = cv2.getRotationMatrix2D((cx, cy), angle_deg, 1.0)
    cos, sin = abs(M[0, 0]), abs(M[0, 1])
    new_w = int(h * sin + w * cos)
    new_h = int(h * cos + w * sin)
    M[0, 2] += new_w / 2 - cx
    M[1, 2] += new_h / 2 - cy
    return cv2.warpAffine(sticker, M, (new_w, new_h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0, 0))


def _place_sticker(img, sticker, anchor_xy, target_width, vy_frac=0.5, vx_frac=0.5, use_centroid=False, rotation_deg=0.0):
    if sticker is None: return img
    cropped = _crop_to_alpha(sticker)
    src_h, src_w = cropped.shape[:2]
    if src_w == 0 or src_h == 0: return img
    target_w = max(int(target_width), 1)
    target_h = max(int(target_w * src_h / src_w), 1)
    if use_centroid:
        cx_norm, cy_norm = _alpha_centroid(cropped)
    else:
        cx_norm, cy_norm = vx_frac, vy_frac
    resized = cv2.resize(cropped, (target_w, target_h), interpolation=cv2.INTER_AREA)
    anchor_local = np.array([target_w * cx_norm, target_h * cy_norm])
    if abs(rotation_deg) >= 0.5:
        rotated = _rotate_rgba(resized, rotation_deg)
        center = np.array([target_w / 2, target_h / 2])
        theta = np.deg2rad(-rotation_deg)
        cos, sin = np.cos(theta), np.sin(theta)
        offset = anchor_local - center
        rotated_offset = np.array([offset[0] * cos - offset[1] * sin, offset[0] * sin + offset[1] * cos])
        rh, rw = rotated.shape[:2]
        anchor_in_rotated = np.array([rw / 2, rh / 2]) + rotated_offset
        x = int(anchor_xy[0] - anchor_in_rotated[0])
        y = int(anchor_xy[1] - anchor_in_rotated[1])
        return overlay_png(img, rotated, x, y, None)
    x = int(anchor_xy[0] - anchor_local[0])
    y = int(anchor_xy[1] - anchor_local[1])
    return overlay_png(img, resized, x, y, None)


def _face_roll_deg(landmarks):
    if not landmarks or len(landmarks) <= max(_LM_LEFT_EYE_OUTER, _LM_RIGHT_EYE_OUTER): return 0.0
    lx, ly = landmarks[_LM_LEFT_EYE_OUTER]
    rx, ry = landmarks[_LM_RIGHT_EYE_OUTER]
    return float(np.degrees(np.arctan2(ry - ly, rx - lx)))

_LM_FOREHEAD = 10
_LM_CHIN = 152
_LM_NOSE_TIP = 4
_LM_NOSE_BRIDGE = 6
_LM_LEFT_CHEEKBONE = 234
_LM_RIGHT_CHEEKBONE = 454
_LM_LEFT_EYE_OUTER = 33
_LM_RIGHT_EYE_OUTER = 263

def _anchor(landmarks, idx, fallback):
    if landmarks and idx < len(landmarks): return landmarks[idx]
    return fallback


def draw_grouped_landmarks(image, grouped_landmarks):
    vis = image.copy()

    colors = {
        "left_eye": (255, 0, 0),
        "right_eye": (0, 255, 0),
        "mouth": (0, 0, 255),
        "jaw": (255, 255, 0),
        "left_eyebrow": (255, 0, 255),
        "right_eyebrow": (0, 255, 255),
    }

    for group_name, points in grouped_landmarks.items():
        color = colors.get(group_name, (255, 255, 255))

        for x, y in points:
            cv2.circle(vis, (int(x), int(y)), 2, color, -1)

    return vis

def apply_sticker_effect(image, effect, landmarks=None):
    img = image.copy()
    h, w = img.shape[:2]
    if effect == "none": return img
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
    sticker_path = {"crown": "crown.png", "cat_ears": "cat_ears.png", "sparkles": "sparkles.png", "freckles": "freckles.png"}.get(effect)
    if sticker_path is None: return img
    sticker = cv2.imread(os.path.join(STICKERS_DIR, sticker_path), cv2.IMREAD_UNCHANGED)
    if sticker is None: return img
    roll = _face_roll_deg(landmarks)
    if effect == "crown":
        anchor = (face_cx, forehead[1])
        img = _place_sticker(img, sticker, anchor, face_scale * 1.50, vy_frac=0.85, rotation_deg=roll)
    elif effect == "cat_ears":
        ears, whiskers = _split_at_horizontal_gap(sticker)
        nose = _anchor(landmarks, _LM_NOSE_TIP, (face_cx, face_cy))
        ear_anchor = (face_cx, forehead[1])
        ear_width = face_scale * 1.10
        if whiskers is not None:
            img = _place_sticker(img, ears, ear_anchor, ear_width, vy_frac=0.95, rotation_deg=roll)
            img = _place_sticker(img, whiskers, nose, face_scale * 1.05, vy_frac=0.5, rotation_deg=roll)
        else:
            img = _place_sticker(img, ears, nose, face_scale * 1.05, vy_frac=0.5, rotation_deg=roll)
    elif effect == "sparkles":
        img = _place_sticker(img, sticker, (face_cx, face_cy), face_scale * 2.20, use_centroid=True, rotation_deg=roll)
    elif effect == "freckles":
        bridge = _anchor(landmarks, _LM_NOSE_BRIDGE, (face_cx, face_cy - face_scale * 0.1))
        theta = np.deg2rad(roll + 90)
        offset_len = face_scale * 0.18
        anchor = (bridge[0] + offset_len * np.cos(theta), bridge[1] + offset_len * np.sin(theta))
        img = _place_sticker(img, sticker, anchor, face_scale * 1.0, use_centroid=True, rotation_deg=roll)
    return img


@app.get("/", response_class=HTMLResponse)
def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


from fastapi.responses import RedirectResponse # En üstteki importlarına bunu eklemeyi unutma!

@app.post("/upload_and_redirect")
async def upload_and_redirect(request: Request, file: UploadFile = File(...)):
    # Rastgele bir ID oluştur
    unique_id = uuid.uuid4().hex
    file_ext = os.path.splitext(file.filename)[1].lower()
    save_path = os.path.join(UPLOAD_DIR, f"{unique_id}{file_ext}")
    
    # Fotoğrafı kaydet
    with open(save_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    
    # Kullanıcıyı fotoğrafın ID'si ile transform sayfasına yolla
    return RedirectResponse(url=f"/transform?image_id={unique_id}&ext={file_ext}", status_code=303)

@app.get("/load_test_image")
def load_test_image(image_name: str):
    src = os.path.join("assets", "test_images", image_name)

    if not os.path.exists(src):
        return RedirectResponse("/", status_code=303)

    unique_id = uuid.uuid4().hex
    ext = os.path.splitext(image_name)[1].lower()

    dst = os.path.join(UPLOAD_DIR, f"{unique_id}{ext}")
    shutil.copy(src, dst)

    return RedirectResponse(
        url=f"/transform?image_id={unique_id}&ext={ext}",
        status_code=303
    )


@app.get("/transform", response_class=HTMLResponse)
def transform_home(
    request: Request,
    image_id: str = None,
    ext: str = ".jpg",
    camera: int = 0
):
    image_url = f"/uploads/{image_id}{ext}" if image_id else None

    return templates.TemplateResponse(
        "transform.html",
        {
            "request": request,
            "image_id": image_id,
            "ext": ext,
            "image_url": image_url,
            "open_camera": camera
        }
    )


@app.post("/transform", response_class=HTMLResponse)
async def transform_image(
    request: Request, file: UploadFile = File(None), camera_data: str = Form(""), mode: str = Form("none"), intensity: float = Form(1.0),
    creative: List[str] = Form(default=[]), glasses_style: str = Form("none"), hair_color_hex: str = Form("#4a2c0a"),
    lip_color_hex: str = Form("#c0392b"), eye_color_hex: str = Form("#1e6db5"), frame_style: str = Form("classic"),
    face_id: str = Form(""), sticker_effect: str = Form("none"),
):
    face_id = face_id if face_id else uuid.uuid4().hex
    uploaded_path = os.path.join(UPLOAD_DIR, f"{face_id}.jpg")

    if camera_data and camera_data.startswith("data:image"):
        try:
            header, encoded = camera_data.split(",", 1)
            data = base64.b64decode(encoded)
            with open(uploaded_path, "wb") as f: f.write(data)
        except Exception as e:
            return templates.TemplateResponse("transform.html", {"request": request, "error_message": f"Kamera verisi çözülemedi: {str(e)}"})
    elif file and file.filename:
        file_ext = os.path.splitext(file.filename)[1].lower()
        uploaded_path = os.path.join(UPLOAD_DIR, f"{face_id}{file_ext}")
        with open(uploaded_path, "wb") as buf: shutil.copyfileobj(file.file, buf)
    else:
        uploaded_path = None
        for ext in ALLOWED_EXTENSIONS.union({".jpg"}):
            p = os.path.join(UPLOAD_DIR, f"{face_id}{ext}")
            if os.path.exists(p): uploaded_path = p; break
        if not uploaded_path: return templates.TemplateResponse("transform.html", {"request": request, "error_message": "Resim dosyası bulunamadı."})

    result = run_preprocessing_pipeline(uploaded_path)
    if not result["success"]: return templates.TemplateResponse("transform.html", {"request": request, "error_message": result["message"]})

    original_image = result["original_image"]
    face_bbox = result["face_bbox"]
    cropped_face = result["cropped_face"]
    resized_face = result["resized_face"]
    grayscale_face = result["grayscale_face"]
    face_image = resized_face if resized_face is not None else cropped_face

    if face_image is None: return templates.TemplateResponse("transform.html", {"request": request, "error_message": "Yüz haritası çıkartılamadı."})
    if len(face_image.shape) == 2: face_image = cv2.cvtColor(face_image, cv2.COLOR_GRAY2BGR)

    detector = FaceLandmarkDetector()
    lm_result = detector.detect(face_image)
    detector.close()

    if not lm_result["success"]: return templates.TemplateResponse("transform.html", {"request": request, "error_message": f"Landmark hatası: {lm_result['message']}"})

    landmarks = lm_result["landmarks"]
    warper = FaceWarper()
    transformed = face_image.copy()

    # Preprocessing Pipeline Çıktı Görsellerini results Klasörüne Hazırlama
    save_image(
    original_image,
    os.path.join(RESULT_DIR, f"{face_id}_original_input.jpg")
)
    save_image(draw_bbox(original_image, face_bbox), os.path.join(RESULT_DIR, f"{face_id}_bbox.jpg"))
    save_image(cropped_face, os.path.join(RESULT_DIR, f"{face_id}_cropped.jpg"))
    save_image(
    resized_face,
    os.path.join(RESULT_DIR, f"{face_id}_resized.jpg")
)
    save_image(grayscale_face, os.path.join(RESULT_DIR, f"{face_id}_grayscale.jpg"))
    
    lm_full_vis = visualize_face_data(face_image, lm_result["bbox"], lm_result["landmarks"], lm_result["grouped_landmarks"], True, True, False)
    lm_grouped_vis = draw_grouped_landmarks(
        face_image,
        lm_result["grouped_landmarks"],
    )

    save_image(
        lm_grouped_vis,
        os.path.join(
            RESULT_DIR,
            f"{face_id}_landmark_grouped.jpg"
        )
    )
    save_image(lm_full_vis, os.path.join(RESULT_DIR, f"{face_id}_landmark_full.jpg"))

    if mode in ["smile", "eyebrow_raise", "lip_widen", "face_slim", "nose_enhance"]:
        transformed = warper.warp_expression(transformed, landmarks, mode, intensity=max(0.1, min(intensity, 2.0)))
    elif mode in ["aging", "deaging"]:
        transformed = aging_deaging_pipeline(transformed, mode=mode, intensity=max(0.0, min(intensity, 2.0)) / 2.0, landmarks=landmarks)

    needs_creative = bool(creative) or glasses_style != "none" or sticker_effect != "none"
    if needs_creative:
        detector2 = FaceLandmarkDetector()
        lm2 = detector2.detect(transformed); detector2.close()
        lm2_pts = lm2["landmarks"] if lm2["success"] else landmarks

        if "lip_color" in creative: transformed = warper.apply_lip_color(transformed, lm2_pts, color=_hex_to_bgr(lip_color_hex))
        if glasses_style != "none": transformed = warper.apply_glasses(transformed, lm2_pts, style=glasses_style, assets_dir=GLASSES_DIR)
        if "eye_color" in creative: transformed = warper.apply_eye_color(transformed, lm2_pts, color=_hex_to_bgr(eye_color_hex))

    result_uid = uuid.uuid4().hex
    x, y, w, h = face_bbox
    final_output = original_image.copy()

    nothing_selected = (mode == "none" and not creative and glasses_style == "none" and sticker_effect == "none")
    if not nothing_selected: final_output[y:y+h, x:x+w] = cv2.resize(transformed, (w, h))

    if "hair_color" in creative or sticker_effect != "none":
        cropped_h, cropped_w = face_image.shape[:2]
        translated_lms = [(x + int(lx * w / max(cropped_w, 1)), y + int(ly * h / max(cropped_h, 1))) for (lx, ly) in (lm2_pts if needs_creative else landmarks)]
        if "hair_color" in creative: final_output = warper.apply_hair_color(final_output, translated_lms, color=_hex_to_bgr(hair_color_hex))
        if sticker_effect != "none": final_output = apply_sticker_effect(final_output, sticker_effect, translated_lms)

    analysis_transformed = face_image if nothing_selected else cv2.resize(final_output[y:y+h, x:x+w], (face_image.shape[1], face_image.shape[0]))
    if "frame_photo" in creative: final_output = warper.apply_frame_photo(final_output, None, style=frame_style)

    save_image(original_image, os.path.join(RESULT_DIR, f"{result_uid}_orig.jpg"))
    save_image(final_output, os.path.join(RESULT_DIR, f"{result_uid}_out.jpg"))

    freq_result = run_frequency_analysis(face_image, analysis_transformed)
    save_image(freq_result["original_spectrum"], os.path.join(RESULT_DIR, f"{result_uid}_orig_spectrum.jpg"))
    save_image(freq_result["processed_spectrum"], os.path.join(RESULT_DIR, f"{result_uid}_proc_spectrum.jpg"))

    evaluation_metrics = run_evaluation(face_image, analysis_transformed)
    evaluation_table_html = f"""
    <table><tr><th>Metric</th><th>Value</th><th>Meaning</th></tr>
    <tr><td>MSE</td><td>{evaluation_metrics['mse']:.4f}</td><td>Pixel-level difference</td></tr>
    <tr><td>PSNR</td><td>{evaluation_metrics['psnr']:.4f} dB</td><td>Signal quality</td></tr>
    <tr><td>SSIM</td><td>{evaluation_metrics['ssim']:.4f}</td><td>Structural similarity</td></tr></table>
    """

    export_data = {
        "mse": evaluation_metrics["mse"], "psnr": evaluation_metrics["psnr"], "ssim": evaluation_metrics["ssim"],
        "total_energy_original": freq_result["original_energy"]["total_energy"], "total_energy_processed": freq_result["processed_energy"]["total_energy"],
        "low_frequency_original": freq_result["original_energy"]["low_frequency_energy"], "low_frequency_processed": freq_result["processed_energy"]["low_frequency_energy"],
        "high_frequency_original": freq_result["original_energy"]["high_frequency_energy"], "high_frequency_processed": freq_result["processed_energy"]["high_frequency_energy"],
        "high_low_ratio_original": freq_result["original_energy"]["high_low_ratio"], "high_low_ratio_processed": freq_result["processed_energy"]["high_low_ratio"],
    }
    csv_path, pdf_path = os.path.join(RESULT_DIR, f"{result_uid}_report.csv"), os.path.join(RESULT_DIR, f"{result_uid}_report.pdf")
    export_metrics_to_csv(export_data, csv_path); export_metrics_to_pdf(export_data, pdf_path)

    pipeline_images = {
    "original_input": f"/results/{face_id}_original_input.jpg",
    "bbox": f"/results/{face_id}_bbox.jpg",
    "cropped": f"/results/{face_id}_cropped.jpg",
    "resized": f"/results/{face_id}_resized.jpg",
    "grayscale": f"/results/{face_id}_grayscale.jpg",
    "landmark_full": f"/results/{face_id}_landmark_full.jpg",
    "landmark_grouped": f"/results/{face_id}_landmark_grouped.jpg",
}

    return templates.TemplateResponse(
        "transform.html", {
            "request": request, "original_image": f"/results/{result_uid}_orig.jpg", "transformed_image": f"/results/{result_uid}_out.jpg",
            "orig_spectrum": f"/results/{result_uid}_orig_spectrum.jpg", "proc_spectrum": f"/results/{result_uid}_proc_spectrum.jpg",
            "frequency_table": freq_result["html_table"], "evaluation_table": evaluation_table_html, "mse": round(evaluation_metrics["mse"], 4),
            "psnr": round(evaluation_metrics["psnr"], 4), "ssim": round(evaluation_metrics["ssim"], 4),
            "ratio": "{:.4f} (Δ {:+.4f})".format(freq_result["processed_energy"]["high_low_ratio"], freq_result["difference"]["ratio_difference"]),
            "applied_mode": mode, "applied_intensity": round(intensity, 2), "csv_report": f"/results/{result_uid}_report.csv", "pdf_report": f"/results/{result_uid}_report.pdf", "image_id": face_id,
            "images": pipeline_images, "face_bbox": str(face_bbox), "landmark_count": len(landmarks)
        }
    )


@app.get("/download-gray")
def download_gray_level_image(image_path: str):
    clean_path = image_path.lstrip("/")
    if not os.path.exists(clean_path): return {"error": "Resim bulunamadı."}
    img_bgr = cv2.imread(clean_path)
    if img_bgr is None: return {"error": "Dosya okuma hatası."}
    img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    base, ext = os.path.splitext(clean_path)
    gray_filename = f"{base}_gray_level.jpg"
    cv2.imwrite(gray_filename, img_gray)
    return FileResponse(path=gray_filename, filename="beauty_dsp_gray_output.jpg", media_type="image/jpeg")