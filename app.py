import json
import os
import shutil
import uuid
from typing import List

import cv2
from fastapi import FastAPI, File, Form, UploadFile
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles

from modules.face_landmark import FaceLandmarkDetector
from modules.preprocessing import run_preprocessing_pipeline
from modules.visualization import visualize_face_data
from modules.warping import FaceWarper

app = FastAPI(title="Face Preprocessing Demo")

UPLOAD_DIR = "uploads"
RESULT_DIR = "results"
GLASSES_DIR = "assets/glasses"

os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(RESULT_DIR, exist_ok=True)
os.makedirs(GLASSES_DIR, exist_ok=True)

# Generate glasses PNGs on startup if they don't exist yet
FaceWarper.ensure_glasses_assets(GLASSES_DIR)

app.mount("/uploads", StaticFiles(directory="uploads"), name="uploads")
app.mount("/results", StaticFiles(directory="results"), name="results")
app.mount("/assets",  StaticFiles(directory="assets"),  name="assets")

ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png"}


def draw_bbox(image, bbox, color=(0, 255, 0), thickness=2):
    image_copy = image.copy()
    x, y, w, h = bbox
    cv2.rectangle(image_copy, (x, y), (x + w, y + h), color, thickness)
    return image_copy


def save_image(image, path):
    cv2.imwrite(path, image)


def render_home_page(
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
    result_section = ""

    if images:
        grouped_landmark_keys = grouped_landmark_keys or []

        result_section = f"""
        <div class="result">
            <p class="{'success' if success else 'fail'}">Success: {success}</p>
            <p><strong>Message:</strong> {message}</p>
            <p><strong>Face BBox:</strong> {face_bbox}</p>
            <p><strong>Metadata:</strong> {metadata}</p>

            <hr style="margin: 18px 0;">

            <p><strong>Landmark Success:</strong> {landmark_success}</p>
            <p><strong>Landmark Count:</strong> {landmark_count}</p>
            <p><strong>Landmark Message:</strong> {landmark_message}</p>
            <p><strong>Grouped Landmark Keys:</strong> {grouped_landmark_keys}</p>

            <div class="grid">
                <div class="card">
                    <h3>Original Image</h3>
                    <img src="{images['original']}" alt="Original Image">
                </div>

                <div class="card">
                    <h3>Detected Face Bounding Box</h3>
                    <img src="{images['bbox']}" alt="Bounding Box Image">
                </div>

                <div class="card">
                    <h3>Cropped Face</h3>
                    <img src="{images['cropped']}" alt="Cropped Face">
                </div>

                <div class="card">
                    <h3>Resized Face</h3>
                    <img src="{images['resized']}" alt="Resized Face">
                </div>

                <div class="card">
                    <h3>Grayscale Face</h3>
                    <img src="{images['grayscale']}" alt="Grayscale Face">
                </div>

                <div class="card">
                    <h3>Landmarks (Full)</h3>
                    <img src="{images['landmark_full']}" alt="Full Landmark Image">
                </div>

                <div class="card">
                    <h3>Landmarks (Grouped)</h3>
                    <img src="{images['landmark_grouped']}" alt="Grouped Landmark Image">
                </div>
            </div>

            <div class="card" style="margin-top: 20px;">
                <h3>Landmark JSON Preview</h3>
                <pre>{landmark_json}</pre>
            </div>
        </div>
        """
    elif message:
        result_section = f"""
        <div class="result">
            <p class="{'success' if success else 'fail'}">Success: {success}</p>
            <p><strong>Message:</strong> {message}</p>
        </div>
        """

    html = f"""
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8" />
        <meta name="viewport" content="width=device-width, initial-scale=1" />
        <title>Face Preprocessing Demo</title>
        <style>
            body {{
                font-family: Arial, sans-serif;
                margin: 40px;
                background: #f7f7f7;
            }}
            .container {{
                max-width: 1200px;
                margin: auto;
                background: white;
                padding: 24px;
                border-radius: 12px;
                box-shadow: 0 2px 10px rgba(0,0,0,0.08);
            }}
            h1 {{
                margin-top: 0;
            }}
            .result {{
                margin-top: 24px;
                padding: 16px;
                border-radius: 10px;
                background: #f2f2f2;
            }}
            .grid {{
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(280px, 1fr));
                gap: 16px;
                margin-top: 20px;
            }}
            .card {{
                background: white;
                padding: 14px;
                border-radius: 10px;
                box-shadow: 0 1px 6px rgba(0,0,0,0.08);
            }}
            img {{
                margin-top: 10px;
                width: 100%;
                border-radius: 10px;
            }}
            .success {{
                color: green;
                font-weight: bold;
            }}
            .fail {{
                color: red;
                font-weight: bold;
            }}
            button {{
                padding: 10px 16px;
                border: none;
                border-radius: 8px;
                background: #222;
                color: white;
                cursor: pointer;
            }}
            input[type="file"] {{
                margin-bottom: 12px;
            }}
            pre {{
                white-space: pre-wrap;
                word-break: break-word;
                background: #fafafa;
                padding: 12px;
                border-radius: 8px;
                overflow-x: auto;
            }}
        </style>
    </head>
    <body>
        <div class="container">
            <h1>Face Preprocessing + Landmark Test Page</h1>
            <p>Upload a JPG or PNG image to test preprocessing and landmark detection.</p>

            <form action="/upload" method="post" enctype="multipart/form-data">
                <input type="file" name="file" accept=".jpg,.jpeg,.png" required />
                <br />
                <button type="submit">Upload and Test</button>
            </form>

            <div style="margin-top:24px; padding:16px; background:#eef4ff; border-radius:10px; border:1px solid #c0d8f8;">
                <strong>&#127774; Expression &amp; Creative Transforms</strong>
                <a href="/transform{f'?image_id={face_id}' if face_id else ''}" style="margin-left:12px; padding:8px 14px; background:#3366cc; color:white;
                   border-radius:8px; text-decoration:none; font-size:13px;">Open Transform Page &rarr;</a>
            </div>

            {result_section}
        </div>
    </body>
    </html>
    """
    return html


@app.get("/", response_class=HTMLResponse)
def home():
    return render_home_page()


@app.post("/upload", response_class=HTMLResponse)
async def upload_image(file: UploadFile = File(...)):
    if not file.filename:
        return render_home_page(
            message="No file selected.",
            success=False
        )

    file_ext = os.path.splitext(file.filename)[1].lower()

    if file_ext not in ALLOWED_EXTENSIONS:
        return render_home_page(
            message="Unsupported file format. Please upload JPG, JPEG, or PNG.",
            success=False
        )

    unique_id = uuid.uuid4().hex
    uploaded_filename = f"{unique_id}{file_ext}"
    uploaded_path = os.path.join(UPLOAD_DIR, uploaded_filename)

    with open(uploaded_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    result = run_preprocessing_pipeline(uploaded_path)

    if not result["success"]:
        return render_home_page(
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

    # Landmark detection için kullanılacak görsel
    face_image = resized_face if resized_face is not None else cropped_face

    landmark_success = False
    landmark_count = 0
    grouped_landmark_keys = []
    landmark_message = "Landmark detection was not run."
    landmark_json_text = "No landmark data."
    landmark_full_image = face_image.copy() if face_image is not None else original_image.copy()
    landmark_grouped_image = face_image.copy() if face_image is not None else original_image.copy()

    if face_image is not None:
        if len(face_image.shape) == 2:
            face_image = cv2.cvtColor(face_image, cv2.COLOR_GRAY2BGR)

        detector = FaceLandmarkDetector()
        landmark_result = detector.detect(face_image)

        landmark_success = landmark_result["success"]
        landmark_count = len(landmark_result["landmarks"])
        grouped_landmark_keys = list(landmark_result["grouped_landmarks"].keys())
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
            landmark_json_text = json.dumps(preview_data, indent=4, ensure_ascii=False)

        detector.close()

    original_result_path = os.path.join(RESULT_DIR, f"{unique_id}_original.jpg")
    bbox_result_path = os.path.join(RESULT_DIR, f"{unique_id}_bbox.jpg")
    cropped_result_path = os.path.join(RESULT_DIR, f"{unique_id}_cropped.jpg")
    resized_result_path = os.path.join(RESULT_DIR, f"{unique_id}_resized.jpg")
    grayscale_result_path = os.path.join(RESULT_DIR, f"{unique_id}_grayscale.jpg")
    landmark_full_result_path = os.path.join(RESULT_DIR, f"{unique_id}_landmark_full.jpg")
    landmark_grouped_result_path = os.path.join(RESULT_DIR, f"{unique_id}_landmark_grouped.jpg")

    save_image(original_image, original_result_path)
    save_image(bbox_image, bbox_result_path)
    save_image(cropped_face, cropped_result_path)
    save_image(resized_face, resized_result_path)
    save_image(grayscale_face, grayscale_result_path)
    save_image(landmark_full_image, landmark_full_result_path)
    save_image(landmark_grouped_image, landmark_grouped_result_path)
    
    if face_image is not None:
        save_image(face_image, os.path.join(UPLOAD_DIR, f"{unique_id}_face.jpg"))

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
    return {"message": "API is running"}


# ---------------------------------------------------------------------------
# Transform page helpers
# ---------------------------------------------------------------------------

def _hex_to_bgr(hex_color: str):
    """Convert #rrggbb string to (B, G, R) tuple for OpenCV."""
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
}

CREATIVE_OPTIONS = {
    "lip_color": "Lip Coloring",
    "glasses": "Glasses Overlay",
    "hair_color": "Hair Coloring",
    "eye_color": "Eye Color",
    "frame_photo": "Photo Frame",
}


def render_transform_page(result_section="", face_id="", face_thumb_url=""):
    # If a face is already loaded, show a banner + hidden ID instead of file upload
    if face_id and face_thumb_url:
        upload_section = f"""
        <div style="display:flex;align-items:center;gap:14px;padding:10px 14px;
                    background:#eef6ee;border:1px solid #a8d5a2;border-radius:10px;margin-bottom:14px;">
          <img src="{face_thumb_url}" style="width:60px;height:60px;object-fit:cover;
               border-radius:8px;margin:0;">
          <div>
            <strong>Image loaded</strong> — transform as many times as you like without re-uploading.<br>
            <a href="/transform" style="font-size:12px;color:#555;">&#10005; Clear and upload a new image</a>
          </div>
          <input type="hidden" name="face_id" value="{face_id}">
        </div>
        """
        file_input = ""
    else:
        upload_section = ""
        file_input = """
        <div style="margin-bottom:14px;">
          <label><strong>Upload Image:</strong></label><br>
          <input type="file" name="file" accept=".jpg,.jpeg,.png" required style="margin-top:6px;">
        </div>
        """

    form_html = f"""
    <h2 style="margin-top:32px;">Expression &amp; Creative Transform</h2>
    <form action="/transform" method="post" enctype="multipart/form-data">
      {upload_section}
      {file_input}

      <div style="margin-bottom:14px;">
        <label><strong>Expression Mode:</strong></label><br>
        <select name="mode" style="padding:6px 12px;border-radius:6px;margin-top:6px;font-size:14px;">
          <option value="none">None</option>
          <option value="smile">Smile Enhancement</option>
          <option value="eyebrow_raise">Eyebrow Raise</option>
          <option value="lip_widen">Lip Widening</option>
          <option value="face_slim">Face Slimming</option>
        </select>
      </div>

      <div style="margin-bottom:14px;">
        <label><strong>Intensity:</strong> <span id="intensityVal">1.0</span></label><br>
        <input type="range" name="intensity" min="0.1" max="2.0" step="0.1" value="1.0"
               oninput="document.getElementById('intensityVal').textContent=parseFloat(this.value).toFixed(1)"
               style="width:220px;margin-top:6px;">
      </div>

      <hr style="margin:18px 0;border:none;border-top:1px solid #ddd;">

      <!-- Glasses selector with preview thumbnails -->
      <div style="margin-bottom:14px;">
        <label><strong>Glasses Style:</strong></label><br>
        <div style="display:flex;gap:12px;flex-wrap:wrap;margin-top:8px;" id="glassesGroup">
          <label class="glasses-opt" style="cursor:pointer;text-align:center;">
            <input type="radio" name="glasses_style" value="none" checked
                   onchange="selectGlasses(this)" style="display:none;">
            <div class="glasses-card selected" id="g-none"
                 style="border:3px solid #333;border-radius:10px;padding:6px;background:#f0f0f0;width:110px;">
              <div style="height:60px;display:flex;align-items:center;justify-content:center;
                          font-size:26px;">&#10006;</div>
              <div style="font-size:12px;margin-top:4px;">None</div>
            </div>
          </label>
          <label class="glasses-opt" style="cursor:pointer;text-align:center;">
            <input type="radio" name="glasses_style" value="round"
                   onchange="selectGlasses(this)" style="display:none;">
            <div class="glasses-card" id="g-round"
                 style="border:3px solid #ccc;border-radius:10px;padding:6px;background:#fff;width:110px;">
              <img src="/assets/glasses/round.png" style="width:100%;height:60px;object-fit:contain;margin:0;">
              <div style="font-size:12px;margin-top:4px;">Round</div>
            </div>
          </label>
          <label class="glasses-opt" style="cursor:pointer;text-align:center;">
            <input type="radio" name="glasses_style" value="square"
                   onchange="selectGlasses(this)" style="display:none;">
            <div class="glasses-card" id="g-square"
                 style="border:3px solid #ccc;border-radius:10px;padding:6px;background:#fff;width:110px;">
              <img src="/assets/glasses/square.png" style="width:100%;height:60px;object-fit:contain;margin:0;">
              <div style="font-size:12px;margin-top:4px;">Square</div>
            </div>
          </label>
          <label class="glasses-opt" style="cursor:pointer;text-align:center;">
            <input type="radio" name="glasses_style" value="aviator"
                   onchange="selectGlasses(this)" style="display:none;">
            <div class="glasses-card" id="g-aviator"
                 style="border:3px solid #ccc;border-radius:10px;padding:6px;background:#fff;width:110px;">
              <img src="/assets/glasses/aviator.png" style="width:100%;height:60px;object-fit:contain;margin:0;">
              <div style="font-size:12px;margin-top:4px;">Aviator</div>
            </div>
          </label>
          <label class="glasses-opt" style="cursor:pointer;text-align:center;">
            <input type="radio" name="glasses_style" value="cateye"
                   onchange="selectGlasses(this)" style="display:none;">
            <div class="glasses-card" id="g-cateye"
                 style="border:3px solid #ccc;border-radius:10px;padding:6px;background:#fff;width:110px;">
              <img src="/assets/glasses/cateye.png" style="width:100%;height:60px;object-fit:contain;margin:0;">
              <div style="font-size:12px;margin-top:4px;">Cat-eye</div>
            </div>
          </label>
        </div>
      </div>

      <hr style="margin:18px 0;border:none;border-top:1px solid #ddd;">

      <!-- Creative features -->
      <div style="margin-bottom:14px;">
        <label><strong>Creative Features:</strong></label><br>
        <div style="display:flex;gap:18px;flex-wrap:wrap;margin-top:8px;">
          <label><input type="checkbox" name="creative" value="lip_color" onchange="toggleSection('lipColorSection', this.checked)"> Lip Coloring</label>
          <label><input type="checkbox" name="creative" value="hair_color" onchange="toggleSection('hairColorSection', this.checked)"> Hair Coloring</label>
          <label><input type="checkbox" name="creative" value="eye_color" onchange="toggleSection('eyeColorSection', this.checked)"> Eye Color</label>
          <label><input type="checkbox" name="creative" value="frame_photo" onchange="toggleSection('frameStyleSection', this.checked)"> Photo Frame</label>
        </div>
      </div>

      <!-- Lip colour picker (hidden until Lip Coloring is checked) -->
      <div style="margin-bottom:20px;display:none;" id="lipColorSection">
        <label><strong>Lip Color:</strong></label><br>
        <div style="display:flex;align-items:center;gap:10px;flex-wrap:wrap;margin-top:8px;">
          <input type="color" name="lip_color_hex" id="lipColorPicker" value="#c0392b"
                 style="width:44px;height:36px;border:none;cursor:pointer;border-radius:6px;">
          <span style="font-size:13px;color:#555;">— or pick a preset:</span>
          <div style="display:flex;gap:6px;flex-wrap:wrap;">
            <button type="button" onclick="setLipColor('#c0392b')"
                style="background:#c0392b;width:32px;height:32px;border-radius:50%;border:2px solid #999;cursor:pointer;"
                title="Classic Red"></button>
            <button type="button" onclick="setLipColor('#e74c6f')"
                style="background:#e74c6f;width:32px;height:32px;border-radius:50%;border:2px solid #999;cursor:pointer;"
                title="Rose Pink"></button>
            <button type="button" onclick="setLipColor('#8e3a59')"
                style="background:#8e3a59;width:32px;height:32px;border-radius:50%;border:2px solid #999;cursor:pointer;"
                title="Berry"></button>
            <button type="button" onclick="setLipColor('#d4736a')"
                style="background:#d4736a;width:32px;height:32px;border-radius:50%;border:2px solid #999;cursor:pointer;"
                title="Coral"></button>
            <button type="button" onclick="setLipColor('#a0522d')"
                style="background:#a0522d;width:32px;height:32px;border-radius:50%;border:2px solid #999;cursor:pointer;"
                title="Nude Brown"></button>
            <button type="button" onclick="setLipColor('#6b1e3a')"
                style="background:#6b1e3a;width:32px;height:32px;border-radius:50%;border:2px solid #999;cursor:pointer;"
                title="Plum"></button>
            <button type="button" onclick="setLipColor('#ff4500')"
                style="background:#ff4500;width:32px;height:32px;border-radius:50%;border:2px solid #999;cursor:pointer;"
                title="Orange Red"></button>
            <button type="button" onclick="setLipColor('#2d0a1e')"
                style="background:#2d0a1e;width:32px;height:32px;border-radius:50%;border:2px solid #999;cursor:pointer;"
                title="Dark Wine"></button>
          </div>
        </div>
      </div>

      <!-- Hair colour picker (hidden until Hair Coloring is checked) -->
      <div style="margin-bottom:20px;display:none;" id="hairColorSection">
        <label><strong>Hair Color:</strong></label><br>
        <div style="display:flex;align-items:center;gap:10px;flex-wrap:wrap;margin-top:8px;">
          <input type="color" name="hair_color_hex" id="hairColorPicker" value="#4a2c0a"
                 style="width:44px;height:36px;border:none;cursor:pointer;border-radius:6px;">
          <span style="font-size:13px;color:#555;">— or pick a preset:</span>
          <div style="display:flex;gap:6px;flex-wrap:wrap;">
            <button type="button" onclick="setHairColor('#0d0d0d')"
                style="background:#0d0d0d;width:32px;height:32px;border-radius:50%;border:2px solid #999;cursor:pointer;"
                title="Black"></button>
            <button type="button" onclick="setHairColor('#3b1f0a')"
                style="background:#3b1f0a;width:32px;height:32px;border-radius:50%;border:2px solid #999;cursor:pointer;"
                title="Dark Brown"></button>
            <button type="button" onclick="setHairColor('#7b4a1e')"
                style="background:#7b4a1e;width:32px;height:32px;border-radius:50%;border:2px solid #999;cursor:pointer;"
                title="Brown"></button>
            <button type="button" onclick="setHairColor('#c9a84c')"
                style="background:#c9a84c;width:32px;height:32px;border-radius:50%;border:2px solid #999;cursor:pointer;"
                title="Blonde"></button>
            <button type="button" onclick="setHairColor('#c0392b')"
                style="background:#c0392b;width:32px;height:32px;border-radius:50%;border:2px solid #999;cursor:pointer;"
                title="Red"></button>
            <button type="button" onclick="setHairColor('#1a237e')"
                style="background:#1a237e;width:32px;height:32px;border-radius:50%;border:2px solid #999;cursor:pointer;"
                title="Dark Blue"></button>
            <button type="button" onclick="setHairColor('#6a0dad')"
                style="background:#6a0dad;width:32px;height:32px;border-radius:50%;border:2px solid #999;cursor:pointer;"
                title="Purple"></button>
            <button type="button" onclick="setHairColor('#e91e8c')"
                style="background:#e91e8c;width:32px;height:32px;border-radius:50%;border:2px solid #999;cursor:pointer;"
                title="Pink"></button>
          </div>
        </div>
      </div>

      <!-- Eye color picker (hidden until Eye Color is checked) -->
      <div style="margin-bottom:20px;display:none;" id="eyeColorSection">
        <label><strong>Eye Color:</strong></label><br>
        <div style="display:flex;align-items:center;gap:10px;flex-wrap:wrap;margin-top:8px;">
          <input type="color" name="eye_color_hex" id="eyeColorPicker" value="#1e6db5"
                 style="width:44px;height:36px;border:none;cursor:pointer;border-radius:6px;">
          <span style="font-size:13px;color:#555;">— or pick a preset:</span>
          <div style="display:flex;gap:6px;flex-wrap:wrap;">
            <button type="button" onclick="setEyeColor('#1e6db5')"
                style="background:#1e6db5;width:32px;height:32px;border-radius:50%;border:2px solid #999;cursor:pointer;"
                title="Blue"></button>
            <button type="button" onclick="setEyeColor('#3a7d3a')"
                style="background:#3a7d3a;width:32px;height:32px;border-radius:50%;border:2px solid #999;cursor:pointer;"
                title="Green"></button>
            <button type="button" onclick="setEyeColor('#7a4a1f')"
                style="background:#7a4a1f;width:32px;height:32px;border-radius:50%;border:2px solid #999;cursor:pointer;"
                title="Brown"></button>
            <button type="button" onclick="setEyeColor('#9c8b6e')"
                style="background:#9c8b6e;width:32px;height:32px;border-radius:50%;border:2px solid #999;cursor:pointer;"
                title="Hazel"></button>
            <button type="button" onclick="setEyeColor('#7a8a99')"
                style="background:#7a8a99;width:32px;height:32px;border-radius:50%;border:2px solid #999;cursor:pointer;"
                title="Gray"></button>
            <button type="button" onclick="setEyeColor('#5a3a8a')"
                style="background:#5a3a8a;width:32px;height:32px;border-radius:50%;border:2px solid #999;cursor:pointer;"
                title="Violet"></button>
            <button type="button" onclick="setEyeColor('#0fa3a3')"
                style="background:#0fa3a3;width:32px;height:32px;border-radius:50%;border:2px solid #999;cursor:pointer;"
                title="Teal"></button>
            <button type="button" onclick="setEyeColor('#b29547')"
                style="background:#b29547;width:32px;height:32px;border-radius:50%;border:2px solid #999;cursor:pointer;"
                title="Amber"></button>
          </div>
        </div>
      </div>

      <!-- Frame style selector (hidden until Photo Frame is checked) -->
      <div style="margin-bottom:20px;display:none;" id="frameStyleSection">
        <label><strong>Frame Style:</strong></label><br>
        <div style="display:flex;gap:12px;flex-wrap:wrap;margin-top:8px;" id="frameGroup">
          <label style="cursor:pointer;text-align:center;">
            <input type="radio" name="frame_style" value="classic" checked
                   onchange="selectFrame(this)" style="display:none;">
            <div class="frame-card selected" id="f-classic"
                 style="border:3px solid #333;border-radius:10px;padding:10px;background:#f0f0f0;width:90px;">
              <div style="height:50px;display:flex;align-items:center;justify-content:center;
                          font-size:22px;">&#127912;</div>
              <div style="font-size:12px;margin-top:4px;">Classic</div>
            </div>
          </label>
          <label style="cursor:pointer;text-align:center;">
            <input type="radio" name="frame_style" value="modern"
                   onchange="selectFrame(this)" style="display:none;">
            <div class="frame-card" id="f-modern"
                 style="border:3px solid #ccc;border-radius:10px;padding:10px;background:#fff;width:90px;">
              <div style="height:50px;display:flex;align-items:center;justify-content:center;
                          font-size:22px;">&#9724;</div>
              <div style="font-size:12px;margin-top:4px;">Modern</div>
            </div>
          </label>
          <label style="cursor:pointer;text-align:center;">
            <input type="radio" name="frame_style" value="polaroid"
                   onchange="selectFrame(this)" style="display:none;">
            <div class="frame-card" id="f-polaroid"
                 style="border:3px solid #ccc;border-radius:10px;padding:10px;background:#fff;width:90px;">
              <div style="height:50px;display:flex;align-items:center;justify-content:center;
                          font-size:22px;">&#128247;</div>
              <div style="font-size:12px;margin-top:4px;">Polaroid</div>
            </div>
          </label>
          <label style="cursor:pointer;text-align:center;">
            <input type="radio" name="frame_style" value="vintage"
                   onchange="selectFrame(this)" style="display:none;">
            <div class="frame-card" id="f-vintage"
                 style="border:3px solid #ccc;border-radius:10px;padding:10px;background:#fff;width:90px;">
              <div style="height:50px;display:flex;align-items:center;justify-content:center;
                          font-size:22px;">&#128444;</div>
              <div style="font-size:12px;margin-top:4px;">Vintage</div>
            </div>
          </label>
        </div>
      </div>

      <button type="submit" style="padding:12px 28px;font-size:15px;">Transform Face</button>
    </form>

    <script>
      function setHairColor(hex) {{
        document.getElementById('hairColorPicker').value = hex;
      }}
      function setLipColor(hex) {{
        document.getElementById('lipColorPicker').value = hex;
      }}
      function setEyeColor(hex) {{
        document.getElementById('eyeColorPicker').value = hex;
      }}
      function selectGlasses(radio) {{
        document.querySelectorAll('.glasses-card').forEach(el => {{
          el.style.border = '3px solid #ccc';
          el.style.background = '#fff';
        }});
        var card = document.getElementById('g-' + radio.value);
        if (card) {{
          card.style.border = '3px solid #333';
          card.style.background = '#f0f0f0';
        }}
      }}
      function selectFrame(radio) {{
        document.querySelectorAll('.frame-card').forEach(el => {{
          el.style.border = '3px solid #ccc';
          el.style.background = '#fff';
        }});
        var card = document.getElementById('f-' + radio.value);
        if (card) {{
          card.style.border = '3px solid #333';
          card.style.background = '#f0f0f0';
        }}
      }}
      function toggleSection(id, show) {{
        document.getElementById(id).style.display = show ? 'block' : 'none';
      }}
    </script>
    """

    html = f"""
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8" />
        <meta name="viewport" content="width=device-width, initial-scale=1" />
        <title>Face Transform</title>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 40px; background: #f7f7f7; }}
            .container {{
                max-width: 1200px; margin: auto; background: white;
                padding: 24px; border-radius: 12px;
                box-shadow: 0 2px 10px rgba(0,0,0,0.08);
            }}
            h1, h2 {{ margin-top: 0; }}
            .result {{ margin-top: 24px; padding: 16px; border-radius: 10px; background: #f2f2f2; }}
            .grid {{
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(320px, 1fr));
                gap: 16px; margin-top: 20px;
            }}
            .card {{
                background: white; padding: 14px; border-radius: 10px;
                box-shadow: 0 1px 6px rgba(0,0,0,0.08);
            }}
            img {{ margin-top: 10px; width: 100%; border-radius: 10px; }}
            .success {{ color: green; font-weight: bold; }}
            .fail {{ color: red; font-weight: bold; }}
            button {{
                padding: 10px 20px; border: none; border-radius: 8px;
                background: #444; color: white; cursor: pointer; font-size: 14px;
            }}
            button:hover {{ background: #222; }}
            a.back {{ display:inline-block; margin-bottom:16px; color:#555; text-decoration:none; }}
            a.back:hover {{ color:#000; }}
        </style>
    </head>
    <body>
        <div class="container">
            <a class="back" href="/">&#8592; Back to Preprocessing</a>
            <h1>Face Expression &amp; Creative Transform</h1>
            <p>Upload a face image, pick an expression mode and optional creative effects, then click Transform.</p>
            {form_html}
            {result_section}
        </div>
    </body>
    </html>
    """
    return html


@app.get("/transform", response_class=HTMLResponse)
def transform_home(image_id: str = None):
    if image_id:
        cropped_path = os.path.join(RESULT_DIR, f"{image_id}_cropped.jpg")
        if os.path.exists(cropped_path):
            face_thumb_url = f"/results/{image_id}_cropped.jpg"
            return render_transform_page(face_id=image_id, face_thumb_url=face_thumb_url)
    return render_transform_page()


@app.post("/transform", response_class=HTMLResponse)
async def transform_image(
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
):
    # --- Resolve face image: reuse stored one or preprocess new upload ---
    face_stored_path = os.path.join(UPLOAD_DIR, f"{face_id}_face.jpg") if face_id else ""

    if face_id and not os.path.exists(face_stored_path):
        fallback_path = os.path.join(RESULT_DIR, f"{face_id}_resized.jpg")
        if not os.path.exists(fallback_path):
            fallback_path = os.path.join(RESULT_DIR, f"{face_id}_cropped.jpg")
        if os.path.exists(fallback_path):
            face_stored_path = fallback_path

    if face_id and os.path.exists(face_stored_path):
        # Reuse previously preprocessed face — no upload needed
        face_image = cv2.imread(face_stored_path)
        if face_image is None:
            face_id = ""  # fall through to upload path
    else:
        face_id = ""

    if not face_id:
        if file is None or not file.filename:
            return render_transform_page(
                '<div class="result"><p class="fail">No file selected.</p></div>'
            )
        file_ext = os.path.splitext(file.filename)[1].lower()
        if file_ext not in ALLOWED_EXTENSIONS:
            return render_transform_page(
                '<div class="result"><p class="fail">Unsupported format. Use JPG or PNG.</p></div>'
            )

        face_id = uuid.uuid4().hex
        uploaded_path = os.path.join(UPLOAD_DIR, f"{face_id}{file_ext}")
        with open(uploaded_path, "wb") as buf:
            shutil.copyfileobj(file.file, buf)

        result = run_preprocessing_pipeline(uploaded_path)
        if not result["success"]:
            return render_transform_page(
                f'<div class="result"><p class="fail">{result["message"]}</p></div>'
            )

        face_image = result.get("resized_face")
        if face_image is None:
            face_image = result.get("cropped_face")

        if face_image is None:
            return render_transform_page(
                '<div class="result"><p class="fail">Could not extract face region.</p></div>'
            )

        if len(face_image.shape) == 2:
            face_image = cv2.cvtColor(face_image, cv2.COLOR_GRAY2BGR)

        # Persist preprocessed face for reuse on subsequent transforms
        face_stored_path = os.path.join(UPLOAD_DIR, f"{face_id}_face.jpg")
        save_image(face_image, face_stored_path)

    detector = FaceLandmarkDetector()
    lm_result = detector.detect(face_image)
    detector.close()

    if not lm_result["success"]:
        return render_transform_page(
            f'<div class="result"><p class="fail">Landmark detection failed: {lm_result["message"]}</p></div>'
        )

    landmarks = lm_result["landmarks"]
    warper = FaceWarper()
    transformed = face_image.copy()

    # Expression warp
    if mode != "none":
        transformed = warper.warp_expression(transformed, landmarks, mode,
                                             intensity=max(0.1, min(intensity, 2.0)))

    # Re-detect landmarks on warped image for overlays
    needs_creative = bool(creative) or glasses_style != "none"
    if needs_creative:
        detector2 = FaceLandmarkDetector()
        lm2 = detector2.detect(transformed)
        detector2.close()
        lm2_pts = lm2["landmarks"] if lm2["success"] else landmarks

        if "lip_color" in creative:
            lip_bgr = _hex_to_bgr(lip_color_hex)
            transformed = warper.apply_lip_color(transformed, lm2_pts, color=lip_bgr)

        if glasses_style != "none":
            transformed = warper.apply_glasses(transformed, lm2_pts,
                                               style=glasses_style,
                                               assets_dir=GLASSES_DIR)

        if "hair_color" in creative:
            hair_bgr = _hex_to_bgr(hair_color_hex)
            transformed = warper.apply_hair_color(transformed, lm2_pts, color=hair_bgr)

        if "eye_color" in creative:
            eye_bgr = _hex_to_bgr(eye_color_hex)
            transformed = warper.apply_eye_color(transformed, lm2_pts, color=eye_bgr)

        if "frame_photo" in creative:
            transformed = warper.apply_frame_photo(transformed, lm2_pts,
                                                   style=frame_style)

    result_uid = uuid.uuid4().hex
    orig_path = os.path.join(RESULT_DIR, f"{result_uid}_orig.jpg")
    out_path  = os.path.join(RESULT_DIR, f"{result_uid}_out.jpg")
    save_image(face_image,  orig_path)
    save_image(transformed, out_path)

    applied = []
    if mode != "none":
        applied.append(EXPRESSION_MODES.get(mode, mode))
    if glasses_style != "none":
        applied.append(f"Glasses ({glasses_style})")
    for c in creative:
        applied.append(CREATIVE_OPTIONS.get(c, c))

    result_section = f"""
    <div class="result">
        <p class="success">Transform complete.</p>
        <p><strong>Applied:</strong> {", ".join(applied) if applied else "None"}</p>
        <p><strong>Intensity:</strong> {intensity}</p>
        <div class="grid">
            <div class="card">
                <h3>Original (preprocessed)</h3>
                <img src="/results/{result_uid}_orig.jpg" alt="Original">
            </div>
            <div class="card">
                <h3>Transformed</h3>
                <img src="/results/{result_uid}_out.jpg" alt="Transformed">
            </div>
        </div>
    </div>
    """
    face_thumb = f"/uploads/{face_id}_face.jpg"
    return render_transform_page(result_section,
                                 face_id=face_id,
                                 face_thumb_url=face_thumb)