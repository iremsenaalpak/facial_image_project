import json
import os
import shutil
import uuid

import cv2
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles

from modules.face_landmark import FaceLandmarkDetector
from modules.preprocessing import run_preprocessing_pipeline
from modules.visualization import visualize_face_data

app = FastAPI(title="Face Preprocessing Demo")

UPLOAD_DIR = "uploads"
RESULT_DIR = "results"

os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(RESULT_DIR, exist_ok=True)

app.mount("/uploads", StaticFiles(directory="uploads"), name="uploads")
app.mount("/results", StaticFiles(directory="results"), name="results")

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
    landmark_json=None
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
        landmark_json=landmark_json_text
    )


@app.get("/api/test")
def api_test():
    return {"message": "API is running"}