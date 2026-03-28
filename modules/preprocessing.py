import cv2
import mediapipe as mp

from modules.image_loader import load_and_validate_image
from utils.image_utils import (
    crop_image,
    resize_image,
    convert_to_grayscale,
    normalize_image
)


def detect_face(image):
    """
    Detects a face in the given image using MediaPipe Face Detection.

    Returns:
        dict:
        {
            "success": bool,
            "message": str,
            "bbox": (x, y, w, h) or None
        }
    """
    mp_face_detection = mp.solutions.face_detection
    image_height, image_width = image.shape[:2]

    with mp_face_detection.FaceDetection(
        model_selection=1,
        min_detection_confidence=0.5
    ) as face_detection:

        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = face_detection.process(rgb_image)

        if not results.detections:
            return {
                "success": False,
                "message": "No face detected in the image.",
                "bbox": None
            }

        largest_bbox = None
        largest_area = 0

        for detection in results.detections:
            bbox = detection.location_data.relative_bounding_box

            x = max(0, int(bbox.xmin * image_width))
            y = max(0, int(bbox.ymin * image_height))
            w = max(1, int(bbox.width * image_width))
            h = max(1, int(bbox.height * image_height))

            area = w * h

            if area > largest_area:
                largest_area = area
                largest_bbox = (x, y, w, h)

        return {
            "success": True,
            "message": "Face detected successfully.",
            "bbox": largest_bbox
        }


def expand_bbox(x, y, w, h, image_width, image_height, padding_ratio=0.15):
    """
    Expands the bounding box by a padding ratio while keeping it inside image boundaries.

    Returns:
        (new_x, new_y, new_w, new_h)
    """
    pad_w = int(w * padding_ratio)
    pad_h = int(h * padding_ratio)

    new_x = max(0, x - pad_w)
    new_y = max(0, y - pad_h)
    new_w = min(image_width - new_x, w + 2 * pad_w)
    new_h = min(image_height - new_y, h + 2 * pad_h)

    return new_x, new_y, new_w, new_h


def crop_face(image, bbox):
    """
    Crops the face region from the image using the given bounding box.

    Returns:
        Cropped face image
    """
    x, y, w, h = bbox
    return crop_image(image, x, y, w, h)


def preprocess_face(image, target_size=(256, 256)):
    """
    Applies preprocessing steps to the cropped face image.

    Steps:
    - Resize
    - Convert to grayscale
    - Normalize color image
    - Normalize grayscale image

    Returns:
        dict containing processed versions of the face image
    """
    resized_face = resize_image(image, target_size)
    grayscale_face = convert_to_grayscale(resized_face)
    normalized_face = normalize_image(resized_face)
    normalized_grayscale_face = normalize_image(grayscale_face)

    return {
        "resized_face": resized_face,
        "grayscale_face": grayscale_face,
        "normalized_face": normalized_face,
        "normalized_grayscale_face": normalized_grayscale_face
    }


def run_preprocessing_pipeline(file_path, target_size=(256, 256), min_width=128, min_height=128):
    """
    Full preprocessing pipeline:
    1. Load and validate image
    2. Detect face
    3. Expand bounding box
    4. Crop face
    5. Resize
    6. Convert to grayscale
    7. Normalize

    Returns:
        dict containing all outputs and metadata
    """
    load_result = load_and_validate_image(file_path, min_width, min_height)

    if not load_result["success"]:
        return {
            "success": False,
            "message": load_result["message"],
            "original_image": None,
            "preview_image": None,
            "face_bbox": None,
            "cropped_face": None,
            "resized_face": None,
            "grayscale_face": None,
            "normalized_face": None,
            "normalized_grayscale_face": None,
            "metadata": {
                **(load_result["metadata"] or {}),
                "face_detected": False,
                "bbox_format": "(x, y, w, h)"
            }
        }

    image = load_result["image"]
    preview_image = load_result["preview_image"]

    face_result = detect_face(image)

    if not face_result["success"]:
        return {
            "success": False,
            "message": face_result["message"],
            "original_image": image,
            "preview_image": preview_image,
            "face_bbox": None,
            "cropped_face": None,
            "resized_face": None,
            "grayscale_face": None,
            "normalized_face": None,
            "normalized_grayscale_face": None,
            "metadata": {
                **load_result["metadata"],
                "face_detected": False,
                "bbox_format": "(x, y, w, h)"
            }
        }

    x, y, w, h = face_result["bbox"]
    image_height, image_width = image.shape[:2]

    expanded_bbox = expand_bbox(x, y, w, h, image_width, image_height, padding_ratio=0.15)
    cropped_face = crop_face(image, expanded_bbox)

    if cropped_face is None or cropped_face.size == 0:
        return {
            "success": False,
            "message": "Face crop failed.",
            "original_image": image,
            "preview_image": preview_image,
            "face_bbox": expanded_bbox,
            "cropped_face": None,
            "resized_face": None,
            "grayscale_face": None,
            "normalized_face": None,
            "normalized_grayscale_face": None,
            "metadata": {
                **load_result["metadata"],
                "target_size": target_size,
                "face_detected": True,
                "bbox_format": "(x, y, w, h)"
            }
        }

    processed = preprocess_face(cropped_face, target_size)

    return {
        "success": True,
        "message": "Preprocessing pipeline completed successfully.",
        "original_image": image,
        "preview_image": preview_image,
        "face_bbox": expanded_bbox,
        "cropped_face": cropped_face,
        "resized_face": processed["resized_face"],
        "grayscale_face": processed["grayscale_face"],
        "normalized_face": processed["normalized_face"],
        "normalized_grayscale_face": processed["normalized_grayscale_face"],
        "metadata": {
            **load_result["metadata"],
            "target_size": target_size,
            "face_detected": True,
            "bbox_format": "(x, y, w, h)"
        }
    }