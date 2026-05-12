import json
import os


def ensure_directory(path):
    if path and not os.path.exists(path):
        os.makedirs(path, exist_ok=True)


def save_landmarks_to_json(data, output_path):
    """
    Saves landmark data into a JSON file.

    Expected data example:
    {
        "bbox": [x, y, w, h],
        "landmarks": [(x1, y1), ...],
        "grouped_landmarks": {
            "left_eye": [(x, y), ...],
            ...
        }
    }
    """
    ensure_directory(os.path.dirname(output_path))

    # tuples -> lists dönüşümü JSON için gerekli
    serializable_data = {
        "bbox": list(data["bbox"]) if data.get("bbox") is not None else None,
        "landmarks": [list(pt) for pt in data.get("landmarks", [])],
        "grouped_landmarks": {
            key: [list(pt) for pt in value]
            for key, value in data.get("grouped_landmarks", {}).items()
        },
        "image_size": data.get("image_size"),
        "message": data.get("message", "")
    }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(serializable_data, f, indent=4, ensure_ascii=False)