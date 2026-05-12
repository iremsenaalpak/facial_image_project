"""Hair segmentation via MediaPipe SelfieMulticlassSegmentation.

Output category indices: 0=background, 1=hair, 2=body-skin, 3=face-skin,
4=clothes, 5=others/accessories.
"""

from pathlib import Path
from threading import Lock

import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision


_MODEL_PATH = Path(__file__).resolve().parents[1] / "selfie_multiclass_256x256.tflite"
_HAIR_CLASS = 1
_segmenter_singleton = None
_segmenter_lock = Lock()


class HairSegmenter:
    def __init__(self, model_path: Path = _MODEL_PATH):
        if not model_path.exists():
            raise FileNotFoundError(f"Segmentation model not found: {model_path}")
        options = mp_vision.ImageSegmenterOptions(
            base_options=mp_python.BaseOptions(model_asset_buffer=model_path.read_bytes()),
            running_mode=mp_vision.RunningMode.IMAGE,
            output_category_mask=True,
            output_confidence_masks=True,
        )
        self._segmenter = mp_vision.ImageSegmenter.create_from_options(options)

    def segment(self, image_bgr: np.ndarray) -> np.ndarray:
        """Return a float32 hair-probability map in [0, 1] sized to image_bgr."""
        h, w = image_bgr.shape[:2]
        rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        result = self._segmenter.segment(mp_image)

        if result.confidence_masks and len(result.confidence_masks) > _HAIR_CLASS:
            mask = np.asarray(result.confidence_masks[_HAIR_CLASS].numpy_view(), dtype=np.float32)
        elif result.category_mask is not None:
            cat = np.asarray(result.category_mask.numpy_view())
            mask = (cat == _HAIR_CLASS).astype(np.float32)
        else:
            return np.zeros((h, w), dtype=np.float32)

        if mask.shape[:2] != (h, w):
            mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_LINEAR)
        return np.clip(mask, 0.0, 1.0)


def get_hair_segmenter() -> HairSegmenter:
    global _segmenter_singleton
    if _segmenter_singleton is None:
        with _segmenter_lock:
            if _segmenter_singleton is None:
                _segmenter_singleton = HairSegmenter()
    return _segmenter_singleton
