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
_BODY_SKIN_CLASS = 2
_FACE_SKIN_CLASS = 3
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
        # The live camera segments on a background thread while the photo path
        # may segment in FastAPI's threadpool. The underlying graph isn't safe
        # for concurrent calls, so serialize them.
        self._call_lock = Lock()

    def _segment_classes(self, image_bgr: np.ndarray, classes: list) -> np.ndarray:
        """Return a float32 probability map covering the union of the given class indices.

        Uses the category mask (argmax classification per pixel) so every pixel
        the model classifies as one of the requested classes gets full weight.
        Wispy hair tips have low *confidence* but are still classified as hair
        in the argmax — using confidence masks alone misses them. Confidence
        masks are added on top as a soft floor for the most certain pixels.
        """
        h, w = image_bgr.shape[:2]
        rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        max_class = max(classes)
        cat_mask = None
        conf_mask = None

        # Hold the lock through the segment call AND the numpy_view reads: those
        # views alias the segmenter's internal buffers, which the next segment
        # call would overwrite. Copy the data out into independent arrays here.
        with self._call_lock:
            result = self._segmenter.segment(mp_image)

            if result.category_mask is not None:
                cat = np.asarray(result.category_mask.numpy_view())
                cat_mask = np.isin(cat, classes).astype(np.float32)

            if result.confidence_masks and len(result.confidence_masks) > max_class:
                shape = np.asarray(result.confidence_masks[classes[0]].numpy_view()).shape
                conf_mask = np.zeros(shape, dtype=np.float32)
                for c in classes:
                    cmask = np.asarray(result.confidence_masks[c].numpy_view(), dtype=np.float32)
                    conf_mask = np.maximum(conf_mask, cmask)

        if cat_mask is None and conf_mask is None:
            return np.zeros((h, w), dtype=np.float32)
        if cat_mask is None:
            mask = conf_mask
        elif conf_mask is None:
            mask = cat_mask
        else:
            # Hard category classification, ORed with confident soft pixels
            mask = np.maximum(cat_mask, conf_mask)

        if mask.shape[:2] != (h, w):
            mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_LINEAR)
        return np.clip(mask, 0.0, 1.0)

    def segment(self, image_bgr: np.ndarray) -> np.ndarray:
        """Return a float32 hair-probability map in [0, 1] sized to image_bgr."""
        return self._segment_classes(image_bgr, [_HAIR_CLASS])

    def segment_skin(self, image_bgr: np.ndarray) -> np.ndarray:
        """Return a float32 skin-probability map (face + neck/body) in [0, 1]."""
        return self._segment_classes(image_bgr, [_FACE_SKIN_CLASS, _BODY_SKIN_CLASS])

    def segment_head(self, image_bgr: np.ndarray) -> np.ndarray:
        """Return a float32 head-probability map (hair + face-skin) in [0, 1]."""
        return self._segment_classes(image_bgr, [_HAIR_CLASS, _FACE_SKIN_CLASS])

    def head_bbox(self, image_bgr: np.ndarray, threshold: float = 0.5):
        """Return (x, y, w, h) bounding box of hair+face, or None if not found."""
        mask = self.segment_head(image_bgr)
        ys, xs = np.where(mask > threshold)
        if len(xs) == 0:
            return None
        return int(xs.min()), int(ys.min()), int(xs.max() - xs.min()), int(ys.max() - ys.min())


def get_hair_segmenter() -> HairSegmenter:
    global _segmenter_singleton
    if _segmenter_singleton is None:
        with _segmenter_lock:
            if _segmenter_singleton is None:
                _segmenter_singleton = HairSegmenter()
    return _segmenter_singleton
