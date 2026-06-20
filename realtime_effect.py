import cv2
import numpy as np
import os
import threading
import time
from modules.face_landmark import FaceLandmarkDetector
from modules.warping import FaceWarper
from modules.aging import aging_deaging_pipeline, grey_hair
from modules.hair_segmenter import get_hair_segmenter
from utils.image_utils import recolor_preserve_luminance, constrain_mask_to_head


GLASSES_DIR = "assets/glasses"
STICKERS_DIR = "assets/stickers"

GLASSES_FILES = ["aviator.png", "cateye.png", "round.png", "square.png"]
STICKER_FILES = ["cat_ears.png", "crown.png", "freckles.png", "sparkles.png"]


def load_effects(folder, files):
    effects = {}
    for file in files:
        path = os.path.join(folder, file)
        img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        if img is not None:
            effects[file] = img
        else:
            print(f"Dosya okunamadi: {path}")
    return effects


def alpha_blend(frame, warped):
    rgb = warped[:, :, :3]
    alpha = warped[:, :, 3:] / 255.0
    frame[:] = frame * (1 - alpha) + rgb * alpha
    return frame


def warp_effect_to_points(frame, effect, src_pts, dst_pts):
    h, w = frame.shape[:2]

    matrix = cv2.getAffineTransform(
        np.float32(src_pts),
        np.float32(dst_pts)
    )

    warped = cv2.warpAffine(
        effect,
        matrix,
        (w, h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(0, 0, 0, 0)
    )

    return alpha_blend(frame, warped)


def get_eye_axis(landmarks):
    left_eye = np.array(landmarks[33], dtype=np.float32)
    right_eye = np.array(landmarks[263], dtype=np.float32)

    v = right_eye - left_eye
    dist = np.linalg.norm(v)

    if dist < 1:
        return None, None, None, None

    v = v / dist
    n = np.array([-v[1], v[0]], dtype=np.float32)

    return left_eye, right_eye, v, n


def apply_glasses(frame, effect, landmarks):
    eh, ew = effect.shape[:2]

    left_eye, right_eye, v, n = get_eye_axis(landmarks)
    if left_eye is None:
        return frame

    eye_dist = np.linalg.norm(right_eye - left_eye)

    center = (left_eye + right_eye) / 2
    center = center + n * eye_dist * 0.03

    width = eye_dist * 1.60

    dst_left = center - v * width * 0.25
    dst_right = center + v * width * 0.25
    dst_bottom = center + n * width * 0.12

    src_left = [ew * 0.28, eh * 0.52]
    src_right = [ew * 0.72, eh * 0.52]
    src_bottom = [ew * 0.50, eh * 0.72]

    return warp_effect_to_points(
        frame,
        effect,
        [src_left, src_right, src_bottom],
        [dst_left, dst_right, dst_bottom]
    )


def apply_cat_ears_or_crown(frame, effect, landmarks, selected_name):
    eh, ew = effect.shape[:2]

    left_eye, right_eye, v, n = get_eye_axis(landmarks)
    if left_eye is None:
        return frame

    eye_dist = np.linalg.norm(right_eye - left_eye)

    forehead = np.array(landmarks[10], dtype=np.float32)
    nose_tip = np.array(landmarks[4], dtype=np.float32)

    if selected_name == "cat_ears.png":
        # Asset icindeki siyah burun noktasini gercek burun ucuna oturtuyoruz.
        dst_nose = nose_tip + n * eye_dist * 0.02

        # Kulaklari kafanin ustune yerlestiriyoruz.
        ear_center = forehead - n * eye_dist * 0.75

        dst_left_ear = ear_center - v * eye_dist * 0.50
        dst_right_ear = ear_center + v * eye_dist * 0.50

        # PNG icindeki referans noktalar:
        # src_nose: sticker icindeki siyah burun
        # src_left_ear / src_right_ear: kulak uclarina yakin noktalar
        src_left_ear = [ew * 0.22, eh * 0.10]
        src_right_ear = [ew * 0.78, eh * 0.10]
        src_nose = [ew * 0.50, eh * 0.82]

        return warp_effect_to_points(
            frame,
            effect,
            [src_left_ear, src_right_ear, src_nose],
            [dst_left_ear, dst_right_ear, dst_nose]
        )

    else:
        # Crown icin ayri ayar
        center = forehead - n * eye_dist * 0.35
        width = eye_dist * 1.80
        height_factor = 0.50

        dst_left = center - v * width * 0.45 + n * width * 0.12
        dst_right = center + v * width * 0.45 + n * width * 0.12
        dst_top = center - n * width * height_factor

        src_left = [ew * 0.08, eh * 0.90]
        src_right = [ew * 0.92, eh * 0.90]
        src_top = [ew * 0.50, eh * 0.08]

        return warp_effect_to_points(
            frame,
            effect,
            [src_left, src_right, src_top],
            [dst_left, dst_right, dst_top]
        )


def apply_freckles(frame, effect, landmarks):
    eh, ew = effect.shape[:2]

    left_eye, right_eye, v, n = get_eye_axis(landmarks)
    if left_eye is None:
        return frame

    eye_dist = np.linalg.norm(right_eye - left_eye)

    nose_top = np.array(landmarks[168], dtype=np.float32)   # burun köprüsü
    nose_tip = np.array(landmarks[4], dtype=np.float32)     # burun ucu

    center = nose_top * 0.57 + nose_tip * 0.45
    center = center - v * eye_dist * 0.05

    width = eye_dist * 1.65

    dst_left = center - v * width * 0.42
    dst_right = center + v * width * 0.42
    dst_bottom = center + n * width * 0.10

    src_left = [ew * 0.08, eh * 0.42]
    src_right = [ew * 0.92, eh * 0.42]
    src_bottom = [ew * 0.50, eh * 0.72]

    return warp_effect_to_points(
        frame,
        effect,
        [src_left, src_right, src_bottom],
        [dst_left, dst_right, dst_bottom]
    )


def apply_sparkles(frame, effect, landmarks):
    eh, ew = effect.shape[:2]

    left_eye, right_eye, v, n = get_eye_axis(landmarks)
    if left_eye is None:
        return frame

    eye_dist = np.linalg.norm(right_eye - left_eye)

    forehead = np.array(landmarks[10], dtype=np.float32)
    chin = np.array(landmarks[152], dtype=np.float32)

    # Yuz/kafa merkezi
    center = (forehead + chin) / 2

    # Biraz yukari al, cunku sparkles kafayi cevrelesin
    center = center - n * eye_dist * 0.15

    # Buyukluk
    width = eye_dist * 2.65
    height = eye_dist * 3.10

    # Sol, sag ve alt referans noktalar
    dst_left = center - v * width * 0.50
    dst_right = center + v * width * 0.50
    dst_bottom = center + n * height * 0.55

    # PNG uzerindeki referans noktalar
    src_left = [ew * 0.05, eh * 0.50]
    src_right = [ew * 0.95, eh * 0.50]
    src_bottom = [ew * 0.50, eh * 0.95]

    return warp_effect_to_points(
        frame,
        effect,
        [src_left, src_right, src_bottom],
        [dst_left, dst_right, dst_bottom]
    )

def draw_menu(frame, mode, selected_name):
    h, w = frame.shape[:2]

    menu_h = 95
    dark = frame.copy()
    cv2.rectangle(dark, (0, h - menu_h), (w, h), (0, 0, 0), -1)
    frame = cv2.addWeighted(dark, 0.55, frame, 0.45, 0)

    cv2.putText(
        frame,
        f"Mode: {mode.upper()} | Selected: {selected_name}",
        (20, h - 60),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (255, 255, 255),
        2
    )

    cv2.putText(
        frame,
        "G: Glasses | S: Stickers | 1-4: Select | 0: None | Q: Quit",
        (20, h - 25),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (255, 255, 255),
        1
    )

    return frame

detector = FaceLandmarkDetector(
    static_image_mode=False,
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

glasses_effects = load_effects(GLASSES_DIR, GLASSES_FILES)
sticker_effects = load_effects(STICKERS_DIR, STICKER_FILES)

warper = FaceWarper()

# Expression reshapes the live warper can run cheaply enough for real time.
# Aging / de-aging stay photo-only (too heavy to stack per frame).
EXPRESSION_MODES = {"smile", "face_slim", "eyebrow_raise", "lip_widen", "nose_enhance"}

# Maps a sticker choice name to its packaged PNG.
STICKER_FILES_BY_NAME = {
    "cat_ears": "cat_ears.png",
    "crown": "crown.png",
    "freckles": "freckles.png",
    "sparkles": "sparkles.png",
}

# Live settings, pushed from the web UI via /set_effect. Colors are BGR tuples
# (or None to disable). This mirrors the photo /transform form so the live
# preview matches the final render.
_settings = {
    "mode": "none",          # expression warp mode
    "intensity": 1.0,
    "lip_color": None,       # BGR tuple or None
    "eye_color": None,       # BGR tuple or None
    "hair_color": None,      # BGR tuple or None
    "glasses_style": "none",
    "sticker": "none",
    "frame_style": "none",   # decorative photo frame; "none" = off
}

# Hair segmentation costs ~250 ms/frame on CPU — far too slow to run inline
# without stalling the video. A background worker re-segments the most recent
# frame as fast as it can (~4 Hz); every frame we translate that cached mask by
# how far the head has moved since it was computed, so the tint tracks the head
# between segmentations and the video stays smooth.
_hair = {
    "lock": threading.Lock(),
    "thread": None,
    "running": False,
    "in_frame": None,        # latest frame queued for the worker
    "in_landmarks": None,    # landmarks for that frame
    "in_seq": 0,             # bumped whenever a new frame is queued
    "mask": None,            # latest computed mask (float32 HxW)
    "mask_pts": None,        # rigid head landmarks when the mask was computed
    "mask_shape": None,      # (h, w) the mask was computed at
}

# Relatively rigid head landmarks used to track head pose between segmentations:
# outer eye corners, nose bridge, forehead top, cheekbones, chin. Avoids mouth /
# jaw points that move with expression and would bias the transform.
_RIGID_IDS = [33, 263, 168, 10, 234, 454, 152]

# Every per-pixel effect (warp displacement maps, LAB hair recolor, lip/eye
# masks) costs in proportion to the frame's pixel count. We process at this
# width and let the browser scale the <img> back up to fill its box — roughly
# halves the work vs. a 640px frame with little visible quality loss. Frames
# already narrower than this are left untouched (never upscaled). Tune higher
# for sharper preview, lower for more fps.
_PROCESS_WIDTH = 480


def set_effect(**kwargs):
    """Update the live settings from the web UI. Unknown keys are ignored."""
    for key, value in kwargs.items():
        if key in _settings:
            _settings[key] = value


def _rigid_points(landmarks):
    if not landmarks or len(landmarks) <= max(_RIGID_IDS):
        return None
    return np.array([landmarks[i] for i in _RIGID_IDS], dtype=np.float32)


def _compute_hair_mask(frame, landmarks=None):
    mask = get_hair_segmenter().segment(frame)
    binary = (mask > 0.5).astype(np.uint8)
    if binary.sum() == 0:
        return None
    # Drop background false positives (e.g. a wall) by keeping only hair blobs
    # near the detected head.
    binary = constrain_mask_to_head(binary, landmarks)
    if binary.sum() == 0:
        return None
    binary = cv2.dilate(binary * 255, np.ones((3, 3), np.uint8), iterations=1)
    return cv2.GaussianBlur(binary.astype(np.float32) / 255.0, (0, 0), 2.0)


def _hair_worker_loop():
    """Continuously re-segment the most recently queued frame. Runs in its own
    thread so the ~250 ms model call never blocks frame delivery."""
    last_seq = -1
    while _hair["running"]:
        with _hair["lock"]:
            seq = _hair["in_seq"]
            frame = _hair["in_frame"]
            landmarks = _hair["in_landmarks"]
        if frame is None or seq == last_seq:
            time.sleep(0.005)
            continue
        last_seq = seq
        mask = _compute_hair_mask(frame, landmarks)
        with _hair["lock"]:
            _hair["mask"] = mask
            _hair["mask_pts"] = _rigid_points(landmarks)
            _hair["mask_shape"] = frame.shape[:2]


def _ensure_hair_worker():
    if _hair["thread"] is not None and _hair["thread"].is_alive():
        return
    _hair["running"] = True
    _hair["thread"] = threading.Thread(target=_hair_worker_loop, daemon=True)
    _hair["thread"].start()


def _reset_hair_cache():
    with _hair["lock"]:
        _hair["in_frame"] = None
        _hair["in_landmarks"] = None
        _hair["mask"] = None
        _hair["mask_pts"] = None
        _hair["mask_shape"] = None


def _tracked_hair_mask(frame, landmarks=None):
    """Queue this frame for the background segmenter and return the latest hair
    mask, warped to follow the head since it was computed. Returns None until the
    first mask is ready (or if it can't be aligned to the current frame).

    The model is too slow to run inline (~250 ms), so the worker keeps a mask
    fresh asynchronously and we track the head with a similarity transform
    (translation + rotation + scale) here every frame — including tilts and
    leaning in/out — between the ~4 Hz mask updates."""
    _ensure_hair_worker()
    cur_pts = _rigid_points(landmarks)

    with _hair["lock"]:
        _hair["in_frame"] = frame
        _hair["in_landmarks"] = landmarks
        _hair["in_seq"] += 1
        mask = _hair["mask"]
        mask_pts = _hair["mask_pts"]
        mask_shape = _hair["mask_shape"]

    if mask is None or mask_shape != frame.shape[:2]:
        return None

    if cur_pts is not None and mask_pts is not None and len(cur_pts) == len(mask_pts):
        M, _ = cv2.estimateAffinePartial2D(mask_pts, cur_pts, method=cv2.LMEDS)
        if M is not None:
            mask = cv2.warpAffine(mask, M, (frame.shape[1], frame.shape[0]))

    return mask


def _apply_hair_color(frame, color, landmarks=None):
    """Recolor hair using the background worker's head-tracked mask."""
    mask = _tracked_hair_mask(frame, landmarks)
    if mask is None:
        return frame
    return recolor_preserve_luminance(frame, color, mask, strength=0.6)


def _apply_hair_grey(frame, landmarks=None, intensity=0.5):
    """Grey hair (for live aging) using the same head-tracked mask."""
    mask = _tracked_hair_mask(frame, landmarks)
    if mask is None:
        return frame
    return grey_hair(frame, mask, intensity)


def _apply_sticker(frame, name, landmarks):
    file_name = STICKER_FILES_BY_NAME.get(name)
    if file_name is None or file_name not in sticker_effects:
        return frame
    effect = sticker_effects[file_name]
    if name in ("cat_ears", "crown"):
        return apply_cat_ears_or_crown(frame, effect, landmarks, file_name)
    if name == "freckles":
        return apply_freckles(frame, effect, landmarks)
    if name == "sparkles":
        return apply_sparkles(frame, effect, landmarks)
    return frame


def process_frame(frame):
    """Apply every currently-selected effect to a single webcam frame.

    Order: geometry warp first, then color/overlay effects on top. All overlays
    reuse the pre-warp landmarks — the warps are subtle and localized, so
    re-running detection (the most expensive step) per effect isn't worth it.
    """
    s = _settings

    # Downscale for speed before any processing. The face is detected on the
    # smaller frame, so landmarks and all effects stay consistent; the browser
    # scales the returned JPEG back up to fill the preview box.
    src_h, src_w = frame.shape[:2]
    if src_w > _PROCESS_WIDTH:
        scale = _PROCESS_WIDTH / float(src_w)
        frame = cv2.resize(
            frame,
            (_PROCESS_WIDTH, max(1, int(round(src_h * scale)))),
            interpolation=cv2.INTER_AREA,
        )

    result = detector.detect(frame)
    landmarks = result["landmarks"] if result["success"] else None

    # 1. Geometry / skin base pass — expression reshape OR aging/de-aging.
    #    These share the single `mode` setting (mutually exclusive in the UI).
    if landmarks and s["mode"] in EXPRESSION_MODES:
        intensity = max(0.1, min(float(s["intensity"]), 2.0))
        frame = warper.warp_expression(frame, landmarks, s["mode"], intensity=intensity)
    elif landmarks and s["mode"] in ("aging", "deaging"):
        # fast=True uses landmark masks instead of the ~235 ms segmenter; the
        # same /2 intensity scaling the photo path applies.
        intensity = max(0.0, min(float(s["intensity"]), 2.0)) / 2.0
        frame = aging_deaging_pipeline(frame, mode=s["mode"], intensity=intensity,
                                       landmarks=landmarks, fast=True)
        # Hair greying isn't done inside the fast pipeline (it would be masked to
        # the face oval) — grey it here with the head-tracked segmenter mask.
        if s["mode"] == "aging":
            frame = _apply_hair_grey(frame, landmarks, intensity)

    # 2. Hair recolor (throttled segmentation; landmarks constrain it to the head)
    if s["hair_color"] is not None:
        frame = _apply_hair_color(frame, s["hair_color"], landmarks)
    elif s["mode"] != "aging":
        # Nothing needs the hair worker (aging greying above also uses it) — let
        # it idle. Resetting while aging is on would wipe the grey mask.
        _reset_hair_cache()

    # 3. Lip color
    if landmarks and s["lip_color"] is not None:
        frame = warper.apply_lip_color(frame, landmarks, color=s["lip_color"])

    # 4. Eye color (iris) — needs the 478-point refined mesh
    if landmarks and s["eye_color"] is not None:
        frame = warper.apply_eye_color(frame, landmarks, color=s["eye_color"])

    # 5. Glasses
    if landmarks and s["glasses_style"] != "none":
        file_name = f"{s['glasses_style']}.png"
        if file_name in glasses_effects:
            frame = apply_glasses(frame, glasses_effects[file_name], landmarks)

    # 6. Sticker
    if landmarks and s["sticker"] != "none":
        frame = _apply_sticker(frame, s["sticker"], landmarks)

    # 7. Decorative photo frame — wraps the whole preview, so it goes last
    if s["frame_style"] != "none":
        frame = warper.apply_frame_photo(frame, style=s["frame_style"])

    return frame
