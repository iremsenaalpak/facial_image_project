import cv2
import numpy as np
import os
from modules.face_landmark import FaceLandmarkDetector


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

mode = "glasses"
selected_index = 2
effect_enabled = False


def set_effect(new_mode, new_index, enabled=True):
    global mode, selected_index, effect_enabled

    mode = new_mode
    selected_index = int(new_index)
    effect_enabled = enabled


def process_frame(frame):
    global mode, selected_index, effect_enabled

    result = detector.detect(frame)

    current_files = GLASSES_FILES if mode == "glasses" else STICKER_FILES
    current_effects = glasses_effects if mode == "glasses" else sticker_effects
    
    selected_index = max(0, min(selected_index, len(current_files) - 1))
    selected_name = current_files[selected_index]

    if result["success"] and effect_enabled and selected_name in current_effects:
        landmarks = result["landmarks"]
        effect = current_effects[selected_name]

        if mode == "glasses":
            frame = apply_glasses(frame, effect, landmarks)
        else:
            if selected_name in ["cat_ears.png", "crown.png"]:
                frame = apply_cat_ears_or_crown(frame, effect, landmarks, selected_name)
            elif selected_name == "freckles.png":
                frame = apply_freckles(frame, effect, landmarks)
            elif selected_name == "sparkles.png":
                frame = apply_sparkles(frame, effect, landmarks)

    return frame
