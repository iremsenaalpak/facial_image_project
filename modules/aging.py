import cv2
import numpy as np

from utils.image_utils import recolor_preserve_luminance


# MediaPipe FaceMesh landmark indices (refine_landmarks=True, 478 points)
FACE_OVAL = [
    10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288,
    397, 365, 379, 378, 400, 377, 152, 148, 176, 149, 150, 136,
    172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109
]
LEFT_EYE_OUTER = 33
RIGHT_EYE_OUTER = 263
LEFT_EYE_LOWER = 145
RIGHT_EYE_LOWER = 374
LEFT_EYEBROW_OUTER = 70
RIGHT_EYEBROW_OUTER = 300
LEFT_EYEBROW_INNER = 55
RIGHT_EYEBROW_INNER = 285
GLABELLA = 9
FOREHEAD_TOP = 10
NOSE_WING_LEFT = 64
NOSE_WING_RIGHT = 294
MOUTH_CORNER_LEFT = 61
MOUTH_CORNER_RIGHT = 291
CHIN = 152
JAW_LEFT = 172
JAW_RIGHT = 397
EYE_HOLE_LEFT = [33, 133, 160, 159, 158, 157, 173, 153, 144, 145, 146]
EYE_HOLE_RIGHT = [362, 263, 387, 386, 385, 384, 398, 373, 374, 380, 381]
LIPS_OUTER = [61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291,
              409, 270, 269, 267, 0, 37, 39, 40, 185]


def clip_uint8(image):
    return np.clip(image, 0, 255).astype(np.uint8)


def normalize_image(image):
    image = image.astype(np.float32)
    image = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX)
    return clip_uint8(image)


def _lm(landmarks, idx):
    """Safe landmark accessor."""
    if landmarks is None or idx >= len(landmarks):
        return None
    return landmarks[idx]


def create_face_mask(image, landmarks=None, feather=21, include_body_skin=True):
    """
    Build a soft skin mask for aging/de-aging.

    Preference order:
      1. MediaPipe selfie-multiclass segmenter (face-skin + body-skin):
         covers full forehead, neck, throat, ears — no chin seam.
      2. MediaPipe face-oval polygon from landmarks (face only).
      3. HSV skin thresholding.

    Eyes and lips are excluded so tonal changes don't tint them.
    """
    h, w = image.shape[:2]

    seg_mask = None
    if include_body_skin:
        try:
            from modules.hair_segmenter import get_hair_segmenter
            segmenter = get_hair_segmenter()
            seg_mask = segmenter.segment_skin(image)
            if seg_mask.sum() < 100:  # segmentation failed / no person
                seg_mask = None
        except Exception:
            seg_mask = None

    if seg_mask is not None:
        mask = (seg_mask * 255).astype(np.uint8)

        # Still cut out eyes and lips from the segmenter mask
        if landmarks is not None:
            for hole_indices in (EYE_HOLE_LEFT, EYE_HOLE_RIGHT, LIPS_OUTER):
                if len(landmarks) > max(hole_indices):
                    hole_pts = np.array(
                        [landmarks[i] for i in hole_indices],
                        dtype=np.int32
                    )
                    cv2.fillPoly(mask, [cv2.convexHull(hole_pts)], 0)

        if feather > 0:
            k = feather if feather % 2 == 1 else feather + 1
            mask = cv2.GaussianBlur(mask, (k, k), 0)

        mask = mask.astype(np.float32) / 255.0
        return np.expand_dims(mask, axis=2)

    if landmarks is not None and len(landmarks) > max(FACE_OVAL):
        mask = np.zeros((h, w), dtype=np.uint8)

        oval_pts = np.array(
            [landmarks[i] for i in FACE_OVAL],
            dtype=np.int32
        )
        cv2.fillConvexPoly(mask, cv2.convexHull(oval_pts), 255)

        for hole_indices in (EYE_HOLE_LEFT, EYE_HOLE_RIGHT, LIPS_OUTER):
            if len(landmarks) > max(hole_indices):
                hole_pts = np.array(
                    [landmarks[i] for i in hole_indices],
                    dtype=np.int32
                )
                cv2.fillPoly(mask, [cv2.convexHull(hole_pts)], 0)

        if feather > 0:
            k = feather if feather % 2 == 1 else feather + 1
            mask = cv2.GaussianBlur(mask, (k, k), 0)

        mask = mask.astype(np.float32) / 255.0
        return np.expand_dims(mask, axis=2)

    # Fallback: HSV skin thresholding
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    lower_skin = np.array([0, 20, 60], dtype=np.uint8)
    upper_skin = np.array([25, 255, 255], dtype=np.uint8)
    mask = cv2.inRange(hsv, lower_skin, upper_skin)
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.GaussianBlur(mask, (15, 15), 0)
    mask = mask.astype(np.float32) / 255.0
    return np.expand_dims(mask, axis=2)


def create_skin_mask(image, landmarks=None):
    """Backwards-compatible alias for create_face_mask."""
    return create_face_mask(image, landmarks=landmarks)


def blend_with_mask(original, processed, mask):
    original = original.astype(np.float32)
    processed = processed.astype(np.float32)

    result = original * (1 - mask) + processed * mask
    return clip_uint8(result)


# =========================================================
# AGING FUNCTIONS
# =========================================================

def high_frequency_boost(image, intensity=0.5):
    """
    Subtle unsharp-mask style boost that emphasizes existing micro-shadows
    (faint creases, pores, eye-corner lines) without amplifying noise.
    """
    intensity = np.clip(intensity, 0, 1)

    img_float = image.astype(np.float32)
    blurred = cv2.GaussianBlur(img_float, (0, 0), sigmaX=2.5)
    high_freq = img_float - blurred

    gain = 0.15 + 0.25 * intensity
    boosted = img_float + high_freq * gain
    return clip_uint8(boosted)


def deepen_skin_shadows(image, intensity=0.5):
    """
    Natural crease/wrinkle deepening — no drawn marks.

    Darkens the regions that are *already* in shadow on the face (the real
    creases: nasolabial folds, under-eye, forehead lines, mouth corners) by
    amplifying wherever local luminance dips below its neighbourhood. Two scales
    catch both fine lines and broader folds. Because it follows the image's own
    structure it reads as natural at any head angle, unlike landmark-drawn lines.
    """
    intensity = np.clip(intensity, 0, 1)

    img = image.astype(np.float32)
    h, w = image.shape[:2]

    # The shadow map is smooth, so build it at half resolution for speed and
    # upscale — roughly 4x fewer pixels through the (costly) large blur.
    small = cv2.resize(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY),
                       (max(w // 2, 1), max(h // 2, 1)),
                       interpolation=cv2.INTER_AREA).astype(np.float32)
    base = max(small.shape[:2])

    shadow = np.zeros_like(small)
    # Fine scale -> pores / fine lines; medium scale -> nasolabial & forehead folds.
    for sigma, weight in ((base * 0.010, 0.6), (base * 0.035, 1.0)):
        blur = cv2.GaussianBlur(small, (0, 0), sigmaX=max(sigma, 1.0))
        deficit = np.clip(blur - small, 0.0, None)   # darker than its surroundings
        shadow += weight * deficit

    # Protect already-dark features (eyebrows, nostrils, hairline): real skin
    # creases are mid-tone dips, so fade the effect out below ~100 luminance.
    feature_gate = np.clip((small - 55.0) / 45.0, 0.0, 1.0)
    shadow *= feature_gate

    # Robust normalise so a handful of very dark pixels don't dominate.
    scale = float(np.percentile(shadow, 99)) + 1e-3
    shadow = np.clip(shadow / scale, 0.0, 1.0)
    shadow = cv2.resize(shadow, (w, h), interpolation=cv2.INTER_LINEAR)
    shadow = cv2.GaussianBlur(shadow, (0, 0), sigmaX=1.5)

    # Darken proportionally to existing shadow depth.
    darken = 1.0 - (0.40 * intensity) * shadow[:, :, np.newaxis]
    return clip_uint8(img * darken)


def fft_high_pass_aging(image, intensity=0.5):
    intensity = np.clip(intensity, 0, 1)

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).astype(np.float32)

    rows, cols = gray.shape
    crow, ccol = rows // 2, cols // 2

    f = np.fft.fft2(gray)
    fshift = np.fft.fftshift(f)

    mask = np.ones((rows, cols), np.float32)

    radius = int(25 + 35 * (1 - intensity))
    cv2.circle(mask, (ccol, crow), radius, 0, -1)

    fshift_filtered = fshift * mask

    img_back = np.fft.ifft2(np.fft.ifftshift(fshift_filtered))
    img_back = np.abs(img_back)

    img_back = normalize_image(img_back)
    detail_map = cv2.cvtColor(img_back, cv2.COLOR_GRAY2BGR)

    aged = cv2.addWeighted(
        image,
        1.0,
        detail_map,
        0.08 + 0.14 * intensity,
        0
    )

    return clip_uint8(aged)


def add_wrinkle_texture(image, intensity=0.5, seed=42):
    intensity = np.clip(intensity, 0, 2)

    rng = np.random.default_rng(seed)
    img = image.astype(np.float32)

    noise = rng.normal(0, 1, img.shape).astype(np.float32)
    noise = cv2.GaussianBlur(noise, (7, 7), 0)

    texture_strength = 8.0 + 15.0 * intensity
    textured = img + noise * texture_strength

    return clip_uint8(textured)


def add_wrinkle_lines(image, intensity=0.5, landmarks=None):
    """
    Draws anatomically-anchored wrinkles at very low opacity. Lines are
    rendered into a temporary overlay then blended at low alpha so they
    read as faint shadows rather than painted marks.
    """
    intensity = np.clip(intensity, 0, 1)

    h, w = image.shape[:2]
    overlay = np.zeros_like(image, dtype=np.uint8)

    # Very faint shadow — barely darker than mid-gray noise
    shadow = int(18 + 20 * intensity)
    shadow_color = (shadow, shadow, shadow)

    needed = [
        FOREHEAD_TOP, GLABELLA, LEFT_EYEBROW_OUTER, RIGHT_EYEBROW_OUTER,
        LEFT_EYE_OUTER, RIGHT_EYE_OUTER, LEFT_EYE_LOWER, RIGHT_EYE_LOWER,
        JAW_LEFT, JAW_RIGHT
    ]
    have_lms = (
        landmarks is not None
        and len(landmarks) > max(needed)
    )

    if not have_lms:
        return image.copy()

    p_forehead_top = np.array(landmarks[FOREHEAD_TOP], dtype=np.float32)
    p_glabella = np.array(landmarks[GLABELLA], dtype=np.float32)
    p_brow_l = np.array(landmarks[LEFT_EYEBROW_OUTER], dtype=np.float32)
    p_brow_r = np.array(landmarks[RIGHT_EYEBROW_OUTER], dtype=np.float32)
    p_eye_lo = np.array(landmarks[LEFT_EYE_OUTER], dtype=np.float32)
    p_eye_ro = np.array(landmarks[RIGHT_EYE_OUTER], dtype=np.float32)
    p_eye_ll = np.array(landmarks[LEFT_EYE_LOWER], dtype=np.float32)
    p_eye_rl = np.array(landmarks[RIGHT_EYE_LOWER], dtype=np.float32)
    p_jaw_l = np.array(landmarks[JAW_LEFT], dtype=np.float32)
    p_jaw_r = np.array(landmarks[JAW_RIGHT], dtype=np.float32)

    face_width = float(np.linalg.norm(p_jaw_r - p_jaw_l))

    # Forehead horizontal wrinkles — 3 thin lines between brow and hairline
    brow_y = (p_brow_l[1] + p_brow_r[1]) * 0.5
    forehead_top_y = p_forehead_top[1]
    forehead_height = max(brow_y - forehead_top_y, face_width * 0.15)
    forehead_center_x = (p_brow_l[0] + p_brow_r[0]) * 0.5
    half_span = float(np.linalg.norm(p_brow_r - p_brow_l)) * 0.48

    n_lines = 5
    for i in range(n_lines):
        t = (i + 1) / (n_lines + 1)
        y = int(forehead_top_y + forehead_height * (0.35 + 0.45 * t))
        amp = int(forehead_height * 0.025)

        pts = []
        for j in range(13):
            u = j / 12.0
            x = int(forehead_center_x + (u - 0.5) * 2 * half_span)
            wave = int(amp * np.sin(u * np.pi * 2 + i))
            pts.append((x, y + wave))

        for k in range(len(pts) - 1):
            cv2.line(overlay, pts[k], pts[k + 1], shadow_color, 1)

    # Glabella verticals — barely visible
    gx = int(p_glabella[0])
    gy = int(p_glabella[1])
    seg = int(face_width * 0.018)
    for dx in (-int(face_width * 0.010), int(face_width * 0.010)):
        cv2.line(
            overlay,
            (gx + dx, gy - seg),
            (gx + dx, gy + seg),
            shadow_color,
            1
        )

    # Crow's feet — 2 short lines from outer eye corners
    for eye_outer, direction in (
        (p_eye_lo, np.array([-1.0, 0.0])),
        (p_eye_ro, np.array([1.0, 0.0]))
    ):
        for dy_frac in (-0.35, 0.35):
            length = face_width * 0.11
            end = eye_outer + direction * length
            end[1] += dy_frac * face_width * 0.022
            cv2.line(
                overlay,
                tuple(eye_outer.astype(int)),
                tuple(end.astype(int)),
                shadow_color,
                1
            )

    # Under-eye soft arc — single thin line
    for eye_lower in (p_eye_ll, p_eye_rl):
        cx = int(eye_lower[0])
        cy = int(eye_lower[1] + face_width * 0.035)

        cv2.ellipse(
            overlay,
            (cx, cy),
            (int(face_width * 0.090), int(face_width * 0.028)),
            0, 0, 180,
            shadow_color,
            2
        )

     # Nasolabial folds — nose side to mouth corner
    nose_l = np.array(landmarks[NOSE_WING_LEFT], dtype=np.float32)
    nose_r = np.array(landmarks[NOSE_WING_RIGHT], dtype=np.float32)
    mouth_l = np.array(landmarks[MOUTH_CORNER_LEFT], dtype=np.float32)
    mouth_r = np.array(landmarks[MOUTH_CORNER_RIGHT], dtype=np.float32)

    for nose, mouth in [(nose_l, mouth_l), (nose_r, mouth_r)]:
        p1 = nose + np.array([0, face_width * 0.02])
        p2 = mouth + np.array([0, -face_width * 0.02])

        mid = (p1 + p2) / 2
        mid[0] = mid[0] * 0.85 + mouth[0] * 0.15

        curve = np.array([p1, mid, p2], dtype=np.int32)

        cv2.polylines(
            overlay,
            [curve],
            False,
            shadow_color,
            3,
            cv2.LINE_AA
        )
        # Strong extra forehead lines
    for i in range(4):
        y = int(forehead_top_y + forehead_height * (0.42 + i * 0.10))
        x1 = int(forehead_center_x - half_span * 0.85)
        x2 = int(forehead_center_x + half_span * 0.85)

        cv2.line(
            overlay,
            (x1, y),
            (x2, y + int(np.sin(i) * 2)),
            shadow_color,
            2,
            cv2.LINE_AA
        )

    # Marionette lines: mouth corners downward
    for mouth in [mouth_l, mouth_r]:
        p1 = mouth + np.array([0, face_width * 0.015])
        p2 = mouth + np.array([0, face_width * 0.12])

        cv2.line(
            overlay,
            tuple(p1.astype(int)),
            tuple(p2.astype(int)),
            shadow_color,
            1,
            cv2.LINE_AA
        )

    # Heavy blur so lines read as soft shadow gradients, not painted marks
    overlay = cv2.GaussianBlur(overlay, (13, 13), 0)

    alpha = 0.32 + 0.18 * intensity
    result = cv2.addWeighted(image, 1.0, overlay, alpha, 0)
    return clip_uint8(result)

def grey_hair(image, hair_mask, intensity=0.5):
    """
    Grey the hair given a hair-probability mask (float32 HxW, values in [0, 1]
    or [0, 255]). Applied to the FULL frame — the caller supplies a real hair
    mask, so unlike the old version this is NOT confined to the face oval.

    Desaturates the hair toward neutral grey (keeping its own luminance
    variation, for a natural salt-and-pepper look rather than a flat painted
    cap), then lifts it slightly so dark hair actually reads as greyer.
    """
    if hair_mask is None:
        return image

    intensity = float(np.clip(intensity, 0, 1))
    mask = hair_mask.astype(np.float32)
    if mask.max() > 1.0:
        mask = mask / 255.0
    mask = np.clip(mask, 0.0, 1.0)

    greyed = recolor_preserve_luminance(image, (190, 190, 190), mask,
                                        strength=0.70 * intensity)

    lift = (0.25 * intensity) * mask[:, :, np.newaxis]
    greyed = greyed.astype(np.float32) * (1.0 - lift) + 205.0 * lift
    return clip_uint8(greyed)


def adjust_aging_tone(image, intensity=0.5):
    """Sallow aging tone: desaturate, push slightly olive/sallow (less rosy,
    less blue), drop a little luminance and flatten contrast. Older skin is
    less saturated and less radiant than young skin."""
    intensity = np.clip(intensity, 0, 1)

    # Flatten contrast (alpha < 1) and darken slightly (beta < 0).
    alpha = 1.0 - 0.06 * intensity
    beta = -6.0 * intensity
    toned = cv2.convertScaleAbs(image, alpha=alpha, beta=beta)

    # Desaturate.
    hsv = cv2.cvtColor(toned, cv2.COLOR_BGR2HSV).astype(np.float32)
    hsv[:, :, 1] *= 1.0 - 0.22 * intensity
    toned = cv2.cvtColor(
        np.clip(hsv, 0, 255).astype(np.uint8),
        cv2.COLOR_HSV2BGR
    )

    # Sallow/olive cast: trim blue and red (rosiness), hold green (BGR order).
    toned = toned.astype(np.float32)
    toned[:, :, 0] *= 1.0 - 0.08 * intensity   # B down
    toned[:, :, 1] *= 1.0 + 0.01 * intensity   # G hold
    toned[:, :, 2] *= 1.0 - 0.05 * intensity   # R down

    return clip_uint8(toned)


def apply_aging(image, intensity=0.5, use_fft=False, use_skin_mask=True,
                landmarks=None, include_body_skin=True):
    """
    Natural skin aging — built from the face's own structure, no drawn marks.

    Layers (all confined to the feathered skin mask):
      1. high-frequency boost  — emphasise pores / micro-detail
      2. deepen_skin_shadows   — darken real creases (nasolabial, under-eye,
                                 forehead) so wrinkles deepen naturally
      3. wrinkle texture        — subtle pore grain
      4. sallow tone            — desaturate + olive cast + flatter, dimmer skin

    Hair greying is NOT done here (it would be reverted by the face-oval blend);
    the caller applies it separately with a real hair mask via grey_hair().

    include_body_skin=False uses the cheap 1 ms landmark face-oval mask instead
    of the ~235 ms segmentation model, for the live camera path.
    """
    if image is None:
        raise ValueError("Input image is empty.")

    intensity = np.clip(intensity, 0, 1)
    original = image.copy()

    soft_intensity = intensity * 0.55

    aged = high_frequency_boost(original, intensity)
    aged = deepen_skin_shadows(aged, intensity)
    aged = add_wrinkle_texture(aged, soft_intensity)
    aged = adjust_aging_tone(aged, soft_intensity)

    if use_skin_mask:
        mask = create_face_mask(original, landmarks=landmarks,
                                include_body_skin=include_body_skin)
        aged = blend_with_mask(original, aged, mask)

    return aged


# =========================================================
# DE-AGING FUNCTIONS
# =========================================================

def edge_preserving_smoothing(image, intensity=0.5):
    intensity = np.clip(intensity, 0, 1)

    d = int(5 + 6 * intensity)

    if d % 2 == 0:
        d += 1

    sigma_color = 30 + 50 * intensity
    sigma_space = 30 + 50 * intensity

    smoothed = cv2.bilateralFilter(
        image,
        d=d,
        sigmaColor=sigma_color,
        sigmaSpace=sigma_space
    )

    return smoothed


def reduce_high_frequency(image, intensity=0.5):
    intensity = np.clip(intensity, 0, 1)

    blurred = cv2.GaussianBlur(
        image,
        (0, 0),
        sigmaX=0.8 + 1.2 * intensity
    )

    result = cv2.addWeighted(
        image,
        1.0 - 0.35 * intensity,
        blurred,
        0.35 * intensity,
        0
    )

    return clip_uint8(result)


def fft_low_pass_deaging(image, intensity=0.5):
    intensity = np.clip(intensity, 0, 1)

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).astype(np.float32)

    rows, cols = gray.shape
    crow, ccol = rows // 2, cols // 2

    f = np.fft.fft2(gray)
    fshift = np.fft.fftshift(f)

    mask = np.zeros((rows, cols), np.float32)

    radius = int(35 + 70 * (1 - intensity))
    cv2.circle(mask, (ccol, crow), radius, 1, -1)

    fshift_filtered = fshift * mask

    img_back = np.fft.ifft2(np.fft.ifftshift(fshift_filtered))
    img_back = np.abs(img_back)

    img_back = normalize_image(img_back)
    smooth_gray = cv2.cvtColor(img_back, cv2.COLOR_GRAY2BGR)

    result = cv2.addWeighted(
        image,
        1.0 - 0.22 * intensity,
        smooth_gray,
        0.22 * intensity,
        0
    )

    return clip_uint8(result)


def brighten_under_eye_area(image, intensity=0.5, landmarks=None):
    """
    Brightens the under-eye region. Uses landmarks when available so the
    highlight lands on the actual under-eye area rather than mid-cheek.
    """
    intensity = np.clip(intensity, 0, 1)

    h, w = image.shape[:2]
    overlay = np.zeros_like(image, dtype=np.uint8)
    color = (22, 22, 22)

    needed = [LEFT_EYE_LOWER, RIGHT_EYE_LOWER, JAW_LEFT, JAW_RIGHT]
    if landmarks is not None and len(landmarks) > max(needed):
        p_l = np.array(landmarks[LEFT_EYE_LOWER], dtype=np.float32)
        p_r = np.array(landmarks[RIGHT_EYE_LOWER], dtype=np.float32)
        face_width = float(np.linalg.norm(
            np.array(landmarks[JAW_RIGHT]) - np.array(landmarks[JAW_LEFT])
        ))
        rx = max(int(face_width * 0.08), 6)
        ry = max(int(face_width * 0.028), 3)
        offset = int(face_width * 0.025)

        for p in (p_l, p_r):
            center = (int(p[0]), int(p[1] + offset))
            cv2.ellipse(overlay, center, (rx, ry), 0, 0, 360, color, -1)
    else:
        # No landmarks — skip, drawing on fixed coordinates causes banding
        return image.copy()

    overlay = cv2.GaussianBlur(overlay, (31, 31), 0)
    alpha = 0.18 + 0.18 * intensity
    result = cv2.addWeighted(image, 1.0, overlay, alpha, 0)
    return clip_uint8(result)


def add_skin_glow(image, intensity=0.5):
    intensity = np.clip(intensity, 0, 1)

    blurred = cv2.GaussianBlur(image, (0, 0), sigmaX=2.0)

    glow = cv2.addWeighted(
        image,
        1.0,
        blurred,
        0.12 + 0.20 * intensity,
        0
    )

    return clip_uint8(glow)


def restore_young_skin_color(image, intensity=0.5):
    intensity = np.clip(intensity, 0, 1)

    img = image.astype(np.float32)

    img[:, :, 0] *= 0.99
    img[:, :, 1] *= 1.01
    img[:, :, 2] *= 1.025 + 0.02 * intensity

    hsv = cv2.cvtColor(clip_uint8(img), cv2.COLOR_BGR2HSV).astype(np.float32)
    hsv[:, :, 1] *= 1.0 + 0.08 * intensity
    hsv[:, :, 2] *= 1.0 + 0.04 * intensity

    return cv2.cvtColor(clip_uint8(hsv), cv2.COLOR_HSV2BGR)


def selective_skin_smoothing(image, intensity=0.5, landmarks=None,
                             include_body_skin=True):
    intensity = np.clip(intensity, 0, 1)

    skin_mask = create_face_mask(image, landmarks=landmarks,
                                 include_body_skin=include_body_skin)

    smoothed = edge_preserving_smoothing(image, intensity)
    smoothed = reduce_high_frequency(smoothed, intensity)

    return blend_with_mask(image, smoothed, skin_mask)


def sharpen_important_features(image, intensity=0.5):
    intensity = np.clip(intensity, 0, 1)

    blurred = cv2.GaussianBlur(image, (0, 0), sigmaX=1.2)
    detail = image.astype(np.float32) - blurred.astype(np.float32)

    sharpened = image.astype(np.float32) + detail * (0.25 + 0.45 * intensity)

    return clip_uint8(sharpened)


def subtle_face_tightening(image, intensity=0.5):
    intensity = np.clip(intensity, 0, 1)

    h, w = image.shape[:2]

    src = np.float32([
        [w * 0.20, h * 0.78],
        [w * 0.80, h * 0.78],
        [w * 0.50, h * 0.45]
    ])

    shift = 3 + 5 * intensity

    dst = np.float32([
        [w * 0.20 + shift, h * 0.78 - shift],
        [w * 0.80 - shift, h * 0.78 - shift],
        [w * 0.50, h * 0.45]
    ])

    matrix = cv2.getAffineTransform(src, dst)
    warped = cv2.warpAffine(image, matrix, (w, h), borderMode=cv2.BORDER_REFLECT)

    alpha = 0.12 + 0.10 * intensity

    return cv2.addWeighted(image, 1 - alpha, warped, alpha, 0)


def enhance_hair_youthfulness(image, intensity=0.5):
    intensity = np.clip(intensity, 0, 1)

    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV).astype(np.float32)

    value = hsv[:, :, 2]
    saturation = hsv[:, :, 1]

    hair_mask = ((value < 95) & (saturation > 20)).astype(np.float32)
    hair_mask = cv2.GaussianBlur(hair_mask, (21, 21), 0)
    hair_mask = np.expand_dims(hair_mask, axis=2)

    img = image.astype(np.float32)

    enhanced = img.copy()
    enhanced[:, :, 0] *= 0.95
    enhanced[:, :, 1] *= 0.95
    enhanced[:, :, 2] *= 0.92

    result = img * (1 - hair_mask * 0.35 * intensity) + enhanced * (hair_mask * 0.35 * intensity)

    return clip_uint8(result)


def adjust_deaging_tone(image, intensity=0.5):
    intensity = np.clip(intensity, 0, 1)

    alpha = 1.0 - 0.03 * intensity
    beta = 5 * intensity

    result = cv2.convertScaleAbs(image, alpha=alpha, beta=beta)

    return result


def apply_deaging(image, intensity=0.5, use_fft=False, use_skin_mask=True,
                  landmarks=None, include_body_skin=True):
    """
    Subtle de-aging: edge-preserving smoothing of skin only. No overlays,
    no color boosting — those produced the orange forehead glow.

    include_body_skin=False uses the cheap landmark face-oval mask for the
    live camera path.
    """
    if image is None:
        raise ValueError("Input image is empty.")

    intensity = np.clip(intensity, 0, 1)
    original = image.copy()

    soft_intensity = intensity * 0.65

    deaged = selective_skin_smoothing(
        original, soft_intensity, landmarks=landmarks,
        include_body_skin=include_body_skin
    )

    if use_skin_mask:
        mask = create_face_mask(original, landmarks=landmarks,
                                include_body_skin=include_body_skin)
        deaged = blend_with_mask(original, deaged, mask)

    return deaged

def add_aging_sagging(image, intensity=0.5, landmarks=None, include_body_skin=True):
    if landmarks is None or len(landmarks) < 468:
        return image.copy()

    intensity = np.clip(intensity, 0, 1)

    h, w = image.shape[:2]
    pts = np.array(landmarks, dtype=np.float64)

    xs, ys = pts[:, 0], pts[:, 1]
    fw = float(xs.max() - xs.min())
    fh = float(ys.max() - ys.min())

    Y, X = np.mgrid[0:h, 0:w].astype(np.float32)
    map_x = X.copy()
    map_y = Y.copy()

    def gauss(cx, cy, sx, sy):
        return np.exp(
            -(((X - cx) ** 2) / (2 * sx ** 2) +
              ((Y - cy) ** 2) / (2 * sy ** 2))
        )

    # Eye outer/lower points
    left_eye = pts[145]
    right_eye = pts[374]

    # Mouth corners
    left_mouth = pts[61]
    right_mouth = pts[291]

    # Cheek/jaw areas
    left_cheek = pts[234]
    right_cheek = pts[454]

    # 1) Under-eye / eyelid droop
    for p in [left_eye, right_eye]:
        g = gauss(
            p[0],
            p[1],
            fw * 0.09,
            fh * 0.045
        )
        map_y -= g * intensity * 0.030 * fh

    # 2) Mouth corners slightly downward
    for p in [left_mouth, right_mouth]:
        g = gauss(
            p[0],
            p[1],
            fw * 0.08,
            fh * 0.055
        )
        map_y -= g * intensity * 0.035 * fh

    # 3) Cheek sagging downward
    for p in [left_cheek, right_cheek]:
        g = gauss(
            p[0],
            p[1],
            fw * 0.13,
            fh * 0.10
        )
        map_y -= g * intensity * 0.025 * fh

    warped = cv2.remap(
        image,
        map_x.astype(np.float32),
        map_y.astype(np.float32),
        cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REFLECT_101
    )

    mask = create_face_mask(image, landmarks=landmarks,
                            include_body_skin=include_body_skin)
    return blend_with_mask(image, warped, mask)

# =========================================================
# MAIN PIPELINE
# =========================================================

def aging_deaging_pipeline(image, mode="aging", intensity=0.5, landmarks=None,
                           fast=False):
    """fast=True swaps the ~235 ms segmentation skin mask for the 1 ms landmark
    face-oval mask, making aging/de-aging cheap enough for the live camera.

    This returns SKIN aging only. Hair greying is the caller's job because it
    must run on the full image, not this (often face-cropped) input: the live
    path greys with the background segmenter's tracked mask, and the photo path
    (app.py) greys the composited full-resolution output via grey_hair().
    """
    if image is None:
        raise ValueError("Input image is empty.")

    intensity = np.clip(intensity, 0, 1)
    include_body_skin = not fast

    if mode == "aging":
        return apply_aging(
            image, intensity=intensity, use_skin_mask=True,
            landmarks=landmarks, include_body_skin=include_body_skin
        )

    elif mode == "deaging":
        return apply_deaging(
            image, intensity=intensity, use_skin_mask=True,
            landmarks=landmarks, include_body_skin=include_body_skin
        )

    else:
        raise ValueError("Mode must be 'aging' or 'deaging'.")