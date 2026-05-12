import cv2
import numpy as np


def clip_uint8(image):
    return np.clip(image, 0, 255).astype(np.uint8)


def normalize_image(image):
    image = image.astype(np.float32)
    image = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX)
    return clip_uint8(image)


def create_skin_mask(image):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    lower_skin = np.array([0, 20, 60], dtype=np.uint8)
    upper_skin = np.array([25, 255, 255], dtype=np.uint8)

    mask = cv2.inRange(hsv, lower_skin, upper_skin)

    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.GaussianBlur(mask, (15, 15), 0)

    mask = mask.astype(np.float32) / 255.0
    mask = np.expand_dims(mask, axis=2)

    return mask


def blend_with_mask(original, processed, mask):
    original = original.astype(np.float32)
    processed = processed.astype(np.float32)

    result = original * (1 - mask) + processed * mask
    return clip_uint8(result)


# =========================================================
# AGING FUNCTIONS
# =========================================================

def high_frequency_boost(image, intensity=0.5):
    intensity = np.clip(intensity, 0, 1)

    img_float = image.astype(np.float32)
    blurred = cv2.GaussianBlur(img_float, (0, 0), sigmaX=2.0)
    high_freq = img_float - blurred

    boosted = img_float + high_freq * (0.6 + 1.2 * intensity)
    return clip_uint8(boosted)


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
    intensity = np.clip(intensity, 0, 1)

    rng = np.random.default_rng(seed)
    img = image.astype(np.float32)

    noise = rng.normal(0, 1, img.shape).astype(np.float32)
    noise = cv2.GaussianBlur(noise, (7, 7), 0)

    texture_strength = 2.0 + 5.0 * intensity
    textured = img + noise * texture_strength

    return clip_uint8(textured)


def add_wrinkle_lines(image, intensity=0.5):
    intensity = np.clip(intensity, 0, 1)

    h, w = image.shape[:2]
    overlay = np.zeros_like(image, dtype=np.uint8)

    shadow_color = (
        int(22 + 28 * intensity),
        int(22 + 28 * intensity),
        int(22 + 28 * intensity)
    )

    thickness = 2

    # Forehead horizontal wrinkles
    forehead_lines = 18
    for i in range(forehead_lines):
        y = int(h * (0.13 + i * 0.020))
        width_scale = 0.20 + (i % 4) * 0.025
        x_shift = int(((-1) ** i) * w * 0.012)

        center = (w // 2 + x_shift, y)
        axes = (int(w * width_scale), int(h * 0.010))

        cv2.ellipse(
            overlay,
            center,
            axes,
            0,
            180,
            360,
            shadow_color,
            thickness
        )

    # Crow's feet / eye corner wrinkles
    eye_data = [
        ((int(w * 0.31), int(h * 0.43)), -1),
        ((int(w * 0.69), int(h * 0.43)), 1)
    ]

    for center, direction in eye_data:
        for offset in [-18, -10, -2, 6, 14, 22]:
            x1, y1 = center
            x2 = x1 + direction * int(w * 0.11)
            y2 = y1 + offset
            cv2.line(overlay, (x1, y1), (x2, y2), shadow_color, 1)

    # Nasolabial folds
    cv2.ellipse(
        overlay,
        (int(w * 0.37), int(h * 0.61)),
        (int(w * 0.065), int(h * 0.22)),
        18,
        270,
        360,
        shadow_color,
        2
    )

    cv2.ellipse(
        overlay,
        (int(w * 0.63), int(h * 0.61)),
        (int(w * 0.065), int(h * 0.22)),
        -18,
        180,
        270,
        shadow_color,
        2
    )

    # Under-eye wrinkles / bags
    for cx in [0.34, 0.66]:
        cv2.ellipse(
            overlay,
            (int(w * cx), int(h * 0.50)),
            (int(w * 0.11), int(h * 0.040)),
            0,
            0,
            360,
            shadow_color,
            1
        )

        cv2.ellipse(
            overlay,
            (int(w * cx), int(h * 0.53)),
            (int(w * 0.09), int(h * 0.030)),
            0,
            0,
            360,
            shadow_color,
            1
        )

    # Jaw sagging shadow
    cv2.ellipse(
        overlay,
        (w // 2, int(h * 0.79)),
        (int(w * 0.24), int(h * 0.055)),
        0,
        0,
        180,
        shadow_color,
        2
    )

    overlay = cv2.GaussianBlur(overlay, (7, 7), 0)

    alpha = 0.50 + 0.35 * intensity
    result = cv2.addWeighted(image, 1.0, overlay, alpha, 0)

    return clip_uint8(result)


def adjust_aging_tone(image, intensity=0.5):
    intensity = np.clip(intensity, 0, 1)

    alpha = 0.96 + 0.08 * intensity
    beta = -14 * intensity

    toned = cv2.convertScaleAbs(image, alpha=alpha, beta=beta)

    hsv = cv2.cvtColor(toned, cv2.COLOR_BGR2HSV).astype(np.float32)
    hsv[:, :, 1] *= 1.0 - 0.18 * intensity
    hsv[:, :, 2] *= 1.0 - 0.04 * intensity

    toned = cv2.cvtColor(
        np.clip(hsv, 0, 255).astype(np.uint8),
        cv2.COLOR_HSV2BGR
    )

    toned = toned.astype(np.float32)

    # Slight yellow/gray older skin tone
    toned[:, :, 0] *= 1.02
    toned[:, :, 1] *= 1.00
    toned[:, :, 2] *= 0.96

    return clip_uint8(toned)


def apply_aging(image, intensity=0.5, use_fft=True, use_skin_mask=True):
    if image is None:
        raise ValueError("Input image is empty.")

    intensity = np.clip(intensity, 0, 1)
    original = image.copy()

    aged = high_frequency_boost(original, intensity)

    if use_fft:
        fft_aged = fft_high_pass_aging(original, intensity)
        aged = cv2.addWeighted(aged, 0.78, fft_aged, 0.22, 0)

    aged = add_wrinkle_texture(aged, intensity)
    aged = add_wrinkle_lines(aged, intensity)
    aged = adjust_aging_tone(aged, intensity)

    if use_skin_mask:
        mask = create_skin_mask(original)
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


def brighten_under_eye_area(image, intensity=0.5):
    intensity = np.clip(intensity, 0, 1)

    h, w = image.shape[:2]
    result = image.copy()

    overlay = np.zeros_like(image, dtype=np.uint8)

    left_center = (int(w * 0.34), int(h * 0.47))
    right_center = (int(w * 0.66), int(h * 0.47))

    color = (18, 18, 18)

    cv2.ellipse(
        overlay,
        left_center,
        (int(w * 0.11), int(h * 0.045)),
        0,
        0,
        360,
        color,
        -1
    )

    cv2.ellipse(
        overlay,
        right_center,
        (int(w * 0.11), int(h * 0.045)),
        0,
        0,
        360,
        color,
        -1
    )

    overlay = cv2.GaussianBlur(overlay, (31, 31), 0)

    alpha = 0.18 + 0.18 * intensity
    result = cv2.addWeighted(result, 1.0, overlay, alpha, 0)

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


def selective_skin_smoothing(image, intensity=0.5):
    intensity = np.clip(intensity, 0, 1)

    skin_mask = create_skin_mask(image)

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


def apply_deaging(image, intensity=0.5, use_fft=True, use_skin_mask=True):
    if image is None:
        raise ValueError("Input image is empty.")

    intensity = np.clip(intensity, 0, 1)
    original = image.copy()

    deaged = selective_skin_smoothing(original, intensity * 0.40)

    if use_fft:
        fft_deaged = fft_low_pass_deaging(original, intensity)
        deaged = cv2.addWeighted(deaged, 0.90, fft_deaged, 0.10, 0)

    deaged = brighten_under_eye_area(deaged, intensity)
    deaged = restore_young_skin_color(deaged, intensity)
    deaged = add_skin_glow(deaged, intensity * 0.20)
    deaged = subtle_face_tightening(deaged, intensity * 0.15)
    deaged = sharpen_important_features(deaged, intensity)
    deaged = enhance_hair_youthfulness(deaged, intensity)
    deaged = adjust_deaging_tone(deaged, intensity)

    if use_skin_mask:
        mask = create_skin_mask(original)
        deaged = blend_with_mask(original, deaged, mask)

    return deaged


# =========================================================
# MAIN PIPELINE
# =========================================================

def aging_deaging_pipeline(image, mode="aging", intensity=0.5):
    if image is None:
        raise ValueError("Input image is empty.")

    intensity = np.clip(intensity, 0, 1)

    if mode == "aging":
        return apply_aging(image, intensity=intensity, use_skin_mask=True)

    elif mode == "deaging":
        return apply_deaging(image, intensity=intensity, use_skin_mask=True)

    else:
        raise ValueError("Mode must be 'aging' or 'deaging'.")