import cv2
import numpy as np


def read_image(file_path):
    """
    Reads an image from the given file path using OpenCV.

    Returns:
        image (numpy.ndarray) if successful, otherwise None
    """
    image = cv2.imread(file_path)
    return image


def get_image_resolution(image):
    """
    Returns the width and height of the image.

    Returns:
        (width, height)
    """
    height, width = image.shape[:2]
    return width, height


def resize_image(image, target_size=(256, 256)):
    """
    Resizes the image to the given target size.

    Args:
        image: Input image
        target_size: (width, height)

    Returns:
        Resized image
    """
    target_width, target_height = target_size
    original_height, original_width = image.shape[:2]

    if original_width > target_width or original_height > target_height:
        interpolation = cv2.INTER_AREA
    else:
        interpolation = cv2.INTER_LINEAR

    resized_image = cv2.resize(image, (target_width, target_height), interpolation=interpolation)
    return resized_image


def convert_to_grayscale(image):
    """
    Converts a BGR image to grayscale.

    Returns:
        Grayscale image
    """
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return gray_image


def normalize_image(image):
    """
    Normalizes image pixel values to the range [0, 1].

    Returns:
        Normalized image as float32
    """
    normalized = image.astype(np.float32) / 255.0
    return normalized


def bgr_to_rgb(image):
    """
    Converts an image from BGR format to RGB format.

    Returns:
        RGB image
    """
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return rgb_image


def crop_image(image, x, y, w, h):
    """
    Crops the image using the given bounding box.

    Args:
        image: Input image
        x, y: Top-left corner
        w, h: Width and height of crop region

    Returns:
        Cropped image
    """
    cropped = image[y:y+h, x:x+w]
    return cropped


def blend_with_mask(base_img, color_bgr, mask_2d, alpha):
    """Alpha-blend a solid color onto base_img using a float32 mask [0,1]."""
    color = np.array(color_bgr, dtype=np.float32)
    weight = (mask_2d.astype(np.float32) * alpha)[:, :, np.newaxis]
    result = base_img.astype(np.float32) * (1.0 - weight) + color * weight
    return np.clip(result, 0, 255).astype(np.uint8)


def recolor_preserve_luminance(image_bgr, target_bgr, mask_float, strength=1.0):
    """Replace chroma (a, b in LAB) with the target color's chroma, keeping the
    image's L channel so highlights and strand detail survive. mask_float is a
    HxW float in [0, 1]; strength scales the effect."""
    lab = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
    target_pixel = np.uint8([[[int(target_bgr[0]), int(target_bgr[1]), int(target_bgr[2])]]])
    target_lab = cv2.cvtColor(target_pixel, cv2.COLOR_BGR2LAB).astype(np.float32)[0, 0]

    weight = np.clip(mask_float.astype(np.float32) * float(strength), 0.0, 1.0)
    lab[:, :, 1] = lab[:, :, 1] * (1.0 - weight) + target_lab[1] * weight
    lab[:, :, 2] = lab[:, :, 2] * (1.0 - weight) + target_lab[2] * weight

    lab = np.clip(lab, 0, 255).astype(np.uint8)
    return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)


def create_polygon_mask(shape_hw, points):
    """Return a uint8 binary mask (255 inside polygon) for image of shape (h, w)."""
    mask = np.zeros(shape_hw[:2], dtype=np.uint8)
    cv2.fillPoly(mask, [np.array(points, dtype=np.int32)], 255)
    return mask


def constrain_mask_to_head(binary_mask, landmarks=None):
    """Drop false-positive blobs from a hair/segmentation mask.

    The selfie segmenter sometimes tags background regions (a wall, a poster)
    as hair. On a cropped face that never shows, but on a full webcam frame it
    paints the room. This keeps only the connected components that fall within
    a generous box around the detected head; with no landmarks it falls back to
    the single largest component. Input/output: uint8, values 0/1.
    """
    bm = (binary_mask > 0).astype(np.uint8)
    num, labels, stats, _ = cv2.connectedComponentsWithStats(bm, connectivity=8)
    if num <= 1:
        return bm  # only background

    if landmarks and len(landmarks) > 0:
        xs = [p[0] for p in landmarks]
        ys = [p[1] for p in landmarks]
        x0, x1 = min(xs), max(xs)
        y0, y1 = min(ys), max(ys)
        fw = max(x1 - x0, 1)
        fh = max(y1 - y0, 1)
        # Generous head region — hair rises well above the forehead.
        rx0, rx1 = x0 - 0.5 * fw, x1 + 0.5 * fw
        ry0, ry1 = y0 - 1.2 * fh, y1 + 0.4 * fh

        keep = np.zeros_like(bm)
        for lbl in range(1, num):
            cx0 = stats[lbl, cv2.CC_STAT_LEFT]
            cy0 = stats[lbl, cv2.CC_STAT_TOP]
            cx1 = cx0 + stats[lbl, cv2.CC_STAT_WIDTH]
            cy1 = cy0 + stats[lbl, cv2.CC_STAT_HEIGHT]
            # Keep the component if its bbox overlaps the head region at all.
            if cx1 >= rx0 and cx0 <= rx1 and cy1 >= ry0 and cy0 <= ry1:
                keep[labels == lbl] = 1
        return keep

    areas = stats[1:, cv2.CC_STAT_AREA]
    largest = 1 + int(np.argmax(areas))
    return (labels == largest).astype(np.uint8)