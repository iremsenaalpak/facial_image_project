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