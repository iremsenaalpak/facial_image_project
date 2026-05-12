import cv2
import numpy as np


def compute_magnitude_spectrum(image):
    """
    Computes log-scaled magnitude spectrum of the image.
    This is used for frequency domain visualization.
    """

    if image is None:
        raise ValueError("Input image is empty.")

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = gray.astype(np.float32)

    f = np.fft.fft2(gray)
    fshift = np.fft.fftshift(f)

    magnitude = np.abs(fshift)

    magnitude_spectrum = 20 * np.log(magnitude + 1)

    magnitude_spectrum = cv2.normalize(
        magnitude_spectrum,
        None,
        0,
        255,
        cv2.NORM_MINMAX
    )

    return magnitude_spectrum.astype(np.uint8)


def compute_frequency_energy(image, radius_ratio=0.15):
    """
    Computes:
    - total spectral energy
    - low-frequency energy
    - high-frequency energy
    - high / low frequency ratio
    """

    if image is None:
        raise ValueError("Input image is empty.")

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = gray.astype(np.float32)

    f = np.fft.fft2(gray)
    fshift = np.fft.fftshift(f)

    magnitude = np.abs(fshift)
    energy = magnitude ** 2

    rows, cols = gray.shape
    crow, ccol = rows // 2, cols // 2

    radius = int(min(rows, cols) * radius_ratio)

    low_mask = np.zeros((rows, cols), dtype=np.uint8)
    cv2.circle(low_mask, (ccol, crow), radius, 1, -1)

    high_mask = 1 - low_mask

    total_energy = np.sum(energy)
    low_energy = np.sum(energy * low_mask)
    high_energy = np.sum(energy * high_mask)

    high_low_ratio = high_energy / (low_energy + 1e-8)

    return {
        "total_energy": float(total_energy),
        "low_frequency_energy": float(low_energy),
        "high_frequency_energy": float(high_energy),
        "high_low_ratio": float(high_low_ratio)
    }


def compare_frequency_analysis(original_image, processed_image):
    """
    Compares original and processed images in frequency domain.
    """

    original_spectrum = compute_magnitude_spectrum(original_image)
    processed_spectrum = compute_magnitude_spectrum(processed_image)

    original_energy = compute_frequency_energy(original_image)
    processed_energy = compute_frequency_energy(processed_image)

    return {
        "original_spectrum": original_spectrum,
        "processed_spectrum": processed_spectrum,
        "original_energy": original_energy,
        "processed_energy": processed_energy
    }