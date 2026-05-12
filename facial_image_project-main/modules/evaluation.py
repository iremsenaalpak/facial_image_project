import cv2
import numpy as np
from skimage.metrics import structural_similarity as ssim


def ensure_same_size(original, processed):
    """
    Ensures both images have the same size.
    If sizes are different, processed image is resized to original size.
    """
    if original is None or processed is None:
        raise ValueError("Input images cannot be None.")

    if original.shape[:2] != processed.shape[:2]:
        processed = cv2.resize(
            processed,
            (original.shape[1], original.shape[0])
        )

    return original, processed


def compute_mse(original, processed):
    """
    Computes Mean Squared Error between original and processed image.
    """
    original, processed = ensure_same_size(original, processed)

    original = original.astype(np.float32)
    processed = processed.astype(np.float32)

    mse_value = np.mean((original - processed) ** 2)

    return float(mse_value)


def compute_psnr(original, processed):
    """
    Computes Peak Signal-to-Noise Ratio based on MSE.
    """
    mse_value = compute_mse(original, processed)

    if mse_value == 0:
        return float("inf")

    max_pixel = 255.0
    psnr_value = 10 * np.log10((max_pixel ** 2) / mse_value)

    return float(psnr_value)


def compute_ssim(original, processed):
    """
    Computes Structural Similarity Index between original and processed image.
    """
    original, processed = ensure_same_size(original, processed)

    if len(original.shape) == 3:
        original_gray = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)
    else:
        original_gray = original

    if len(processed.shape) == 3:
        processed_gray = cv2.cvtColor(processed, cv2.COLOR_BGR2GRAY)
    else:
        processed_gray = processed

    ssim_value = ssim(
        original_gray,
        processed_gray,
        data_range=processed_gray.max() - processed_gray.min()
    )

    return float(ssim_value)


def run_evaluation(original, processed):
    """
    Runs all quantitative evaluation metrics.
    """
    mse_value = compute_mse(original, processed)
    psnr_value = compute_psnr(original, processed)
    ssim_value = compute_ssim(original, processed)

    return {
        "mse": mse_value,
        "psnr": psnr_value,
        "ssim": ssim_value
    }


def evaluation_table_to_html(metrics):
    """
    Converts evaluation metrics into an HTML table.
    """
    html = f"""
    <table style="width:100%; border-collapse: collapse; margin-top: 12px;">
        <thead>
            <tr style="background:#eeeeee;">
                <th style="border:1px solid #ccc; padding:8px;">Metric</th>
                <th style="border:1px solid #ccc; padding:8px;">Value</th>
                <th style="border:1px solid #ccc; padding:8px;">Meaning</th>
            </tr>
        </thead>
        <tbody>
            <tr>
                <td style="border:1px solid #ccc; padding:8px;">MSE</td>
                <td style="border:1px solid #ccc; padding:8px;">{metrics["mse"]:.4f}</td>
                <td style="border:1px solid #ccc; padding:8px;">Pixel-level difference between original and processed image</td>
            </tr>
            <tr>
                <td style="border:1px solid #ccc; padding:8px;">PSNR</td>
                <td style="border:1px solid #ccc; padding:8px;">{metrics["psnr"]:.4f} dB</td>
                <td style="border:1px solid #ccc; padding:8px;">Signal quality based on MSE</td>
            </tr>
            <tr>
                <td style="border:1px solid #ccc; padding:8px;">SSIM</td>
                <td style="border:1px solid #ccc; padding:8px;">{metrics["ssim"]:.4f}</td>
                <td style="border:1px solid #ccc; padding:8px;">Structural similarity between images</td>
            </tr>
        </tbody>
    </table>
    """

    return html