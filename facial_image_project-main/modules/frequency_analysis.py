import cv2
import numpy as np


def convert_to_gray(image):
    if image is None:
        raise ValueError("Input image is empty.")

    if len(image.shape) == 2:
        return image.astype(np.float32)

    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).astype(np.float32)


def compute_fft(image):
    gray = convert_to_gray(image)

    f = np.fft.fft2(gray)
    fshift = np.fft.fftshift(f)

    return fshift


def compute_magnitude_spectrum(image):
    """
    Computes log-scaled magnitude spectrum of the image.
    """

    fshift = compute_fft(image)

    magnitude = np.abs(fshift)
    magnitude_spectrum = np.log1p(magnitude)

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

    gray = convert_to_gray(image)
    fshift = compute_fft(gray)

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
    high_percent = (high_energy / (total_energy + 1e-8)) * 100
    low_percent = (low_energy / (total_energy + 1e-8)) * 100

    return {
        "total_energy": float(total_energy),
        "low_frequency_energy": float(low_energy),
        "high_frequency_energy": float(high_energy),
        "high_low_ratio": float(high_low_ratio),
        "low_frequency_percent": float(low_percent),
        "high_frequency_percent": float(high_percent)
    }


def compare_frequency_analysis(original_image, processed_image):
    """
    Compares original and processed images in frequency domain.
    """

    original_spectrum = compute_magnitude_spectrum(original_image)
    processed_spectrum = compute_magnitude_spectrum(processed_image)

    original_energy = compute_frequency_energy(original_image)
    processed_energy = compute_frequency_energy(processed_image)

    difference = {
        "total_energy_difference": processed_energy["total_energy"] - original_energy["total_energy"],
        "low_energy_difference": processed_energy["low_frequency_energy"] - original_energy["low_frequency_energy"],
        "high_energy_difference": processed_energy["high_frequency_energy"] - original_energy["high_frequency_energy"],
        "ratio_difference": processed_energy["high_low_ratio"] - original_energy["high_low_ratio"],
        "high_percent_difference": processed_energy["high_frequency_percent"] - original_energy["high_frequency_percent"],
        "low_percent_difference": processed_energy["low_frequency_percent"] - original_energy["low_frequency_percent"]
    }

    return {
        "original_spectrum": original_spectrum,
        "processed_spectrum": processed_spectrum,
        "original_energy": original_energy,
        "processed_energy": processed_energy,
        "difference": difference
    }


def format_scientific(value):
    """
    Formats large energy values for UI tables.
    """

    return f"{value:.4e}"


def create_frequency_table(original_energy, processed_energy, difference):
    """
    Creates a table-friendly list for HTML/UI display.
    """

    table = [
        {
            "metric": "Total Spectral Energy",
            "original": format_scientific(original_energy["total_energy"]),
            "processed": format_scientific(processed_energy["total_energy"]),
            "difference": format_scientific(difference["total_energy_difference"])
        },
        {
            "metric": "Low Frequency Energy",
            "original": format_scientific(original_energy["low_frequency_energy"]),
            "processed": format_scientific(processed_energy["low_frequency_energy"]),
            "difference": format_scientific(difference["low_energy_difference"])
        },
        {
            "metric": "High Frequency Energy",
            "original": format_scientific(original_energy["high_frequency_energy"]),
            "processed": format_scientific(processed_energy["high_frequency_energy"]),
            "difference": format_scientific(difference["high_energy_difference"])
        },
        {
            "metric": "High / Low Frequency Ratio",
            "original": f"{original_energy['high_low_ratio']:.6f}",
            "processed": f"{processed_energy['high_low_ratio']:.6f}",
            "difference": f"{difference['ratio_difference']:.6f}"
        },
        {
            "metric": "Low Frequency Percent",
            "original": f"{original_energy['low_frequency_percent']:.2f}%",
            "processed": f"{processed_energy['low_frequency_percent']:.2f}%",
            "difference": f"{difference['low_percent_difference']:.2f}%"
        },
        {
            "metric": "High Frequency Percent",
            "original": f"{original_energy['high_frequency_percent']:.2f}%",
            "processed": f"{processed_energy['high_frequency_percent']:.2f}%",
            "difference": f"{difference['high_percent_difference']:.2f}%"
        }
    ]

    return table


def frequency_table_to_html(table):
    """
    Converts frequency table into an HTML table string.
    This is useful for FastAPI app.py result display.
    """

    rows = ""

    for item in table:
        rows += f"""
        <tr>
            <td>{item["metric"]}</td>
            <td>{item["original"]}</td>
            <td>{item["processed"]}</td>
            <td>{item["difference"]}</td>
        </tr>
        """

    html = f"""
    <table style="width:100%; border-collapse: collapse; margin-top: 12px;">
        <thead>
            <tr style="background:#eeeeee;">
                <th style="border:1px solid #ccc; padding:8px;">Metric</th>
                <th style="border:1px solid #ccc; padding:8px;">Original</th>
                <th style="border:1px solid #ccc; padding:8px;">Processed</th>
                <th style="border:1px solid #ccc; padding:8px;">Difference</th>
            </tr>
        </thead>
        <tbody>
            {rows}
        </tbody>
    </table>
    """

    return html


def run_frequency_analysis(original_image, processed_image):
    """
    Main function for frequency analysis.
    Returns spectra, energy values, difference values and HTML table.
    """

    result = compare_frequency_analysis(original_image, processed_image)

    table = create_frequency_table(
        result["original_energy"],
        result["processed_energy"],
        result["difference"]
    )

    html_table = frequency_table_to_html(table)

    result["table"] = table
    result["html_table"] = html_table

    return result