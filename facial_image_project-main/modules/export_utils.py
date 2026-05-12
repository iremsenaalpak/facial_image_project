import os
import csv
from fpdf import FPDF


def export_metrics_to_csv(metrics, output_path):
    """
    Exports evaluation metrics into CSV file.
    """

    with open(output_path, mode="w", newline="") as file:
        writer = csv.writer(file)

        writer.writerow(["Metric", "Value"])

        for key, value in metrics.items():
            writer.writerow([key, value])


def export_metrics_to_pdf(metrics, output_path):
    """
    Exports evaluation metrics into PDF file.
    """

    pdf = FPDF()

    pdf.add_page()

    pdf.set_font("Arial", size=16)

    pdf.cell(200, 10, txt="Face Transformation Evaluation Report", ln=True, align="C")

    pdf.ln(10)

    pdf.set_font("Arial", size=12)

    for key, value in metrics.items():
        line = f"{key.upper()}: {value}"

        pdf.cell(200, 10, txt=line, ln=True)

    pdf.output(output_path)