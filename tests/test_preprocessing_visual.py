import cv2
import matplotlib.pyplot as plt

from modules.preprocessing import run_preprocessing_pipeline
from utils.image_utils import bgr_to_rgb


def draw_bbox(image, bbox, color=(0, 255, 0), thickness=2):
    """
    Draws a bounding box on a copy of the image.
    """
    image_copy = image.copy()
    x, y, w, h = bbox
    cv2.rectangle(image_copy, (x, y), (x + w, y + h), color, thickness)
    return image_copy


file_path = "assets/test_images/sample.jpg"

result = run_preprocessing_pipeline(file_path)

print("Success:", result["success"])
print("Message:", result["message"])

if not result["success"]:
    print("Pipeline failed. No visualization will be shown.")
else:
    original_image = result["original_image"]
    face_bbox = result["face_bbox"]
    cropped_face = result["cropped_face"]
    resized_face = result["resized_face"]
    grayscale_face = result["grayscale_face"]

    boxed_image = draw_bbox(original_image, face_bbox)

    original_rgb = bgr_to_rgb(original_image)
    boxed_rgb = bgr_to_rgb(boxed_image)
    cropped_rgb = bgr_to_rgb(cropped_face)
    resized_rgb = bgr_to_rgb(resized_face)

    plt.figure(figsize=(12, 8))

    plt.subplot(2, 3, 1)
    plt.imshow(original_rgb)
    plt.title("Original Image")
    plt.axis("off")

    plt.subplot(2, 3, 2)
    plt.imshow(boxed_rgb)
    plt.title("Detected Face Bounding Box")
    plt.axis("off")

    plt.subplot(2, 3, 3)
    plt.imshow(cropped_rgb)
    plt.title("Cropped Face")
    plt.axis("off")

    plt.subplot(2, 3, 4)
    plt.imshow(resized_rgb)
    plt.title("Resized Face (256x256)")
    plt.axis("off")

    plt.subplot(2, 3, 5)
    plt.imshow(grayscale_face, cmap="gray")
    plt.title("Grayscale Face")
    plt.axis("off")

    plt.tight_layout()
    plt.show()