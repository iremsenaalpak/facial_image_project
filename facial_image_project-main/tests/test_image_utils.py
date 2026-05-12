from utils.image_utils import (
    read_image,
    get_image_resolution,
    resize_image,
    convert_to_grayscale,
    normalize_image,
    bgr_to_rgb
)

file_path = "assets/test_images/sample.jpg"

image = read_image(file_path)

if image is None:
    print("Image could not be read.")
else:
    width, height = get_image_resolution(image)
    print("Original resolution:", width, "x", height)

    resized = resize_image(image, (256, 256))
    resized_width, resized_height = get_image_resolution(resized)
    print("Resized resolution:", resized_width, "x", resized_height)

    gray = convert_to_grayscale(image)
    print("Grayscale shape:", gray.shape)

    normalized = normalize_image(image)
    print("Normalized dtype:", normalized.dtype)
    print("Min pixel value:", normalized.min())
    print("Max pixel value:", normalized.max())

    rgb = bgr_to_rgb(image)
    print("RGB shape:", rgb.shape)

    print("All image utility functions work successfully.")