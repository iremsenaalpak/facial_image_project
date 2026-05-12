from utils.file_utils import validate_image_file
from utils.image_utils import read_image, get_image_resolution, bgr_to_rgb


def load_input_image(file_path):
    """
    Loads an image from the given file path.

    Returns:
        image if successful, otherwise None
    """
    return read_image(file_path)


def validate_image_resolution(image, min_width=128, min_height=128):
    """
    Validates whether the image resolution meets the minimum requirements.

    Returns:
        dict with success status and resolution info
    """
    width, height = get_image_resolution(image)

    if width < min_width or height < min_height:
        return {
            "success": False,
            "message": f"Image resolution is too low. Minimum required is {min_width}x{min_height}.",
            "width": width,
            "height": height
        }

    return {
        "success": True,
        "message": "Image resolution is valid.",
        "width": width,
        "height": height
    }


def prepare_preview_image(image):
    """
    Prepares the image for display by converting it from BGR to RGB.

    Returns:
        RGB image
    """
    return bgr_to_rgb(image)


def load_and_validate_image(file_path, min_width=128, min_height=128):
    """
    Loads and validates an input image.

    Steps:
    1. Check file existence and extension
    2. Read image
    3. Check if image is readable
    4. Validate resolution
    5. Prepare preview image

    Returns:
        dict containing validation results, image data, and metadata
    """
    file_validation = validate_image_file(file_path)

    if not file_validation["success"]:
        return {
            "success": False,
            "message": file_validation["message"],
            "image": None,
            "preview_image": None,
            "metadata": None
        }

    image = load_input_image(file_path)

    if image is None:
        return {
            "success": False,
            "message": "Image could not be read. The file may be corrupted or unsupported.",
            "image": None,
            "preview_image": None,
            "metadata": None
        }

    resolution_validation = validate_image_resolution(image, min_width, min_height)

    if not resolution_validation["success"]:
        return {
            "success": False,
            "message": resolution_validation["message"],
            "image": None,
            "preview_image": None,
            "metadata": {
                "width": resolution_validation["width"],
                "height": resolution_validation["height"],
                "extension": file_validation["extension"]
            }
        }

    preview_image = prepare_preview_image(image)

    return {
        "success": True,
        "message": "Image loaded and validated successfully.",
        "image": image,
        "preview_image": preview_image,
        "metadata": {
            "width": resolution_validation["width"],
            "height": resolution_validation["height"],
            "extension": file_validation["extension"]
        }
    }