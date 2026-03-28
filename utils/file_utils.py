import os

ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png"}


def get_file_extension(file_path):
    """
    Returns the file extension in lowercase.
    Example: 'photo.JPG' -> '.jpg'
    """
    _, extension = os.path.splitext(file_path)
    return extension.lower()


def file_exists(file_path):
    """
    Checks whether the given file exists.
    """
    return os.path.isfile(file_path)


def is_allowed_extension(file_path):
    """
    Checks whether the file extension is supported.
    """
    extension = get_file_extension(file_path)
    return extension in ALLOWED_EXTENSIONS


def validate_image_file(file_path):
    """
    Validates whether the file exists and has a supported image extension.

    Returns:
        dict: {
            "success": bool,
            "message": str,
            "file_path": str,
            "extension": str
        }
    """
    if not file_exists(file_path):
        return {
            "success": False,
            "message": "File does not exist.",
            "file_path": file_path,
            "extension": None
        }

    extension = get_file_extension(file_path)

    if extension not in ALLOWED_EXTENSIONS:
        return {
            "success": False,
            "message": "Unsupported file format. Only JPG, JPEG, and PNG are allowed.",
            "file_path": file_path,
            "extension": extension
        }

    return {
        "success": True,
        "message": "File is valid.",
        "file_path": file_path,
        "extension": extension
    }