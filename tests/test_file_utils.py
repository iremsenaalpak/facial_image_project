from utils.file_utils import validate_image_file

test_path = "assets/test_images/sample.jpg"
result = validate_image_file(test_path)
print(result)