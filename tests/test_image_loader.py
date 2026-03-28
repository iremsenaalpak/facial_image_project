from modules.image_loader import load_and_validate_image

file_path = "assets/test_images/sample.jpg"

result = load_and_validate_image(file_path)

print("Success:", result["success"])
print("Message:", result["message"])
print("Metadata:", result["metadata"])

if result["image"] is not None:
    print("Original image shape:", result["image"].shape)

if result["preview_image"] is not None:
    print("Preview image shape:", result["preview_image"].shape)