from modules.preprocessing import run_preprocessing_pipeline

file_path = "assets/test_images/sample.jpg"

result = run_preprocessing_pipeline(file_path)

print("Success:", result["success"])
print("Message:", result["message"])
print("Metadata:", result["metadata"])
print("Face bbox:", result["face_bbox"])

if result["cropped_face"] is not None:
    print("Cropped face shape:", result["cropped_face"].shape)

if result["resized_face"] is not None:
    print("Resized face shape:", result["resized_face"].shape)

if result["grayscale_face"] is not None:
    print("Grayscale face shape:", result["grayscale_face"].shape)

if result["normalized_face"] is not None:
    print("Normalized face dtype:", result["normalized_face"].dtype)
    print("Normalized face min:", result["normalized_face"].min())
    print("Normalized face max:", result["normalized_face"].max())