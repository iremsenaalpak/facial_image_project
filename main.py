from modules.preprocessing import run_preprocessing_pipeline


def main():
    file_path = "assets/test_images/sample.jpg"

    result = run_preprocessing_pipeline(file_path)

    print("=== PREPROCESSING RESULT ===")
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


if __name__ == "__main__":
    main()