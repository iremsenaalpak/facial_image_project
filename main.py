import cv2

from modules.preprocessing import run_preprocessing_pipeline
from modules.face_landmark import FaceLandmarkDetector
from modules.visualization import visualize_face_data
from utils.helpers import save_landmarks_to_json, ensure_directory
from modules.aging import aging_deaging_pipeline

def main():
    file_path = "assets/test_images/sample.jpg"

    # 1) Preprocessing pipeline
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

    if not result["success"]:
        print("\nPreprocessing failed. Landmark detection skipped.")
        return

    # 2) Landmark detection için kullanılacak görüntüyü seç
    # En mantıklısı resized_face, yoksa cropped_face kullanmak
    face_image = None

    if result["resized_face"] is not None:
        face_image = result["resized_face"]
    elif result["cropped_face"] is not None:
        face_image = result["cropped_face"]

    if face_image is None:
        print("\nNo valid face image found for landmark detection.")
        return

    # Eğer grayscale ise BGR'ye çevir, çünkü MediaPipe renkli görüntü bekler
    if len(face_image.shape) == 2:
        face_image = cv2.cvtColor(face_image, cv2.COLOR_GRAY2BGR)

    # 3) Face + landmark detection
    detector = FaceLandmarkDetector()
    landmark_result = detector.detect(face_image)

    print("\n=== FACE LANDMARK RESULT ===")
    print("Success:", landmark_result["success"])
    print("Message:", landmark_result["message"])
    print("Bounding box:", landmark_result["bbox"])
    print("Landmark count:", len(landmark_result["landmarks"]))
    print("Grouped landmark keys:", list(landmark_result["grouped_landmarks"].keys()))

    if landmark_result["success"]:
        # 4) Sonuç görselleştirme
        visualized_image = visualize_face_data(
            image=face_image,
            bbox=landmark_result["bbox"],
            landmarks=landmark_result["landmarks"],
            grouped_landmarks=landmark_result["grouped_landmarks"],
            show_bbox=True,
            show_landmarks=True,
            use_grouped=True
        )
        # Aging işlemi
        aged_image = aging_deaging_pipeline(
            face_image,
            mode="aging",
            intensity=0.7
        )

        # De-aging işlemi
        deaged_image = aging_deaging_pipeline(
           face_image,
           mode="deaging",
           intensity=0.45
        )
        # 5) Kayıt klasörleri
        ensure_directory("results/images")
        ensure_directory("results/json")

        output_image_path = "results/images/landmark_result.jpg"
        output_json_path = "results/json/landmarks.json"

        # Görsel kaydet
        cv2.imwrite(output_image_path, visualized_image)
        aged_output_path = "results/images/aged_result.jpg"
        deaged_output_path = "results/images/deaged_result.jpg"

        cv2.imwrite(aged_output_path, aged_image)
        cv2.imwrite(deaged_output_path, deaged_image)

        print("Saved aged image:", aged_output_path)
        print("Saved de-aged image:", deaged_output_path)

        # JSON kaydet
        save_landmarks_to_json(landmark_result, output_json_path)

        print("Saved landmark image:", output_image_path)
        print("Saved landmark json:", output_json_path)

    detector.close()


if __name__ == "__main__":
    main()