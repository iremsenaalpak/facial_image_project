Image Input & Preprocessing modülü hazır.

Ana fonksiyon:
run_preprocessing_pipeline(file_path, target_size=(256,256), min_width=128, min_height=128)

Çıktılar:
- original_image
- preview_image
- face_bbox
- cropped_face
- resized_face
- grayscale_face
- normalized_face
- normalized_grayscale_face
- metadata

Yapılan işlemler:
- file validation
- image loading
- resolution check
- preview preparation
- face detection (MediaPipe)
- bbox expansion
- face crop
- resize
- grayscale conversion
- normalization

Not:
- bbox formatı: (x, y, w, h)
- birden fazla yüz varsa en büyük yüz seçiliyor
- metadata içinde width, height, extension, target_size ve face_detected bilgileri var