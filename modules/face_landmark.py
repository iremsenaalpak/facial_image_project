import cv2
import mediapipe as mp
import numpy as np


class FaceLandmarkDetector:
    """
    Detects face bounding box and facial landmarks using MediaPipe.
    Returns:
        - bounding box
        - full landmark list
        - grouped landmark dictionary
    """

    # MediaPipe landmark index groups
    # These are practical subsets for facial manipulation / warping tasks.
    LANDMARK_GROUPS = {
        "left_eye": [33, 133, 160, 159, 158, 157, 173, 153, 144, 145, 146],
        "right_eye": [362, 263, 387, 386, 385, 384, 398, 373, 374, 380, 381],
        "mouth": [61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291],
        "jaw": [127, 234, 93, 132, 58, 172, 136, 150, 149, 176, 148, 152,
                377, 400, 378, 379, 365, 397, 288, 361, 323, 454, 356],
        "left_eyebrow": [46, 53, 52, 65, 55, 70, 63, 105, 66, 107],
        "right_eyebrow": [276, 283, 282, 295, 285, 300, 293, 334, 296, 336]
    }

    def __init__(self,
                 static_image_mode=True,
                 max_num_faces=1,
                 refine_landmarks=True,
                 min_detection_confidence=0.5,
                 min_tracking_confidence=0.5):
        self.mp_face_detection = mp.solutions.face_detection
        self.mp_face_mesh = mp.solutions.face_mesh

        self.face_detection = self.mp_face_detection.FaceDetection(
            model_selection=1,
            min_detection_confidence=min_detection_confidence
        )

        self.face_mesh = self.mp_face_mesh.FaceMesh(
            static_image_mode=static_image_mode,
            max_num_faces=max_num_faces,
            refine_landmarks=refine_landmarks,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence
        )

    def _normalized_to_pixel(self, landmark, width, height):
        x = min(max(int(landmark.x * width), 0), width - 1)
        y = min(max(int(landmark.y * height), 0), height - 1)
        return (x, y)

    def _extract_bbox_from_detection(self, detection, width, height):
        bbox = detection.location_data.relative_bounding_box

        x = int(bbox.xmin * width)
        y = int(bbox.ymin * height)
        w = int(bbox.width * width)
        h = int(bbox.height * height)

        x = max(0, x)
        y = max(0, y)
        w = min(w, width - x)
        h = min(h, height - y)

        return (x, y, w, h)

    def _group_landmarks(self, full_landmarks):
        grouped = {}
        for group_name, indices in self.LANDMARK_GROUPS.items():
            grouped[group_name] = [full_landmarks[i] for i in indices if i < len(full_landmarks)]
        return grouped

    def detect(self, image):
        """
        Parameters:
            image: BGR image (OpenCV format)

        Returns:
            dict with:
            {
                "success": bool,
                "bbox": (x, y, w, h) or None,
                "landmarks": [(x, y), ...],
                "grouped_landmarks": {...},
                "image_size": {"width": w, "height": h}
            }
        """
        if image is None:
            return {
                "success": False,
                "bbox": None,
                "landmarks": [],
                "grouped_landmarks": {},
                "image_size": None,
                "message": "Input image is None."
            }

        height, width = image.shape[:2]
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        detection_results = self.face_detection.process(rgb_image)
        mesh_results = self.face_mesh.process(rgb_image)

        bbox = None
        if detection_results.detections:
            bbox = self._extract_bbox_from_detection(
                detection_results.detections[0], width, height
            )

        if not mesh_results.multi_face_landmarks:
            return {
                "success": False,
                "bbox": bbox,
                "landmarks": [],
                "grouped_landmarks": {},
                "image_size": {"width": width, "height": height},
                "message": "No face landmarks detected."
            }

        face_landmarks = mesh_results.multi_face_landmarks[0]
        full_landmarks = [
            self._normalized_to_pixel(lm, width, height)
            for lm in face_landmarks.landmark
        ]

        # If face detection failed but landmarks exist, estimate bbox from landmarks
        if bbox is None and full_landmarks:
            xs = [pt[0] for pt in full_landmarks]
            ys = [pt[1] for pt in full_landmarks]
            x_min, x_max = min(xs), max(xs)
            y_min, y_max = min(ys), max(ys)
            bbox = (x_min, y_min, x_max - x_min, y_max - y_min)

        grouped_landmarks = self._group_landmarks(full_landmarks)

        return {
            "success": True,
            "bbox": bbox,
            "landmarks": full_landmarks,
            "grouped_landmarks": grouped_landmarks,
            "image_size": {"width": width, "height": height},
            "message": "Face and landmarks detected successfully."
        }

    def close(self):
        self.face_detection.close()
        self.face_mesh.close()