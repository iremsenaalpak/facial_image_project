import cv2


def draw_bounding_box(image, bbox, color=(0, 255, 0), thickness=2):
    """
    Draws a bounding box on the image.
    bbox format: (x, y, w, h)
    """
    if image is None or bbox is None:
        return image

    output = image.copy()
    x, y, w, h = bbox
    cv2.rectangle(output, (x, y), (x + w, y + h), color, thickness)
    return output


def draw_landmarks(image, landmarks, color=(0, 0, 255), radius=1, show_index=False):
    """
    Draws all landmarks on the image.
    landmarks format: [(x, y), ...]
    """
    if image is None:
        return None

    output = image.copy()

    for idx, (x, y) in enumerate(landmarks):
        cv2.circle(output, (x, y), radius, color, -1)
        if show_index:
            cv2.putText(
                output,
                str(idx),
                (x + 2, y - 2),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.3,
                (255, 255, 255),
                1,
                cv2.LINE_AA
            )

    return output


def draw_grouped_landmarks(image, grouped_landmarks, color_map=None, radius=2):
    """
    Draw landmark groups with different colors.
    grouped_landmarks format:
    {
        "left_eye": [(x, y), ...],
        ...
    }
    """
    if image is None:
        return None

    output = image.copy()

    default_color_map = {
        "left_eye": (255, 0, 0),
        "right_eye": (0, 255, 0),
        "mouth": (0, 0, 255),
        "jaw": (255, 255, 0),
        "left_eyebrow": (255, 0, 255),
        "right_eyebrow": (0, 255, 255)
    }

    if color_map is None:
        color_map = default_color_map

    for group_name, points in grouped_landmarks.items():
        color = color_map.get(group_name, (200, 200, 200))
        for (x, y) in points:
            cv2.circle(output, (x, y), radius, color, -1)

    return output


def visualize_face_data(image, bbox=None, landmarks=None, grouped_landmarks=None,
                        show_bbox=True, show_landmarks=True, use_grouped=False):
    """
    Master visualization function with toggle support.
    """
    if image is None:
        return None

    output = image.copy()

    if show_bbox and bbox is not None:
        output = draw_bounding_box(output, bbox)

    if show_landmarks:
        if use_grouped and grouped_landmarks is not None:
            output = draw_grouped_landmarks(output, grouped_landmarks)
        elif landmarks is not None:
            output = draw_landmarks(output, landmarks)

    return output