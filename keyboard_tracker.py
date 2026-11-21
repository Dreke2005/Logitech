"""Keyboard localization and fingertip-to-key mapping with OpenCV.

The detector finds a keyboard-shaped quadrilateral in a camera frame and
rectifies it to a top-down view. Key labels are generated from a selected
layout (AZERTY/QWERTY); this does not perform OCR on key legends.

Coordinates passed to :func:`finger_pose_to_key` are image pixel coordinates
of the fingertip (for example, a hand-landmark index-tip point).
"""

from dataclasses import dataclass
from typing import Optional, Tuple

import cv2
import numpy as np

Point = Tuple[float, float]

# Key centers measured in the bundled azerty.jpg reference (source pixels).
# These are scaled into the rectified image for fingertip interpolation.
_AZERTY_CENTERS = [
    (82, 177, "²"), (117, 177, "&"), (157, 177, "é"), (197, 177, '"'),
    (236, 177, "'"), (276, 177, "("), (315, 177, "-"), (354, 177, "è"),
    (394, 177, "_"), (433, 177, "ç"), (472, 177, "à"), (508, 177, ")"),
    (100, 214, "A"), (139, 214, "Z"), (178, 214, "E"), (217, 214, "R"),
    (257, 214, "T"), (296, 214, "Y"), (335, 214, "U"), (373, 214, "I"),
    (412, 214, "O"), (450, 214, "P"), (489, 214, "^"), (528, 214, "$"),
    (110, 252, "Q"), (149, 252, "S"), (188, 252, "D"), (227, 252, "F"),
    (266, 252, "G"), (305, 252, "H"), (344, 252, "J"), (384, 252, "K"),
    (423, 252, "L"), (462, 252, "M"), (501, 252, "ù"),
    (129, 291, "W"), (168, 291, "X"), (207, 291, "C"), (246, 291, "V"),
    (285, 291, "B"), (324, 291, "N"), (363, 291, ","), (402, 291, ";"),
    (441, 291, ":"), (480, 291, "!"),
]


@dataclass
class KeyboardDetection:
    """Detected keyboard polygon, rectified image, and perspective transforms."""

    corners: np.ndarray  # ordered TL, TR, BR, BL in source image
    warped: np.ndarray
    image_to_keyboard: np.ndarray
    keyboard_to_image: np.ndarray
    confidence: float


def _order_corners(points: np.ndarray) -> np.ndarray:
    points = np.asarray(points, dtype=np.float32).reshape(4, 2)
    sums = points.sum(axis=1)
    diffs = np.diff(points, axis=1).ravel()
    # diff is y-x: minimum is top-right, maximum is bottom-left.
    return np.array([points[np.argmin(sums)], points[np.argmin(diffs)],
                     points[np.argmax(sums)], points[np.argmax(diffs)]], dtype=np.float32)


def detect_keyboard(frame: np.ndarray, min_area_ratio: float = 0.035,
                    output_size: Tuple[int, int] = (1000, 284)
                    ) -> Optional[KeyboardDetection]:
    """Find the largest plausible keyboard-like quadrilateral in ``frame``.

    Returns ``None`` if no sufficiently large quadrilateral is found. For best
    results, show the complete keyboard with reasonable contrast from the desk.
    The contour heuristic is intended as a first-stage localizer, not a
    general object classifier; clutter can produce false detections.
    """
    if frame is None or frame.size == 0:
        return None
    height, width = frame.shape[:2]
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if frame.ndim == 3 else frame
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(gray, 45, 135)
    edges = cv2.dilate(edges, np.ones((5, 5), np.uint8), iterations=2)
    contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    min_area = width * height * min_area_ratio
    candidates = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if area < min_area:
            continue
        perimeter = cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, 0.025 * perimeter, True)
        if len(approx) != 4 or not cv2.isContourConvex(approx):
            continue
        quad = _order_corners(approx.reshape(4, 2))
        top = np.linalg.norm(quad[1] - quad[0])
        bottom = np.linalg.norm(quad[2] - quad[3])
        left = np.linalg.norm(quad[3] - quad[0])
        right = np.linalg.norm(quad[2] - quad[1])
        aspect = ((top + bottom) / max(left + right, 1.0))
        if not 1.25 <= aspect <= 4.5:
            continue
        # Favor a large candidate with an ordinary keyboard aspect ratio.
        score = area * np.exp(-abs(np.log(aspect / 3.5)))
        candidates.append((score, area, quad))
    if not candidates:
        return None

    score, area, corners = max(candidates, key=lambda item: item[0])
    out_w, out_h = output_size
    destination = np.array([[0, 0], [out_w - 1, 0],
                            [out_w - 1, out_h - 1], [0, out_h - 1]], dtype=np.float32)
    transform = cv2.getPerspectiveTransform(corners, destination)
    inverse = cv2.getPerspectiveTransform(destination, corners)
    warped = cv2.warpPerspective(frame, transform, (out_w, out_h))
    confidence = float(min(1.0, (area / (width * height)) / 0.18))
    return KeyboardDetection(corners, warped, transform, inverse, confidence)


def finger_pose_to_key(finger_xy: Point, detection: KeyboardDetection,
                       layout: str = "azerty") -> Optional[str]:
    """Map an image-space fingertip point to a key label, or ``None``.

    ``finger_xy`` is the fingertip's ``(x, y)`` position in the original
    camera frame. The point is perspective-corrected into keyboard space.
    The AZERTY map uses key centers measured from the bundled reference.
    Large keys (space, enter, shift, etc.) are not mapped yet.
    """
    layout = layout.lower()
    if layout not in ("azerty", "qwerty"):
        raise ValueError("layout must be 'azerty' or 'qwerty'")
    if layout == "qwerty":
        raise NotImplementedError("QWERTY key calibration is not yet available")
    point = np.array([[[float(finger_xy[0]), float(finger_xy[1])]]], dtype=np.float32)
    x, y = cv2.perspectiveTransform(point, detection.image_to_keyboard)[0, 0]
    out_h, out_w = detection.warped.shape[:2]
    if x < 0 or x >= out_w or y < 0 or y >= out_h:
        return None
    scale_x = (out_w - 1) / (911 - 19)
    scale_y = (out_h - 1) / (356 - 103)
    distances = []
    for cx, cy, _label in _AZERTY_CENTERS:
        kx, ky = (cx - 19) * scale_x, (cy - 103) * scale_y
        distances.append(((x - kx) / (18 * scale_x)) ** 2
                         + ((y - ky) / (17 * scale_y)) ** 2)
    nearest = int(np.argmin(distances))
    return _AZERTY_CENTERS[nearest][2] if distances[nearest] <= 1.0 else None


def render_keyboard_map(reference_path: str = "azerty.jpg",
                        highlighted_key: Optional[str] = None) -> np.ndarray:
    """Load the reference keyboard image and overlay each calibrated key name."""
    image = cv2.imread(reference_path)
    if image is None:
        raise FileNotFoundError(f"Cannot read keyboard reference image: {reference_path}")
    for cx, cy, label in _AZERTY_CENTERS:
        selected = label == highlighted_key
        color = (0, 80, 255) if selected else (40, 190, 40)
        cv2.circle(image, (cx, cy), 12 if selected else 9, color, 2)
        cv2.putText(image, label, (cx - 7, cy + 5), cv2.FONT_HERSHEY_SIMPLEX,
                    0.45, color, 1, cv2.LINE_AA)
    return image


def draw_keyboard_view(detection: KeyboardDetection,
                       highlighted_key: Optional[str] = None) -> np.ndarray:
    """Draw labels on the rectified live keyboard image and highlight one key."""
    image = detection.warped.copy()
    height, width = image.shape[:2]
    scale_x = (width - 1) / (911 - 19)
    scale_y = (height - 1) / (356 - 103)
    for cx, cy, label in _AZERTY_CENTERS:
        point = (round((cx - 19) * scale_x), round((cy - 103) * scale_y))
        selected = label == highlighted_key
        color = (0, 80, 255) if selected else (40, 220, 40)
        cv2.circle(image, point, 12 if selected else 8, color, 2)
        cv2.putText(image, label, (point[0] - 7, point[1] + 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)
    return image


def draw_detection(frame: np.ndarray, detection: KeyboardDetection,
                   fingertip: Optional[Point] = None, key: Optional[str] = None) -> np.ndarray:
    """Return a copy annotated with the keyboard outline and optional key."""
    result = frame.copy()
    polygon = np.round(detection.corners).astype(np.int32).reshape(-1, 1, 2)
    cv2.polylines(result, [polygon], True, (40, 220, 40), 3)
    if fingertip is not None:
        point = tuple(np.round(fingertip).astype(int))
        cv2.circle(result, point, 8, (0, 80, 255), -1)
        if key:
            cv2.putText(result, key, (point[0] + 10, point[1] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 80, 255), 2)
    return result

