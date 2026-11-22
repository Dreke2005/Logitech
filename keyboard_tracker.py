"""Keyboard localization and fingertip-to-key mapping with OpenCV.

The detector finds a keyboard-shaped quadrilateral in a camera frame and
rectifies it to a top-down view. Key labels are generated from a selected
layout (AZERTY/QWERTY); this does not perform OCR on key legends.

Coordinates passed to :func:`finger_pose_to_key` are image pixel coordinates
of the fingertip (for example, a hand-landmark index-tip point).
"""

from dataclasses import dataclass
from typing import Iterable, Optional, Tuple, Union

import cv2
import numpy as np

from keyboard_extraction import H as CLEAR_TO_BLUR_H
from keyboard_extraction import KEYS as EXTRACTED_KEYS
from keyboard_extraction import closest_key

Point = Tuple[float, float]
EXTRACTED_LAYOUT_SIZE = (950.0, 300.0)

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
    """Find a keyboard's dark outer frame around its light key bed.

    The candidate must be a large, wide quadrilateral whose border is darker
    than its mostly light interior. This suppresses contours from individual
    key legends and light backgrounds.
    """
    if frame is None or frame.size == 0:
        return None
    height, width = frame.shape[:2]
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if frame.ndim == 3 else frame.copy()
    gray = cv2.GaussianBlur(gray, (5, 5), 0)

    # Threshold dark pixels so the keyboard's black surround becomes one
    # connected component even when the keys and background are both white.
    threshold, _ = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
    threshold = min(max(threshold, 55), 175)
    dark = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY_INV)[1]
    close_w = max(3, (width // 160) | 1)
    close_h = max(3, (height // 160) | 1)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (close_w, close_h))
    dark = cv2.morphologyEx(dark, cv2.MORPH_CLOSE, kernel, iterations=2)
    contours, _ = cv2.findContours(dark, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    min_area = width * height * min_area_ratio
    max_area = width * height * 0.88
    candidates = []
    out_w, out_h = 320, 100
    for contour in contours:
        area = cv2.contourArea(contour)
        if not min_area <= area <= max_area:
            continue
        perimeter = cv2.arcLength(contour, True)
        for epsilon_ratio in (0.02, 0.03, 0.04, 0.055, 0.07):
            approx = cv2.approxPolyDP(contour, epsilon_ratio * perimeter, True)
            if len(approx) != 4 or not cv2.isContourConvex(approx):
                continue
            quad = _order_corners(approx.reshape(4, 2))
            top = np.linalg.norm(quad[1] - quad[0])
            bottom = np.linalg.norm(quad[2] - quad[3])
            left = np.linalg.norm(quad[3] - quad[0])
            right = np.linalg.norm(quad[2] - quad[1])
            aspect = ((top + bottom) / max(left + right, 1.0))
            if not 1.6 <= aspect <= 5.0:
                continue

            destination = np.array([[0, 0], [out_w - 1, 0],
                                    [out_w - 1, out_h - 1], [0, out_h - 1]],
                                   dtype=np.float32)
            rectifier = cv2.getPerspectiveTransform(quad, destination)
            sample = cv2.warpPerspective(gray, rectifier, (out_w, out_h))
            border = np.concatenate((sample[:8, :].ravel(), sample[-8:, :].ravel(),
                                     sample[:, :8].ravel(), sample[:, -8:].ravel()))
            interior = sample[20:80, 24:-24]
            contrast = float(np.mean(interior) - np.mean(border))
            if contrast < 12.0:
                continue

            # Prefer a standard keyboard shape and a clearly dark surround.
            shape_score = np.exp(-abs(np.log(aspect / 3.5)))
            area_score = min(1.0, area / (width * height * 0.20))
            score = (contrast / 255.0) * shape_score * area_score
            candidates.append((score, area, quad))
            break
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
    camera frame. The point is perspective-corrected into keyboard space and
    looked up against the rectangles in :mod:`keyboard_extraction`.
    """
    layout = layout.lower()
    if layout != "azerty":
        raise ValueError("keyboard_extraction currently defines only the AZERTY layout")
    point = np.array([[[float(finger_xy[0]), float(finger_xy[1])]]], dtype=np.float32)
    x, y = cv2.perspectiveTransform(point, detection.image_to_keyboard)[0, 0]
    out_h, out_w = detection.warped.shape[:2]
    if x < 0 or x >= out_w or y < 0 or y >= out_h:
        return None
    # Convert the manually rectified point into keyboard_extraction's clear
    # keyboard plane, then use its calibrated homography and rectangle lookup.
    clear_w, clear_h = EXTRACTED_LAYOUT_SIZE
    clear_point = np.array([[[x * clear_w / out_w,
                              y * clear_h / out_h]]], dtype=np.float32)
    blur_point = cv2.perspectiveTransform(clear_point, CLEAR_TO_BLUR_H)[0, 0]
    label, distance = closest_key(float(blur_point[0]), float(blur_point[1]), max_dist=12.0)
    return label if distance <= 12.0 else None


def _draw_extracted_key_boxes(image: np.ndarray,
                              highlighted_key: Optional[Union[str, Iterable[str]]] = None,
                              bounds: Optional[Tuple[float, float, float, float]] = None
                              ) -> np.ndarray:
    """Draw key rectangles into a target rectangle (x0, y0, x1, y1)."""
    result = image.copy()
    height, width = result.shape[:2]
    x0, y0, x1, y1 = bounds or (0.0, 0.0, float(width), float(height))
    scale_x = (x1 - x0) / EXTRACTED_LAYOUT_SIZE[0]
    scale_y = (y1 - y0) / EXTRACTED_LAYOUT_SIZE[1]
    selected_keys = ({highlighted_key} if isinstance(highlighted_key, str)
                     else set(highlighted_key or ()))
    for name, (kx0, ky0, kx1, ky1) in EXTRACTED_KEYS.items():
        p0 = (round(x0 + kx0 * scale_x), round(y0 + ky0 * scale_y))
        p1 = (round(x0 + kx1 * scale_x), round(y0 + ky1 * scale_y))
        selected = name in selected_keys
        color = (0, 80, 255) if selected else (40, 220, 40)
        cv2.rectangle(result, p0, p1, color, 2 if selected else 1)
        thickness = 2
        box_width, box_height = p1[0] - p0[0], p1[1] - p0[1]
        font_scale = 0.82 if len(name) == 1 else 0.68
        (text_width, text_height), baseline = cv2.getTextSize(
            name, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
        while font_scale > 0.30 and (text_width > box_width - 4
                                     or text_height + baseline > box_height - 2):
            font_scale *= 0.90
            (text_width, text_height), baseline = cv2.getTextSize(
                name, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
        text_x = p0[0] + max(2, (box_width - text_width) // 2)
        text_y = p0[1] + max(text_height + 1, (box_height + text_height - baseline) // 2)
        cv2.putText(result, name, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX,
                    font_scale, color, thickness, cv2.LINE_AA)
    return result


def render_keyboard_map(reference_path: str = "azerty.jpg",
                        highlighted_key: Optional[Union[str, Iterable[str]]] = None) -> np.ndarray:
    """Overlay extracted key rectangles on the bundled AZERTY reference."""
    image = cv2.imread(reference_path)
    if image is None:
        raise FileNotFoundError(f"Cannot read keyboard reference image: {reference_path}")
    return _draw_extracted_key_boxes(image, highlighted_key, (19.0, 103.0, 911.0, 356.0))


def draw_keyboard_view(detection: KeyboardDetection,
                       highlighted_key: Optional[Union[str, Iterable[str]]] = None) -> np.ndarray:
    """Draw extracted key rectangles on the rectified keyboard and highlight one."""
    return _draw_extracted_key_boxes(detection.warped, highlighted_key)


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

