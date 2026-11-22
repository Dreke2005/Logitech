"""Select a marked region from a captured RGB page using OpenCV's BGR image."""

from dataclasses import dataclass
from typing import Optional, Tuple

import cv2
import numpy as np

Bounds = Tuple[int, int, int, int]


@dataclass(frozen=True)
class PaperSelection:
    """A crop and its half-open pixel bounds in the unmodified source frame."""

    image: np.ndarray
    bounds: Optional[Bounds]
    kind: str
    reason: str = ""


def select_paper_region(frame: np.ndarray, roi: Optional[Bounds] = None, *, allow_dashed=False) -> PaperSelection:
    """Use a manual rectangle, the contents of a drawn loop, or the full page.

    Draw one oval around the required ink, leaving a gap between the writing and
    the oval. The original frame stays available for review or another selection.
    """
    if frame.dtype != np.uint8 or frame.ndim != 3 or frame.shape[2] != 3 or not frame.size:
        raise ValueError("Select paper from a non-empty BGR image.")
    height, width = frame.shape[:2]
    if roi is not None:
        if len(roi) != 4:
            raise ValueError("A selection needs two corners.")
        ax, ay, bx, by = roi
        left, right = sorted((max(0, min(width, ax)), max(0, min(width, bx))))
        top, bottom = sorted((max(0, min(height, ay)), max(0, min(height, by))))
        if right - left < 8 or bottom - top < 8:
            raise ValueError("Drag a larger selection around the writing.")
        bounds = int(left), int(top), int(right), int(bottom)
        x0, y0, x1, y1 = bounds
        return PaperSelection(frame[y0:y1, x0:x1].copy(), bounds, "manual")
    if min(height, width) < 48:
        return PaperSelection(frame, None, "full")

    scale = min(1.0, 1100 / max(width, height))
    preview = cv2.resize(frame, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA) if scale < 1 else frame
    gray = cv2.cvtColor(preview, cv2.COLOR_BGR2GRAY)
    block = min(81, max(31, (min(gray.shape) // 10) | 1))
    ink = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                cv2.THRESH_BINARY_INV, block, 9)
    candidates = _loops(ink, 3)
    if allow_dashed:
        for gap in (25, 17, 9):
            dashed = _loops(ink, gap)
            if dashed:
                candidates = dashed
                break
    if len(candidates) != 1:
        reason = "multiple circles" if candidates else ""
        return PaperSelection(frame, None, "full", reason)

    contour, (x, y, w, h) = candidates[0]
    source_scale = np.array([width / gray.shape[1], height / gray.shape[0]])
    margin = max(6, round(0.035 * min(w, h) / scale))
    x0 = max(0, int(x * source_scale[0]) - margin)
    y0 = max(0, int(y * source_scale[1]) - margin)
    x1 = min(width, int(np.ceil((x + w) * source_scale[0])) + margin)
    y1 = min(height, int(np.ceil((y + h) * source_scale[1])) + margin)
    mask = np.zeros((y1 - y0, x1 - x0), dtype=np.uint8)
    local_contour = np.rint(contour * source_scale).astype(np.int32) - np.array([x0, y0], dtype=np.int32)
    cv2.drawContours(mask, [local_contour], -1, 255, cv2.FILLED)
    inset = max(1, round(2 / scale))
    mask = cv2.erode(mask, np.ones((2 * inset + 1, 2 * inset + 1), dtype=np.uint8))
    cropped = frame[y0:y1, x0:x1].copy()
    cropped[mask == 0] = 255
    return PaperSelection(cropped, (x0, y0, x1, y1), "circle")


def _loops(ink, gap):
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (gap, gap))
    strokes = (cv2.morphologyEx(ink, cv2.MORPH_CLOSE, kernel) if gap == 3
               else cv2.dilate(ink, kernel))
    contours, hierarchy = cv2.findContours(strokes, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    if hierarchy is None:
        return []
    candidates = []
    joined = set()
    for inner, (_, _, child, parent) in enumerate(hierarchy[0]):
        if (gap == 3 and child < 0) or parent < 0 or hierarchy[0, parent, 3] >= 0 or parent in joined:
            continue
        contour = contours[inner]
        outer = contours[parent]
        if gap > 3:
            if len(outer) < 5:
                continue
            center, diameters, angle = cv2.fitEllipse(outer)
            axes = tuple(round(axis / 2 - gap / 2 - 4) for axis in diameters)
            if min(axes) < 12:
                continue
            contour = cv2.ellipse2Poly(tuple(round(v) for v in center), axes, round(angle), 0, 360, 5)
        x, y, w, h = cv2.boundingRect(contour)
        if (w < max(48, ink.shape[1] * 0.08) or h < max(24, ink.shape[0] * 0.045)
                or max(w / h, h / w) > 10):
            continue
        area = cv2.contourArea(contour)
        outer_area = cv2.contourArea(outer)
        if (not outer_area or area / outer_area < (0.64 if gap == 3 else 0.45)
                or not 0.50 <= area / (w * h) <= 0.93
                or area > ink.size * 0.78):
            continue
        if len(cv2.approxPolyDP(contour, max(1.5, 0.07 * min(w, h)), True)) < 5:
            continue
        hull_area = cv2.contourArea(cv2.convexHull(contour))
        if not hull_area or area / hull_area < 0.86:
            continue
        # A loop is a hollow stroke with separate ink inside it.
        enclosed = np.zeros((h, w), dtype=np.uint8)
        cv2.drawContours(enclosed, [contour], -1, 255, cv2.FILLED, offset=(-x, -y))
        clear_interior = cv2.erode(enclosed, np.ones((5, 5), dtype=np.uint8))
        interior_ink = cv2.countNonZero(cv2.bitwise_and(ink[y:y+h, x:x+w], clear_interior))
        if max(15, area * 0.003) <= interior_ink <= area * 0.38:
            if gap > 3:
                perimeter = np.zeros_like(ink)
                cv2.drawContours(perimeter, [contour], -1, 255, gap + 8)
                original = cv2.bitwise_and(perimeter, ink)
                _, _, stats, centroids = cv2.connectedComponentsWithStats(original)
                fragments = centroids[1:][stats[1:, cv2.CC_STAT_AREA] >= 5]
                quadrants = {(cx >= center[0], cy >= center[1]) for cx, cy in fragments}
                if len(quadrants) < 4:
                    continue
                joined.add(parent)
            candidates.append((contour, (x, y, w, h)))
    return candidates
