"""Estimate a pen tip near a tracked writing grip in camera coordinates."""

from dataclasses import dataclass

import cv2
import numpy as np


def locate_pen(frame, grip, wrist, radius):
    """Find a narrow pen shaft aligned with the grip, then estimate its exposed endpoint."""
    height, width = frame.shape[:2]
    x, y = grip
    reach = max(28, round(1.4 * radius))
    left, top = max(0, x - reach), max(0, y - reach)
    right, bottom = min(width, x + reach + 1), min(height, y + reach + 1)
    if right - left < 15 or bottom - top < 15:
        return None
    patch = frame[top:bottom, left:right]
    hue, saturation, value = cv2.split(cv2.cvtColor(patch, cv2.COLOR_BGR2HSV))
    # Saturated blue/green/purple and red shafts stand out from skin and paper.
    colored = (saturation >= 85) & ((hue >= 36) | (hue <= 5)) & (value >= 35)
    dark = value < 85
    origin = np.array((x - left, y - top), dtype=float)
    forward = np.subtract(grip, wrist).astype(float)
    norm = np.linalg.norm(forward)
    if norm < 1:
        return None
    forward /= norm
    best = None
    for mask, color_weight in ((colored, 1.0), (dark, 0.78)):
        strokes = cv2.Canny(mask.astype(np.uint8) * 255, 40, 120)
        lines = cv2.HoughLinesP(strokes, 1, np.pi / 180, max(10, radius // 4),
                                minLineLength=max(15, radius // 2), maxLineGap=max(3, radius // 8))
        if lines is None:
            continue
        for segment in lines.reshape(-1, 4):
            points = segment.reshape(2, 2).astype(float) - origin
            progress = points @ forward
            tip, back = points[np.argmax(progress)], points[np.argmin(progress)]
            if np.linalg.norm(tip) > radius * 1.25 or max(progress) < radius * 0.14:
                continue
            shaft = tip - back
            length = np.linalg.norm(shaft)
            if not length or (shaft @ forward) / length < 0.25:
                continue
            distance_to_grip = abs(shaft[0] * back[1] - shaft[1] * back[0]) / length
            if distance_to_grip > radius * 0.22 or min(progress) > radius * 0.36:
                continue
            score = color_weight * min(1.0, length / radius) * (1 - distance_to_grip / radius)
            if best is None or score > best[0]:
                best = score, tip + origin, shaft / length
        # A visible colored shaft is easier to distinguish from letters, shadows, and the desk.
        if best is not None and best[0] >= 0.47:
            break
    if best is None or best[0] < 0.40:
        return None
    tip = _centered_endpoint(patch, best[1], best[2], radius)
    return tuple(np.rint(tip + (left, top)).astype(int)) if tip is not None else None


def _centered_endpoint(patch, end, axis, radius):
    """Walk across the shaft's center toward the narrow tip until foreground ends."""
    normal = np.array((-axis[1], axis[0]))
    along = np.arange(-max(9, round(radius * 0.38)), max(9, round(radius * 0.50)) + 1)
    half = max(6, min(18, round(radius * 0.28)))
    across = np.arange(-half, half + 1)
    grid = end + along[:, None, None] * axis + across[None, :, None] * normal
    gray = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY)
    strip = cv2.remap(gray, grid[:, :, 0].astype(np.float32), grid[:, :, 1].astype(np.float32),
                      cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=255)
    background = np.median(np.c_[strip[:, :2], strip[:, -2:]], axis=1)
    foreground = strip < np.minimum(196, background - 20)[:, None]
    center, width, misses, last = 0.0, None, 0, None
    for step, row in zip(along, foreground):
        x, y = end + step * axis + center * normal
        if not (0.5 <= x < gray.shape[1] - 1 and 0.5 <= y < gray.shape[0] - 1):
            return None
        edges = np.flatnonzero(np.diff(np.r_[False, row, False]))
        runs = [(across[start], across[stop - 1]) for start, stop in edges.reshape(-1, 2)]
        nearby = [(first, last_pixel) for first, last_pixel in runs
                  if first > -half and last_pixel < half and first - 3 <= center <= last_pixel + 3
                  and last_pixel - first + 1 <= (half if width is None else max(4, width * 1.7))]
        if not nearby:
            misses += 1
            if last is not None and misses >= 3:
                return last[1] if last[0] >= -max(5, radius * 0.15) else None
            continue
        first, last_pixel = min(nearby, key=lambda run: abs(np.mean(run) - center))
        center = (first + last_pixel) / 2
        span = last_pixel - first + 1
        width = span if width is None else (0.75 * width + 0.25 * span)
        misses = 0
        last = step, end + step * axis + center * normal
    # The shaft is clipped or still continues at the edge of the search window.
    if last is not None and misses >= 2 and last[0] >= -max(5, radius * 0.15):
        return last[1]
    return None


def draw_trajectory(samples, area, frame_size, grip_samples):
    """Render ordered segments, leaving gaps where a tip was lost or moved suddenly."""
    if len(samples) < 10:
        return None
    paths = [path for path in split_strokes(samples, frame_size[0]) if len(path) >= 3
             and np.sum(np.linalg.norm(np.diff(np.asarray(path, float), axis=0), axis=1)) >= 8]
    tight = stroke_bounds(paths)
    if tight is None:
        return None
    left, top, right, bottom = tight
    if right - left < 15 or bottom - top < 7:
        return None
    margin = max(12, round((bottom - top) * 0.22))
    width, height = frame_size
    x0, y0, x1, y1 = area or (0, 0, width, height)
    bounds = max(x0, left - margin), max(y0, top - margin), min(x1, right + margin), min(y1, bottom + margin)
    if bounds[2] <= bounds[0] or bounds[3] <= bounds[1]:
        return None
    count = sum(len(path) for path in paths)
    reliable = len(paths) >= 2 and count >= 18 and count >= grip_samples * 0.58
    return render_strokes(paths, bounds), bounds, reliable


def split_strokes(samples, frame_width, gap=0.22):
    """Split ordered (time, point[, stroke id]) samples at time gaps, jumps, or stroke changes."""
    strokes, previous = [], None
    for sample in samples:
        now, point = sample[0], tuple(sample[1])
        stroke = sample[2] if len(sample) > 2 else None
        if (previous is None or now - previous[0] > gap or stroke != previous[2]
                or np.linalg.norm(np.subtract(point, previous[1])) > max(18, frame_width * 0.027)):
            strokes.append([])
        strokes[-1].append(point)
        previous = now, point, stroke
    return [stroke for stroke in strokes if len(stroke) >= 2]


def stroke_bounds(strokes):
    """Tight half-open bounds around all stroke points, or None."""
    if not strokes:
        return None
    points = np.vstack([np.asarray(stroke) for stroke in strokes])
    left, top = points.min(axis=0)
    right, bottom = points.max(axis=0)
    return int(left), int(top), int(right) + 1, int(bottom) + 1


def render_strokes(strokes, bounds, thickness=2):
    """Draw pen-down strokes on white paper in the coordinates of a crop's bounds."""
    left, top, right, bottom = bounds
    image = np.full((max(1, bottom - top), max(1, right - left), 3), 255, np.uint8)
    for stroke in strokes:
        cv2.polylines(image, [(np.asarray(stroke) - (left, top)).astype(np.int32)], False,
                      (12, 12, 12), thickness, cv2.LINE_AA)
    return image


def rotate_points(points, rotation):
    """Turn image coordinates clockwise by 0, 90, 180, or 270 degrees (up to a translation)."""
    x, y = np.asarray(points, float).reshape(-1, 2).T
    return {0: np.c_[x, y], 90: np.c_[-y, x], 180: np.c_[-x, -y], 270: np.c_[y, -x]}[rotation % 360]


def rotate_image(image, rotation):
    codes = {90: cv2.ROTATE_90_CLOCKWISE, 180: cv2.ROTATE_180, 270: cv2.ROTATE_90_COUNTERCLOCKWISE}
    return image if image is None or rotation % 360 == 0 else cv2.rotate(image, codes[rotation % 360])


def render_trace(strokes, rotation=0, text_height=None, glyph=80, longest=1400):
    """Clean pen-down strokes, upright for the writer, cropped tight and scaled to ~``glyph`` px glyphs.

    Strokes are drawn black on white in writing order with a light smoothing of the tip
    jitter; nothing else (hand, paper, earlier ink) is in the image.
    """
    strokes = [rotate_points(stroke, rotation) for stroke in strokes if len(stroke)]
    if not strokes:
        return None
    if not text_height:
        heights = [np.ptp(stroke[:, 1]) for stroke in strokes if len(stroke) > 1]
        text_height = float(np.median(heights)) if heights else 20.0
    points = np.vstack(strokes)
    low, high = points.min(axis=0), points.max(axis=0)
    scale = float(np.clip(glyph / max(text_height, 1.0), 0.2, 8.0))
    pad = round(0.45 * glyph)
    size = (high - low) * scale + 2 * pad
    if size.max() > longest:
        shrink = (longest - 2 * pad) / max((high - low).max() * scale, 1)
        scale *= shrink
        size = (high - low) * scale + 2 * pad
    width, height = (max(96, int(np.ceil(v)) + 1) for v in size)
    image = np.full((height, width, 3), 255, np.uint8)
    offset = np.array((width, height)) / 2 - (low + high) / 2 * scale
    thickness = max(2, round(glyph * 0.06))
    for stroke in strokes:
        if len(stroke) >= 3:
            smooth = stroke.copy()
            smooth[1:-1] = (stroke[:-2] + 2 * stroke[1:-1] + stroke[2:]) / 4
            stroke = smooth
        drawn = np.rint(stroke * scale + offset).astype(np.int32)
        if len(drawn) == 1 or np.ptp(drawn, axis=0).max() < 2:
            cv2.circle(image, tuple(int(v) for v in drawn[0]), thickness, (0, 0, 0), -1, cv2.LINE_AA)
        else:
            cv2.polylines(image, [drawn.reshape(-1, 1, 2)], False, (0, 0, 0), thickness, cv2.LINE_AA)
    return image


def find_closed_loop(points, min_radius, max_radius):
    """Find a final closed, roughly round loop in ordered points.

    Returns (start index, center, radius) for the earliest start whose segment to the
    last point turns at least 1.75 full half-turns around its center and closes near
    its start. Later points may only dwell near the ring.
    """
    points = np.asarray(points, float)
    for start in range(max(0, len(points) - 100), len(points) - 4):
        segment = points[start:]
        center = (segment.min(axis=0) + segment.max(axis=0)) / 2
        distances = np.linalg.norm(segment - center, axis=1)
        radius = float(np.median(distances))
        if not min_radius <= radius <= max_radius or np.std(distances) > 0.35 * radius:
            continue
        angles = np.unwrap(np.arctan2(segment[:, 1] - center[1], segment[:, 0] - center[0]))
        turned = np.max(np.abs(angles - angles[0]))
        closing = np.min(np.linalg.norm(segment[len(segment) // 2:] - segment[0], axis=1))
        if turned >= 1.75 * np.pi and closing <= 0.75 * radius:
            return start, tuple(center), radius
    return None


@dataclass(frozen=True)
class PenTip:
    point: tuple
    grip: tuple
    radius: int
    scale: float
    visual: bool


def _landmark(hand, index):
    landmarks = hand.get("landmarks") or ()
    return np.asarray(landmarks[index]["pixel"], float) if index < len(landmarks) else None


def pen_grip(hand):
    """Grip center, pen-axis rest point, grip search radius, and hand scale, or None.

    MediaPipe landmarks: 0 wrist, 2 thumb MCP, 5 index MCP, 9 middle MCP. A pen passes
    between the thumb and index tips and rests in the web near the index knuckle.
    """
    thumb = np.asarray(hand["fingertips"]["thumb"]["pixel"], float)
    index = np.asarray(hand["fingertips"]["index"]["pixel"], float)
    wrist = _landmark(hand, 0)
    size = np.linalg.norm(index - wrist)
    if size < 12 or np.linalg.norm(index - thumb) > 0.7 * size:
        return None
    grip = (thumb + index) / 2
    knuckle, thumb_base, middle = _landmark(hand, 5), _landmark(hand, 2), _landmark(hand, 9)
    rest = 0.6 * knuckle + 0.4 * thumb_base if knuckle is not None and thumb_base is not None else wrist
    if np.linalg.norm(grip - rest) < 4:
        rest = wrist
    scale = np.linalg.norm(middle - wrist) if middle is not None else 0.55 * size
    return grip, rest, int(np.clip(size * 0.7, 24, 100)), max(float(scale), 8.0)


def estimate_pen_tip(hand, frame=None, bias=(0.0, 0.0)):
    """Place a landmark at the pen tip rather than the index fingertip.

    The geometric estimate continues the rest-to-grip pen axis past the grip by about a
    quarter of the hand scale (the exposed pen). A visible shaft found near the grip
    replaces it when the two agree. ``bias`` is a learned (along, across) correction in
    hand-scale units, used when the shaft is hidden.
    """
    found = pen_grip(hand)
    if found is None:
        return None
    grip, rest, radius, scale = found
    axis = (grip - rest) / np.linalg.norm(grip - rest)
    normal = np.array((-axis[1], axis[0]))
    estimate = grip + axis * scale * (0.28 + bias[0]) + normal * scale * bias[1]
    grip_point = tuple(int(v) for v in np.rint(grip))
    if frame is not None:
        seen = locate_pen(frame, grip_point, tuple(int(v) for v in np.rint(rest)), radius)
        if seen is not None and np.linalg.norm(np.subtract(seen, estimate)) <= max(radius * 0.9, scale * 0.35):
            return PenTip(tuple(int(v) for v in seen), grip_point, radius, scale, True)
        height, width = frame.shape[:2]
        estimate = np.clip(estimate, 0, (width - 1, height - 1))
    return PenTip(tuple(int(v) for v in np.rint(estimate)), grip_point, radius, scale, False)


class OneEuro:
    """One Euro filter: steady when the pen is slow, responsive when it moves quickly."""

    def __init__(self, min_cutoff=2.5, beta=0.03, derivative_cutoff=1.5, reset_after=0.35):
        self.min_cutoff, self.beta, self.derivative_cutoff = min_cutoff, beta, derivative_cutoff
        self.reset_after = reset_after
        self.time = self.value = self.speed = None

    @staticmethod
    def _alpha(dt, cutoff):
        return 1 / (1 + 1 / (2 * np.pi * cutoff * dt))

    def __call__(self, now, value):
        value = np.asarray(value, float)
        if self.time is None or now - self.time > self.reset_after or now <= self.time:
            self.time, self.value, self.speed = now, value, np.zeros_like(value)
            return value
        dt = now - self.time
        self.speed = self.speed + self._alpha(dt, self.derivative_cutoff) * ((value - self.value) / dt - self.speed)
        cutoff = self.min_cutoff + self.beta * np.linalg.norm(self.speed)
        self.value = self.value + self._alpha(dt, cutoff) * (value - self.value)
        self.time = now
        return self.value


class PenTipTracker:
    """Smoothed pen-tip landmark per hand, with a grip-to-tip offset learned from visible shafts."""

    def __init__(self):
        self.filters, self.bias = {}, {}

    def update(self, side, hand, frame, now):
        bias = self.bias.get(side, np.zeros(2))
        tip = estimate_pen_tip(hand, frame, bias)
        if tip is None:
            return None
        if tip.visual:
            plain = np.asarray(estimate_pen_tip(hand).point, float)
            axis = plain - tip.grip
            if np.linalg.norm(axis) > 1:
                axis /= np.linalg.norm(axis)
                offset = (np.asarray(tip.point, float) - plain) / tip.scale
                observed = np.array((offset @ axis, offset @ np.array((-axis[1], axis[0]))))
                self.bias[side] = np.clip(0.8 * bias + 0.2 * observed, -0.3, 0.3)
        smooth = self.filters.setdefault(side, OneEuro())(now, tip.point)
        return PenTip(tuple(int(v) for v in np.rint(smooth)), tip.grip, tip.radius, tip.scale, tip.visual)
