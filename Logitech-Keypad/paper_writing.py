"""Collect in-memory writing evidence from synchronized RGB frames and host hand landmarks.

While a recording runs, every writing-hand sample places a smoothed pen-tip landmark
(paper_pen.PenTipTracker). A tip sample counts as pen-down only once new ink is visible
under it compared with the reference frame captured when the key was pressed, so hover
moves drop out of the trajectory. The recognition crop is the pen-down trajectory joined
with nearby new ink. A small closed circle drawn last ends the recording on its own.
"""

from collections import deque
from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np

from paper_pen import PenTipTracker, find_closed_loop, pen_grip, render_strokes, stroke_bounds
from paper_selection import PaperSelection, select_paper_region

INK_DELTA = 28  # Gray levels a pixel must darken against the reference frame to count as ink.
# Exposure-normalized ink tests: darkness is 1 - pixel / local paper brightness, so an
# auto-exposure or lighting change that scales the whole page does not create ink.
INK_DARK = 0.07  # New ink is at least this much darker than the paper around it ...
INK_GAIN = 0.07  # ... and darker by at least this much than the same spot before writing.
# Faint ballpoint on an overexposed page measured only 0.07-0.09 darker than the paper on the
# LVNext camera, while frame-to-frame noise stays below that in components of a few pixels.
PAPER_SHARE = 0.6  # A tip sample over a region darker than this share of the paper is off the page.


@dataclass(frozen=True)
class WritingSelection:
    page: PaperSelection
    evidence: np.ndarray = None
    trace: np.ndarray = None
    trace_reliable: bool = False
    end_marked: bool = False
    strokes: tuple = ()  # Pen-down strokes in frame coordinates, in writing order.
    rotation: int = 0  # Clockwise degrees that turn the frame upright for the writer.
    text_height: float = 0.0  # Typical glyph height in frame pixels.


@dataclass
class TipSample:
    time: float
    point: tuple
    visual: bool = False
    down: Optional[bool] = None  # Unknown until the paper under the sample is visible.
    wrist: Optional[tuple] = None


@dataclass(frozen=True)
class EndCircle:
    side: str
    start: float
    center: tuple
    radius: float


def _gray(frame):
    height, width = frame.shape[:2]
    scale = min(1, 960 / max(height, width))
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return cv2.resize(gray, (round(width * scale), round(height * scale)), interpolation=cv2.INTER_AREA)


def paper_bounds(frame):
    """Find a bright, paper-shaped region. A largely paper-filled frame can be used directly."""
    gray = _gray(frame)
    rows, cols = gray.shape
    bright = cv2.inRange(gray, 175, 255)
    height, width = frame.shape[:2]
    if np.mean(bright != 0) > 0.8:
        return 0, 0, width, height
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (17, 17))
    joined = cv2.morphologyEx(bright, cv2.MORPH_CLOSE, kernel)
    contours, _ = cv2.findContours(joined, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    for contour in sorted(contours, key=cv2.contourArea, reverse=True):
        if not 0.045 * rows * cols <= cv2.contourArea(contour) <= 0.68 * rows * cols:
            continue
        polygon = cv2.approxPolyDP(contour, 0.024 * cv2.arcLength(contour, True), True)
        if len(polygon) != 4 or not cv2.isContourConvex(polygon):
            continue
        x, y, w, h = cv2.boundingRect(polygon)
        if x < 2 or y < 2 or x + w > cols - 2 or y + h > rows - 2 or max(w / h, h / w) > 3:
            continue
        return (round(x * width / cols), round(y * height / rows),
                round((x + w) * width / cols), round((y + h) * height / rows))
    return None


def _segment_distance(point, start, end):
    point, start, end = (np.asarray(value, float) for value in (point, start, end))
    span = end - start
    length = span @ span
    if length < 1e-6:
        return float(np.linalg.norm(point - start))
    along = np.clip((point - start) @ span / length, 0, 1)
    return float(np.linalg.norm(point - start - along * span))


class WritingSession:
    def __init__(self, baseline=None, area=None, *, end_circle=True):
        self.history = deque(maxlen=50)
        self.path = deque(maxlen=2000 if baseline is not None else 180)
        self.baseline = _gray(baseline) if baseline is not None else None
        # Full-resolution reference for thin ink under individual pen-tip samples.
        self.reference = cv2.cvtColor(baseline, cv2.COLOR_BGR2GRAY) if baseline is not None else None
        self.area = area
        # Some workflows (branch diagrams, copy marks) use circles as content, not as an end mark.
        self.end_circle_enabled = end_circle
        self.paper_level = None
        if self.reference is not None:
            x0, y0, x1, y1 = area or (0, 0, self.reference.shape[1], self.reference.shape[0])
            region = self.reference[y0:y1, x0:x1]
            self.paper_level = float(np.percentile(region, 90)) if region.size else 255.0
        self.last_sample = 0.0
        self.last_path = {}
        self.last_check = 0.0
        self.tips = {}
        self.tracker = PenTipTracker()
        self.current_tips = {}
        self.occluders = []
        self.occluded_at = float("-inf")
        self.end_marked = False
        self.end_circle = None
        self.finish_reason = None
        self.gain = 1.0
        width = baseline.shape[1] if baseline is not None else 640
        self.ink_radius = max(4, round(width * 0.006))

    # Observation -------------------------------------------------------------------------

    def observe(self, frame, hand_sample, now):
        checkpoint = now - self.last_sample >= 0.24
        if checkpoint:
            self.history.append((now, _gray(frame)))
            self.last_sample = now
        hands = hand_sample["hands"] if hand_sample else ()
        occluders = []
        for hand in hands:
            found = pen_grip(hand)
            side = hand["handedness"]
            if found is not None:
                grip, _, radius, _ = found
                x, y = point = tuple(int(v) for v in np.rint(grip))
                inside = not self.area or (self.area[0] <= x < self.area[2] and self.area[1] <= y < self.area[3])
                if inside and now - self.last_path.get(side, 0.0) >= 0.03:
                    self.path.append((now, side, point, radius))
                    self.last_path[side] = now
                    if self.reference is not None:
                        tip = self.tracker.update(side, hand, frame, now)
                        if tip is not None:
                            self.current_tips[side] = now, tip
                            if self._contains(tip.point):
                                wrist = hand.get("landmarks") or ()
                                wrist = tuple(wrist[0]["pixel"]) if wrist else None
                                self.tips.setdefault(side, deque(maxlen=3000)).append(
                                    TipSample(now, tip.point, tip.visual, wrist=wrist))
            occluders.append(self._occluder(hand, side, now))
        if hands:
            self.occluders, self.occluded_at = occluders, now
        elif now - self.occluded_at > 0.3:
            self.occluders = []  # Hold the last hand shape through brief detection dropouts.
        if checkpoint and self.reference is not None and self.reference.shape == frame.shape[:2]:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            self._classify(gray, now)
            if self.end_circle_enabled and not self.end_marked:
                self._detect_end_circle(gray, now)

    def _occluder(self, hand, side, now):
        landmarks = [item["pixel"] for item in hand.get("landmarks", ())]
        points = landmarks + [item["pixel"] for item in hand.get("fingertips", {}).values()]
        scale = max(12.0, float(np.ptp(np.asarray(points, float), axis=0).max(initial=0)) * 0.5)
        if len(landmarks) > 9:
            # The forearm is not a landmark but covers the page behind the wrist.
            wrist, middle = np.asarray(landmarks[0], float), np.asarray(landmarks[9], float)
            back = wrist - middle
            if np.linalg.norm(back) > 1:
                back /= np.linalg.norm(back)
                across = np.array((-back[1], back[0])) * scale * 0.55
                reach = wrist + back * scale * 2.5
                points += [tuple(wrist + across), tuple(wrist - across), tuple(reach + across), tuple(reach - across)]
        hull = cv2.convexHull(np.rint(np.asarray(points, float)).astype(np.int32).reshape(-1, 1, 2))
        recent = self.current_tips.get(side)
        pen = None
        if recent is not None and now - recent[0] < 0.2:
            tip = recent[1]
            pen = tip.grip, tip.point, max(5.0, tip.scale * 0.1)
        return hull, max(6.0, scale * 0.12), pen

    def _contains(self, point):
        if self.area is None:
            return True
        x0, y0, x1, y1 = self.area
        return x0 <= point[0] < x1 and y0 <= point[1] < y1

    def _occluded(self, point):
        for hull, margin, pen in self.occluders:
            if cv2.pointPolygonTest(hull, (float(point[0]), float(point[1])), True) >= -margin:
                return True
            if pen is not None and _segment_distance(point, pen[0], pen[1]) <= pen[2] + self.ink_radius:
                return True
        return False

    def _ink_near(self, gray, point):
        """True for new thin ink close to a point, False for clean paper, None when covered."""
        r = self.ink_radius
        height, width = gray.shape
        x, y = (int(round(v)) for v in point)
        x0, y0, x1, y1 = max(0, x - r), max(0, y - r), min(width, x + r + 1), min(height, y + r + 1)
        if x1 <= x0 or y1 <= y0:
            return False
        before = self.reference[y0:y1, x0:x1].astype(np.float32)
        after = gray[y0:y1, x0:x1].astype(np.float32)
        paper_before, paper = np.percentile(before, 85), np.percentile(after, 85)
        if self.paper_level is not None and paper_before < PAPER_SHARE * self.paper_level:
            return False  # Off the page: desk, lap, keypad, or the other hand.
        if paper < 0.86 * self.gain * paper_before:
            return None  # A hand or shadow darkens the paper itself.
        # Compare darkness relative to the local paper so exposure changes cancel out.
        dark_before = 1 - before / max(paper_before, 1.0)
        dark_after = 1 - after / max(paper, 1.0)
        new = (dark_after - dark_before >= INK_GAIN) & (dark_after >= INK_DARK)
        # Only ink in the core counts, so hovering beside earlier strokes stays pen-up.
        rows, cols = np.ogrid[y0 - y:y1 - y, x0 - x:x1 - x]
        core = rows * rows + cols * cols <= max(2, round(0.75 * r)) ** 2
        return bool(np.count_nonzero(new & core) >= 3)

    def _ink_at(self, gray, point, previous=None):
        """Pen-down test for a tip sample: new ink at the point and halfway back along its motion."""
        here = self._ink_near(gray, point)
        if not here or previous is None:
            return here
        if np.linalg.norm(np.subtract(point, previous)) <= self.ink_radius:
            return here
        middle = self._ink_near(gray, np.add(point, previous) / 2)
        return here if middle is None else middle

    def _update_gain(self, gray):
        """Global brightness ratio against the reference, measured on paper pixels."""
        small, reference = cv2.resize(gray, None, fx=0.25, fy=0.25, interpolation=cv2.INTER_AREA), \
            cv2.resize(self.reference, None, fx=0.25, fy=0.25, interpolation=cv2.INTER_AREA)
        if self.area:
            x0, y0, x1, y1 = (round(v / 4) for v in self.area)
            small, reference = small[y0:y1, x0:x1], reference[y0:y1, x0:x1]
        paper = reference >= 0.85 * (self.paper_level or 255)
        if np.count_nonzero(paper) >= 50:
            self.gain = float(np.clip(np.median(small[paper]) / max(np.median(reference[paper]), 1), 0.5, 2.0))

    def _classify(self, gray, now):
        self._update_gain(gray)
        for samples in self.tips.values():
            previous = None
            for sample in samples:
                if sample.down is None and now - sample.time >= 0.15 and not self._occluded(sample.point):
                    linked = previous is not None and sample.time - previous.time <= 0.3
                    sample.down = self._ink_at(gray, sample.point, previous.point if linked else None)
                previous = sample

    # Strokes and end circle ------------------------------------------------------------------

    def _writer_path(self):
        paths = {}
        for item in self.path:
            paths.setdefault(item[1], []).append(item)

        def motion(path):
            return sum(distance for previous, current in zip(path, path[1:])
                       if current[0] - previous[0] < 0.35
                       and 2 <= (distance := np.linalg.norm(np.subtract(current[2], previous[2]))) < 85)

        def pen_down(side):
            return sum(1 for sample in self.tips.get(side, ()) if sample.down)

        return max(paths.values(), key=lambda path: (pen_down(path[0][1]), motion(path)), default=[])

    def _stroke_samples(self, side, gray=None, pending=False):
        """Pen-down tip samples split into strokes at lifts, time gaps, and jumps.

        Undecided samples are resolved against ``gray`` (a final frame) when given, or
        kept as provisional pen-down samples with ``pending``. Samples from the end
        circle onward are never part of the writing.
        """
        width = self.reference.shape[1] if self.reference is not None else 640
        jump = max(18, width * 0.027)
        strokes, previous, raw = [], None, None
        for sample in self.tips.get(side, ()):
            if self.end_circle is not None and self.end_circle.side == side and sample.time >= self.end_circle.start:
                break
            down = sample.down
            if gray is not None and down is not False:
                # Re-check against the final frame: a pen or hand shadow that looked like ink
                # while writing has moved away by now, real ink is still there.
                linked = raw is not None and sample.time - raw.time <= 0.3
                seen = None if self._occluded(sample.point) else self._ink_at(
                    gray, sample.point, raw.point if linked else None)
                down = bool(down) if seen is None else seen
            elif down is None:
                down = pending
            raw = sample
            if not down:
                previous = None
                continue
            if (previous is None or sample.time - previous.time > 0.3
                    or np.linalg.norm(np.subtract(sample.point, previous.point)) > jump):
                strokes.append([])
            strokes[-1].append(sample)
            previous = sample
        return strokes

    def _detect_end_circle(self, gray, now):
        path = self._writer_path()
        if not path:
            return
        side = path[0][1]
        strokes = self._stroke_samples(side, pending=True)
        if len(strokes) < 2 or not any(sample.down for stroke in strokes[:-1] for sample in stroke):
            return  # The end mark must follow some writing.
        last = strokes[-1]
        width = self.reference.shape[1]
        points = np.asarray([sample.point for sample in last], float)
        loop = find_closed_loop(points, max(4.0, width * 0.005), width * 0.07)
        if loop is None:
            return
        start, center, radius = loop
        segment = points[start:]
        closing = start + len(segment) // 2 + int(np.argmin(
            np.linalg.norm(segment[len(segment) // 2:] - segment[0], axis=1)))
        if now - last[closing].time < 0.5:
            return  # Wait for a short pause so a written "o" followed by more text is not taken.
        earlier = [np.asarray([sample.point for sample in stroke], float) for stroke in strokes[:-1]]
        if any(np.min(np.linalg.norm(stroke - center, axis=1)) < 1.2 * radius for stroke in earlier):
            return  # A loop around existing writing is a selection, not an end mark.
        heights = [np.ptp(stroke[:, 1]) for stroke in earlier if len(stroke) >= 3]
        if heights and 2 * radius < 0.8 * float(np.median(heights)):
            return  # Loops inside letters such as "e" or "a" are smaller than the text.
        ring = self._ring_in_ink(gray, center, radius)
        if ring is not None:
            self.end_circle = EndCircle(side, last[start].time, *ring)
            self.end_marked = True

    def _occlusion_mask(self, shape, origin, grow=1.0):
        mask = np.zeros(shape, np.uint8)
        x0, y0 = origin
        for hull, margin, pen in self.occluders:
            shifted = (hull.reshape(-1, 2) - (x0, y0)).astype(np.int32)
            cv2.fillConvexPoly(mask, shifted, 1)
            if margin * grow >= 1:
                cv2.polylines(mask, [shifted.reshape(-1, 1, 2)], True, 1, int(2 * margin * grow))
            if pen is not None:
                grip, tip = np.asarray(pen[0], float), np.asarray(pen[1], float)
                # With grow, also cover the barrel behind the grip, which lies over the page.
                back = grip + (grip - tip) * (3.5 if grow > 1 else 0.0)
                cv2.line(mask, tuple(int(v) for v in np.rint(back - origin)),
                         tuple(int(v) for v in np.rint(tip - origin)), 1,
                         int(2 * (pen[2] + self.ink_radius) * grow))
        return mask

    def _ring_in_ink(self, gray, center, radius):
        """Confirm the traced loop as a visible ring of new ink; return its refined center and radius."""
        height, width = gray.shape
        reach = int(2 * radius + self.ink_radius)
        cx, cy = int(round(center[0])), int(round(center[1]))
        x0, y0, x1, y1 = max(0, cx - reach), max(0, cy - reach), min(width, cx + reach + 1), min(height, cy + reach + 1)
        before = self.reference[y0:y1, x0:x1].astype(np.int16)
        after = gray[y0:y1, x0:x1].astype(np.int16)
        if not after.size:
            return None
        hidden = self._occlusion_mask(after.shape, (x0, y0)) != 0
        new = (before - after >= INK_DELTA) & (after <= np.percentile(after, 85) - 25) & ~hidden
        ys, xs = np.nonzero(new)
        offsets = np.c_[xs + x0 - center[0], ys + y0 - center[1]]
        distance = np.linalg.norm(offsets, axis=1)
        keep = (distance >= 0.35 * radius) & (distance <= 1.8 * radius)
        if np.count_nonzero(keep) < 8:
            return None
        # Least-squares circle through the visible ink; corrects a constant tip offset.
        px, py = xs[keep] + x0, ys[keep] + y0
        system = np.c_[2 * px, 2 * py, np.ones(len(px))]
        try:
            (fx, fy, c), *_ = np.linalg.lstsq(system, px ** 2 + py ** 2, rcond=None)
        except np.linalg.LinAlgError:
            return None
        fitted = np.sqrt(max(c + fx * fx + fy * fy, 0))
        if not 0.5 * radius <= fitted <= 1.8 * radius or np.hypot(fx - center[0], fy - center[1]) > 1.2 * radius:
            return None
        angles = np.arctan2(py - fy, px - fx)
        distance = np.hypot(px - fx, py - fy)
        on_ring = np.abs(distance - fitted) <= max(0.4 * fitted, self.ink_radius * 0.6)
        if np.mean(on_ring) < 0.7:
            return None
        sectors = 12
        inked = np.bincount(((angles[on_ring] + np.pi) / (2 * np.pi) * sectors).astype(int) % sectors,
                            minlength=sectors) >= 2
        visible = []
        for index in range(sectors):
            angle = (index + 0.5) / sectors * 2 * np.pi - np.pi
            x, y = int(round(fx + fitted * np.cos(angle))) - x0, int(round(fy + fitted * np.sin(angle))) - y0
            visible.append(0 <= x < hidden.shape[1] and 0 <= y < hidden.shape[0] and not hidden[y, x])
        visible = np.asarray(visible)
        if visible.sum() < 6 or np.mean(inked[visible]) < 0.75:
            return None
        inner = (np.hypot(xs + x0 - fx, ys + y0 - fy) < 0.4 * fitted).sum()
        if inner > 0.15 * np.pi * (0.4 * fitted) ** 2:
            return None  # A filled dot or a letter, not an open ring.
        return (float(fx), float(fy)), float(fitted)

    # Drawing -------------------------------------------------------------------------------------

    def draw(self, frame, now):
        canvas = frame.copy()
        for side in ("Left", "Right"):
            samples = [(t, point) for t, hand, point, _ in self.path if hand == side and now - t <= 1.4]
            for (previous, start), (current, end) in zip(samples, samples[1:]):
                if current - previous < 0.25 and np.linalg.norm(np.subtract(start, end)) < 110:
                    cv2.line(canvas, start, end, (60, 210, 245), 3, cv2.LINE_AA)
            for stroke in self._stroke_samples(side):
                cv2.polylines(canvas, [np.asarray([s.point for s in stroke], np.int32)], False,
                              (205, 110, 25), 2, cv2.LINE_AA)
            recent = self.current_tips.get(side)
            if recent is not None and now - recent[0] < 0.25:
                self._draw_tip(canvas, recent[1])
        if self.end_circle is not None:
            center = tuple(int(round(v)) for v in self.end_circle.center)
            cv2.circle(canvas, center, int(round(self.end_circle.radius * 1.3)), (0, 140, 255), 3, cv2.LINE_AA)
        return canvas

    @staticmethod
    def _draw_tip(canvas, tip):
        """Magenta crosshair at the pen tip: filled when the shaft is seen, hollow when estimated."""
        x, y = tip.point
        size = max(9, canvas.shape[1] // 150)
        for color, width in (((25, 25, 25), 4), ((230, 40, 230), 2)):
            cv2.circle(canvas, (x, y), size, color, width, cv2.LINE_AA)
            cv2.line(canvas, (x - 2 * size, y), (x - size - 2, y), color, width, cv2.LINE_AA)
            cv2.line(canvas, (x + size + 2, y), (x + 2 * size, y), color, width, cv2.LINE_AA)
            cv2.line(canvas, (x, y - 2 * size), (x, y - size - 2), color, width, cv2.LINE_AA)
            cv2.line(canvas, (x, y + size + 2), (x, y + 2 * size), color, width, cv2.LINE_AA)
        if tip.visual:
            cv2.circle(canvas, (x, y), 3, (230, 40, 230), -1, cv2.LINE_AA)

    # Selection -----------------------------------------------------------------------------------

    def select(self, frame, roi=None, *, narrow=True, marked_only=False, prefer_recent=False):
        if prefer_recent:
            selected = self._select_recording(frame, roi)
            if selected is not None and self.finish_reason is None:
                self.finish_reason = "circle" if self.end_marked else "key"
            return selected
        chosen = select_paper_region(frame, roi, allow_dashed=marked_only)
        if chosen.reason or (marked_only and chosen.kind == "full"):
            return WritingSelection(chosen)
        changes = self._visible_changes(frame)
        if changes is None:
            return WritingSelection(chosen)
        if narrow and chosen.bounds is None:
            bounds = self._recent_bounds(changes)
            if bounds is not None:
                chosen = select_paper_region(frame, bounds)
        return WritingSelection(chosen, self._ink_crop(chosen, changes))

    def _end_center(self):
        return tuple(int(round(v)) for v in self.end_circle.center)

    def _end_erase_radius(self):
        return int(round(self.end_circle.radius * 1.4 + self.ink_radius))

    def _erase_end_circle(self, chosen):
        """Paint the end circle with the paper color so it is not transcribed."""
        image = chosen.image.copy()
        x0, y0 = chosen.bounds[:2] if chosen.bounds else (0, 0)
        paper = np.median(image.reshape(-1, 3)[np.min(image.reshape(-1, 3), axis=1) >= np.percentile(image, 60)],
                          axis=0) if image.size else (255, 255, 255)
        center = tuple(np.subtract(self._end_center(), (x0, y0)))
        cv2.circle(image, center, self._end_erase_radius(), tuple(int(v) for v in paper), -1)
        return image

    def _select_recording(self, frame, roi):
        """Crop only what was written in this recording.

        The pen-down strokes define the area. New ink extends it only where a mark touches
        the strokes (within half a glyph height); without strokes, new ink near the pen
        tip defines it. New ink is limited to thin, compact marks on paper, so a hand,
        forearm, shadow, exposure change, or shifted page cannot grow the crop, and the
        page or paper bounds are never used.
        """
        height, width = frame.shape[:2]
        if self.reference is None or self.reference.shape != (height, width):
            return None
        path = self._writer_path()
        strokes, grip_samples, side = [], 0, None
        if path:
            side = path[0][1]
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            strokes = [[sample.point for sample in stroke] for stroke in self._stroke_samples(side, gray)]
            grip_samples = len(path)
        trace_bounds = stroke_bounds(strokes)
        if trace_bounds is not None and max(trace_bounds[2] - trace_bounds[0], trace_bounds[3] - trace_bounds[1]) < 6:
            strokes, trace_bounds = [], None
        rotation = self._rotation(side)
        ink = self._new_ink(frame) if path else None
        if roi is not None:
            bounds, text_height = roi, self._text_height(strokes, rotation)
        else:
            bounds, text_height = self._writing_bounds(ink, strokes, side, rotation, (width, height))
        if bounds is None:
            return None
        chosen = select_paper_region(frame, bounds)
        if self.end_circle is not None:
            chosen = PaperSelection(self._erase_end_circle(chosen), chosen.bounds, chosen.kind, chosen.reason)
        trace = render_strokes(strokes, chosen.bounds) if strokes and chosen.bounds else None
        count = sum(len(stroke) for stroke in strokes)
        reliable = len(strokes) >= 2 and count >= 12 and count >= 0.3 * grip_samples
        evidence = None
        if ink is not None and chosen.bounds:
            x0, y0, x1, y1 = chosen.bounds
            local = ink[y0:y1, x0:x1]
            if np.count_nonzero(local) >= 25:
                evidence = cv2.cvtColor(np.where(local != 0, 0, 255).astype(np.uint8), cv2.COLOR_GRAY2BGR)
        return WritingSelection(chosen, evidence, trace, reliable, self.end_marked,
                                tuple(np.asarray(stroke, np.int32) for stroke in strokes), rotation, text_height)

    def _rotation(self, side):
        """Clockwise turn that makes the writing upright for the writer.

        A writer's hand rests below the line with the pen pointing away from the body,
        so the wrist-to-tip direction is the writer's "up" in the camera image. A camera
        facing the writer from across the desk sees the page upside down.
        """
        samples = [sample for sample in self.tips.get(side, ()) if sample.wrist is not None]
        if len(samples) < 3:
            return 0
        down = [sample for sample in samples if sample.down]
        vectors = np.asarray([np.subtract(sample.point, sample.wrist) for sample in (down or samples)], float)
        dx, dy = np.median(vectors, axis=0)
        if abs(dy) >= abs(dx):
            return 0 if dy <= 0 else 180
        return 90 if dx < 0 else 270

    def _text_height(self, strokes, rotation=0):
        """Typical glyph height: the median vertical extent of the taller strokes."""
        floor = 2.0 * self.ink_radius
        if not strokes:
            return 3.0 * self.ink_radius
        axis = 0 if rotation in (90, 270) else 1
        extents = np.asarray([np.ptp(np.asarray(stroke, float)[:, axis]) for stroke in strokes if len(stroke) >= 2])
        if not extents.size:
            return floor
        overall = float(np.ptp(np.vstack([np.asarray(stroke, float) for stroke in strokes])[:, axis]))
        tall = extents[extents >= 0.15 * max(overall, 1.0)]
        return float(np.clip(np.median(tall) if tall.size else overall, floor, max(floor, overall)))

    def _ink_window(self, width, height):
        """Region worth searching for new ink: around the writer's tip and grip samples."""
        points = [item[2] for item in self.path]
        points += [sample.point for samples in self.tips.values() for sample in samples]
        x0, y0, x1, y1 = self.area or (0, 0, width, height)
        if not points:
            return x0, y0, x1, y1
        points = np.asarray(points, float)
        pad = 0.08 * width
        left, top = points.min(axis=0) - pad
        right, bottom = points.max(axis=0) + pad
        return (int(max(x0, left)), int(max(y0, top)), int(min(x1, right)), int(min(y1, bottom)))

    def _page_shift(self, frame):
        """Small translation of the page since the reference frame."""
        if self.baseline is None:
            return 0.0, 0.0
        after = _gray(frame)
        if after.shape != self.baseline.shape:
            return 0.0, 0.0
        height, width = after.shape
        scale = frame.shape[1] / width
        window = cv2.createHanningWindow((width, height), cv2.CV_32F)
        (dx, dy), response = cv2.phaseCorrelate(self.baseline.astype(np.float32), after.astype(np.float32), window)
        if response < 0.24 or abs(dx) > width * 0.025 or abs(dy) > height * 0.025 or np.hypot(dx, dy) < 0.5:
            return 0.0, 0.0
        return dx * scale, dy * scale

    def _new_ink(self, frame):
        """Full-frame 0/1 map of thin, dark marks that appeared on the paper since the key press.

        Darkness is measured against the local paper brightness (a closing removes thin
        strokes from the background estimate), so auto-exposure and broad lighting or
        shadow changes cancel. Hands, forearm, pen, skin-colored pixels, and anything off
        the paper are masked, and only compact components with a pen-stroke width remain.
        """
        height, width = frame.shape[:2]
        x0, y0, x1, y1 = self._ink_window(width, height)
        if x1 - x0 < 8 or y1 - y0 < 8:
            return None
        gray = cv2.cvtColor(frame[y0:y1, x0:x1], cv2.COLOR_BGR2GRAY)
        dx, dy = self._page_shift(frame)
        shift = np.array([[1, 0, dx - x0], [0, 1, dy - y0]], dtype=np.float32)
        before = cv2.warpAffine(self.reference, shift, (x1 - x0, y1 - y0), borderMode=cv2.BORDER_REPLICATE)
        size = max(9, round(width * 0.01)) | 1
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size))

        def light(image):
            closed = cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel)
            return np.maximum(cv2.GaussianBlur(closed, (0, 0), size / 3).astype(np.float32), 1.0)

        light_before, light_after = light(before), light(gray)
        dark_before = 1 - before.astype(np.float32) / light_before
        dark_after = 1 - gray.astype(np.float32) / light_after
        new = (dark_after >= INK_DARK) & (dark_after - dark_before >= INK_GAIN - 0.01)
        paper = self.paper_level or 255.0
        new &= (light_before >= PAPER_SHARE * paper) & (light_after >= 0.45 * paper * self.gain)
        hue, saturation, value = cv2.split(cv2.cvtColor(frame[y0:y1, x0:x1], cv2.COLOR_BGR2HSV))
        new &= ~((hue >= 3) & (hue <= 25) & (saturation >= 35) & (saturation <= 170) & (value >= 60))
        new &= self._occlusion_mask(new.shape, (x0, y0), grow=1.8) == 0
        if self.end_circle is not None:
            ring = np.zeros(new.shape, np.uint8)
            cv2.circle(ring, tuple(int(v) for v in np.subtract(self._end_center(), (x0, y0))),
                       self._end_erase_radius(), 1, -1)
            new &= ring == 0
        new = new.astype(np.uint8)
        count, labels, stats, _ = cv2.connectedComponentsWithStats(new, connectivity=8)
        result = np.zeros((height, width), np.uint8)
        if count <= 1:
            return result
        # Pixels farther than a pen-stroke half-width from the background belong to a
        # blob (hand, sleeve, pen barrel, shadow); stroke crossings leave only a few.
        core = cv2.distanceTransform(new, cv2.DIST_L2, 3) > max(3.5, width * 0.0035)
        area = stats[:, cv2.CC_STAT_AREA]
        thick = np.bincount(labels[core], minlength=count)
        longest = np.maximum(stats[:, cv2.CC_STAT_WIDTH], stats[:, cv2.CC_STAT_HEIGHT])
        keep = ((area >= max(3, round(1.3 * (width / 640) ** 2)))
                & (thick <= np.maximum(6, 0.08 * area)) & (longest <= 0.3 * width))
        keep[0] = False
        result[y0:y1, x0:x1] = keep[labels]
        return result

    def _writing_bounds(self, ink, strokes, side, rotation, frame_size):
        """Stroke bounds joined with touching new ink, plus half a glyph height. Returns (bounds, height)."""
        width, height = frame_size
        ax0, ay0, ax1, ay1 = self.area or (0, 0, width, height)
        components = None
        if ink is not None and ink.any():
            _, labels, stats, _ = cv2.connectedComponentsWithStats(ink, connectivity=8)
            components = labels, stats
        if strokes and sum(len(stroke) for stroke in strokes) >= 8:
            text = self._text_height(strokes, rotation)
            left, top, right, bottom = stroke_bounds(strokes)
            if components is not None:
                labels, stats = components
                reach = int(round(max(2 * self.ink_radius, 0.5 * text)))
                zone = np.zeros(ink.shape, np.uint8)
                for stroke in strokes:
                    cv2.polylines(zone, [np.asarray(stroke, np.int32).reshape(-1, 1, 2)], False, 1, 2 * reach + 1)
                    for point in stroke:
                        cv2.circle(zone, tuple(int(v) for v in point), reach, 1, -1)
                for label in np.unique(labels[(zone != 0) & (ink != 0)]):
                    if label:
                        x, y, w, h = stats[label, :4]
                        left, top = min(left, x), min(top, y)
                        right, bottom = max(right, x + w), max(bottom, y + h)
        else:
            if components is None:
                return None, 0.0
            labels, stats = components
            anchors = [sample.point for sample in self.tips.get(side, ())]
            reach = 4 * self.ink_radius
            if not anchors:
                anchors = [item[2] for item in self.path if item[1] == side]
                reach = max([item[3] for item in self.path if item[1] == side], default=reach)
            near = np.zeros(ink.shape, np.uint8)
            for point in anchors:
                cv2.circle(near, tuple(int(v) for v in point), int(reach), 1, -1)
            chosen = [label for label in np.unique(labels[(near != 0) & (ink != 0)]) if label]
            if not chosen:
                return None, 0.0
            column = 2 if rotation in (90, 270) else 3
            boxes = stats[1:, :4]
            for _ in range(2):
                # Marks within half a glyph of the chosen ones continue the same writing.
                box = stats[chosen, :4]
                left, top = box[:, 0].min(), box[:, 1].min()
                right, bottom = (box[:, 0] + box[:, 2]).max(), (box[:, 1] + box[:, 3]).max()
                reach = 0.5 * max(np.median(box[:, column]), 2 * self.ink_radius)
                close = ((boxes[:, 0] <= right + reach) & (boxes[:, 0] + boxes[:, 2] >= left - reach)
                         & (boxes[:, 1] <= bottom + reach) & (boxes[:, 1] + boxes[:, 3] >= top - reach))
                chosen = sorted(set(chosen) | set(np.flatnonzero(close) + 1))
            box = stats[chosen, :4]
            left, top = box[:, 0].min(), box[:, 1].min()
            right, bottom = (box[:, 0] + box[:, 2]).max(), (box[:, 1] + box[:, 3]).max()
            if strokes:
                sx0, sy0, sx1, sy1 = stroke_bounds(strokes)
                left, top, right, bottom = min(left, sx0), min(top, sy0), max(right, sx1), max(bottom, sy1)
            span = right - left if column == 2 else bottom - top
            text = float(np.clip(np.median(box[:, column]), 2 * self.ink_radius, max(2 * self.ink_radius, span)))
        margin = int(round(np.clip(0.5 * text, 6, 0.05 * width)))
        bounds = (int(max(ax0, left - margin)), int(max(ay0, top - margin)),
                  int(min(ax1, right + margin)), int(min(ay1, bottom + margin)))
        if bounds[2] - bounds[0] < 8 or bounds[3] - bounds[1] < 8:
            return None, 0.0
        return bounds, text

    def _ink_crop(self, chosen, changes):
        if changes is None:
            return None
        if chosen.bounds:
            x0, y0, x1, y1 = chosen.bounds
            changes = changes[y0:y1, x0:x1]
            if chosen.kind == "circle":
                changes = np.where(np.min(chosen.image, axis=2) < 250, changes, 255).astype(np.uint8)
        if np.count_nonzero(changes < 190) >= 25:
            return cv2.cvtColor(changes, cv2.COLOR_GRAY2BGR)
        return None

    def _recent_bounds(self, changes):
        x, y, width, height = cv2.boundingRect((changes < 190).astype(np.uint8))
        rows, cols = changes.shape
        if not width or not height or width * height > rows * cols * 0.35:
            return None
        pad = min(round(cols * 0.075), max(round(cols * 0.025), height // 3))
        left, top, right, bottom = self.area or (0, 0, cols, rows)
        return max(left, x - pad), max(top, y - pad), min(right, x + width + pad), min(bottom, y + height + pad)

    def ready(self, frame, now, *, require_pen=False):
        """A selection once the writer ends: an end circle, or the hand leaving a settled page."""
        if self.baseline is None or now - self.last_check < (0.2 if self.end_marked else 0.45):
            return None
        if self.end_marked:
            self.last_check = now
            selected = self._select_recording(frame, None)
            if selected is not None and (selected.trace is not None or selected.evidence is not None):
                self.finish_reason = "circle"
                return selected
            return None
        if len(self.path) < 5:
            return None
        self.last_check = now
        path = self._writer_path()
        if len(path) < 5 or now - path[-1][0] < 1.6 or len(self.history) < 3:
            return None
        after = self.history[-1][1]
        before = self.history[-3][1]
        if self.area:
            height, width = frame.shape[:2]
            rows, cols = after.shape
            x0, y0, x1, y1 = self.area
            xs = slice(round(x0 * cols / width), round(x1 * cols / width))
            ys = slice(round(y0 * rows / height), round(y1 * rows / height))
            after, before = after[ys, xs], before[ys, xs]
        if np.mean(cv2.absdiff(after, before) > 20) > 0.015:
            return None
        selected = self._select_recording(frame, None)
        if selected is not None and (selected.trace_reliable or (not require_pen and selected.evidence is not None)):
            self.finish_reason = "lifted"
            return selected
        return None

    def _visible_changes(self, frame):
        if len(self.history) < 3 or len(self.path) < 5:
            return None
        after = _gray(frame)
        height, width = after.shape
        full_height, full_width = frame.shape[:2]
        now = self.history[-1][0]
        path = self._writer_path() if self.baseline is not None else [item for item in self.path if now - item[0] <= 10]
        if len(path) < 5:
            return None
        before = self.baseline
        if before is None:
            before = next((gray for time, gray in self.history if time >= path[0][0] - 1), self.history[0][1])
        if before.shape != after.shape:
            return None
        near_hand = np.zeros(after.shape, dtype=np.uint8)
        for _, _, (x, y), radius in path:
            cv2.circle(near_hand, (round(x * width / full_width), round(y * height / full_height)),
                       max(8, round(radius * width / full_width)), 1, -1)
        tip_reach = max(6, round(3 * self.ink_radius * width / full_width))
        for sample in self.tips.get(path[0][1], ()):
            x, y = sample.point
            cv2.circle(near_hand, (round(x * width / full_width), round(y * height / full_height)), tip_reach, 1, -1)
        if self.area:
            x0, y0, x1, y1 = self.area
            x0, x1 = round(x0 * width / full_width), round(x1 * width / full_width)
            y0, y1 = round(y0 * height / full_height), round(y1 * height / full_height)
            outside = np.ones_like(near_hand, dtype=bool)
            outside[y0:y1, x0:x1] = False
            near_hand[outside] = 0
        # Align the visible surroundings, where the new strokes have less influence.
        window = cv2.createHanningWindow((width, height), cv2.CV_32F)
        window *= 1 - cv2.GaussianBlur(near_hand.astype(np.float32), (0, 0), 9)
        (dx, dy), response = cv2.phaseCorrelate(before.astype(np.float32), after.astype(np.float32), window)
        if response < 0.24 or abs(dx) > width * 0.025 or abs(dy) > height * 0.025:
            untouched = near_hand == 0
            differences = cv2.absdiff(before, after)[untouched]
            if not differences.size or np.mean(differences > 18) > 0.015:
                return None
            dx = dy = 0
        transform = np.array([[1, 0, dx], [0, 1, dy]], dtype=np.float32)
        before = cv2.warpAffine(before, transform, (width, height), borderMode=cv2.BORDER_REPLICATE)
        # Flatten lighting before differencing so wide shadows do not become strokes.
        ink = after.astype(np.int16)
        old = before.astype(np.int16)
        ink = cv2.GaussianBlur(after, (0, 0), 8).astype(np.int16) - ink
        old = cv2.GaussianBlur(before, (0, 0), 8).astype(np.int16) - old
        visible = (ink >= 9) & (ink - old >= 9) & (near_hand != 0)
        if self.area:
            # Keys, printed graphics, and lifted hands often leave strong edges but
            # also create broad intensity changes. Only retain marks on light paper.
            backdrop = cv2.GaussianBlur(before, (0, 0), 5)
            broad_change = cv2.GaussianBlur(cv2.absdiff(before, after), (0, 0), 8)
            visible &= (backdrop >= 150) & (broad_change < 45)
        if np.count_nonzero(visible) < 20:
            return None
        changes = np.full(after.shape, 255, dtype=np.uint8)
        changes[visible] = np.clip(255 - 7 * ink[visible], 0, 255).astype(np.uint8)
        return cv2.resize(changes, (full_width, full_height), interpolation=cv2.INTER_NEAREST)
