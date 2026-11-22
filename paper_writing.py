"""Collect in-memory writing evidence from synchronized RGB frames and host hand landmarks.

While a recording runs, every writing-hand sample places a smoothed pen-tip landmark
(paper_pen.PenTipTracker). A tip sample counts as pen-down only once new ink is visible
under it compared with the reference frame captured when the key was pressed, so hover
moves drop out of the trajectory. The recognition crop is the pen-down trajectory joined
with nearby new ink. A small closed circle drawn last ends the recording on its own.
"""

from collections import deque
from dataclasses import dataclass, replace
from typing import Optional

import cv2
import numpy as np

from paper_pen import (OneEuro, PenTip, PenTipTracker, blend_colour, find_bars, find_closed_loop, find_pen,
                       hue_distance, pen_grip, render_strokes, stroke_bounds)
from paper_selection import PaperSelection, select_paper_region
from paper_motion import sheet_motion
from paper_trace import ink_strokes

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
    ink: np.ndarray = None  # Clean picture of the ink written in this recording (dark on white), same crop.


@dataclass
class TipSample:
    time: float
    point: tuple
    visual: bool = False
    down: Optional[bool] = None  # Unknown until the paper under the sample is visible.
    wrist: Optional[tuple] = None
    # From the pen itself (paper_pen.find_pen), not guessed from a hand: ``visual`` then says whether its
    # point was seen, or hidden by the fingers and put where the hand ends on the pen's line.
    pen: bool = False
    over: bool = False  # Taken where ink already lay: the pen passed over writing, it did not write it.


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


def upright_turn(dx, dy):
    """Clockwise turn (0, 90, 180, 270) that makes writing upright, from the direction pointing away
    from the writer in the image (wrist to tip, or the pen's back to its nib)."""
    if abs(dy) >= abs(dx):
        return 0 if dy <= 0 else 180
    return 90 if dx < 0 else 270


def _segment_distance(point, start, end):
    point, start, end = (np.asarray(value, float) for value in (point, start, end))
    span = end - start
    length = span @ span
    if length < 1e-6:
        return float(np.linalg.norm(point - start))
    along = np.clip((point - start) @ span / length, 0, 1)
    return float(np.linalg.norm(point - start - along * span))


class WritingSession:
    # The pen's colour (hue, saturation, value), learned whenever a detected hand holds the barrel and
    # shared by all sessions; the workbench keeps it in its settings. It tells the pen from a cable.
    pen_colour = None

    def __init__(self, baseline=None, area=None, *, end_circle=True, settle=0.9, sheet=None):
        # Seconds the hand must stay off the page before a recording ends by itself. Drawings with
        # pauses between their parts (branch diagrams recorded without a key press) get longer.
        self.settle = settle
        self.pen = self.pen_seen = None  # The pen's barrel as last seen (paper_pen.PenEnd), and when.
        self.pen_filters = {}  # Smoothing of the nib per hand, like the tracker's for estimated tips.
        self.nib_trail = deque(maxlen=60)  # (time, point) of the nib, drawn as the yellow trail.
        # The sheet's outline (4 x 2 frame points). Only ink inside it counts: the desk, a keypad,
        # or a wrapper that moves is never writing. Its own edge is left out with a small margin.
        self.sheet_mask = None
        if sheet is not None and baseline is not None:
            mask = np.zeros(baseline.shape[:2], np.uint8)
            cv2.fillConvexPoly(mask, np.rint(np.asarray(sheet, float)).astype(np.int32), 1)
            edge = max(5, round(0.02 * baseline.shape[1]))  # The sheet's own edge shifts when it is pushed.
            self.sheet_mask = cv2.erode(mask, np.ones((2 * edge + 1, 2 * edge + 1), np.uint8))
        self.history = deque(maxlen=50)
        self.path = deque(maxlen=2000 if baseline is not None else 180)
        self.baseline = _gray(baseline) if baseline is not None else None
        # Full-resolution reference for thin ink under individual pen-tip samples.
        self.reference = cv2.cvtColor(baseline, cv2.COLOR_BGR2GRAY) if baseline is not None else None
        self.reference_bgr = baseline.copy() if baseline is not None else None  # For re-aligning a moved sheet.
        self.realigned = False  # True once the reference was moved onto where the sheet went.
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
        # Ink registered live. ink_count: in how many checkpoints a pixel was seen as new ink while
        # uncovered. ink_dark: its strongest darkness (0-255), which keeps the stroke's soft edges.
        self.ink_count = None
        self.ink_dark = None
        self._ink_level = None
        # Correction from the tracked pen tip to the real nib, learned from the ink (see _calibrate_tip).
        self.tip_offset = np.zeros(2)
        self._calibrated_at = 0.0
        self.gain = 1.0
        width = baseline.shape[1] if baseline is not None else 640
        self.ink_radius = max(4, round(width * 0.006))
        # Bars that lay in the picture at the key press (a cable, a ruler, a pen put down): never the pen.
        self.still_bars = find_bars(baseline, self._pen_region(baseline.shape)) if baseline is not None else []
        self._other_colour = 0  # Sightings of a held pen whose colour is not the remembered one.

    # Observation -------------------------------------------------------------------------

    def observe(self, frame, hand_sample, now):
        checkpoint = now - self.last_sample >= 0.24
        if checkpoint:
            self.history.append((now, _gray(frame)))
            self.last_sample = now
        hands = hand_sample["hands"] if hand_sample else ()
        occluders = []
        # The pen itself, found by its barrel: the landmark goes on the nib, with or without a detected hand.
        pen = self._find_pen(frame, hands, now)
        pen_side = None
        if pen is not None:
            pen_side = self._hand_of(pen, hands)
            self._pen_sample(frame, pen, "Pen", hands, now)
            occluders.append(self._pen_occluder(pen))
        # Only while no pen has been seen for a while is the tip guessed from the hand's landmarks: with a
        # pen that can be seen, a moment without it means it is hidden, and a guess would land on a finger.
        guess = pen is None and (self.pen_seen is None or now - self.pen_seen > 2.5)
        for hand in hands:
            found = pen_grip(hand)
            side = hand["handedness"]
            if found is not None and side != pen_side:
                grip, _, radius, _ = found
                x, y = point = tuple(int(v) for v in np.rint(grip))
                inside = not self.area or (self.area[0] <= x < self.area[2] and self.area[1] <= y < self.area[3])
                # Without a visible barrel the tip is estimated from the grip (thumb and index along the pen).
                if inside and guess and now - self.last_path.get(side, 0.0) >= 0.03:
                    self.path.append((now, side, point, radius))
                    self.last_path[side] = now
                    # The tip is followed even between recordings: a snapshot reads what it points at.
                    tip = self.tracker.update(side, hand, frame, now)
                    if tip is not None:
                        self.current_tips[side] = now, tip
                        if self.reference is not None:
                            if self._contains(tip.point):
                                wrist = hand.get("landmarks") or ()
                                wrist = tuple(wrist[0]["pixel"]) if wrist else None
                                point = tuple(int(round(v)) for v in np.add(tip.point, self.tip_offset))
                                self.tips.setdefault(side, deque(maxlen=3000)).append(
                                    TipSample(now, point, tip.visual, wrist=wrist))
            occluders.append(self._occluder(hand, side, now))
        fresh = [tip for time, tip in self.current_tips.values() if time == now]
        if fresh:
            self.nib_trail.append((now, self._nib(fresh[0])))
        if hands or pen is not None:
            self.occluders, self.occluded_at = occluders, now
        elif now - self.occluded_at > 0.3:
            self.occluders = []  # Hold the last hand shape through brief detection dropouts.
        if checkpoint and self.reference is not None and self.reference.shape == frame.shape[:2] \
                and self._sheet_track(frame):  # Not while the sheet is in motion (see _sheet_track).
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            if self.path:
                self._register_ink(frame)
                self._mark_written(now)
            self._classify(gray, now)
            if self.end_circle_enabled and not self.end_marked:
                self._detect_end_circle(gray, now)
            if self.path:
                self._calibrate_tip(now)

    def _find_pen(self, frame, hands, now):
        """The pen in this frame (paper_pen.find_pen): first around where it was a moment ago, which is
        quick and keeps to the same pen, then anywhere around the sheet."""
        height, width = frame.shape[:2]
        colour = WritingSession.pen_colour
        inside = None
        if self.area is not None:
            x0, y0, x1, y1 = self.area
            inside = (x0 - 60, y0 - 60, x1 + 60, y1 + 60)  # The nib is over the sheet; the back may stick out.
        previous = self.pen if self.pen is not None and self.pen_seen is not None and now - self.pen_seen < 0.6 else None
        if previous is not None:
            reach = 3 * previous.width + 90
            xs, ys = zip(previous.point, previous.back)
            window = (max(0, min(xs) - reach), max(0, min(ys) - reach), min(width, max(xs) + reach),
                      min(height, max(ys) + reach))
            pen = find_pen(frame, window, hands, colour=colour, previous=previous, ignore=self.still_bars,
                           inside=inside)
            if pen is not None:
                return pen
        return find_pen(frame, self._pen_region(frame.shape), hands, colour=colour, previous=previous,
                        ignore=self.still_bars, inside=inside)

    def _pen_region(self, shape):
        """Where to look for the pen: around the sheet (or the writing area), and around where it was."""
        height, width = shape[:2]
        boxes = []
        if self.area is not None:
            x0, y0, x1, y1 = self.area
            margin = 0.35 * max(x1 - x0, y1 - y0)
            boxes.append((x0 - margin, y0 - margin, x1 + margin, y1 + margin))
        if self.pen is not None and self.pen_seen is not None:
            reach = 6 * self.pen.width + 120
            xs, ys = zip(self.pen.point, self.pen.back)
            boxes.append((min(xs) - reach, min(ys) - reach, max(xs) + reach, max(ys) + reach))
        if not boxes:
            return None
        x0, y0 = min(box[0] for box in boxes), min(box[1] for box in boxes)
        x1, y1 = max(box[2] for box in boxes), max(box[3] for box in boxes)
        return max(0, x0), max(0, y0), min(width, x1), min(height, y1)

    @staticmethod
    def _hand_of(pen, hands):
        """The handedness of the hand holding the pen, or "Pen" when no hand was detected near it."""
        best, side = None, "Pen"
        for hand in hands:
            tips = [hand["fingertips"][name]["pixel"] for name in ("thumb", "index") if name in hand.get("fingertips", {})]
            if not tips:
                continue
            distance = min(_segment_distance(tip, pen.back, pen.point) for tip in tips)
            if distance < 4 * pen.width + 40 and (best is None or distance < best):
                best, side = distance, hand["handedness"]
        return side

    def _learn_colour(self, pen):
        """Remember the pen's colour once a detected hand holds it and its point is seen. A pen of another
        colour (or a colour remembered wrongly) replaces the memory after it was seen like that a few times."""
        if not (pen.held and pen.coloured and pen.tapered and pen.colour is not None):
            return
        known = WritingSession.pen_colour
        if known is not None and len(known) == 3 and hue_distance(known[0], pen.colour[0]) <= 20 \
                and abs(known[1] - pen.colour[1]) <= 90:
            WritingSession.pen_colour = blend_colour(known, pen.colour)
            self._other_colour = 0
            return
        self._other_colour += 1
        if known is None or self._other_colour >= 6:
            WritingSession.pen_colour = tuple(float(v) for v in pen.colour)
            self._other_colour = 0

    def _pen_sample(self, frame, pen, side, hands, now):
        self.pen, self.pen_seen = pen, now
        self._learn_colour(pen)
        if pen.tapered:
            # The landmark is on the point itself. (A pen whose point the fingers hide gives none: the end of
            # the hand on the pen's line is only a guess, kept below as an estimated sample.)
            point = self.pen_filters.setdefault(side, OneEuro(min_cutoff=4.0, beta=0.06))(now, pen.point)
            tip = PenTip(tuple(int(v) for v in np.rint(point)), pen.back, int(2 * pen.width),
                         max(8.0, 4 * pen.width), True)
            self.current_tips[side] = now, tip
            sample = tip.point
        else:
            sample = tuple(int(round(v)) for v in pen.point)
        if now - self.last_path.get(side, 0.0) < 0.03:
            return
        self.last_path[side] = now
        self.path.append((now, side, pen.back, int(2 * pen.width)))
        if self.reference is not None and self._contains(sample):
            # The back of the barrel stands in for the wrist: back-to-nib points away from the writer.
            shown, x, y = getattr(self, "_shown", None), int(sample[0]), int(sample[1])
            over = bool(pen.tapered and shown is not None and 0 <= y < shown.shape[0] and 0 <= x < shown.shape[1]
                        and shown[max(0, y - 3):y + 4, max(0, x - 3):x + 4].any())
            self.tips.setdefault(side, deque(maxlen=3000)).append(
                TipSample(now, sample, pen.tapered, False if over else None, wrist=pen.back, pen=True, over=over))

    def _nib(self, tip):
        """Where the nib is for a tracked tip: the point itself when it was seen on the pen, else the
        tip estimated from the hand plus the offset the ink taught (see _calibrate_tip)."""
        if tip.visual:
            return tuple(int(v) for v in tip.point)
        return tuple(int(round(v)) for v in np.add(tip.point, self.tip_offset))

    def _pen_occluder(self, pen):
        """The barrel covers the page under it, like a hand."""
        back, nib = np.asarray(pen.back, float), np.asarray(pen.point, float)
        axis = nib - back
        length = max(float(np.linalg.norm(axis)), 1.0)
        normal = np.array((-axis[1], axis[0])) / length * pen.width * 0.8
        far = back - axis / length * 4 * pen.width  # The barrel continues under the hand.
        # ... and the point itself (graphite, a metal tip) is as dark and thin as ink: cover it too.
        front = nib + axis / length * 0.6 * pen.width
        hull = np.rint([far + normal, far - normal, front - normal * 0.5, front + normal * 0.5]).astype(np.int32)
        return hull.reshape(-1, 1, 2), max(4.0, pen.width * 0.3), None

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
                # (A sample on the pen's own point is judged from the registered ink: see _mark_written.)
                if (sample.down is None and not sample.pen and now - sample.time >= 0.15
                        and not self._occluded(sample.point)):
                    linked = previous is not None and sample.time - previous.time <= 0.3
                    sample.down = self._ink_at(gray, sample.point, previous.point if linked else None)
                previous = sample

    def _pen_followed(self, samples):
        """True when the samples mostly come from the pen itself, not from a tip guessed from a hand."""
        return len(samples) >= 3 and float(np.mean([sample.pen for sample in samples])) >= 0.6

    def _written(self, ink, samples):
        """The part of ``ink`` (0/1, new marks on the paper) that the pen wrote.

        Ink comes out of the nib: a mark is writing only when the nib travelled along it. Each mark must
        lie, for the most part, within a nib's width of the path of the pen's point, and the point must
        have moved along it, not just rested beside it. A pen that hovers, points, or rests leaves no ink,
        and whatever else changed on the page (a shadow, old writing on a sheet that moved, the sheet's
        edge) was not under a moving nib. Marks next to accepted ones on the same text line are taken
        with them: the last letters are often written while the point is hidden by the fingers. (Where
        the fingers hide the point its place is only known roughly, so twice the distance is allowed.)"""
        result = np.zeros(ink.shape, np.uint8)
        count, labels, stats, _ = cv2.connectedComponentsWithStats(ink, connectivity=8)
        seen = [sample for sample in samples if sample.pen]
        if count <= 1 or not seen:
            return result
        reach = max(4, round(0.75 * self.ink_radius))
        trail = np.zeros(ink.shape, np.uint8)
        previous = None
        for sample in seen:
            point = int(sample.point[0]), int(sample.point[1])
            if not sample.visual:
                cv2.circle(trail, point, 2 * reach, 1, -1)
                previous = None
                continue
            if (previous is not None and sample.time - previous[0] <= 0.35
                    and abs(point[0] - previous[1][0]) + abs(point[1] - previous[1][1]) <= 4 * self.ink_radius):
                cv2.line(trail, previous[1], point, 1, 2 * reach + 1)
            else:
                cv2.circle(trail, point, reach, 1, -1)
            previous = sample.time, point
        area = stats[:, cv2.CC_STAT_AREA]
        covered = np.bincount(labels[trail != 0], minlength=count)
        points = np.asarray([sample.point for sample in seen], float)
        accepted = np.zeros(count, bool)
        for label in np.flatnonzero(covered >= 0.5 * area):
            if label == 0:
                continue
            x, y, w, h = stats[label, :4]
            near = ((points[:, 0] >= x - reach) & (points[:, 0] < x + w + reach)
                    & (points[:, 1] >= y - reach) & (points[:, 1] < y + h + reach))
            # A resting nib writes nothing: the point moved, and about as far as the mark is long.
            if near.any() and max(np.ptp(points[near, 0]), np.ptp(points[near, 1])) >= max(3.0, 0.4 * max(w, h) - 3):
                accepted[label] = True
        longest = np.maximum(stats[:, cv2.CC_STAT_WIDTH], stats[:, cv2.CC_STAT_HEIGHT])
        if not accepted.any() or longest[accepted].max() < self.ink_radius:
            return result  # Nothing, or specks only: the shortest stroke of a letter is longer than that.
        result = accepted[labels].astype(np.uint8)
        return self._continue_lines(ink, result)

    def _mark_written(self, now):
        """Pen-down for the samples on the pen's own point, from the ink registered so far: down where
        ink the pen wrote lies under the sample, up where the paper there is seen and has none. Also keeps
        that ink for the live picture (see draw)."""
        self._shown = None
        path = self._writer_path()
        if self.ink_count is None or not path:
            return
        samples = self.tips.get(path[0][1], ())
        if not self._pen_followed(samples):
            return
        self._shown = self._written((self.ink_count >= 1).astype(np.uint8), samples)
        self._mark_samples(samples, self._shown, now)

    def _mark_samples(self, samples, written, now=None):
        """Pen-down per sample, from the ink the pen wrote. A sample is down when that ink appears under
        it once its spot can be seen; a spot seen empty stays pen-up, whatever is written there later
        (the pen hovered there first), and a sample taken over ink that already lay there stays pen-up
        too (the pen passed over its own writing). Without ``now`` (the end) every sample is settled."""
        height, width = written.shape
        reach = max(3, round(0.5 * self.ink_radius))
        near = cv2.dilate(written, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * reach + 1,) * 2)) \
            if written.any() else written
        # Where ink cannot be registered yet (under the pen and the hand, as _new_ink masks them).
        hidden = self._occlusion_mask(written.shape, (0, 0), grow=1.8) if now is not None else None
        for sample in samples:
            if not sample.pen or sample.over or (sample.down is False):
                continue
            x, y = int(sample.point[0]), int(sample.point[1])
            if not (0 <= x < width and 0 <= y < height):
                sample.down = False
            elif near[y, x]:
                sample.down = True
            elif now is None or (now - sample.time > 0.5
                                 and not hidden[max(0, y - reach):y + reach + 1, max(0, x - reach):x + reach + 1].any()):
                sample.down = False
            else:
                sample.down = None

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
        if self.ink_count is not None and self.ink_count.shape == canvas.shape[:2]:
            # The writing as registered: exactly the ink the reader will get, painted blue.
            shown = getattr(self, "_shown", None)
            if shown is not None and shown.shape == self.ink_count.shape:
                canvas[shown != 0] = (205, 110, 25)  # With the pen's point followed: only what the pen wrote.
            else:
                canvas[self.ink_count >= 1] = (205, 110, 25)
        # The nib's path over the last moments (not the hand's), and the barrel the nib was found on.
        trail = [(t, point) for t, point in self.nib_trail if now - t <= 1.4]
        for (previous, start), (current, end) in zip(trail, trail[1:]):
            if current - previous < 0.25 and np.linalg.norm(np.subtract(start, end)) < 110:
                cv2.line(canvas, start, end, (60, 210, 245), 3, cv2.LINE_AA)
        if self.pen is not None and self.pen_seen is not None and now - self.pen_seen < 0.25:
            cv2.line(canvas, self.pen.back, self.pen.point, (60, 210, 245), 2, cv2.LINE_AA)
        for side in ("Left", "Right", "Pen"):
            if self.ink_count is None:
                for stroke in self._stroke_samples(side):
                    cv2.polylines(canvas, [np.asarray([s.point for s in stroke], np.int32)], False,
                                  (205, 110, 25), 2, cv2.LINE_AA)
            recent = self.current_tips.get(side)
            if recent is not None and now - recent[0] < 0.25:
                # At the nib: the tracked tip plus the offset the ink taught (see _calibrate_tip).
                point = self._nib(recent[1])
                self._draw_tip(canvas, replace(recent[1], point=point))
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

    def _select_recording(self, frame, roi, follow_sheet=True):
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
        strokes, grip_samples, side, samples = [], 0, None, []
        if path:
            side = path[0][1]
            samples = [sample for sample in self.tips.get(side, ())
                       if self.end_circle is None or sample.time < self.end_circle.start]
            grip_samples = len(path)
        # With the pen's own point followed, the ink alone says what was written (see _written); only a
        # tip guessed from the hand needs its samples as strokes where the hand still hides the ink.
        followed = self._pen_followed(samples)
        if path and not followed:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            strokes = [[sample.point for sample in stroke] for stroke in self._stroke_samples(side, gray)]
        trace_bounds = stroke_bounds(strokes)
        if trace_bounds is not None and max(trace_bounds[2] - trace_bounds[0], trace_bounds[3] - trace_bounds[1]) < 6:
            strokes, trace_bounds = [], None
        rotation = self._rotation(side)
        if follow_sheet and path:
            self._follow_sheet(frame)
        ink, darkness = self._ink_record(frame) if path else (None, None)
        if followed:
            ink = self._written(ink, samples) if ink is not None else np.zeros((height, width), np.uint8)
            if np.count_nonzero(ink) < 30:
                ink[:] = 0  # A speck is no writing.
            self._mark_samples(samples, ink)
            self._shown = ink
            if not ink.any():
                return None  # The pen moved, but nothing appeared under it: nothing was written.
            darkness = np.where(cv2.dilate(ink, np.ones((3, 3), np.uint8)) != 0, darkness, 0).astype(np.uint8)
        if strokes and ink is not None:
            strokes = self._match_ink(strokes, ink)
            trace_bounds = stroke_bounds(strokes)
        if ink is not None and side is not None:
            traced = self._trace_on_ink(ink, side, strokes)
            if traced:
                strokes, trace_bounds = traced, stroke_bounds(traced)
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
        clean = None
        if evidence is not None and darkness is not None:
            # The registered ink alone, dark on white with its soft edges: what the reader gets first.
            local = darkness[y0:y1, x0:x1]
            marked = local[local >= 20]
            if marked.size >= 25:
                # Faint pencil is only a little darker than the paper: stretch so its firmest strokes are black.
                gain = float(np.clip(235.0 / max(float(np.percentile(marked, 98)), 1.0), 1.0, 4.0))
                local = np.clip(local.astype(np.float32) * gain, 0, 255).astype(np.uint8)
            if np.count_nonzero(local >= 60) >= 25:
                clean = cv2.cvtColor(255 - local, cv2.COLOR_GRAY2BGR)
        return WritingSelection(chosen, evidence, trace, reliable, self.end_marked,
                                tuple(np.asarray(stroke, np.int32) for stroke in strokes), rotation, text_height,
                                clean)

    def _trace_on_ink(self, ink, side, tip_strokes):
        """The strokes of this recording traced on the ink that appeared (see paper_trace).

        Only ink the pen tip came near belongs to the writing. Where the hand still covers
        the page, the ink cannot be seen yet, so the tip strokes are kept for those parts.
        Returns None when there is too little ink to trace."""
        samples = [sample for sample in self.tips.get(side, ())
                   if self.end_circle is None or sample.time < self.end_circle.start]
        if self._pen_followed(samples) and sum(1 for sample in samples if sample.down) >= 3:
            # The order of the strokes comes from when the nib wrote them, not from when it passed over them.
            samples = [sample for sample in samples if sample.down]
        if np.count_nonzero(ink) < 30 or not samples:
            return None
        near = np.zeros(ink.shape, np.uint8)
        for sample in samples:
            cv2.circle(near, tuple(int(v) for v in sample.point), 4 * self.ink_radius, 1, -1)
        count, labels = cv2.connectedComponents(ink, connectivity=8)
        touched = np.unique(labels[(near != 0) & (ink != 0)])
        writing = np.isin(labels, touched[touched != 0]).astype(np.uint8)
        if np.count_nonzero(writing) < 30:
            return None
        traced = [stroke for stroke in ink_strokes(writing, [(sample.time, sample.point) for sample in samples],
                                                   reach=3.0 * self.ink_radius, spur=0.75 * self.ink_radius)
                  if len(stroke) >= 2]
        if not traced:
            return None
        hidden = self._occlusion_mask(ink.shape, (0, 0), grow=1.8) != 0
        if hidden.any():
            distance = cv2.distanceTransform((ink == 0).astype(np.uint8), cv2.DIST_L2, 3)
            height, width = ink.shape
            for stroke in tip_strokes:
                points = np.asarray(stroke, int).reshape(-1, 2)
                xs, ys = np.clip(points[:, 0], 0, width - 1), np.clip(points[:, 1], 0, height - 1)
                unseen = hidden[ys, xs] & (distance[ys, xs] > self.ink_radius)
                edges = np.flatnonzero(np.diff(np.r_[0, unseen.astype(np.int8), 0]))
                traced += [points[start:end] for start, end in zip(edges[::2], edges[1::2]) if end - start >= 2]
        return [np.asarray(stroke, np.int32) for stroke in traced]

    @staticmethod
    def _continue_lines(current, accepted, rounds=3):
        """``accepted`` (0/1 ink near the pen's path) grown by new marks that continue the same text line:
        overlapping it vertically by half a letter and at most two letter heights beside it. The last
        letters are often written while the pen itself is hidden, so no nib sample lies on them."""
        count, labels, stats, _ = cv2.connectedComponentsWithStats(current, connectivity=8)
        if count <= 1:
            return accepted
        chosen = set(np.unique(labels[(accepted != 0) & (current != 0)])) - {0}
        heights = stats[list(chosen), cv2.CC_STAT_HEIGHT] if chosen else np.array(())
        letter = float(np.median(heights)) if len(heights) else 0.0
        if letter < 3:
            return accepted
        for _ in range(rounds):
            boxes = stats[list(chosen), :4]
            added = False
            for label in range(1, count):
                if label in chosen or stats[label, cv2.CC_STAT_AREA] < 3:
                    continue
                x, y, w, h = stats[label, :4]
                overlap = np.minimum(boxes[:, 1] + boxes[:, 3], y + h) - np.maximum(boxes[:, 1], y)
                gap = np.maximum(np.maximum(boxes[:, 0] - (x + w), x - (boxes[:, 0] + boxes[:, 2])), 0)
                if np.any((overlap >= 0.5 * min(h, letter)) & (gap <= 2 * letter)) and h <= 3 * letter:
                    chosen.add(label)
                    added = True
            if not added:
                break
        return np.isin(labels, list(chosen)).astype(np.uint8) | accepted

    def _follow_sheet(self, frame):
        """Have the reference where the sheet is in the frame the selection is made from.

        The sheet is followed through the whole recording (see _sheet_track), and the record moves with
        it; this looks once more, at the final frame. Only when the tracker could not keep the sheet (it
        lost it, or had to start again) the older one-shot alignment is tried (paper_motion.sheet_motion):
        the reference is moved onto where the sheet went, the ink registered at the checkpoints was
        measured against another position and is dropped, and the final frame alone gives the new ink."""
        if self.reference_bgr is None or self.area is None or self.reference_bgr.shape != frame.shape:
            return
        if hasattr(self, "_sheet_frame"):
            if self._sheet_frame is not frame:
                self._sheet_track(frame)
            if self._sheet_kept:
                return
        motion, response = sheet_motion(self.reference_bgr, frame, self.area, self.hidden_mask(frame.shape))
        shift = float(np.abs(motion[:, 2] + (motion[:, :2] - np.eye(2)) @ (np.asarray(frame.shape[1::-1]) / 2)).max())
        angle = abs(np.degrees(np.arctan2(motion[1, 0], motion[0, 0])))
        if response < 0.15 or (shift < 3 and angle < 0.5):
            return
        height, width = frame.shape[:2]
        self.reference_bgr = cv2.warpAffine(self.reference_bgr, motion, (width, height), borderMode=cv2.BORDER_REPLICATE)
        self.reference = cv2.cvtColor(self.reference_bgr, cv2.COLOR_BGR2GRAY)
        self.baseline = _gray(self.reference_bgr)
        if self.sheet_mask is not None:
            self.sheet_mask = cv2.warpAffine(self.sheet_mask, motion, (width, height), flags=cv2.INTER_NEAREST)
        x0, y0, x1, y1 = self.area
        corners = np.float32([[x0, y0], [x1, y0], [x1, y1], [x0, y1]]) @ motion[:, :2].T + motion[:, 2]
        self.area = (int(max(0, corners[:, 0].min())), int(max(0, corners[:, 1].min())),
                     int(min(width, corners[:, 0].max())), int(min(height, corners[:, 1].max())))
        self.ink_count = self.ink_dark = None
        self.realigned = True

    def _sheet_track(self, frame):
        """Follow the sheet to this frame and move the record with it (see paper_motion.SheetTracker).

        Everything the session keeps is in frame coordinates: the reference, the ink registered so far,
        the pen samples, the sheet's region. When the sheet slides or turns, all of it is moved onto
        where the sheet is now, so new ink is always measured against the same spot of the sheet and
        the record stays valid. Returns False while the sheet is visibly not where the pose says (it is
        in motion, or cannot be found): such a frame is not compared with the reference at all."""
        from paper_motion import SheetTracker, displacement

        if self.reference_bgr is None or self.reference_bgr.shape != frame.shape:
            return True
        self._sheet_frame = frame
        height, width = frame.shape[:2]
        watch = getattr(self, "_sheet_watch", None)
        if watch is not None:
            # The workbench's calibrated sheet (see _sheet_use): its pose now, relative to the key press.
            state = watch.locate(frame)
            if watch.calibrated and watch.generation == self._sheet_generation:
                pose = watch.pose @ self._sheet_base
                if displacement(pose, self._sheet_pose, self.area or (0, 0, width, height)) >= 1.0:
                    self._sheet_apply(pose)
                self._sheet_kept = state != "lost"
                return self._sheet_kept
            self._sheet_watch = None  # It started over on another sheet: follow from here on our own.
            self._sheet_tracker = None
        tracker = getattr(self, "_sheet_tracker", None)
        if tracker is None:
            self._sheet_origin = self.reference_bgr, self.sheet_mask, self.area  # As at the key press.
            self._sheet_pose = np.eye(3)
            tracker = self._sheet_tracker = SheetTracker(self.reference_bgr, self.area)
        state = tracker.update(frame)
        self._sheet_kept = tracker.tied and state != "lost"
        written = self.ink_count is not None and np.count_nonzero(self.ink_count >= 2) >= 30
        if not tracker.tied and not written and not getattr(self, "_sheet_restarted", False):
            # What the camera shows never matched the reference (blurred at the key press, or the sheet
            # was pulled away before it could be followed) and nothing is written yet: this frame is
            # the reference from here on.
            self._sheet_restarted = True
            self.reference_bgr = frame.copy()
            self.reference = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            self.baseline = _gray(frame)
            self.ink_count = self.ink_dark = None
            for samples in self.tips.values():
                for sample in samples:
                    sample.down = None  # Judged against the wrong picture: judge again.
            self._sheet_origin = self.reference_bgr, self.sheet_mask, self.area
            self._sheet_pose = np.eye(3)
            self._sheet_tracker = SheetTracker(frame, self.area)
            self._sheet_kept = True
            return True
        if displacement(tracker.pose, self._sheet_pose, self.area or (0, 0, width, height)) >= 1.0:
            self._sheet_apply(tracker.pose)
        return state != "lost"

    def _sheet_use(self, watch):
        """Follow the sheet with the workbench's SheetWatch (paper_motion), which was calibrated on a clear
        view of the sheet, instead of a tracker that starts from the key press with a hand on the sheet.
        Call it right after creating the session, once the watch has seen the reference frame
        (``watch.locate(baseline)``). Returns False, and changes nothing, when the watch holds no sheet."""
        if watch is None or not watch.calibrated or watch.state == "lost" or self.reference_bgr is None:
            return False
        self._sheet_watch, self._sheet_generation = watch, watch.generation
        self._sheet_base = np.linalg.inv(watch.pose)  # Its pose at the key press.
        self._sheet_origin = self.reference_bgr, self.sheet_mask, self.area
        self._sheet_pose = np.eye(3)
        return True

    def _sheet_apply(self, pose):
        """Move what the session keeps in frame coordinates onto the sheet's pose (3x3, key press -> now)."""
        from paper_motion import move_points

        reference, mask, area = self._sheet_origin
        height, width = reference.shape[:2]
        step = pose @ np.linalg.inv(self._sheet_pose)  # From where the record is to where the sheet is now.
        matrix = pose[:2].astype(np.float32)
        # The reference, the sheet's region, and the area come from their originals: no blur builds up.
        self.reference_bgr = cv2.warpAffine(reference, matrix, (width, height), borderMode=cv2.BORDER_REPLICATE)
        self.reference = cv2.cvtColor(self.reference_bgr, cv2.COLOR_BGR2GRAY)
        self.baseline = _gray(self.reference_bgr)
        if mask is not None:
            self.sheet_mask = cv2.warpAffine(mask, matrix, (width, height), flags=cv2.INTER_NEAREST)
        if area is not None:
            x0, y0, x1, y1 = area
            corners = move_points(pose, [[x0, y0], [x1, y0], [x1, y1], [x0, y1]])
            self.area = (int(max(0, corners[:, 0].min())), int(max(0, corners[:, 1].min())),
                         int(min(width, corners[:, 0].max())), int(min(height, corners[:, 1].max())))
        moved = step[:2].astype(np.float32)
        if self.ink_count is not None and self.ink_count.shape == (height, width):
            self.ink_count = cv2.warpAffine(self.ink_count, moved, (width, height), flags=cv2.INTER_NEAREST)
            self.ink_dark = cv2.warpAffine(self.ink_dark, moved, (width, height), flags=cv2.INTER_LINEAR)

        def carried(point):
            x, y = move_points(step, [point])[0]
            return int(round(x)), int(round(y))

        # Where the pen wrote stays with the sheet.
        for samples in self.tips.values():
            for sample in samples:
                sample.point = carried(sample.point)
        self.path = deque(((time, side, carried(point), radius) for time, side, point, radius in self.path),
                          maxlen=self.path.maxlen)
        if self.end_circle is not None:
            self.end_circle = replace(self.end_circle, center=tuple(float(v) for v in
                                                                    move_points(step, [self.end_circle.center])[0]))
        self._sheet_pose = pose.copy()

    def _match_ink(self, strokes, ink):
        """Register the pen trace on the visible new ink, then cut out moves that left no ink.

        The tip landmark sits a few pixels off the nib, so the pen-down test has to accept
        ink anywhere within the ink radius. With small handwriting that also accepts a
        hover hop to the next letter, which joins glyphs in the trace. Here the whole trace
        is first shifted onto the ink (one offset for the recording), and each segment is
        then checked against the ink with a tight tolerance. Points still under the hand
        cannot be judged and are kept. A faint or missing ink map leaves the trace as is.
        """
        height, width = ink.shape
        if np.count_nonzero(ink) < 30:
            return strokes
        distance = cv2.distanceTransform((ink == 0).astype(np.uint8), cv2.DIST_L2, 3)
        hidden = self._occlusion_mask(ink.shape, (0, 0), grow=1.8) != 0
        tolerance = max(2.5, 0.35 * self.ink_radius)
        reach = int(self.ink_radius)

        def lookup(points):
            xs = np.clip(np.rint(points[:, 0]).astype(int), 0, width - 1)
            ys = np.clip(np.rint(points[:, 1]).astype(int), 0, height - 1)
            return distance[ys, xs], hidden[ys, xs]

        points = np.vstack([np.asarray(stroke, float) for stroke in strokes])
        best, offset = None, np.zeros(2)
        for dy in range(-reach, reach + 1, 2):
            for dx in range(-reach, reach + 1, 2):
                gaps, covered = lookup(points + (dx, dy))
                if np.count_nonzero(~covered) < 8:
                    continue
                # Prefer the smallest shift among equally good fits.
                score = float(np.mean(np.minimum(gaps[~covered], 2.0 * reach))) + 0.02 * np.hypot(dx, dy)
                if best is None or score < best:
                    best, offset = score, np.array((dx, dy), float)
        if best is None:
            return strokes
        gaps, covered = lookup(points + offset)
        if np.mean(gaps[~covered] <= tolerance) < 0.5:
            return strokes  # The ink map does not explain the trace; do not let it cut strokes.
        result = []
        for stroke in strokes:
            moved = np.asarray(stroke, float) + offset
            pieces = [[moved[0]]]
            for start, end in zip(moved, moved[1:]):
                length = float(np.linalg.norm(end - start))
                if length > tolerance:
                    steps = max(2, int(np.ceil(length / 2.0)) + 1)
                    along = start + (end - start) * np.linspace(0, 1, steps)[:, None]
                    gaps, covered = lookup(along)
                    judged = ~covered
                    if np.count_nonzero(judged) >= 2 and np.mean(gaps[judged] > tolerance) > 0.5:
                        pieces.append([])  # Mostly clean paper under this move: the pen was up.
                pieces[-1].append(end)
            for piece in pieces:
                piece = np.asarray(piece, float)
                gaps, covered = lookup(piece)
                on_ink = np.flatnonzero(covered | (gaps <= tolerance))
                if len(on_ink) >= 2:  # Trim hover samples at both ends of the piece.
                    piece = piece[on_ink[0]:on_ink[-1] + 1]
                    result.append([tuple(int(v) for v in np.rint(point)) for point in piece])
        return result or strokes

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
        return upright_turn(*np.median(vectors, axis=0))

    def pen_rotation(self, now, within=3.0):
        """The writer's orientation from the pen seen just now (its back-to-nib direction), or None."""
        if self.pen is None or self.pen_seen is None or now - self.pen_seen > within:
            return None
        return upright_turn(*np.subtract(self.pen.point, self.pen.back))

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
        # What was already written counts where it was and two pixels around: a sheet that crept by a pixel
        # or a reference moved onto it (and softened by that) must not make every old stroke look new.
        dark_before = cv2.dilate(dark_before, np.ones((5, 5), np.uint8))
        dark_after = 1 - gray.astype(np.float32) / light_after
        new = (dark_after >= INK_DARK) & (dark_after - dark_before >= INK_GAIN - 0.01)
        paper = self.paper_level or 255.0
        new &= (light_before >= PAPER_SHARE * paper) & (light_after >= 0.45 * paper * self.gain)
        hue, saturation, value = cv2.split(cv2.cvtColor(frame[y0:y1, x0:x1], cv2.COLOR_BGR2HSV))
        # Ink is grey, black, or blue; red, orange, skin, and wood are a hand, a pencil's barrel, or its cone.
        warm = ((hue <= 25) | (hue >= 165)) & (saturation >= 35) & (value >= 60)
        new &= ~(cv2.dilate(warm.astype(np.uint8), np.ones((5, 5), np.uint8)) != 0)
        new &= self._occlusion_mask(new.shape, (x0, y0), grow=1.8) == 0
        if self.sheet_mask is not None:
            new &= self.sheet_mask[y0:y1, x0:x1] != 0
        if self.end_circle is not None:
            ring = np.zeros(new.shape, np.uint8)
            cv2.circle(ring, tuple(int(v) for v in np.subtract(self._end_center(), (x0, y0))),
                       self._end_erase_radius(), 1, -1)
            new &= ring == 0
        new = new.astype(np.uint8)
        count, labels, stats, _ = cv2.connectedComponentsWithStats(new, connectivity=8)
        result = np.zeros((height, width), np.uint8)
        # Darkness of this window against the paper, for the clean ink picture (see _register_ink).
        self._ink_level = (x0, y0, x1, y1), np.clip(dark_after * (255 * 2.4), 0, 255).astype(np.uint8)
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

    def _register_ink(self, frame, ink=None):
        """Add the new ink visible in this frame to the live ink record.

        The hand hides what is being written, so each checkpoint only contributes the ink it
        can see. Over the recording every stroke is uncovered at some moment, and the record
        ends up with the complete writing even if the hand still rests on the page at the end.
        """
        if ink is None:
            ink = self._new_ink(frame)
        if ink is None or self._ink_level is None:
            return None
        if self.ink_count is None or self.ink_count.shape != ink.shape:
            self.ink_count = np.zeros(ink.shape, np.uint8)
            self.ink_dark = np.zeros(ink.shape, np.uint8)
        seen = ink != 0
        self.ink_count[seen] = np.minimum(self.ink_count[seen], 250) + 1
        (x0, y0, x1, y1), level = self._ink_level
        # One pixel around each mark keeps the anti-aliased edge of the stroke.
        near = cv2.dilate(ink[y0:y1, x0:x1], np.ones((3, 3), np.uint8)) != 0
        target = self.ink_dark[y0:y1, x0:x1]
        np.maximum(target, np.where(near, level, 0).astype(np.uint8), out=target)
        return ink

    def _calibrate_tip(self, now):
        """Learn where the real nib is relative to the tracked pen tip, from the ink itself.

        The tip landmark is estimated from the hand and usually sits several pixels off the
        nib, which makes the pen-down test miss real writing. The ink that has appeared is
        ground truth for where the nib was: the tip samples of the last seconds are shifted
        as a whole onto that ink, and the shift that fits best becomes a standing correction
        for the samples already recorded (which are judged again) and for those to come.
        """
        if self.ink_count is None or now - self._calibrated_at < 0.7:
            return
        path = self._writer_path()
        if not path:
            return
        samples = self.tips.get(path[0][1], ())
        if samples and np.mean([sample.visual for sample in list(samples)[-40:]]) > 0.5:
            return  # The nib is seen on the pen itself: there is no landmark offset to learn.
        # Only samples whose spot the hand has uncovered: next to the newest samples the ink is still
        # hidden, and fitting them to the ink that is visible would drag the tip back onto older strokes.
        settled = [sample for sample in samples
                   if now - sample.time > 0.6 and not self._occluded(sample.point)][-160:]
        if len(settled) < 15:
            return
        self._calibrated_at = now
        points = np.asarray([sample.point for sample in settled], float)
        reach = 3 * self.ink_radius
        height, width = self.ink_count.shape
        x0, y0 = np.maximum(points.min(axis=0).astype(int) - 2 * reach, 0)
        x1, y1 = np.minimum(points.max(axis=0).astype(int) + 2 * reach + 1, (width, height))
        seen = self.ink_count[y0:y1, x0:x1] >= 1
        if np.count_nonzero(seen) < 40:
            return
        distance = np.minimum(cv2.distanceTransform((~seen).astype(np.uint8), cv2.DIST_L2, 3), 2.0 * reach)
        local = points - (x0, y0)

        def cost(dx, dy):
            xs = np.clip(np.rint(local[:, 0] + dx).astype(int), 0, distance.shape[1] - 1)
            ys = np.clip(np.rint(local[:, 1] + dy).astype(int), 0, distance.shape[0] - 1)
            # The median: pen-up moves between letters are far from any ink and must not decide the fit.
            return float(np.median(distance[ys, xs]))

        base = cost(0, 0)
        best, shift = base, (0, 0)
        for dy in range(-reach, reach + 1, 2):
            for dx in range(-reach, reach + 1, 2):
                value = cost(dx, dy) + 0.05 * np.hypot(dx, dy)
                if value < best:
                    best, shift = value, (dx, dy)
        if best > 0.6 * base or base - best < 1.5 or np.hypot(*shift) < 2:
            return  # Already on the ink, or the ink does not explain the samples clearly better.
        delta = np.rint(0.7 * np.asarray(shift, float)).astype(int)  # Damped: the estimate settles over a few rounds.
        self.tip_offset = np.clip(self.tip_offset + delta, -2 * reach, 2 * reach)
        for sample in samples:
            sample.point = (sample.point[0] + int(delta[0]), sample.point[1] + int(delta[1]))
            if not sample.down:
                sample.down = None  # Judge again at the corrected position.

    def _ink_record(self, frame):
        """(mask, darkness) of the ink written so far: marks seen at least twice during the
        recording, plus whatever new ink the current frame shows."""
        current = self._new_ink(frame)
        if current is None:
            return None, None
        level_window, level = self._ink_level
        x0, y0, x1, y1 = level_window
        # Marks seen only now are accepted only next to the pen's last positions: the last letters,
        # written just before the end. Anywhere else they are a hand or its shadow on the page.
        latest = max((samples[-1].time for samples in self.tips.values() if samples), default=None)
        recent = np.zeros(current.shape, np.uint8)
        if latest is not None:
            for samples in self.tips.values():
                for sample in samples:
                    # After re-aligning, all of this recording's writing is measured on the final frame.
                    if self.realigned or latest - sample.time <= 2.0:
                        cv2.circle(recent, tuple(int(v) for v in sample.point), 3 * self.ink_radius, 1, -1)
        mask = (current & recent) if latest is not None else current.copy()
        if latest is not None and mask.any():
            mask = self._continue_lines(current, mask)
        dark = np.zeros(current.shape, np.uint8)
        near = cv2.dilate(mask[y0:y1, x0:x1], np.ones((3, 3), np.uint8)) != 0
        dark[y0:y1, x0:x1] = np.where(near, level, 0)
        if self.ink_count is not None and self.ink_count.shape == current.shape:
            steady = self.ink_count >= 2
            # Ink does not disappear. A mark registered earlier that is gone now, where nothing covers
            # the page, was a finger or a shadow that rested there for a moment.
            hidden = self._occlusion_mask(current.shape, (0, 0), grow=1.8) != 0
            faint = np.zeros(current.shape, bool)
            faint[y0:y1, x0:x1] = level < 0.5 * INK_DARK * 255 * 2.4
            steady &= ~(faint & ~hidden)
            mask[steady] = 1
            steady_near = cv2.dilate(steady.astype(np.uint8), np.ones((3, 3), np.uint8)) != 0
            np.maximum(dark, np.where(steady_near, self.ink_dark, 0).astype(np.uint8), out=dark)
        if self.end_circle is not None:
            cv2.circle(mask, self._end_center(), self._end_erase_radius(), 0, -1)
            cv2.circle(dark, self._end_center(), self._end_erase_radius(), 0, -1)
        return mask, dark

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
                # Writing the pen-down test missed still left ink where the tip was tracked.
                for sample in self.tips.get(side, ()):
                    cv2.circle(zone, tuple(int(v) for v in sample.point), 4 * self.ink_radius, 1, -1)
                for label in np.unique(labels[(zone != 0) & (ink != 0)]):
                    if label:
                        x, y, w, h = stats[label, :4]
                        left, top = min(left, x), min(top, y)
                        right, bottom = max(right, x + w), max(bottom, y + h)
        else:
            if components is None:
                return None, 0.0
            labels, stats = components
            # Ink counts as this recording's writing when it lies where the pen tip was tracked or
            # where the grip rested: the estimated tip can be off by a good part of the hand.
            near = np.zeros(ink.shape, np.uint8)
            for sample in self.tips.get(side, ()):
                cv2.circle(near, tuple(int(v) for v in sample.point), 4 * self.ink_radius, 1, -1)
            for item in self.path:
                if item[1] == side:
                    cv2.circle(near, tuple(int(v) for v in item[2]), int(max(0.6 * item[3], 3 * self.ink_radius)), 1, -1)
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

    def pointer(self, now, within=1.2):
        """Where the pen points now: the latest tip, corrected by the offset learned from the ink. None
        without a recent tip."""
        recent = [(time, tip) for time, tip in self.current_tips.values() if now - time <= within]
        # The pen over the sheet, not a hand resting elsewhere with its fingers together.
        recent = [item for item in recent if self._on_sheet(self._nib(item[1]))]
        if not recent:
            return None
        _, tip = max(recent, key=lambda item: item[0])
        return self._nib(tip)

    def _on_sheet(self, point):
        x, y = (int(round(v)) for v in point)
        if self.sheet_mask is not None:
            height, width = self.sheet_mask.shape
            return 0 <= x < width and 0 <= y < height and bool(
                self.sheet_mask[max(0, y - 20):y + 21, max(0, x - 20):x + 21].any())
        return self._contains((x, y))

    def hidden_mask(self, shape):
        """0/1 map of the page covered by the hand and pen, as last seen."""
        return self._occlusion_mask(shape[:2], (0, 0), grow=1.4)

    def hand_over_page(self, now):
        """True while a hand was seen in the last moments, so ink may still be covered."""
        return bool(self.occluders) and now - self.occluded_at < 0.4

    def motion_bounds(self, frame_size):
        """Where the writing hand worked during this recording: the pen-tip samples (or, without
        any, the grip positions) with a margin. Used to read a photo of that region when
        neither strokes nor new ink could be detected. None when no hand was tracked."""
        width, height = frame_size
        path = self._writer_path()
        if not path:
            return None
        side = path[0][1]
        points = [sample.point for sample in self.tips.get(side, ())]
        reach = [item[3] for item in path]
        if len(points) < 5:
            points = [item[2] for item in path]
        points = np.asarray(points, float)
        # Drop the approach and the retreat of the hand: keep the middle 90 % on each axis.
        low, high = np.percentile(points, 5, axis=0), np.percentile(points, 95, axis=0)
        pad = max(0.03 * width, 0.5 * float(np.median(reach)))
        x0, y0, x1, y1 = self.area or (0, 0, width, height)
        bounds = (int(max(x0, low[0] - pad)), int(max(y0, low[1] - pad)),
                  int(min(x1, high[0] + pad)), int(min(y1, high[1] + pad)))
        return bounds if bounds[2] - bounds[0] >= 16 and bounds[3] - bounds[1] >= 16 else None

    def tip_count(self):
        return sum(len(samples) for samples in self.tips.values())

    def last_tip_time(self):
        return max((samples[-1].time for samples in self.tips.values() if samples), default=0.0)

    def summary(self):
        """Counts for a status line: tip samples and how many counted as pen-down."""
        tips = [sample for samples in self.tips.values() for sample in samples]
        return len(tips), sum(1 for sample in tips if sample.down)

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
        if len(path) < 5 or now - path[-1][0] < self.settle or len(self.history) < 3:
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
