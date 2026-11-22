"""Fit a keyboard around relaxed fingers using the two thumbs as its anchor."""

import math
from dataclasses import dataclass
from statistics import median

ROWS = (
    tuple("1234567890"),
    tuple("AZERTYUIOP"),
    tuple("QSDFGHJKLM"),
    tuple("WXCVBN',."),
    ("Shift", "Backspace", "Space", "Enter"),
)
FINGER_IDS = (8, 12, 16, 20)
KNUCKLE_IDS = (5, 9, 13, 17)
POSE_IDS = (0, 4, *KNUCKLE_IDS, *FINGER_IDS)
MAX_FRAME_GAP_MS = 220
MIN_STABLE_FRAMES = 5
THUMB_JOIN = 1.35
THUMB_HOLD = 1.65
SETUP_HINT = "Bring both thumbs close together and relax your fingers."


def _delta(a, b):
    return a[0] - b[0], a[1] - b[1]


def _dot(a, b):
    return a[0] * b[0] + a[1] * b[1]


def _mean_point(points):
    xs, ys = zip(*points)
    return sum(xs) / len(xs), sum(ys) / len(ys)


def ordered_hands(tracking, row_axis):
    """Order hands along the keyboard, independent of image orientation."""
    hands = tracking["hands"]
    if len(hands) != 2:
        return None
    return tuple(sorted(hands, key=lambda hand: _dot(hand["landmarks"][0]["pixel"], row_axis)))


def finger_reach(wrist, knuckle, tip):
    """Finger extension along its own wrist-to-knuckle direction, in pixels."""
    direction = _delta(knuckle, wrist)
    return _dot(_delta(tip, knuckle), direction) / max(1, math.hypot(*direction))


def _axes(points):
    # Average the two palm directions. The thumb-to-thumb line is unstable
    # when the thumbs are close enough to touch.
    palms = [_delta(hand[3], hand[0]) for hand in points]
    lengths = [math.hypot(*palm) for palm in palms]
    if min(lengths) < 1:
        return None
    forward = tuple(sum(palm[i] / length for palm, length in zip(palms, lengths))
                    for i in range(2))
    length = math.hypot(*forward)
    if length < 0.35:
        return None
    fx, fy = (component / length for component in forward)
    return (-fy, fx), (-fx, -fy)


def _control_edges(left, right, pitch):
    """Center Space at the thumb anchor, leaving room for the outer controls."""
    half_space = min(2 * pitch, (right - left) * 0.22)
    padding = (right - left) * 0.14
    left = min(left, -half_space - padding)
    right = max(right, half_space + padding)
    return left, (left - half_space) / 2, -half_space, half_space, right


@dataclass(frozen=True)
class Key:
    label: str
    corners: tuple
    row_axis: tuple
    column_axis: tuple

    @property
    def bounds(self):
        xs, ys = zip(*self.corners)
        return math.floor(min(xs)), math.floor(min(ys)), math.ceil(max(xs)), math.ceil(max(ys))

    def contains(self, x, y):
        origin, right, _, bottom = self.corners
        offset = _delta((x, y), origin)
        return (
            0 <= _dot(offset, self.row_axis) < _dot(_delta(right, origin), self.row_axis)
            and 0 <= _dot(offset, self.column_axis) < _dot(_delta(bottom, origin), self.column_axis)
        )


@dataclass(frozen=True)
class TypingReference:
    frame_size: tuple
    keys: tuple
    wrists: tuple
    palm_sizes: tuple
    home_tips: tuple
    thumb_anchor: tuple
    pitch: float
    row_axis: tuple
    column_axis: tuple

    def key_at(self, x, y):
        return next((key.label for key in self.keys if key.contains(x, y)), None)


def _fit_pose(frame_size, points, thumb_limit):
    axes = _axes(points)
    if axes is None:
        return None, "Keep both hands visible to the camera."
    row_axis, column_axis = axes
    points = sorted(points, key=lambda hand: _dot(hand[0], row_axis))
    wrists = tuple(hand[0] for hand in points)
    thumbs = tuple(hand[1] for hand in points)
    palm_sizes = tuple(math.dist(hand[0], hand[3]) for hand in points)
    palm_size = sum(palm_sizes) / 2
    if min(palm_sizes) < max(5, min(frame_size) * 0.017):
        return None, "Bring both hands into clearer view."
    if math.dist(*thumbs) > thumb_limit * palm_size:
        return None, "Bring your thumbs a little closer together."

    tips = tuple(hand[6:] for hand in points)
    for hand, palm in zip(points, palm_sizes):
        extended = sum(finger_reach(hand[0], knuckle, tip) > 0.08 * palm
                       for knuckle, tip in zip(hand[2:6], hand[6:]))
        if extended < 2:
            return None, "Let your fingers relax out of the grip."

    origin = _mean_point(thumbs)
    left_centers = [_dot(_delta(finger, origin), row_axis) for finger in reversed(tips[0])]
    right_centers = [_dot(_delta(finger, origin), row_axis) for finger in tips[1]]
    f, j = left_centers[-1], right_centers[0]
    center_gap = (j - f) / 3
    centers = [*left_centers, f + center_gap, f + 2 * center_gap, *right_centers]
    # A relaxed pose can project adjacent fingers almost onto each other.
    # Space the affected columns around their original group center.
    min_step = max(6, 0.32 * palm_size)
    adjusted = [centers[0]]
    for x in centers[1:]:
        adjusted.append(max(x, adjusted[-1] + min_step))
    offset = median(new - old for new, old in zip(adjusted, centers))
    centers = [x - offset for x in adjusted]
    pitch = median(b - a for a, b in zip(centers, centers[1:]))
    edges = [centers[0] - (centers[1] - centers[0]) / 2,
             *((a + b) / 2 for a, b in zip(centers, centers[1:])),
             centers[-1] + (centers[-1] - centers[-2]) / 2]

    def position(column):
        if column == 10:
            return edges[-1]
        start = int(column)
        return edges[start] + (edges[start + 1] - edges[start]) * (column - start)

    home_depths = [_dot(_delta(finger, origin), column_axis) for hand in tips for finger in hand]
    home_depth = (min(home_depths) + max(home_depths)) / 2
    minimum_row = max(8, max(home_depths) - min(home_depths) + 0.2 * pitch)
    upper_step = max(minimum_row, 0.82 * pitch)
    preferred_height = max(10, min(0.82 * pitch, -home_depth / 2))
    home_height = max(minimum_row, preferred_height)
    space_half_height = preferred_height / 2
    lower_letter_height = max(6, 0.28 * palm_size)
    # The bottom row always straddles the thumbs. If their projection is
    # close to the home row, give the letters enough room above Space.
    home_depth = min(home_depth, -space_half_height - lower_letter_height - home_height / 2)
    home_top = home_depth - home_height / 2
    row_edges = (
        home_top - 2 * upper_step,
        home_top - upper_step,
        home_top,
        home_top + home_height,
        -space_half_height,
        space_half_height,
    )
    width, height = frame_size
    keys = []
    for row_index, row in enumerate(ROWS):
        if row_index == len(ROWS) - 1:
            horizontal_edges = _control_edges(edges[0], edges[-1], pitch)
        else:
            column = (10 - len(row)) / 2
            horizontal_edges = [position(column + i) for i in range(len(row) + 1)]
        top, bottom = row_edges[row_index:row_index + 2]
        for label, left, right in zip(row, horizontal_edges, horizontal_edges[1:]):
            gap = min(0.07 * pitch, 0.15 * (right - left), 0.15 * (bottom - top))
            corners = tuple(
                (origin[0] + u * row_axis[0] + v * column_axis[0],
                 origin[1] + u * row_axis[1] + v * column_axis[1])
                for u, v in ((left + gap, top + gap), (right - gap, top + gap),
                             (right - gap, bottom - gap), (left + gap, bottom - gap))
            )
            if any(not (0 <= px < width and 0 <= py < height) for px, py in corners):
                return None, "Center your hands or move the camera back to fit the keys."
            keys.append(Key(label, corners, row_axis, column_axis))
    reference = TypingReference(frame_size, tuple(keys), wrists, palm_sizes, tips, origin, pitch,
                                row_axis, column_axis)
    return reference, ""


def _pose_points(hands, frame_size):
    width, height = frame_size
    points = tuple(tuple(tuple(hand["landmarks"][i]["pixel"]) for i in POSE_IDS)
                   for hand in hands)
    if any(not (0 <= x < width and 0 <= y < height) for hand in points for x, y in hand):
        return None
    return points


class PoseCalibration:
    """Collect a stable pose, then freeze the average reference until reset."""

    def __init__(self, hold_ms=650):
        if hold_ms <= 0:
            raise ValueError("hold_ms must be positive")
        self.hold_ms = hold_ms
        self.reset()

    def reset(self):
        self.reference = None
        self.progress = 0.0
        self.hint = SETUP_HINT
        self._frame_size = None
        self._last = None
        self._steady_ms = 0
        self._interrupted = False
        self._samples = []

    def _reset_hold(self):
        self._samples.clear()
        self._steady_ms = 0
        self.progress = 0.0

    def update(self, tracking):
        frame_size = tuple(tracking["frame_size"])
        if frame_size != self._frame_size:
            self.reset()
            self._frame_size = frame_size
        if self.reference is not None:
            return self.reference

        timestamp = tracking["timestamp_ms"]
        if self._samples and not 0 < timestamp - self._last <= MAX_FRAME_GAP_MS:
            self._reset_hold()

        hands = tracking["hands"]
        points = _pose_points(hands, frame_size) if len(hands) == 2 else None
        if len(hands) != 2:
            candidate, self.hint = None, "Show both hands. Bring your thumbs close together."
        elif points is None:
            candidate, self.hint = None, "Keep all fingertips and both wrists in the frame."
        else:
            thumb_limit = THUMB_HOLD if self._samples else THUMB_JOIN
            candidate, self.hint = _fit_pose(frame_size, points, thumb_limit)
        if candidate is None:
            # Keep a partially completed hold through a brief missed frame,
            # while excluding time spent untracked from the hold duration.
            self._interrupted = True
            return None
        points = tuple(sorted(points, key=lambda hand: _dot(hand[0], candidate.row_axis)))

        if self._samples:
            if not self._stable(points, candidate.palm_sizes):
                self._reset_hold()
                # Requiring the closer thumb distance for each fresh pose
                # prevents the relaxed hold threshold from arming a new one.
                candidate, self.hint = _fit_pose(frame_size, points, THUMB_JOIN)
                if candidate is None:
                    return None
            elif not self._interrupted:
                self._steady_ms += timestamp - self._last
        self._last = timestamp
        self._interrupted = False
        self._samples.append(points)
        self.hint = "Thumb reference found. Hold both hands still."
        self.progress = min(1.0, self._steady_ms / self.hold_ms)
        if self.progress >= 1 and len(self._samples) >= MIN_STABLE_FRAMES:
            averaged = tuple(tuple(_mean_point(sample[side][index] for sample in self._samples)
                                   for index in range(len(points[side]))) for side in range(2))
            self.reference, issue = _fit_pose(frame_size, averaged, THUMB_HOLD)
            if issue:
                self.hint = issue
            self._reset_hold()
            if self.reference is not None:
                self.progress = 1.0
        return self.reference

    def _stable(self, points, palm_sizes):
        baseline = self._samples[0]
        thumbs = _mean_point(hand[1] for hand in points)
        original_thumbs = _mean_point(hand[1] for hand in baseline)
        return math.dist(thumbs, original_thumbs) <= max(5, 0.34 * max(palm_sizes)) and all(
            math.dist(current, original) <= max(4, 0.34 * palm)
            for hand, first, palm in zip(points, baseline, palm_sizes)
            for index, (current, original) in enumerate(zip(hand, first))
            if index != 1
        )
