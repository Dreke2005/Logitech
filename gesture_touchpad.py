"""Translate one hand in the RGB preview into desktop touchpad actions."""

import math
from dataclasses import dataclass

MODE_HOLD_MS = 80
MAX_FRAME_GAP_MS = 180
PINCH_CLOSED = 0.30
PINCH_OPEN = 0.44
MIN_CLICK_MS = 65
DRAG_HOLD_MS = 430
SCROLL_STEP = 0.38
ZOOM_STEP = 0.18
SWIPE_TIME_MS = 550

FINGERS = ((5, 6, 8), (9, 10, 12), (13, 14, 16), (17, 18, 20))
STATUS = {
    "point": "Point to move. Pinch to click or hold to drag.",
    "scroll": "Move two fingers together to scroll.",
    "zoom": "Pinch or spread your thumb and index finger to zoom.",
    "swipe": "Swipe three fingers to switch tabs or views.",
}


def _delta(point, anchor):
    return point[0] - anchor[0], point[1] - anchor[1]


def _average(first, second):
    return (first[0] + second[0]) / 2, (first[1] + second[1]) / 2


def _extended(points, finger):
    base, joint, tip = (points[i] for i in finger)
    lower_x, lower_y = _delta(joint, base)
    upper_x, upper_y = _delta(tip, joint)
    lower = math.hypot(lower_x, lower_y)
    upper = math.hypot(upper_x, upper_y)
    return bool(lower and upper and lower_x * upper_x + lower_y * upper_y > 0.42 * lower * upper)


@dataclass
class _Hand:
    wrist: tuple
    palm: tuple
    index: tuple
    middle: tuple
    size: float
    fingers: tuple
    index_pinch: float
    middle_pinch: float

    @classmethod
    def from_tracking(cls, tracking):
        if len(tracking["hands"]) != 1:
            return None
        landmarks = tracking["hands"][0]["landmarks"]
        points = [tuple(landmark["pixel"]) for landmark in landmarks]
        width, height = tracking["frame_size"]
        wrist, thumb, index, middle = (points[i] for i in (0, 4, 8, 12))
        if any(not (0 <= x < width and 0 <= y < height) for x, y in (wrist, thumb, index, middle)):
            return None
        size = max(math.dist(wrist, points[9]), 0.9 * math.dist(points[5], points[17]))
        if size < max(12, 0.028 * min(width, height)):
            return None
        return cls(wrist, _average(wrist, points[9]), index, middle, size,
                   tuple(_extended(points, finger) for finger in FINGERS),
                   math.dist(thumb, index) / size, math.dist(thumb, middle) / size)

    @property
    def scroll_point(self):
        return _average(self.index, self.middle)

    @property
    def right_closed(self):
        return (self.fingers[0] and self.middle_pinch <= PINCH_CLOSED
                and self.index_pinch >= max(PINCH_CLOSED, self.middle_pinch + 0.19))


class TouchpadController:
    """Accept preview landmarks and emit commands through an injected output object."""

    def __init__(self, output):
        self.output = output
        self._dragging = False
        self.reset()

    def reset(self):
        """End a drag and discard all gesture history."""
        self._clear_mode()
        self._last_ms = None
        self._last_hand = None
        self._candidate = None
        self._candidate_ms = 0

    def _clear_mode(self):
        if self._dragging:
            self.output.button_up("left")
            self._dragging = False
        self._mode = None
        self._anchor = None
        self._swipe_started_ms = None
        self._swiped = False
        self._pinch = None
        self._pinch_ready = False

    def _pose(self, hand):
        index, middle, ring, pinky = hand.fingers
        if middle and ring and pinky:
            return "zoom"
        if index and middle and ring and not pinky:
            return "swipe"
        if not ring and not pinky:
            left_close = hand.index_pinch <= PINCH_OPEN and not middle
            # Maintain the point pose across pinch release, even when the
            # raised middle finger briefly resembles a scroll pose.
            if self._pinch or hand.right_closed or left_close or (index and not middle):
                return "point"
            if index and middle:
                return "scroll"
        return None

    def update(self, tracking):
        timestamp = tracking["timestamp_ms"]
        hand = _Hand.from_tracking(tracking)
        if hand is None:
            self.reset()
            return "Show one hand fully in view."
        previous = self._last_hand
        if self._last_ms is not None and (
            not 0 < timestamp - self._last_ms <= MAX_FRAME_GAP_MS
            or math.dist(hand.wrist, previous.wrist) > 0.95 * previous.size
            or not 0.62 * previous.size <= hand.size <= 1.60 * previous.size
        ):
            self.reset()
        self._last_ms, self._last_hand = timestamp, hand
        pose = self._pose(hand)
        if pose is None:
            self.reset()
            return "Pose unclear. Raise your index finger to move."
        if pose != self._mode:
            if self._pinch:
                self._clear_mode()
            if pose != self._candidate:
                self._candidate, self._candidate_ms = pose, timestamp
            if timestamp - self._candidate_ms < MODE_HOLD_MS:
                return "Hold the gesture briefly."
            self._clear_mode()
            self._mode = pose
        elif pose != self._candidate:
            # A fleeting pose change freezes input. Rebase on return instead
            # of sending the displacement accumulated during the transition.
            self._candidate, self._candidate_ms = pose, timestamp
            self._anchor = None
            self._swipe_started_ms = None

        if pose == "point":
            return self._point(hand, timestamp, tracking["frame_size"][0])
        if pose == "scroll":
            self._scroll(hand)
        elif pose == "zoom":
            self._zoom(hand)
        else:
            return self._swipe(hand, timestamp)
        return STATUS[pose]

    def _point(self, hand, timestamp, frame_width):
        left, right = hand.index_pinch, hand.middle_pinch
        if self._pinch:
            button, started_ms = self._pinch
            gap = left if button == "left" else right
            elapsed = timestamp - started_ms
            if gap >= PINCH_OPEN:
                if self._dragging:
                    self.output.button_up("left")
                    self._dragging = False
                elif MIN_CLICK_MS <= elapsed < DRAG_HOLD_MS:
                    self.output.click(button)
                self._anchor = hand.index
                self._pinch = None
                self._pinch_ready = False
                return STATUS["point"]
            if button == "left" and elapsed >= DRAG_HOLD_MS:
                if not self._dragging:
                    self.output.button_down("left")
                    self._dragging = True
                    self._anchor = hand.palm
                self._move(hand.palm, hand.size, frame_width)
                return "Dragging. Open the pinch to release."
            return "Open the pinch to click."

        right_close = hand.right_closed
        left_close = left <= PINCH_CLOSED and not hand.fingers[1]
        if left_close or right_close:
            if self._pinch_ready:
                self._pinch = ("right" if right_close else "left", timestamp)
            return "Open the pinch to click." if self._pinch else "Open your pinch to get ready."
        if left >= PINCH_OPEN and right >= PINCH_OPEN:
            self._pinch_ready = True
            self._move(hand.index, hand.size, frame_width)
        return STATUS["point"]

    def _move(self, point, size, frame_width):
        if self._anchor is None:
            self._anchor = point
            return
        dx, dy = _delta(point, self._anchor)
        if math.hypot(dx, dy) <= max(1.5, size * 0.038):
            return
        gain = 3.1 * 640 / frame_width
        # Keep a little lag behind the target to soften noisy landmark steps.
        self._anchor = (self._anchor[0] + 0.62 * dx, self._anchor[1] + 0.62 * dy)
        movement = tuple(round(max(-96, min(96, delta * gain * 0.62))) for delta in (dx, dy))
        if any(movement):
            self.output.move(*movement)

    def _scroll(self, hand):
        point = hand.scroll_point
        if self._anchor is None:
            self._anchor = point
            return
        dx, dy = _delta(point, self._anchor)
        step = SCROLL_STEP * hand.size
        x, y = math.trunc(dx / step), math.trunc(dy / step)
        if x or y:
            self._anchor = (self._anchor[0] + x * step, self._anchor[1] + y * step)
            self.output.scroll(vertical=-y, horizontal=x)

    def _zoom(self, hand):
        distance = hand.index_pinch
        if self._anchor is None:
            self._anchor = distance
            return
        notches = math.trunc((distance - self._anchor) / ZOOM_STEP)
        if notches:
            self._anchor += notches * ZOOM_STEP
            self.output.zoom(notches)

    def _swipe(self, hand, timestamp):
        if self._swiped:
            return "Lower a finger before swiping again."
        if self._anchor is None:
            self._anchor = hand.palm
        dx, dy = _delta(hand.palm, self._anchor)
        if self._swipe_started_ms is None and math.hypot(dx, dy) > 0.20 * hand.size:
            self._swipe_started_ms = timestamp
        if self._swipe_started_ms is None:
            return STATUS["swipe"]
        if timestamp - self._swipe_started_ms > SWIPE_TIME_MS:
            self._anchor = hand.palm
            self._swipe_started_ms = None
            return STATUS["swipe"]
        if abs(dx) > max(1.10 * hand.size, 1.45 * abs(dy)):
            keys = ("ctrl", "tab") if dx > 0 else ("ctrl", "shift", "tab")
        elif abs(dy) > max(1.10 * hand.size, 1.45 * abs(dx)):
            keys = ("win", "tab") if dy < 0 else ("win", "d")
        else:
            return STATUS["swipe"]
        self.output.hotkey(*keys)
        self._swiped = True
        return "Lower a finger before swiping again."
