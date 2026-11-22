"""Use one hand for scrolling, zooming, and switching tabs."""

import math
from dataclasses import dataclass
from itertools import combinations
from statistics import median
from typing import Optional

from pinch_zoom import PinchZoom

MODE_HOLD_MS = 80
RESET_HOLD_MS = 110
MAX_FRAME_GAP_MS = 240
SCROLL_STEP = 0.24
SWIPE_TIME_MS = 550
SWIPE_MIN_MS = 55
OVERVIEW_HOLD_MS = 175
OVERVIEW_RELEASE_MS = 140

FINGERS = ((5, 6, 8), (9, 10, 12), (13, 14, 16), (17, 18, 20))
FINGERTIPS = (4, 8, 12, 16, 20)
STATUS = {
    "scroll": "Scroll with fingers together. Spread them flat or make a fist to reset.",
    "zoom": "Tuck three fingers. Spread or pinch thumb and index to zoom.",
    "swipe": "Swipe three fingers left or right to switch tabs.",
}
NAVIGATION_HINT = "Open hand to scroll. Three fingers swipe. Gather all five fingertips for tabs."


def _delta(point, anchor):
    return point[0] - anchor[0], point[1] - anchor[1]


def _average(first, second):
    return (first[0] + second[0]) / 2, (first[1] + second[1]) / 2


def _extension(points, finger):
    base, joint, tip = (points[i] for i in finger)
    lower_x, lower_y = _delta(joint, base)
    upper_x, upper_y = _delta(tip, joint)
    lower = math.hypot(lower_x, lower_y)
    upper = math.hypot(upper_x, upper_y)
    if not lower or not upper:
        return 0.0
    return (lower_x * upper_x + lower_y * upper_y) / (lower * upper)


@dataclass
class _Hand:
    wrist: tuple
    palm: tuple
    size: float
    fingers: tuple
    extension: tuple
    scroll_tips: tuple
    index_pinch: float
    index_reach: float
    thumb_reach: float
    thumb_clearance: float
    finger_spread: float
    tip_diameter: float
    four_tip_diameter: float
    tips_outside_palm: bool
    handedness: str

    @classmethod
    def from_landmarks(cls, landmarks, width, height, handedness=""):
        if len(landmarks) < 21:
            return None
        points = [tuple(landmark["pixel"]) for landmark in landmarks]
        wrist, thumb, index, middle = (points[i] for i in (0, 4, 8, 12))
        if any(not (0 <= x < width and 0 <= y < height) for x, y in (wrist, thumb, index, middle)):
            return None
        size = max(math.dist(wrist, points[9]), 0.9 * math.dist(points[5], points[17]))
        if size < max(12, 0.028 * min(width, height)):
            return None
        knuckles = [points[base] for base, _, _ in FINGERS]
        knuckle_center = tuple(sum(axis) / len(axis) for axis in zip(*knuckles))
        palm = _average(wrist, knuckle_center)
        extension = tuple(_extension(points, finger) for finger in FINGERS)
        tips = [points[i] for i in FINGERTIPS]
        all_visible = all(0 <= x < width and 0 <= y < height for x, y in tips)
        tip_diameter = max(math.dist(a, b) for a, b in combinations(tips, 2)) / size if all_visible else math.inf
        four_tip_diameter = (
            max(math.dist(a, b) for a, b in combinations(tips[1:], 2)) / size if all_visible else math.inf
        )
        fx, fy = _delta(knuckle_center, wrist)
        forward_scale = math.hypot(fx, fy) * size
        # All five can project close together in a fist. A gathered pose
        # presents several fingertips in front of their base knuckles.
        outward = sum((tip[0] - base[0]) * fx + (tip[1] - base[1]) * fy > 0.15 * forward_scale
                      for tip, base in zip(tips[1:], knuckles))
        return cls(wrist, palm, size, tuple(score > 0.42 for score in extension),
                   extension, tuple(points[tip] for _, _, tip in FINGERS),
                   math.dist(thumb, index) / size, math.dist(points[5], index) / size,
                   math.dist(points[2], thumb) / size,
                   min(math.dist(thumb, points[tip]) for tip in (12, 16, 20)) / size,
                   math.dist(index, points[20]) / max(0.55 * size, math.dist(points[5], points[17])),
                   tip_diameter, four_tip_diameter, outward >= 3, handedness)

    def matches(self, other):
        if self.handedness and other.handedness:
            return self.handedness == other.handedness
        return math.dist(self.wrist, other.wrist) <= 1.9 * self.size


@dataclass
class _PoseHold:
    hand: _Hand
    since: int
    last_ms: int
    samples: int = 1

    @classmethod
    def observe(cls, previous, hand, timestamp, *, steady=False):
        if (previous is None or not 0 < timestamp - previous.last_ms <= MAX_FRAME_GAP_MS
                or not previous.hand.matches(hand)
                or (steady and math.dist(previous.hand.palm, hand.palm) > 0.28 * previous.hand.size)):
            return cls(hand, timestamp, timestamp)
        previous.last_ms = timestamp
        previous.samples += 1
        return previous


@dataclass
class _SwipePath:
    hand: _Hand
    started_ms: Optional[int] = None
    direction: int = 0
    last_dx: float = 0
    segments: int = 0


@dataclass
class _ScrollPosition:
    palm: tuple
    tips: tuple
    size: float
    sent_x: int = 0
    sent_y: int = 0

    def displacement(self, hand, axis):
        palm = hand.palm[axis] - self.palm[axis]
        changes = [current[axis] - initial[axis] for current, initial in zip(hand.scroll_tips, self.tips)]
        tip = median(changes)
        tolerance = max(2, 0.10 * self.size)
        coherent = sum(abs(change - tip) <= tolerance for change in changes) >= 3
        if not coherent or abs(tip) <= abs(palm):
            return palm
        if tip * palm >= 0:
            return palm + 0.82 * (tip - palm)
        if abs(palm) < tolerance:
            # Small palm jitter should not hide coordinated finger motion.
            return palm + 0.82 * tip * (1 - abs(palm) / tolerance)
        return palm


class TouchpadController:
    """Accept preview landmarks and emit commands through an injected output object."""

    def __init__(self, output):
        self.output = output
        self._pinch_zoom = PinchZoom()
        self.reset()

    def reset(self):
        """Clear the gesture session after an explicit user pause."""
        self._overview_owner = None
        self._swipe_owner = None
        self.tracking_lost()
        self._pinch_zoom.reset()

    def tracking_lost(self):
        """Discard sampled motion without treating lost tracking as a reset pose."""
        self._suspend_navigation()
        self._overview_hold = self._overview_release = None
        self._swipe_release = None

    def _suspend_navigation(self):
        self._clear_mode()
        self._last_ms = None
        self._last_hand = None
        self._candidate = None
        self._candidate_ms = 0
        self._pinch_zoom.inactive()
        self._zoom_start = None
        self._reset_pose = None
        self._reset_since = self._last_reset_ms = None

    def _clear_mode(self):
        self._mode = None
        self._scroll_position = None
        self._swipe_path = None

    def _pose(self, hand):
        index, middle, ring, pinky = hand.fingers
        if (not any(hand.fingers) and hand.index_reach < 0.34
                and sum(score <= -0.25 for score in hand.extension) >= 3):
            return "fist" if hand.index_pinch < 0.80 else None
        gather_limit = 0.60 if self._overview_hold or self._overview_owner else 0.48
        if hand.tips_outside_palm and hand.tip_diameter <= gather_limit:
            return "overview"
        if hand.tips_outside_palm and hand.four_tip_diameter <= 0.40:
            return "gathering"
        strength = min(hand.extension)
        open_hand = strength >= 0.10 or (sum(hand.fingers) >= 3 and strength >= -0.08)
        # Spread the fingers for a flat reset. An open hand with fingers
        # together is the separate scroll pose.
        flat_spread = 1.28 if self._reset_pose == "flat" else 1.43
        if open_hand and hand.finger_spread >= flat_spread:
            return "flat"
        same_hand = self._last_hand is not None and math.dist(hand.wrist, self._last_hand.wrist) < hand.size
        continuing_scroll = "scroll" in (self._mode, self._candidate) and same_hand
        if open_hand or (continuing_scroll and sum(hand.fingers) >= 3 and strength > -0.24):
            return "scroll"
        if index and middle and ring and hand.extension[3] <= -0.25:
            return "swipe"
        continuing_zoom = "zoom" in (self._mode, self._candidate) and same_hand
        reach = 0.24 if continuing_zoom else 0.32
        # An index point can have long thumb reach because the thumb crosses
        # the palm. Keep it distinct from a free thumb-index pinch.
        clearance = 0.20 if continuing_zoom else 0.30
        if (not (middle or ring or pinky) and min(hand.index_reach, hand.thumb_reach) >= reach
                and hand.thumb_clearance >= clearance):
            return "zoom"
        return None

    def _observe_reset(self, pose, timestamp):
        since = self._reset_since
        if (pose != self._reset_pose or self._last_reset_ms is None
                or not 0 < timestamp - self._last_reset_ms <= MAX_FRAME_GAP_MS):
            since = timestamp
        # Any stop-pose sample suppresses movement. Require a short hold before
        # it can clear the zoom latch and enable another gesture.
        self._suspend_navigation()
        self._reset_pose, self._reset_since, self._last_reset_ms = pose, since, timestamp
        if timestamp - since < RESET_HOLD_MS:
            label = "Fist" if pose == "fist" else "Flat hand"
            return f"{label}: paused. Hold briefly to reset."
        self._pinch_zoom.complete(pose)
        return "Reset. Reposition your hand, then start the next gesture."

    def _observe_overview(self, hand, timestamp):
        self._suspend_navigation()
        self._overview_release = None
        if self._overview_owner is not None:
            return "Tab overview shortcut sent. Open your fingers to rearm."
        self._overview_hold = _PoseHold.observe(self._overview_hold, hand, timestamp, steady=True)
        if self._overview_hold.samples < 3 or timestamp - self._overview_hold.since < OVERVIEW_HOLD_MS:
            return "Hold all five fingertips together briefly to open tab overview."
        self.output.hotkey("ctrl", "shift", "a")
        self._overview_owner = hand
        self._overview_hold = None
        return "Tab overview shortcut sent. Open your fingers to rearm."

    def _rearm_overview(self, poses, timestamp):
        self._suspend_navigation()
        releases = [hand for hand, pose in poses if self._overview_owner.matches(hand)
                    and ((pose in ("scroll", "flat") and hand.tip_diameter >= 0.72) or pose == "fist")]
        if len(releases) != 1:
            self._overview_release = None
        else:
            self._overview_release = _PoseHold.observe(self._overview_release, releases[0], timestamp)
            if self._overview_release.samples >= 3 and timestamp - self._overview_release.since >= OVERVIEW_RELEASE_MS:
                self._overview_owner = self._overview_release = None
                return NAVIGATION_HINT
        return "Tab overview shortcut sent. Open your fingers to rearm."

    def _rearm_swipe(self, poses, timestamp):
        if self._swipe_owner is None:
            return
        releases = [hand for hand, pose in poses if self._swipe_owner.matches(hand)
                    and pose != "swipe" and (pose in ("fist", "flat", "overview") or not all(hand.fingers[:3]))]
        if len(releases) != 1:
            self._swipe_release = None
            return
        self._swipe_release = _PoseHold.observe(self._swipe_release, releases[0], timestamp)
        if timestamp - self._swipe_release.since >= MODE_HOLD_MS:
            self._swipe_owner = self._swipe_release = None

    def update(self, tracking):
        timestamp = tracking["timestamp_ms"]
        hands = [_Hand.from_landmarks(hand["landmarks"], *tracking["frame_size"], hand.get("handedness", ""))
                 for hand in tracking["hands"]]
        usable = [hand for hand in hands if hand is not None]
        if not usable:
            self.tracking_lost()
            return "Show a hand with its wrist and fingertips visible."
        poses = [(hand, self._pose(hand)) for hand in usable]
        recognized = [(hand, pose) for hand, pose in poses if pose in STATUS]
        gathered = [hand for hand, pose in poses if pose == "overview"]
        self._rearm_swipe(poses, timestamp)
        if gathered:
            if len(gathered) != 1 or recognized:
                self.tracking_lost()
                return "Use one hand at a time for tab overview."
            if self._pinch_zoom.direction:
                self.tracking_lost()
                return "Finish zoom with a spread flat hand or a fist."
            return self._observe_overview(gathered[0], timestamp)
        self._overview_hold = None
        if self._overview_owner is not None:
            return self._rearm_overview(poses, timestamp)
        if any(pose == "gathering" for _, pose in poses):
            self._suspend_navigation()
            return "Bring your thumb and all four fingertips together for tab overview."
        if not recognized:
            stop_pose = next((pose for _, pose in poses if pose in ("flat", "fist")), None)
            if stop_pose:
                return self._observe_reset(stop_pose, timestamp)
        if len(recognized) != 1:
            self._suspend_navigation()
            if recognized:
                self._swipe_release = None
                return "Use one hand at a time for navigation gestures."
            return NAVIGATION_HINT
        hand, pose = recognized[0]
        if pose != "zoom" and self._pinch_zoom.direction:
            self.tracking_lost()
            return "Finish zoom with a spread flat hand or a fist."
        self._reset_pose = self._reset_since = self._last_reset_ms = None
        previous = self._last_hand
        if self._last_ms is not None and (
            not 0 < timestamp - self._last_ms <= MAX_FRAME_GAP_MS
            or math.dist(hand.wrist, previous.wrist) > 1.9 * previous.size
            or not 0.62 * previous.size <= hand.size <= 1.60 * previous.size
        ):
            self.tracking_lost()
        if pose != "zoom":
            self._pinch_zoom.inactive(timestamp)
        self._last_ms, self._last_hand = timestamp, hand
        if pose != self._mode:
            if pose != self._candidate:
                self._candidate, self._candidate_ms = pose, timestamp
                self._zoom_start = (hand.index_pinch, timestamp) if pose == "zoom" else None
            if timestamp - self._candidate_ms < MODE_HOLD_MS:
                if pose == "zoom":
                    self._pinch_zoom.inactive(timestamp)
                return f"Hold the pose briefly. {STATUS[pose]}"
            self._clear_mode()
            self._mode = pose
            if self._zoom_start is not None:
                # Debounce the pose, but measure from its first valid aperture
                # so a short, deliberate motion isn't lost during arming.
                self._pinch_zoom.update(*self._zoom_start)
                self._zoom_start = None
        elif pose != self._candidate:
            # A fleeting pose change freezes input. Rebase on return instead
            # of sending the displacement accumulated during the transition.
            self._candidate, self._candidate_ms = pose, timestamp
            self._scroll_position = None
            self._swipe_path = None

        if pose == "scroll":
            self._scroll(hand)
        elif pose == "zoom":
            return self._zoom(hand, timestamp)
        else:
            return self._swipe(hand, timestamp)
        return STATUS[pose]

    def _scroll(self, hand):
        if self._scroll_position is None:
            self._scroll_position = _ScrollPosition(hand.palm, hand.scroll_tips, hand.size)
            return
        position = self._scroll_position
        step = max(6, SCROLL_STEP * position.size)
        dx = position.displacement(hand, 0) - position.sent_x * step
        dy = position.displacement(hand, 1) - position.sent_y * step
        x, y = (max(-3, min(3, math.trunc(delta / step))) for delta in (dx, dy))
        if x or y:
            position.sent_x += x
            position.sent_y += y
            self.output.scroll(vertical=-y, horizontal=x)

    def _zoom(self, hand, timestamp):
        notches = self._pinch_zoom.update(hand.index_pinch, timestamp)
        if notches:
            self.output.zoom(notches)
        if self._pinch_zoom.direction > 0:
            return "Zoom in: spread again to continue. Flat hand or fist to reset."
        if self._pinch_zoom.direction < 0:
            return "Zoom out: pinch again to continue. Flat hand or fist to reset."
        return STATUS["zoom"]

    def _swipe(self, hand, timestamp):
        if self._swipe_owner is not None:
            return "Lower a finger before swiping again."
        if self._swipe_path is None:
            self._swipe_path = _SwipePath(hand)
            return STATUS["swipe"]
        path = self._swipe_path
        dx, dy = _delta(hand.palm, path.hand.palm)
        size = path.hand.size
        direction = 1 if dx > 0 else -1
        horizontal = abs(dx) > 1.45 * abs(dy)
        if path.started_ms is None:
            if horizontal and abs(dx) > 0.20 * size:
                path.started_ms = timestamp
                path.direction, path.last_dx, path.segments = direction, dx, 1
            return STATUS["swipe"]
        duration = timestamp - path.started_ms
        segment = path.direction * (dx - path.last_dx)
        if duration > SWIPE_TIME_MS or segment < -0.30 * size:
            self._swipe_path = _SwipePath(hand)
            return STATUS["swipe"]
        if segment > 0.10 * size:
            path.segments += 1
            path.last_dx = dx
        if not horizontal or abs(dx) <= 1.10 * size or duration < SWIPE_MIN_MS or path.segments < 2:
            return STATUS["swipe"]
        tip_movements = [_delta(tip, start) for tip, start in zip(hand.scroll_tips[:3], path.hand.scroll_tips[:3])]
        coherent = sum(sx * direction >= 0.60 * abs(dx) and abs(sx) >= 1.20 * abs(sy) for sx, sy in tip_movements)
        if coherent < 2:
            return "Swipe sideways with your hand and all three fingertips."
        keys = ("ctrl", "tab") if direction > 0 else ("ctrl", "shift", "tab")
        self.output.hotkey(*keys)
        self._swipe_owner = hand
        return "Lower a finger before swiping again."
