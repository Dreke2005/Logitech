"""Recognize a tap and release from fingertip motion relative to its knuckle."""

import math
from dataclasses import dataclass
from typing import Optional

from typing_calibration import finger_reach, ordered_hands

FINGER_KNUCKLES = {"index": 5, "middle": 9, "ring": 13, "pinky": 17, "thumb": 2}
HOME_FINGERS = ("index", "middle", "ring", "pinky")
OPEN_TAP = 0.08
DOWN_TAP = 0.19
MAX_TAP = 0.62
MAX_SIDEWAYS = 0.30
MAX_KNUCKLE_MOVE = 0.26
MAX_FRAME_GAP_MS = 220
MIN_CYCLE_MS = 45
MAX_CYCLE_MS = 320
MAX_DESCENT_MS = 145


def _along(origin, point, axis):
    return (point[0] - origin[0]) * axis[0] + (point[1] - origin[1]) * axis[1]


@dataclass
class _FingerSample:
    tip: tuple
    knuckle: tuple
    depth: float
    palm_axis: tuple


@dataclass
class _Motion:
    start: _FingerSample
    started_ms: int
    key: Optional[str]
    pressed: bool = False
    cancelled: bool = False

    def curl_axis(self):
        x = self.start.knuckle[0] - self.start.tip[0]
        y = self.start.knuckle[1] - self.start.tip[1]
        length = math.hypot(x, y)
        return (x / length, y / length) if length >= 1 else self.start.palm_axis

    def down(self, sample):
        tip_dx = sample.tip[0] - self.start.tip[0]
        tip_dy = sample.tip[1] - self.start.tip[1]
        dx = tip_dx - (sample.knuckle[0] - self.start.knuckle[0])
        dy = tip_dy - (sample.knuckle[1] - self.start.knuckle[1])
        finger_x, finger_y = self.curl_axis()
        palm_x, palm_y = self.start.palm_axis
        # Model depth adds a signal when the dip is mostly into the image.
        return max(dx * finger_x + dy * finger_y, dx * palm_x + dy * palm_y,
                   0.60 * (sample.depth - self.start.depth))


class TapDetector:
    """Hold input armed while both hands stay near the calibrated typing pose."""

    def __init__(self):
        self.reset()

    def reset(self):
        self.active = False
        self.hovered_key = None
        self._last_ms = None
        self._motions = [{}, {}]

    def update(self, tracking, layout):
        """Return keys from completed down-and-return tap motions in this frame."""
        hands = ordered_hands(tracking, layout.row_axis)
        timestamp = tracking["timestamp_ms"]
        if hands is None or tuple(tracking["frame_size"]) != tuple(layout.frame_size):
            self.reset()
            return []
        if self._last_ms is not None and not 0 < timestamp - self._last_ms <= MAX_FRAME_GAP_MS:
            self.reset()
            return []

        frames = [self._active_hand(hand, layout, side) for side, hand in enumerate(hands)]
        if any(frame is None for frame in frames):
            self.reset()
            return []

        self.active = True
        self._last_ms = timestamp
        self.hovered_key = None
        keys = []
        for side, (samples, palm) in enumerate(frames):
            motions = self._motions[side]
            down_together = sum(
                name in motions and motions[name].down(sample) > 0.12 * palm
                for name, sample in samples.items()
                if name != "thumb"
            )
            if down_together >= 3:
                # Curling several fingers together is a grip, not a set of taps.
                self.reset()
                return []
            # An isolated overlap in projection can be harmless. If it occurs
            # during an unfinished tap, require a fresh typing pose.
            if math.dist(samples["index"].tip, samples["thumb"].tip) < 0.16 * palm:
                if any(motions[name].pressed for name in ("index", "thumb") if name in motions):
                    self.reset()
                    return []
                motions.pop("index", None)
                motions.pop("thumb", None)
                samples = {name: samples[name] for name in ("middle", "ring", "pinky")}
            for name, sample in samples.items():
                key = self._observe(motions, name, sample, palm, timestamp, layout)
                if key is not None:
                    keys.append(key)
        return list(dict.fromkeys(keys))

    @staticmethod
    def _active_hand(hand, layout, side):
        points = hand["landmarks"]
        wrist, middle_knuckle = points[0]["pixel"], points[9]["pixel"]
        palm = layout.palm_sizes[side]
        size = math.dist(wrist, middle_knuckle)
        if (
            math.dist(wrist, layout.wrists[side]) > max(16, 1.55 * palm)
            or not 0.48 * palm <= size <= 1.75 * palm
        ):
            return None
        palm_axis = ((wrist[0] - middle_knuckle[0]) / size, (wrist[1] - middle_knuckle[1]) / size)
        samples = {}
        for name, joint in FINGER_KNUCKLES.items():
            tip = hand["fingertips"][name]
            knuckle = points[joint]
            depth = (tip.get("z", 0) - knuckle.get("z", 0)) * layout.frame_size[0]
            samples[name] = _FingerSample(tuple(tip["pixel"]), tuple(knuckle["pixel"]), depth, palm_axis)
        curled = sum(finger_reach(wrist, samples[name].knuckle, samples[name].tip) <= 0.08 * size
                     for name in HOME_FINGERS)
        if curled >= 3:
            return None
        return samples, palm

    def _observe(self, motions, name, sample, palm, timestamp, layout):
        motion = motions.get(name)
        key = layout.key_at(*sample.tip)
        if name == "thumb" and key != "Space":
            key = None

        # Moving sideways or moving the knuckle to reach another key starts a
        # fresh baseline. Use the target under the fingertip before flexing.
        if motion is None:
            motions[name] = _Motion(sample, timestamp, key)
            return None
        curl_x, curl_y = motion.curl_axis()
        palm_x, palm_y = motion.start.palm_axis
        sideways = min(
            abs(_along(motion.start.tip, sample.tip, (-curl_y, curl_x))),
            abs(_along(motion.start.tip, sample.tip, (-palm_y, palm_x))),
        )
        if (
            sideways > MAX_SIDEWAYS * palm
            or math.dist(motion.start.knuckle, sample.knuckle) > MAX_KNUCKLE_MOVE * palm
        ):
            motions[name] = _Motion(sample, timestamp, key)
            return None

        down = motion.down(sample) / palm
        age = timestamp - motion.started_ms
        if motion.cancelled:
            if down <= OPEN_TAP or key != motion.key:
                motions[name] = _Motion(sample, timestamp, key)
            return None
        if (
            down > MAX_TAP
            or down < -0.14
            or age > MAX_CYCLE_MS
            or (not motion.pressed and age > MAX_DESCENT_MS)
        ):
            if down > OPEN_TAP and key is not None and key == motion.key:
                # A held or overextended finger must come back up before it can
                # register another tap at this key.
                motion.cancelled = True
            else:
                motions[name] = _Motion(sample, timestamp, key)
            return None

        if motion.pressed and down <= OPEN_TAP:
            motions[name] = _Motion(sample, timestamp, key)
            near_edge = key is None and math.dist(motion.start.tip, sample.tip) <= max(3, 0.12 * palm)
            if age >= MIN_CYCLE_MS and (key == motion.key or near_edge):
                self.hovered_key = motion.key
                return motion.key
            return None

        if down <= OPEN_TAP:
            motions[name] = _Motion(sample, timestamp, key)
            return None

        if motion.key is not None:
            if down >= DOWN_TAP:
                motion.pressed = True
            self.hovered_key = motion.key
        return None
