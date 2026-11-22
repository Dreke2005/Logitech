"""Turn palm-normalized thumb/index strokes into discrete zoom notches."""

import math

ZOOM_STEP = 0.085
DECISIVE_STEP = 0.13
RETURN_TRAVEL = 0.06
MAX_SAMPLE_GAP_MS = 240
MAX_SAMPLE_JUMP = 1.0
MAX_NOTCHES = 3


class PinchZoom:
    """Latch the first stroke direction until a completion gesture.

    Travel against that direction repositions the fingers for another stroke.
    An occlusion or nonzoom pose rebases the aperture without ending the stroke.
    """

    def __init__(self):
        self.reset()

    @property
    def direction(self):
        """Return +1 for opening strokes, -1 for closing, or zero if unlocked."""
        return self._direction

    def reset(self):
        self._direction = 0
        self._clear_samples()

    def _clear_samples(self):
        self._anchor = None
        self._record = None
        self._last_distance = None
        self._last_ms = None
        self._candidate = 0
        self._returning = False

    def inactive(self, _timestamp_ms=None):
        """Discard sampled motion while retaining the chosen zoom direction."""
        self._clear_samples()

    def complete(self, pose):
        """Reset either zoom direction with a flat hand or a fist."""
        if pose in ("fist", "flat"):
            self.reset()
            return True
        return False

    def update(self, distance, timestamp_ms):
        """Return signed zoom notches for one aperture sample, or zero."""
        if not math.isfinite(distance) or distance < 0:
            self.inactive(timestamp_ms)
            return 0

        if self._last_ms is not None and (
            not 0 < timestamp_ms - self._last_ms <= MAX_SAMPLE_GAP_MS
            or abs(distance - self._last_distance) > MAX_SAMPLE_JUMP
        ):
            # A skipped frame or landmark jump isn't a measurable stroke.
            self._clear_samples()
        self._last_ms = timestamp_ms
        self._last_distance = distance

        if self._anchor is None:
            self._anchor = self._record = distance
            return 0

        if self._direction:
            progress = self._direction * (distance - self._record)
            if not self._returning and progress <= -RETURN_TRAVEL + 1e-9:
                self._returning = True
                self._anchor = distance
                self._candidate = 0
                return 0
            if self._returning and self._direction * (distance - self._anchor) <= 0:
                # Follow the end of the repositioning motion to its full extent.
                self._anchor = distance
                self._candidate = 0
                return 0
            if not self._returning and progress > 0:
                self._record = distance

        travel = distance - self._anchor
        direction = self._direction or (1 if travel > 0 else -1)
        span = direction * travel
        if span + 1e-9 < ZOOM_STEP:
            self._candidate = 0
            return 0
        if span < DECISIVE_STEP and self._candidate != direction:
            # Confirm short movements on a second frame to reject threshold jitter.
            self._candidate = direction
            return 0

        steps = math.floor((span + 1e-9) / ZOOM_STEP)
        notches = direction * min(MAX_NOTCHES, steps)
        if steps > MAX_NOTCHES:
            self._anchor = distance
        else:
            self._anchor += notches * ZOOM_STEP
        if not self._direction or self._returning:
            self._record = distance
        self._direction = direction
        self._returning = False
        self._candidate = direction
        return notches
