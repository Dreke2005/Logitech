"""Use the left index finger as a clutch for relative pointer dragging."""

# Author: Victor Lancieux
# Date: 2026-11-22 (hand pointer clutch added)
# Purpose: Catch the pointer with a left-pointing left index, drag relatively, release to re-anchor.

import math


class HandPointerController:
    """Catch at the current cursor position, drag relatively, and release on pose break."""

    def __init__(self, output, smoothing=0.35, straight_angle=22.0,
                 horizontal_angle=18.0, sensitivity=1.5):
        self.output = output
        self.smoothing = smoothing
        self.sensitivity = sensitivity
        self.straight_cosine = math.cos(math.radians(straight_angle))
        self.horizontal_tangent = math.tan(math.radians(horizontal_angle))
        self.cursor = None
        self.last_sent = None
        self.hand_anchor = None
        self.cursor_anchor = None
        self.button_held = False

    @staticmethod
    def _point(landmarks, index):
        pixel = landmarks[index].get("pixel")
        if not pixel or len(pixel) < 2:
            return None
        return float(pixel[0]), float(pixel[1])

    def _points_left(self, landmarks):
        base, pip, dip, tip = (self._point(landmarks, index) for index in (5, 6, 7, 8))
        if any(point is None for point in (base, pip, dip, tip)):
            return False
        for start, end in ((base, pip), (pip, dip), (dip, tip)):
            dx, dy = end[0] - start[0], end[1] - start[1]
            length = math.hypot(dx, dy)
            if length <= 1.0 or -dx / length < self.straight_cosine:
                return False
        dx, dy = tip[0] - base[0], tip[1] - base[1]
        return dx < 0 and abs(dy) <= self.horizontal_tangent * -dx

    def update(self, tracking, enabled=True):
        """Catch with the pose, move relative to its start, release to re-anchor."""
        if not enabled or not tracking:
            self.release()
            return "Hand pointer paused"

        frame_width, frame_height = tracking["frame_size"]
        hand = next((item for item in tracking.get("hands", ())
                     if str(item.get("handedness", "")).strip().lower() == "left"
                     and float(item.get("score", 0.0)) >= 0.35
                     and len(item.get("landmarks", ())) >= 21), None)
        if hand is None or frame_width <= 1 or frame_height <= 1:
            self.release()
            return "Left hand missing; pointer released"

        landmarks = hand["landmarks"]
        tip = self._point(landmarks, 8)
        if tip is None:
            self.release()
            return "Left index missing; pointer released"

        should_catch = self._points_left(landmarks)
        if not should_catch:
            if self.button_held:
                self.release()
                return "Pointer released; reposition and point right to catch again"
            return "Point left index horizontally left to catch pointer"

        if not self.button_held:
            # Re-anchor from the current cursor so catching never jumps it.
            self.hand_anchor = tip
            self.cursor_anchor = self.output.cursor_position()
            self.cursor = self.cursor_anchor
            self.last_sent = self.cursor
            self.output.button_down("left")
            self.button_held = True
            return "Pointer caught; move left hand to drag"

        delta = ((tip[0] - self.hand_anchor[0]) / (frame_width - 1),
                 (tip[1] - self.hand_anchor[1]) / (frame_height - 1))
        target = tuple(min(1.0, max(0.0, anchor + self.sensitivity * change))
                       for anchor, change in zip(self.cursor_anchor, delta))
        self.cursor = tuple(current + self.smoothing * (wanted - current)
                            for current, wanted in zip(self.cursor, target))
        if (self.last_sent is None
                or max(abs(a - b) for a, b in zip(self.cursor, self.last_sent)) >= 0.001):
            self.output.move_to(*self.cursor)
            self.last_sent = self.cursor
        return "Pointer caught; moving mouse"

    def release(self):
        """Release the mouse and clear the anchor, including on tracking loss."""
        if self.button_held:
            self.output.button_up("left")
        self.button_held = False
        self.hand_anchor = None
        self.cursor_anchor = None
        self.cursor = None
        self.last_sent = None

    def close(self):
        self.release()
