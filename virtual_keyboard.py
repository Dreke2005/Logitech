"""Render the calibrated tap keyboard and keep its typed text."""

import math

import cv2
import numpy as np

from typing_calibration import MAX_FRAME_GAP_MS, PoseCalibration
from typing_taps import TapDetector

FONT = cv2.FONT_HERSHEY_SIMPLEX
RESUME_HOLD_MS = 120
PAUSE_NOTICE_MS = 200
THUMB_RELEASE = 2.0
THUMB_RELEASE_HOLD_MS = 400


class VirtualKeyboard:
    def __init__(self):
        self.text = ""
        self.shift = False
        self.calibration = PoseCalibration()
        self.taps = TapDetector()
        self.reference = None
        self.last_key = None
        self.highlight_until = 0
        self._reset_gate()

    @property
    def keys(self):
        return () if self.reference is None else self.reference.keys

    def reset_calibration(self):
        self.reference = None
        self.calibration.reset()
        self.taps.reset()
        self.shift = False
        self.last_key = None
        self.highlight_until = 0
        self._reset_gate()

    def _reset_gate(self):
        self._frame_ms = None
        self._pause_since = None
        self._resume_since = None
        self._opening_since = None
        self._opening_last_ms = None

    def _pause(self, timestamp):
        if self._pause_since is None:
            self._pause_since = timestamp
        self.last_key = None
        self._resume_since = None

    def _handle_thumb_release(self, tracking, timestamp):
        """Keep the layout until a wide thumb gap is observed continuously."""
        width, height = self.reference.frame_size
        tips = [hand["fingertips"]["thumb"]["pixel"] for hand in tracking["hands"]]
        palms = self.reference.palm_sizes
        separated = (
            len(tips) == 2
            and all(0 <= x < width and 0 <= y < height for x, y in tips)
            and math.dist(*tips) > THUMB_RELEASE * (sum(palms) / 2)
        )
        if not separated:
            self._opening_since = None
            self._opening_last_ms = None
            return False
        if self._opening_last_ms is None or not 0 < timestamp - self._opening_last_ms <= MAX_FRAME_GAP_MS:
            self._opening_since = timestamp
        self._opening_last_ms = timestamp
        self._pause(timestamp)
        self.taps.reset()
        if timestamp - self._opening_since >= THUMB_RELEASE_HOLD_MS:
            self.reset_calibration()
        return True

    def _show_pause(self):
        return self._pause_since is not None and self._frame_ms - self._pause_since >= PAUSE_NOTICE_MS

    def update(self, tracking):
        if self.reference is not None and tuple(tracking["frame_size"]) != self.reference.frame_size:
            self.reset_calibration()
        if self.reference is None:
            self.reference = self.calibration.update(tracking)
            if self.reference is None:
                return

        timestamp = tracking["timestamp_ms"]
        self._frame_ms = timestamp
        if self._handle_thumb_release(tracking, timestamp):
            return
        pressed_keys = self.taps.update(tracking, self.reference)
        if not self.taps.active:
            self._pause(timestamp)
            return

        if self._pause_since is not None:
            if self._resume_since is None:
                self._resume_since = timestamp
            if timestamp - self._resume_since >= RESUME_HOLD_MS:
                # A finger movement that began while paused cannot become a
                # tap after resuming. Seed a fresh baseline at the current pose.
                self.taps.reset()
                self.taps.update(tracking, self.reference)
                self._pause_since = None
                self._resume_since = None
            return

        for label in pressed_keys:
            self._type(label)
            self.last_key = label
            self.highlight_until = timestamp + 200
        if timestamp >= self.highlight_until:
            self.last_key = None

    def _type(self, label):
        if label == "Shift":
            self.shift = not self.shift
        elif label == "Backspace":
            self.text = self.text[:-1]
        elif label == "Space":
            self.text += " "
        elif label == "Enter":
            self.text += "\n"
        else:
            self.text += label.upper() if self.shift else label.lower()
            if label.isalpha():
                self.shift = False

    def draw_keys(self, frame):
        if self.reference is None:
            return

        layer = frame.copy()
        show_pause = self._show_pause()
        for key in self.keys:
            fill = (55, 56, 62) if show_pause else (37, 42, 53)
            if key.label == self.last_key:
                fill = (92, 230, 160)
            elif self._pause_since is None and key.label == self.taps.hovered_key:
                fill = (117, 105, 63)
            elif key.label == "Shift" and self.shift:
                fill = (170, 102, 64)
            corners = np.rint(key.corners).astype(np.int32)
            cv2.fillConvexPoly(layer, corners, fill, lineType=cv2.LINE_AA)

        cv2.addWeighted(layer, 0.84, frame, 0.16, 0, dst=frame)
        for key in self.keys:
            top_left, top_right, bottom_right, bottom_left = np.asarray(key.corners)
            horizontal = top_right - top_left
            vertical = bottom_left - top_left
            key_width = np.linalg.norm(horizontal)
            key_height = np.linalg.norm(vertical)
            # Leave the middle of each letter key free for the fingertip marker.
            if len(key.label) == 1:
                center = top_left + 0.24 * horizontal + 0.29 * vertical
                width_fraction = 0.40
            else:
                center = (top_left + bottom_right) / 2
                width_fraction = 0.80
            (text_width, text_height), _ = cv2.getTextSize(key.label, FONT, 1, 1)
            scale = min(0.75, key_width * width_fraction / text_width, key_height * 0.48 / text_height)
            (width, height), _ = cv2.getTextSize(key.label, FONT, scale, 1)
            color = (19, 28, 38) if key.label == self.last_key else (245, 245, 245)
            origin = (round(center[0] - width / 2), round(center[1] + height / 2))
            cv2.putText(frame, key.label, origin, FONT, scale, color, 1, cv2.LINE_AA)

    def add_text_panel(self, frame):
        width = frame.shape[1]
        scale = min(0.6, width / 960)
        top = frame.shape[0]
        panel = cv2.copyMakeBorder(frame, 0, 99, 0, 0, cv2.BORDER_CONSTANT, value=(25, 30, 41))
        show_pause = self._show_pause()
        if show_pause and self._opening_since is not None:
            hint = "Keep your thumbs apart to dismiss the keyboard."
        elif show_pause:
            hint = "Paused. Bring both hands back to the keyboard."
        elif self.reference is not None:
            hint = "Ready. Tap a finger down and lift it."
        else:
            hint = self.calibration.hint
        hint_width = cv2.getTextSize(hint, FONT, 1, 1)[0][0]
        hint_scale = min(scale, max(1, width - 24) / hint_width)
        cv2.putText(panel, hint, (12, top + 20), FONT, hint_scale, (183, 194, 209), 1, cv2.LINE_AA)
        if self.reference is None and self.calibration.progress:
            x = 12 + round(max(0, width - 24) * self.calibration.progress)
            cv2.line(panel, (12, top + 28), (x, top + 28), (92, 230, 160), 2)

        char_width = max(1, cv2.getTextSize("W", FONT, scale, 1)[0][0])
        columns = max(1, (width - 28) // char_width)
        lines = []
        for line in (self.text[-3 * columns:] + "|").split("\n"):
            lines.extend([line[i:i + columns] for i in range(0, len(line), columns)] or [""])
        for i, line in enumerate(lines[-3:]):
            cv2.putText(panel, line, (12, top + 45 + i * 20), FONT, scale,
                        (250, 250, 250), 1, cv2.LINE_AA)
        return panel
