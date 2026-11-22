"""Camera exposure for writing on paper, through the UVC controls of the SDK's OpenCV capture.

The camera's own auto exposure follows the whole scene: a hand or a dark laptop coming into
view brightens the picture, which washes out faint ink and, during a recording, changes the
paper's brightness so that the whole page looks changed. ``PaperExposure`` takes manual control
instead: it keeps the brightest part of the scene (the paper) between ``LOW`` and ``HIGH`` -
bright, but not clipped, so the ink keeps its contrast - and changes nothing while a recording
runs. A camera that refuses manual exposure is left on its own automatic exposure.
"""

import json
import os
import time
from pathlib import Path

import cv2
import numpy as np

LOW, HIGH = 170, 242  # Brightness of the paper (97th percentile of the scene) to keep.
MANUAL, AUTO = 0.25, 0.75  # DirectShow values of CAP_PROP_AUTO_EXPOSURE.


class PaperExposure:
    def __init__(self, cap, interval=1.0, settle=0.6):
        self.cap, self.interval, self.settle = cap, interval, settle
        self.checked = self.changed = time.monotonic()
        self.value = cap.get(cv2.CAP_PROP_EXPOSURE)
        self.manual = False
        # DirectShow exposure is log2 seconds (e.g. -6 = 1/64 s). Without a usable value, leave the camera alone.
        self.enabled = np.isfinite(self.value) and -13 <= self.value <= 0
        self.range = (-11.0, -6.0)  # Never longer than 1/64 s: a moving hand or pen would blur.

    @staticmethod
    def paper_level(frame):
        gray = cv2.cvtColor(frame[::8, ::8], cv2.COLOR_BGR2GRAY)
        return float(np.percentile(gray, 97))

    def update(self, frame, locked=False):
        """Check the picture about once a second and step the exposure when the paper is clipped or dim.
        Returns the new exposure value when it changed, else None."""
        now = time.monotonic()
        if not self.enabled or locked or now - self.checked < self.interval or now - self.changed < self.settle:
            return None
        self.checked = now
        level = self.paper_level(frame)
        step = -1.0 if level >= HIGH else 1.0 if level < LOW else 0.0
        if not step:
            return None
        target = float(np.clip(self.value + step, *self.range))
        if target == self.value:
            return None
        if not self.manual:
            if not self.cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, MANUAL):
                self.enabled = False
                return None
            self.manual = True
        if not self.cap.set(cv2.CAP_PROP_EXPOSURE, target):
            self.enabled = False
            self.cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, AUTO)
            return None
        self.value, self.changed = target, now
        return target

    def release(self):
        """Give the exposure back to the camera, so other programs get its automatic exposure."""
        if self.manual:
            self.cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, AUTO)
            self.manual = False


FOCUS_FILE = Path(os.environ.get("LOCALAPPDATA", Path.home())) / "LogiInsightPaper" / "focus.json"


def sharpness(frame):
    """Variance of the Laplacian of a quarter-size grey picture: high when the scene is in focus."""
    small = cv2.resize(frame, None, fx=0.25, fy=0.25, interpolation=cv2.INTER_AREA)
    return float(cv2.Laplacian(cv2.cvtColor(small, cv2.COLOR_BGR2GRAY), cv2.CV_32F).var())


class PaperFocus:
    """A fixed focus on the desk instead of the camera's autofocus.

    The autofocus hunts between the hand (closer to the camera) and the paper: on the LogiInsight
    camera the sharpness of the page jumped between about 90 and 700 from one recording to the next,
    and a blurred recording cannot be read. So the focus is locked: when the camera starts, a short
    sweep (coarse, then fine around the best) finds the sharpest setting, which is kept for the next
    start. While a recording runs nothing changes; between recordings a clear loss of sharpness (the
    desk was rearranged) starts a new fine sweep. A camera without manual focus is left alone."""

    COARSE = tuple(range(350, 751, 50))
    SETTLE = 0.35  # Seconds for the lens to get to a new setting.

    def __init__(self, cap):
        self.cap = cap
        self.enabled = bool(cap.set(cv2.CAP_PROP_AUTOFOCUS, 0))
        self.best = None  # (sharpness, focus) of the locked setting
        self.queue, self.scores, self.target, self.since = [], [], None, 0.0
        self.checked = time.monotonic()
        saved = None
        try:
            saved = float(json.loads(FOCUS_FILE.read_text(encoding="utf-8"))["focus"])
        except (OSError, ValueError, KeyError, TypeError):
            pass
        if self.enabled:
            # Start sharp at once from the last setting, then refine; without one, sweep the range.
            self._plan(saved + np.arange(-50, 51, 25) if saved is not None else self.COARSE, coarse=saved is None)

    def _plan(self, values, coarse=False):
        self.queue = [float(value) for value in values if 0 <= value <= 1023]
        self.scores, self.coarse = [], coarse
        self._next()

    def _next(self):
        self.target = self.queue.pop(0) if self.queue else None
        if self.target is not None:
            self.cap.set(cv2.CAP_PROP_FOCUS, self.target)
            self.since = time.monotonic()

    def update(self, frame, locked=False):
        """Called for every frame. Returns the focus value when one was locked, else None."""
        if not self.enabled:
            return None
        now = time.monotonic()
        if self.target is not None:
            if now - self.since < self.SETTLE:
                return None
            self.scores.append((sharpness(frame), self.target))
            if self.queue:
                self._next()
                return None
            score, focus = max(self.scores)
            if self.coarse:
                self._plan(focus + np.arange(-40, 41, 20))  # Fine sweep around the best coarse setting.
                return None
            self.target = None
            self.best = score, focus
            self.cap.set(cv2.CAP_PROP_FOCUS, focus)
            try:
                FOCUS_FILE.parent.mkdir(parents=True, exist_ok=True)
                FOCUS_FILE.write_text(json.dumps({"focus": focus}), encoding="utf-8")
            except OSError:
                pass
            return focus
        if locked or self.best is None or now - self.checked < 4.0:
            return None
        self.checked = now
        if sharpness(frame) < 0.45 * self.best[0]:
            self._plan(self.best[1] + np.arange(-60, 61, 30))  # The scene changed: focus again.
        return None

    def release(self):
        """Give the focus back to the camera's autofocus for other programs."""
        if self.enabled:
            self.cap.set(cv2.CAP_PROP_AUTOFOCUS, 1)
