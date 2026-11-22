"""The calibrated sheet of paper for the workbench, kept up to date off the UI thread.

``paper_motion.SheetWatch`` finds the sheet's four edges when the view allows it and follows the
sheet from then on. Looking for a sheet can take a tenth of a second or more, which the UI thread
(camera preview, key presses) must never wait for, so ``SheetCalibration`` runs the watch in a
background thread on the newest camera frame and publishes the result: the outline, its box, and
how sure it is. The workbench shows the outline in the preview and starts a recording from it,
instead of looking for the paper at the key press, when hands and pen usually cover part of it.
"""

import time
from threading import Event, Lock, Thread

import numpy as np


class SheetCalibration:
    def __init__(self, interval=0.25):
        self.interval = interval
        self._latest = None  # (frame, hand sample, time) of the newest camera frame
        self._lock, self._stop, self._thread = Lock(), Event(), None
        self._published = {"outline": None, "bounds": None, "confidence": 0.0, "state": "none", "generation": 0}
        self.error = None
        self.cost = 0.0  # Seconds the last look took, for the debug record.

    def start(self):
        if self._thread is None or not self._thread.is_alive():
            self._stop.clear()
            self._thread = Thread(target=self._run, daemon=True)
            self._thread.start()

    def stop(self):
        self._stop.set()
        with self._lock:
            self._latest = None
            self._published = {"outline": None, "bounds": None, "confidence": 0.0, "state": "none", "generation": 0}

    def feed(self, frame, hands, now):
        """Hand over the newest frame. Costs nothing: the thread picks it up when it is ready."""
        self._latest = frame, hands, now

    def _run(self):
        from paper_motion import SheetWatch

        watch, seen = SheetWatch(), None
        while not self._stop.wait(self.interval):
            latest = self._latest
            if latest is None or latest[0] is seen:
                continue
            frame, hands, now = latest
            seen = frame
            try:
                start = time.perf_counter()
                watch.locate(frame, hands, now)
                self.cost = time.perf_counter() - start
                outline = watch.outline()
                with self._lock:
                    self._published = {
                        "outline": None if outline is None else np.asarray(outline, np.float32).copy(),
                        "bounds": watch.bounds(), "confidence": float(watch.confidence), "state": watch.state,
                        "generation": watch.generation}
                self.error = None
            except Exception as error:  # A failed look must never stop the camera or the UI.
                self.error = f"{type(error).__name__}: {error}"
                watch = SheetWatch()

    def sheet(self, confidence=0.5):
        """(bounds, outline) of the calibrated sheet when it is held with at least ``confidence``, else None."""
        with self._lock:
            held = self._published
        if held["outline"] is None or held["bounds"] is None or held["confidence"] < confidence:
            return None
        return held["bounds"], held["outline"]

    def status(self):
        """(outline or None, confidence, state) for the preview and the debug record."""
        with self._lock:
            held = self._published
        return held["outline"], held["confidence"], held["state"]
