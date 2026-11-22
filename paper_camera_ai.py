"""The LogiInsight camera's own AI (face detection on its NPU), read through the SDK.

The camera runs a face detector on every RGB frame and publishes one face box per result as
JSON over its USB network link (LogiWebcamClient, see docs/API.md sections 5, 7 and 8).
PaperBridge uses it for what the picture of the desk cannot tell by itself: where the writer
sits. The side of the frame on which the face appears is the writer's side, and handwriting
is upright seen from there, so a snapshot of writing can be turned the right way up even
when no pen is in view. The face box also marks a region that is never paper.

``CameraAI`` polls in a background thread, so the UI never waits for the network. It needs
the SSH user name and password from the kit card in LOGIWEBCAM_USER and LOGIWEBCAM_PASSWORD;
without them, or with the camera unplugged, it stays idle and everything else works as before.
"""

import os
import time
from threading import Event, Lock, Thread


def writer_turn(face, frame_size):
    """Clockwise turn (0, 90, 180, 270) that makes the page upright for a writer whose face box
    ``face`` = (x, y, w, h) is seen in a frame of ``frame_size`` = (width, height).

    A desk camera sees the face of someone sitting across the desk, facing it, wherever along that
    side they sit: their writing is upside down in the picture (turn 180). Only a face at the far
    left or right edge at mid height is someone sitting at the side of the desk (turn 270 or 90).
    A writer on the camera's own side is not in the picture at all."""
    x, y, w, h = face
    width, height = frame_size
    cx, cy = (x + w / 2) / width, (y + h / 2) / height
    if 0.35 <= cy <= 0.8 and cx <= 0.15:
        return 270
    if 0.35 <= cy <= 0.8 and cx >= 0.85:
        return 90
    return 180


class CameraAI:
    def __init__(self, interval=0.4):
        self.interval = interval
        self.face = None  # (x, y, w, h) in stream pixels
        self.confidence = 0.0
        self.seen = 0.0  # time.monotonic() of the last result with a face
        self.updated = 0.0  # time.monotonic() of the last result at all
        self.firmware = None
        self.error = None
        self._lock, self._stop, self._thread = Lock(), Event(), None

    @staticmethod
    def configured():
        """True when the SDK credentials are available. A program started before they were set (the
        keypad service starts the workbench) does not have them in its environment yet, so the user's
        environment in the registry is looked up as well; they are never stored anywhere else."""
        names = ("LOGIWEBCAM_USER", "LOGIWEBCAM_PASSWORD")
        if not all(os.environ.get(name) for name in names):
            try:
                import winreg

                with winreg.OpenKey(winreg.HKEY_CURRENT_USER, "Environment") as key:
                    for name in names:
                        if not os.environ.get(name):
                            os.environ[name] = str(winreg.QueryValueEx(key, name)[0])
            except (ImportError, OSError):
                pass
        return all(os.environ.get(name) for name in names)

    def start(self):
        """Begin polling. Returns False when the credentials are not set (nothing is started)."""
        if not self.configured():
            self.error = "LOGIWEBCAM_USER and LOGIWEBCAM_PASSWORD are not set."
            return False
        if self._thread is None or not self._thread.is_alive():
            self._stop.clear()
            self._thread = Thread(target=self._run, daemon=True)
            self._thread.start()
        return True

    def stop(self):
        self._stop.set()

    def _run(self):
        from logiwebcam import LogiWebcamClient

        delay = 1.0
        while not self._stop.is_set():
            try:
                with LogiWebcamClient() as client:
                    self.firmware = client.get_firmware_version()
                    # Only the face box is needed: the 478-point mesh costs NPU time and heat.
                    client.set_ai_config(enable_face_detection=True, enable_face_mesh=False)
                    self.error, delay = None, 1.0
                    while not self._stop.wait(self.interval):
                        data = client.get_latest_ai_data()
                        if not data or not data.get("plugin_loaded"):
                            continue  # Being rewritten, or the RGB stream is not open yet.
                        now = time.monotonic()
                        faces = data.get("faces") or ()
                        with self._lock:
                            self.updated = now
                            if faces and len(faces[0].get("bbox", ())) == 4:
                                self.face = tuple(float(v) for v in faces[0]["bbox"])
                                self.confidence = float(faces[0].get("confidence", 0.0))
                                self.seen = now
            except Exception as error:  # Unplugged, wrong password, link down: retry later, never crash.
                self.error = f"{type(error).__name__}: {error}"
            if self._stop.wait(delay):
                break
            delay = min(delay * 2, 30.0)

    def writer_rotation(self, frame_size, within=5.0, confidence=0.5):
        """The writer's side as a turn, from a face seen in the last ``within`` seconds, else None."""
        with self._lock:
            if self.face is None or time.monotonic() - self.seen > within or self.confidence < confidence:
                return None
            return writer_turn(self.face, frame_size)

    def face_box(self, within=1.0):
        """The face box (x, y, w, h) seen in the last ``within`` seconds, or None."""
        with self._lock:
            return self.face if self.face is not None and time.monotonic() - self.seen <= within else None

    def status(self):
        if self.error:
            return f"Camera AI unavailable ({self.error})"
        if not self.updated:
            return "Camera AI connecting..."
        return f"Camera AI running (firmware {str(self.firmware).split()[-3] if self.firmware else '?'})"
