"""Optional, calibrated gaze pointing and held-wink clicking from RGB video."""

import math
import shutil
import time
import urllib.request
from collections import deque
from pathlib import Path
from tempfile import NamedTemporaryFile

import cv2
import numpy as np

MODEL_URL = "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task"
DEFAULT_MODEL = Path(__file__).resolve().parent / "models" / "face_landmarker.task"
TARGETS = ((0.5, 0.5), (0.11, 0.15), (0.89, 0.15), (0.89, 0.85), (0.11, 0.85))
MAX_FRAME_GAP_MS = 200
MIN_WINK_MS = 270
MAX_WINK_MS = 1200
MIN_OPEN_MS = 150
CLICK_COOLDOWN_MS = 460
SCORE_NAMES = ("eyeBlinkLeft", "eyeBlinkRight", "eyeSquintLeft", "eyeSquintRight")


def ensure_eye_model():
    """Download Google's versioned Face Landmarker bundle when first needed."""
    if DEFAULT_MODEL.is_file():
        return DEFAULT_MODEL
    DEFAULT_MODEL.parent.mkdir(parents=True, exist_ok=True)
    partial = None
    print("Downloading Google's face model (first use) ...", flush=True)
    try:
        with NamedTemporaryFile(mode="wb", suffix=".download", dir=DEFAULT_MODEL.parent, delete=False) as target:
            partial = Path(target.name)
            with urllib.request.urlopen(MODEL_URL, timeout=30) as source:
                shutil.copyfileobj(source, target)
        partial.replace(DEFAULT_MODEL)
    except (OSError, ValueError) as error:
        raise RuntimeError("Face model download failed. Check your connection or supply --eye-model PATH.") from error
    finally:
        if partial is not None:
            partial.unlink(missing_ok=True)
    return DEFAULT_MODEL


def _eye_coordinates(points, corner1, corner2, iris, aspect_scale):
    # MediaPipe normalized x/y have different scale factors; use square pixels.
    p, q, r = [np.array((points[i].x, points[i].y * aspect_scale)) for i in (corner1, corner2, iris)]
    axis = q - p
    length = np.linalg.norm(axis)
    if not np.isfinite(length) or length < 0.012:
        return None
    horizontal = axis / length
    vertical = np.array((-horizontal[1], horizontal[0]))
    offset = r - (p + q) / 2
    return np.array((offset @ horizontal, offset @ vertical)) / length


def extract_eye_sample(result, timestamp_ms, frame_size):
    """Keep only the calibrated gaze feature, face position, and eyelid scores."""
    lost = {"timestamp_ms": timestamp_ms, "gaze": None}
    if len(result.face_landmarks) != 1 or len(result.face_blendshapes) != 1:
        return lost
    points = result.face_landmarks[0]
    if len(points) < 478:
        return lost
    width, height = frame_size
    aspect_scale = height / width
    eye_a = _eye_coordinates(points, 33, 133, 468, aspect_scale)
    eye_b = _eye_coordinates(points, 362, 263, 473, aspect_scale)
    if eye_a is None or eye_b is None:
        return lost
    scores = {value.category_name: float(value.score) for value in result.face_blendshapes[0]}
    if any(name not in scores for name in SCORE_NAMES):
        return lost
    gaze = tuple((eye_a + eye_b) / 2)
    blink = tuple(scores[name] for name in SCORE_NAMES[:2])
    squint = tuple(scores[name] for name in SCORE_NAMES[2:])
    head = (points[1].x, points[1].y * aspect_scale, math.hypot(
        points[263].x - points[33].x, (points[263].y - points[33].y) * aspect_scale,
    ))
    if not all(math.isfinite(value) for value in (*gaze, *head, *blink, *squint)):
        return lost
    return {"timestamp_ms": timestamp_ms, "gaze": gaze, "head": head, "blink": blink, "squint": squint}


class EyeLandmarker:
    """Run MediaPipe Face Landmarker locally on RGB preview frames."""

    def __init__(self, model_path):
        import mediapipe as mp

        self.mp = mp
        self._last_ms = -1
        options = mp.tasks.vision.FaceLandmarkerOptions(
            base_options=mp.tasks.BaseOptions(model_asset_path=str(model_path)),
            running_mode=mp.tasks.vision.RunningMode.VIDEO,
            num_faces=2,  # Suspend pointing if someone else enters the frame.
            output_face_blendshapes=True,
        )
        self.detector = mp.tasks.vision.FaceLandmarker.create_from_options(options)

    def process(self, frame):
        timestamp = max(self._last_ms + 1, time.monotonic_ns() // 1_000_000)
        self._last_ms = timestamp
        image = self.mp.Image(image_format=self.mp.ImageFormat.SRGB,
                              data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        return extract_eye_sample(self.detector.detect_for_video(image, timestamp),
                                  timestamp, (frame.shape[1], frame.shape[0]))

    def close(self):
        self.detector.close()


def _fit_calibration(samples):
    gaze = np.array([point[:2] for point in samples])
    centered = gaze - gaze.mean(axis=0)
    if np.linalg.svd(centered, compute_uv=False)[-1] < 0.035:
        return None, None
    features = np.column_stack((gaze, np.ones(len(gaze))))
    fit = np.linalg.lstsq(features, np.array(TARGETS), rcond=None)[0]
    if np.max(np.linalg.norm(features @ fit - TARGETS, axis=1)) > 0.16:
        return None, None
    head = np.median(np.array(samples)[:, 2:], axis=0)
    if head[2] < 0.025:
        return None, None
    return fit, head


def _eye_action(blink, squint, held_side):
    """Return 'open', 'blocked', or the index of a unilateral wink."""
    for side in (0, 1):
        other = 1 - side
        closed = blink[side] >= (0.45 if side == held_side else 0.66)
        narrowed = squint[side] >= (0.48 if side == held_side else 0.62)
        asymmetric = squint[side] - squint[other] >= 0.24
        if (closed or (narrowed and asymmetric)) and blink[other] <= 0.35 and squint[other] <= 0.45:
            return side
    if max(blink) <= 0.33 and (max(squint) <= 0.45 or abs(squint[0] - squint[1]) <= 0.18):
        return "open"
    return "blocked"


class EyePointer:
    """Calibration and UI state, driven exclusively by the injected control output."""

    def __init__(self, output):
        self.output = output
        self.reset()

    def reset(self):
        self._fit = self._head_reference = None
        self.pause()

    def pause(self):
        """Cancel incomplete actions while retaining the last successful calibration."""
        self._recent = deque()
        self._samples = None
        self._last_frame = self._received_at = None
        self._cursor = None
        self._cooldown_until = 0
        self._cancel_wink()

    def _cancel_wink(self):
        self._wink = self._open_since = None

    def begin_calibration(self):
        """Move the real pointer to the first target, where the user should look."""
        self.pause()
        self._samples = []
        self.output.move_to(*TARGETS[0])
        return self._target_hint()

    def _target_hint(self):
        return f"Eye {len(self._samples) + 1}/{len(TARGETS)}: look at the cursor, then press Space."

    def capture_target(self):
        if self._samples is None:
            return "Eye: press C to start calibration."
        # Refuse the sample if video has stalled between update and keyboard input.
        if (not self._recent or time.monotonic() - self._received_at > 0.25
                or len(self._recent) < 5 or self._recent[-1][0] - self._recent[0][0] < 280):
            return "Eye: keep looking at the cursor, then press Space."
        data = np.array([value for _, value in self._recent])
        if max(np.ptp(np.quantile(data[:, :2], (0.1, 0.9), axis=0), axis=0)) > 0.065:
            return "Eye: keep your gaze steady, then press Space."
        self._samples.append(np.median(data, axis=0))
        self._recent.clear()
        if len(self._samples) < len(TARGETS):
            self.output.move_to(*TARGETS[len(self._samples)])
            return self._target_hint()
        samples = self._samples
        self.pause()
        fit, head = _fit_calibration(samples)
        if fit is None:
            return "Eye: calibration unclear. Press C to retry."
        self._fit, self._head_reference = fit, head
        return "Eye ready: hold one eye closed, then open it to click."

    def update(self, sample):
        timestamp = sample["timestamp_ms"]
        if self._last_frame is not None and not 0 < timestamp - self._last_frame <= MAX_FRAME_GAP_MS:
            self._cancel_wink()
            self._recent.clear()
        self._last_frame = timestamp
        self._received_at = time.monotonic()
        if sample["gaze"] is None:
            self._cancel_wink()
            self._recent.clear()
            return "Eye: show one face with both eyes visible."

        held_side = self._wink[0] if self._wink is not None else None
        action = _eye_action(sample["blink"], sample["squint"], held_side)
        if self._samples is not None:
            if action == "open":
                self._recent.append((timestamp, (*sample["gaze"], *sample["head"])))
                while self._recent and timestamp - self._recent[0][0] > 550:
                    self._recent.popleft()
            else:
                self._recent.clear()
            return self._target_hint()
        if self._fit is None:
            return "Eye: press C to calibrate the gaze pointer."
        if self._head_moved(sample["head"]):
            self._cancel_wink()
            self._cursor = None
            return "Eye: return to your calibration position or press C."
        return self._point_or_click(sample["gaze"], action, timestamp)

    def _head_moved(self, head):
        ref_x, ref_y, ref_width = self._head_reference
        x, y, width = head
        return math.hypot(x - ref_x, y - ref_y) > 0.60 * ref_width or not 0.70 <= width / ref_width <= 1.42

    def _point_or_click(self, gaze, action, now):
        if action == "blocked":
            self._cancel_wink()
            return "Eye: reopen both eyes to resume."
        if isinstance(action, int):
            if self._wink is not None:
                side, started = self._wink
                if action != side or now - started > MAX_WINK_MS:
                    self._cancel_wink()
            elif self._open_since is not None and now - self._open_since >= MIN_OPEN_MS and now >= self._cooldown_until:
                self._wink = (action, now)
            self._open_since = None
            return "Eye: hold the wink, then reopen to click."

        wink = self._wink
        self._wink = None
        if self._open_since is None:
            self._open_since = now
        if wink is not None and MIN_WINK_MS <= now - wink[1] <= MAX_WINK_MS:
            # Preserve the last open-eye target while the eye was closed.
            self.output.click(button="left", count=1)
            self._cooldown_until = now + CLICK_COOLDOWN_MS
            return "Eye: clicked."

        target = np.clip(np.array((*gaze, 1)) @ self._fit, 0, 1)
        previous = self._cursor if self._cursor is not None else target
        self._cursor = previous + 0.31 * (target - previous)
        self.output.move_to(*self._cursor)
        return "Eye ready: look to point. Hold and release a wink to click."
