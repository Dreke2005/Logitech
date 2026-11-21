"""Optional, calibrated gaze pointing and one-eye blink clicks from RGB video."""

import math
import shutil
import time
import urllib.request
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from tempfile import NamedTemporaryFile

import cv2
import numpy as np

from gaze_pose import (
    MAX_ROLL,
    MAX_TURN,
    MIN_TURN,
    POSE_DIRECTIONS,
    fit_head_compensation,
    orientation_from_matrix,
)

MODEL_URL = "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task"
DEFAULT_MODEL = Path(__file__).resolve().parent / "models" / "face_landmarker.task"
TARGETS = (
    (0.50, 0.50),
    # Sample the perimeter for reach and the interior for interpolation.
    (0.06, 0.50), (0.06, 0.06), (0.50, 0.06), (0.94, 0.06),
    (0.94, 0.50), (0.94, 0.94), (0.50, 0.94), (0.06, 0.94),
    (0.27, 0.50), (0.27, 0.27), (0.50, 0.27), (0.73, 0.27),
    (0.73, 0.50), (0.73, 0.73), (0.50, 0.73), (0.27, 0.73),
)
TARGET_SETTLE_MS = 300
TARGET_DWELL_MS = 1850
POSE_DWELL_MS = 2350
POSE_SETTLE_MS = 500
POSE_WAIT_MS = 4800
SAMPLE_WINDOW_MS = 1800
MIN_FIXATION_MS = 1100
MIN_SAMPLE_COUNT = 8
MAX_SAMPLE_CREDIT_MS = 100
MAX_SAMPLE_AGE_MS = 300
MAX_TARGET_ERROR = 0.14
MAX_FRAME_GAP_MS = 200
CURSOR_WINDOW_MS = 360
MIN_WINK_MS = 55
MIN_WINK_FRAMES = 2
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
    """Keep gaze, face position and orientation, and eyelid scores."""
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
    matrices = getattr(result, "facial_transformation_matrixes", ())
    orientation = orientation_from_matrix(matrices[0]) if len(matrices) == 1 else None
    return {"timestamp_ms": timestamp_ms, "gaze": gaze, "head": head, "orientation": orientation,
            "blink": blink, "squint": squint}


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
            output_facial_transformation_matrixes=True,
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


def _gaze_features(normalized, quadratic):
    x, y = np.moveaxis(normalized, -1, 0)
    features = (x, y, np.ones_like(x))
    if quadratic:
        features += (x * x, y * y, x * y)
    return np.stack(features, axis=-1)


@dataclass(frozen=True)
class _GazeMap:
    center: np.ndarray
    scale: np.ndarray
    limits: np.ndarray
    coefficients: np.ndarray

    @property
    def quadratic(self):
        return len(self.coefficients) == 6

    def predict(self, gaze):
        # Keep the map local to the sampled area, with enough margin to reach
        # the ends of the display. Unbounded extrapolation can reverse a move.
        normalized = np.clip((np.asarray(gaze) - self.center) / self.scale, *self.limits)
        return _gaze_features(normalized, self.quadratic) @ self.coefficients


def _gaze_coefficients(features, targets):
    if features.shape[1] == 6:
        # Penalize curvature in normalized coordinates. Allow the scale,
        # orientation and offset of the affine part to fit freely.
        penalty = np.column_stack((np.zeros((3, 3)), math.sqrt(0.18) * np.eye(3)))
    else:
        penalty = np.empty((0, features.shape[1]))
    weights = np.ones(len(targets))
    for _ in range(5):
        roots = np.sqrt(weights[:, None])
        fit = np.linalg.lstsq(
            np.vstack((features * roots, penalty)),
            np.vstack((targets * roots, np.zeros((len(penalty), 2)))),
            rcond=None,
        )[0]
        # A missed dot should not distort all the other regions of the screen.
        error = np.linalg.norm(features @ fit - targets, axis=1)
        weights = 0.055 / np.maximum(error, 0.055)
    return fit


def _held_out_errors(features, targets):
    errors = []
    for i in range(len(targets)):
        other = np.arange(len(targets)) != i
        fit = _gaze_coefficients(features[other], targets[other])
        errors.append(np.linalg.norm(features[i] @ fit - targets[i]))
    return np.array(errors)


def _validation_score(errors):
    # Keep a single wrong dot from deciding whether a stable curve generalizes.
    return float(np.sqrt(np.mean(np.square(np.minimum(errors, MAX_TARGET_ERROR)))))


def _has_stable_gains(curved, affine, limits):
    """Reject folds and excessive motion gain around the sampled iris range."""
    inverse_affine = np.linalg.pinv(affine[:2])
    for x in np.linspace(*limits[:, 0], 3):
        for y in np.linspace(*limits[:, 1], 3):
            slopes = np.array((curved[0] + 2 * x * curved[3] + y * curved[5],
                               curved[1] + 2 * y * curved[4] + x * curved[5]))
            relative = inverse_affine @ slopes
            singular_values = np.linalg.svd(relative, compute_uv=False)
            if np.linalg.det(relative) <= 0 or singular_values[-1] < 0.32 or singular_values[0] > 1.75:
                return False
    return True


def _fit_calibration(samples):
    gaze = np.array([point[:2] for point in samples])
    center = gaze.mean(axis=0)
    travel = np.ptp(gaze, axis=0)
    # Vertical iris travel can be smaller than horizontal travel. Check for
    # independent movement after accounting for the different scales.
    if np.min(travel) < 0.012:
        return None
    scale = travel / 2
    normalized = (gaze - center) / scale
    if np.linalg.svd(normalized, compute_uv=False)[-1] < 0.40:
        return None
    limits = np.array((normalized.min(axis=0) - 0.25, normalized.max(axis=0) + 0.25))
    targets = np.array(TARGETS)
    affine = _gaze_features(normalized, False)
    coefficients = _gaze_coefficients(affine, targets)
    errors = _held_out_errors(affine, targets)
    quadratic = _gaze_features(normalized, True)
    strength = np.linalg.svd(quadratic, compute_uv=False)
    if strength[-1] >= 0.075 * strength[0]:
        curved = _gaze_coefficients(quadratic, targets)
        curve_errors = _held_out_errors(quadratic, targets)
        affine_error = _validation_score(errors)
        curve_error = _validation_score(curve_errors)
        if (
            curve_error <= min(affine_error * 0.85, affine_error - 0.008)
            and _has_stable_gains(curved, coefficients, limits)
        ):
            coefficients = curved
            errors = curve_errors
    fit = _GazeMap(center, scale, limits, coefficients)
    head = np.median(np.array(samples)[:, 2:], axis=0)
    if head[2] < 0.025:
        return None
    return fit, head, errors


def _eye_action(blink, squint, held_side):
    """Classify both-open, bilateral/uncertain, or unilateral blink phases."""
    for side in (0, 1):
        other = 1 - side
        closed = blink[side] >= (0.45 if side == held_side else 0.66)
        narrowed = squint[side] >= (0.48 if side == held_side else 0.62)
        asymmetric = squint[side] - squint[other] >= 0.24
        if (closed or (narrowed and asymmetric)) and blink[other] <= 0.35 and squint[other] <= 0.45:
            return side
    if max(blink) <= 0.33 and max(squint) <= 0.45:
        return "open"
    if held_side is not None:
        other = 1 - held_side
        if blink[other] <= 0.33 and squint[other] <= 0.45:
            return "transition"
    elif max(blink) <= 0.33 and abs(squint[0] - squint[1]) <= 0.18:
        return "open"
    return "blocked"


class GazeSmoother:
    """Reject brief spikes and hold still at fixation without slowing large moves."""

    def __init__(self, deadband):
        self.deadband = deadband
        self.cursor = self._sent = None
        self.discard_history()

    def discard_history(self):
        # Keep the last position through a blink, a tracking gap, or head motion.
        self._recent = deque(maxlen=5)
        self._last_ms = None
        self._moving = False

    def update(self, target, timestamp):
        last = self._last_ms
        if last is not None and not 0 < timestamp - last <= MAX_FRAME_GAP_MS:
            self.discard_history()
            last = None
        self._last_ms = timestamp
        self._recent.append((timestamp, target))
        while timestamp - self._recent[0][0] > CURSOR_WINDOW_MS:
            self._recent.popleft()
        if len(self._recent) < 3:
            return None
        target = np.median([position for _, position in self._recent], axis=0)
        if self.cursor is None:
            self.cursor = target
        else:
            change = target - self.cursor
            distance = np.linalg.norm(change / self.deadband)
            if distance <= (0.36 if self._moving else 1.0):
                self._moving = False
                return None
            self._moving = True
            # Faster response to a saccade, stronger smoothing for small corrections.
            tau = float(np.interp(distance, (1.0, 8.0), (0.23, 0.075)))
            interval = min(0.10, (timestamp - last) / 1000)
            self.cursor += -math.expm1(-interval / tau) * change
        if self._sent is None or np.max(np.abs(self.cursor - self._sent)) >= 0.0008:
            self._sent = self.cursor.copy()
            return self.cursor
        return None


class EyePointer:
    """Calibration and UI state, driven exclusively by the injected control output."""

    def __init__(self, output):
        self.output = output
        self.reset()

    def reset(self):
        self._fit = self._head_reference = None
        self._head_assist = None
        self._deadband = np.array((0.010, 0.012))
        self._setup_hint = "Eye: press C to calibrate the gaze pointer."
        self.pause()

    @property
    def calibrating(self):
        return self._samples is not None

    @property
    def ready(self):
        return self._fit is not None and not self.calibrating

    @property
    def target(self):
        if self._pose_step is not None:
            return TARGETS[0]
        return TARGETS[self._pending[0]] if self.calibrating else None

    @property
    def target_progress(self):
        if self._target_started is None or self._last_frame is None:
            return 0.0
        elapsed = self._last_frame - self._target_started
        dwell = TARGET_DWELL_MS if self._pose_step is None else POSE_DWELL_MS
        return min(1.0, elapsed / dwell, self._fixation_ms() / MIN_FIXATION_MS,
                   len(self._recent) / MIN_SAMPLE_COUNT)

    def pause(self):
        """Cancel incomplete actions while retaining the last successful calibration."""
        self._recent = deque()
        self._samples = None
        self._pending = deque()
        self._sample_noise = []
        self._sample_orientations = []
        self._pose_step = self._pose_started = self._base_calibration = None
        self._pose_samples = []
        self._rechecking = False
        self._target_started = None
        self._last_frame = self._received_at = None
        self._smoother = GazeSmoother(self._deadband)
        self._cooldown_until = 0
        self._cancel_wink()

    def _cancel_wink(self):
        self._wink = self._open_since = None

    def begin_calibration(self):
        """Start collecting gaze for the preview's visible calibration targets."""
        self.pause()
        self._samples = [None] * len(TARGETS)
        self._sample_noise = [None] * len(TARGETS)
        self._sample_orientations = [None] * len(TARGETS)
        self._pending.extend(range(len(TARGETS)))
        return self._target_hint()

    def _target_hint(self):
        if self._pose_step is not None:
            direction, _ = POSE_DIRECTIONS[self._pose_step]
            return f"Head assist {self._pose_step + 1}/{len(POSE_DIRECTIONS)}. {direction.capitalize()}. Keep looking at the dot."
        position = f"{self._pending[0] + 1}/{len(TARGETS)}"
        if self._rechecking:
            return f"Eye: recheck dot {position}."
        return f"Eye {position}: look at the dot."

    def _fixation_ms(self):
        stamps = [timestamp for timestamp, _ in self._recent]
        # A blink or missing detection shouldn't be counted as fixation time,
        # even when usable frames happen to bracket a long gap.
        return sum(min(MAX_SAMPLE_CREDIT_MS, b - a) for a, b in zip(stamps, stamps[1:]))

    def capture_target(self):
        if self._samples is None:
            return "Eye: press C to start calibration."
        # Refuse the sample if video has stalled between update and keyboard input.
        if (not self._recent or time.monotonic() - self._received_at > 0.35
                or self._last_frame - self._recent[-1][0] > MAX_SAMPLE_AGE_MS
                or self.target_progress < 1):
            return self._target_hint() + " Keep both eyes visible."
        data = np.array([value for _, value in self._recent])
        if max(np.ptp(np.quantile(data[:, :2], (0.1, 0.9), axis=0), axis=0)) > 0.065:
            return self._target_hint() + " Hold your gaze steady."
        head_span = np.quantile(data[:, 2:5], 0.9, axis=0) - np.quantile(data[:, 2:5], 0.1, axis=0)
        median = np.median(data[:, :5], axis=0)
        if np.linalg.norm(head_span[:2]) > 0.32 * median[4] or head_span[2] > 0.22 * median[4]:
            return self._target_hint() + " Keep your head still."
        rotations = data[np.isfinite(data[:, 5:]).all(axis=1), 5:]
        orientation = None
        if len(rotations) >= max(MIN_SAMPLE_COUNT, len(data) / 2):
            orientation = np.median(rotations, axis=0)
            spans = np.ptp(np.quantile(rotations, (0.1, 0.9), axis=0), axis=0)
            if np.any(spans > 0.075):
                return self._target_hint() + " Hold your head still."
        if self._pose_step is not None:
            if orientation is None:
                return self._target_hint() + " Keep your face visible."
            self._pose_samples.append((orientation, median[:2]))
            return self._next_pose()
        index = self._pending.popleft()
        self._samples[index] = median
        self._sample_noise[index] = data[:, (0, 1, 5, 6, 7)]
        self._sample_orientations[index] = orientation
        self._recent.clear()
        self._target_started = None
        if self._pending:
            return self._target_hint()
        return self._complete_calibration()

    def _complete_calibration(self):
        result = _fit_calibration(self._samples)
        if result is None:
            return self._reject_calibration()
        fit, head, errors = result
        retry = np.flatnonzero(errors > MAX_TARGET_ERROR)
        if np.median(errors) > 0.085 or len(retry) > 3 or (self._rechecking and len(retry)):
            return self._reject_calibration()
        if len(retry):
            self._rechecking = True
            self._pending.extend(int(i) for i in sorted(retry, key=lambda i: -errors[i]))
            return self._target_hint()
        self._base_calibration = fit, head
        if self._sample_orientations[0] is not None:
            self._pose_step = 0
            return self._target_hint()
        return self._finish_calibration()

    def _next_pose(self):
        self._recent.clear()
        self._target_started = self._pose_started = None
        self._pose_step += 1
        if self._pose_step == len(POSE_DIRECTIONS):
            return self._finish_calibration()
        return self._target_hint()

    def _finish_calibration(self):
        fit, head = self._base_calibration
        samples = self._samples
        sample_noise = self._sample_noise
        orientations = self._sample_orientations
        orientation = orientations[0]
        assistance = None
        if orientation is not None:
            poses = [sample for sample in self._pose_samples if sample is not None]
            assistance = fit_head_compensation(orientation, samples[0][:2], poses, fit)
        self.pause()
        self._fit, self._head_reference = fit, head
        self._head_assist = assistance
        # Measure noise where each point was sampled. The same raw iris
        # variation can map to different screen movement near the corners.
        # Include any head-pose noise introduced by the optional correction.
        deviations = []
        for median, rotation, observations in zip(samples, orientations, sample_noise):
            anchor = median[:2]
            iris = observations[:, :2]
            if assistance is not None and rotation is not None:
                pose_data = observations[:, 2:]
                valid = np.isfinite(pose_data).all(axis=1)
                iris = assistance.correct(iris[valid], pose_data[valid])
                anchor = assistance.correct(anchor, rotation)
            deviations.append(fit.predict(iris) - fit.predict(anchor))
        noise = np.quantile(np.abs(np.concatenate(deviations)), 0.75, axis=0)
        self._deadband = np.clip(2.0 * noise, (0.010, 0.012), (0.029, 0.034))
        self._smoother = GazeSmoother(self._deadband)
        return self._ready_hint()

    def _ready_hint(self):
        if self._head_assist is not None:
            return "Eye ready with head assist: look to point. Blink one eye to click."
        return "Eye ready: look to point. Blink one eye to click."

    def _reject_calibration(self):
        self.pause()
        self._setup_hint = "Eye: calibration unclear. Follow each dot with your eyes. Press C to retry."
        return self._setup_hint

    def update(self, sample):
        timestamp = sample["timestamp_ms"]
        if self._last_frame is not None and not 0 < timestamp - self._last_frame <= MAX_FRAME_GAP_MS:
            self._cancel_wink()
            if self.calibrating:
                self._recent.clear()
                self._target_started = None
        self._last_frame = timestamp
        self._received_at = time.monotonic()
        if self.calibrating:
            return self._calibration_frame(sample, timestamp)
        if sample["gaze"] is None:
            self._cancel_wink()
            self._smoother.discard_history()
            return "Eye: show one face with both eyes visible."

        held_side = self._wink[0] if self._wink is not None else None
        action = _eye_action(sample["blink"], sample["squint"], held_side)
        if self._fit is None:
            return self._setup_hint
        orientation = sample.get("orientation")
        gaze = sample["gaze"]
        if self._head_moved(sample["head"]):
            self._cancel_wink()
            self._smoother.discard_history()
            return "Eye: return to your calibration position or press C."
        if self._head_assist is not None:
            if orientation is None or not self._head_assist.within_range(orientation):
                self._cancel_wink()
                self._smoother.discard_history()
                return "Eye: face the screen within your calibration angles."
            gaze = self._head_assist.correct(gaze, orientation)
        return self._point_or_click(gaze, action, timestamp)

    def _calibration_frame(self, sample, timestamp):
        pose = self._pose_step is not None
        if pose:
            if self._pose_started is None:
                self._pose_started = timestamp
            elif timestamp - self._pose_started >= POSE_WAIT_MS:
                # Head assistance is optional. If a turn can't be tracked,
                # finish through the eye-only calibration instead of getting stuck.
                self._pose_samples.append(None)
                return self._next_pose()
        if self._target_started is None:
            self._target_started = timestamp
        elapsed = timestamp - self._target_started
        while self._recent and timestamp - self._recent[0][0] > SAMPLE_WINDOW_MS:
            self._recent.popleft()
        if sample["gaze"] is None:
            if self._recent and timestamp - self._recent[-1][0] > MAX_SAMPLE_AGE_MS:
                self._recent.clear()
                self._target_started = timestamp
            return self._target_hint() + " Show one face with both eyes visible."
        orientation = sample.get("orientation")
        if pose:
            issue = self._guided_pose_issue(sample)
            if issue:
                self._recent.clear()
                self._target_started = None
                return self._target_hint() + " " + issue
        elif orientation is not None and self._sample_orientations[0] is not None:
            offset = np.abs(orientation - self._sample_orientations[0])
            if np.any(offset > (0.15, 0.15, 0.22)):
                self._recent.clear()
                self._target_started = None
                return self._target_hint() + " Keep your head centered and move only your eyes."
        # A blink skips the frame instead of dropping the whole target.
        settle = POSE_SETTLE_MS if pose else TARGET_SETTLE_MS
        if elapsed >= settle and max(sample["blink"]) < 0.50:
            angles = orientation if orientation is not None else (math.nan,) * 3
            self._recent.append((timestamp, (*sample["gaze"], *sample["head"], *angles)))
        dwell = POSE_DWELL_MS if pose else TARGET_DWELL_MS
        return self.capture_target() if elapsed >= dwell else self._target_hint()

    def _guided_pose_issue(self, sample):
        orientation = sample.get("orientation")
        if orientation is None:
            return "Keep your face visible."
        if self._head_moved(sample["head"], self._base_calibration[1]):
            return "Rotate your head without moving closer or farther away."
        _, axis = POSE_DIRECTIONS[self._pose_step]
        reference = self._sample_orientations[0]
        offset = orientation - reference
        if np.any(np.abs(offset[:2]) > MAX_TURN) or abs(offset[2]) > MAX_ROLL:
            return "Make a smaller turn."
        if abs(offset[axis]) < MIN_TURN:
            return "Turn a little farther."
        if abs(offset[1 - axis]) > abs(offset[axis]):
            return "Follow the direction shown."
        if self._pose_step % 2 and self._pose_samples[-1] is not None:
            previous, _ = self._pose_samples[-1]
            if (previous[axis] - reference[axis]) * offset[axis] >= 0:
                return "Move in the opposite direction."
        return None

    def _head_moved(self, head, reference=None):
        ref_x, ref_y, ref_width = self._head_reference if reference is None else reference
        x, y, width = head
        return math.hypot(x - ref_x, y - ref_y) > 0.60 * ref_width or not 0.70 <= width / ref_width <= 1.42

    def _point_or_click(self, gaze, action, now):
        if action == "blocked":
            self._cancel_wink()
            self._smoother.discard_history()
            return "Eye: reopen both eyes to resume."
        if action == "transition":
            # Eyelid scores pass through a gray zone during reopening. Keep a
            # candidate only after consecutive, clearly closed frames.
            if self._wink[2] < MIN_WINK_FRAMES:
                self._cancel_wink()
            return "Eye: reopen to click."
        if isinstance(action, int):
            self._smoother.discard_history()
            if self._wink is not None:
                side, started, closed_frames = self._wink
                if action != side or now - started > MAX_WINK_MS:
                    self._cancel_wink()
                else:
                    self._wink = (side, started, closed_frames + 1)
            elif (self._smoother.cursor is not None and self._open_since is not None
                  and now - self._open_since >= MIN_OPEN_MS and now >= self._cooldown_until):
                self._wink = (action, now, 1)
            self._open_since = None
            return "Eye: reopen to click."

        wink = self._wink
        self._wink = None
        if self._open_since is None:
            self._open_since = now
        if wink is not None and wink[2] >= MIN_WINK_FRAMES and MIN_WINK_MS <= now - wink[1] <= MAX_WINK_MS:
            # Preserve the last open-eye target while the eye was closed.
            self.output.click(button="left", count=1)
            self._cooldown_until = now + CLICK_COOLDOWN_MS
            return "Eye: clicked."

        target = np.clip(self._fit.predict(gaze), 0, 1)
        position = self._smoother.update(target, now)
        if position is not None:
            self.output.move_to(*position)
        return self._ready_hint()
