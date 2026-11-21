"""Control desktop scrolling, zoom, and mouse clicks with camera 0 gestures.

Run with ``uv run python pose_gesture_detector.py``. Show a prayer pose to
start control. While active, move one open palm vertically to scroll; pinch
with both hands and move them apart/together to zoom in/out. Wink right eye for right-click or left eye for left-click; both eyes closed does nothing. Show both open
palms facing the camera to stop. Keep shoulders, arms, wrists, and hands in
frame. Press Q or ESC to quit.
"""

import argparse
import time
from typing import Any, Dict, Optional

import cv2
import numpy as np

from camera_utils import open_camera
from hand_pose_capture import HAND_CONNECTIONS, _draw_chain, _ensure_model, extract_body_points

POSE_EDGES = ((11, 13), (13, 15), (12, 14), (14, 16), (11, 12))
FINGERS = {
    "index": (5, 6, 8),
    "middle": (9, 10, 12),
    "ring": (13, 14, 16),
    "pinky": (17, 18, 20),
}


def _xy(landmark: Any, width: int, height: int) -> np.ndarray:
    return np.array([landmark.x * width, landmark.y * height], dtype=np.float32)


def _distance(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.linalg.norm(a - b))


def _hand_metrics(landmarks: Any, width: int, height: int) -> Dict[str, Any]:
    points = [_xy(point, width, height) for point in landmarks]
    wrist_to_middle_mcp = max(_distance(points[0], points[9]), 1.0)
    palm_width = _distance(points[5], points[17])
    extended = {}
    for name, (mcp, pip, tip) in FINGERS.items():
        full_length = _distance(points[mcp], points[tip])
        first_segment = max(_distance(points[mcp], points[pip]), 1.0)
        extended[name] = full_length > first_segment * 1.45
    thumb_extended = _distance(points[4], points[5]) > palm_width * 0.45
    finger_spread = _distance(points[8], points[20]) / max(palm_width, 1.0)
    palm_facing_score = palm_width / wrist_to_middle_mcp
    pinch_ratio = _distance(points[4], points[8]) / max(palm_width, 1.0)
    return {
        "points": points,
        "length": wrist_to_middle_mcp,
        "palm_width": palm_width,
        "extended": extended,
        "thumb_extended": thumb_extended,
        "finger_spread": finger_spread,
        "palm_facing_score": palm_facing_score,
        "open": all(extended.values()) and thumb_extended and finger_spread > 1.15,
        "facing_camera": palm_facing_score > 0.55,
        "pinching": pinch_ratio < 0.30,
        "pinch_center": (points[4] + points[8]) / 2.0,
    }


def detect_prayer_pose(results: Any, frame_size: tuple) -> bool:
    """True when both hands are close together with fingers pointing upward."""
    width, height = frame_size
    left = results.left_hand_landmarks
    right = results.right_hand_landmarks
    if not left or not right:
        return False
    left_metrics = _hand_metrics(left, width, height)
    right_metrics = _hand_metrics(right, width, height)
    avg_length = (left_metrics["length"] + right_metrics["length"]) / 2.0
    for metrics in (left_metrics, right_metrics):
        pts = metrics["points"]
        if not all(metrics["extended"].values()):
            return False
        if pts[8][1] > pts[5][1] - 0.12 * metrics["length"]:
            return False
    corresponding_tip_gaps = [
        _distance(left_metrics["points"][idx], right_metrics["points"][idx])
        for idx in (8, 12, 16, 20)
    ]
    palm_gap = _distance(left_metrics["points"][9], right_metrics["points"][9])
    return (float(np.median(corresponding_tip_gaps)) < 0.75 * avg_length
            and palm_gap < 1.25 * avg_length)


def detect_stop_pose(results: Any, frame_size: tuple) -> bool:
    """True when both hands are open and their palms face the camera."""
    width, height = frame_size
    left = results.left_hand_landmarks
    right = results.right_hand_landmarks
    if not left or not right:
        return False
    hands = (_hand_metrics(left, width, height), _hand_metrics(right, width, height))
    return all(hand["open"] and hand["facing_camera"] for hand in hands)


class PoseStateMachine:
    """Debounce pose events and alternate between waiting, active, and stopped."""

    def __init__(self, hold_frames: int = 6):
        self.state = "waiting"
        self.hold_frames = hold_frames
        self.consecutive = 0

    def update(self, prayer: bool, stop: bool) -> Optional[str]:
        target = prayer if self.state in ("waiting", "stopped") else stop
        self.consecutive = self.consecutive + 1 if target else 0
        if self.consecutive < self.hold_frames:
            return None
        self.consecutive = 0
        if self.state in ("waiting", "stopped"):
            self.state = "active"
            return "START"
        self.state = "stopped"
        return "STOP"


class DesktopGestureControls:
    """Translate active hand motion to scroll and Ctrl+wheel zoom actions."""

    def __init__(self, pyautogui_module: Any):
        self.gui = pyautogui_module
        self.scroll_side = None
        self.scroll_y = None
        self.scroll_remainder = 0.0
        self.zoom_distance = None
        self.last_zoom_time = 0.0

    def reset(self) -> None:
        self.scroll_side = None
        self.scroll_y = None
        self.scroll_remainder = 0.0
        self.zoom_distance = None

    def update(self, results: Any, frame_size: tuple, active: bool) -> str:
        if not active:
            self.reset()
            return ""
        width, height = frame_size
        raw_hands = {
            side: getattr(results, f"{side}_hand_landmarks")
            for side in ("left", "right")
        }
        metrics = {
            side: _hand_metrics(hand, width, height)
            for side, hand in raw_hands.items() if hand
        }

        # Both hands pinch together: increasing separation zooms in, decreasing
        # separation zooms out. The baseline is re-anchored after each zoom step.
        if all(side in metrics and metrics[side]["pinching"] for side in ("left", "right")):
            left_center = metrics["left"]["pinch_center"]
            right_center = metrics["right"]["pinch_center"]
            distance = _distance(left_center, right_center)
            self.scroll_side = None
            self.scroll_y = None
            self.scroll_remainder = 0.0
            if self.zoom_distance is None:
                self.zoom_distance = distance
                return "ZOOM: pinch both hands and move apart/together"
            change = distance - self.zoom_distance
            if abs(change) >= max(24.0, width * 0.035):
                if time.monotonic() - self.last_zoom_time >= 0.25:
                    direction = 1 if change > 0 else -1
                    self.gui.keyDown("ctrl")
                    try:
                        self.gui.scroll(direction)
                    finally:
                        self.gui.keyUp("ctrl")
                    self.zoom_distance = distance
                    self.last_zoom_time = time.monotonic()
                    return "ZOOM IN" if direction > 0 else "ZOOM OUT"
            return "ZOOM: move pinched hands apart/together"
        self.zoom_distance = None

        # Exactly one open, front-facing palm controls vertical page scrolling.
        open_hands = [
            side for side, hand in metrics.items()
            if hand["open"] and hand["facing_camera"]
        ]
        if len(open_hands) == 1:
            side = open_hands[0]
            y = float(metrics[side]["points"][9][1])
            if self.scroll_side != side or self.scroll_y is None:
                self.scroll_side = side
                self.scroll_y = y
                self.scroll_remainder = 0.0
                return "SCROLL: move one open hand up/down"
            self.scroll_remainder += y - self.scroll_y
            self.scroll_y = y
            steps = int(self.scroll_remainder / 28.0)
            if steps:
                # PyAutoGUI uses positive wheel units for scrolling upward.
                self.gui.scroll(-steps)
                self.scroll_remainder -= steps * 28.0
                return "SCROLL DOWN" if steps > 0 else "SCROLL UP"
            return "SCROLL: move one open hand up/down"

        self.scroll_side = None
        self.scroll_y = None
        self.scroll_remainder = 0.0
        return ""



# MediaPipe Face Mesh eye landmarks: (outer-left, upper-left, upper-right,
# outer-right, lower-right, lower-left), expressed in image coordinates.
EYE_LANDMARKS = {
    "left": (362, 385, 387, 263, 373, 380),
    "right": (33, 160, 158, 133, 153, 144),
}
EYE_CLOSED_EAR = 0.20


def _eye_aspect_ratio(face_landmarks: Any, indices: tuple, width: int, height: int) -> float:
    points = [_xy(face_landmarks[i], width, height) for i in indices]
    horizontal = max(_distance(points[0], points[3]), 1.0)
    vertical = _distance(points[1], points[5]) + _distance(points[2], points[4])
    return vertical / (2.0 * horizontal)


class EyeClickController:
    """Convert a single-eye wink to a click at the current mouse position."""

    def __init__(self, pyautogui_module: Any, hold_frames: int = 3):
        self.gui = pyautogui_module
        self.hold_frames = hold_frames
        self.armed = True
        self.side = None
        self.frames = 0
        self.both_closed = False

    def reset(self) -> None:
        self.armed = True
        self.side = None
        self.frames = 0
        self.both_closed = False

    def update(self, face_landmarks: Any, frame_size: tuple, active: bool) -> str:
        if not active:
            self.reset()
            return ""
        if not face_landmarks or len(face_landmarks) <= max(max(v) for v in EYE_LANDMARKS.values()):
            self.side = None
            self.frames = 0
            return ""

        width, height = frame_size
        left_ear = _eye_aspect_ratio(face_landmarks, EYE_LANDMARKS["left"], width, height)
        right_ear = _eye_aspect_ratio(face_landmarks, EYE_LANDMARKS["right"], width, height)
        left_closed = left_ear < EYE_CLOSED_EAR
        right_closed = right_ear < EYE_CLOSED_EAR

        if left_closed and right_closed:
            first_frame = not self.both_closed
            self.both_closed = True
            self.armed = False
            self.side = None
            self.frames = 0
            return "BOTH EYES CLOSED - ignored" if first_frame else ""

        if not left_closed and not right_closed:
            self.both_closed = False
            self.armed = True
            self.side = None
            self.frames = 0
            return ""

        # After a double blink, wait until both eyes reopen before re-arming.
        if not self.armed:
            return ""
        self.both_closed = False
        side = "left" if left_closed else "right"
        self.frames = self.frames + 1 if self.side == side else 1
        self.side = side
        if self.frames < self.hold_frames:
            return ""

        button = "left" if side == "left" else "right"
        self.gui.click(button=button)
        self.armed = False
        self.frames = 0
        label = "LEFT CLICK (left eye wink)" if side == "left" else "RIGHT CLICK (right eye wink)"
        return label

def run(camera_index: int = 0, confidence: float = 0.55) -> None:
    try:
        import mediapipe as mp
        import pyautogui
    except ImportError as exc:
        raise RuntimeError("Required package missing. Run: uv add mediapipe pyautogui") from exc

    model_path = _ensure_model()
    vision = mp.tasks.vision
    options = vision.HolisticLandmarkerOptions(
        base_options=mp.tasks.BaseOptions(model_asset_path=str(model_path)),
        running_mode=vision.RunningMode.VIDEO,
        min_pose_detection_confidence=confidence,
        min_hand_landmarks_confidence=confidence,
    )
    landmarker = vision.HolisticLandmarker.create_from_options(options)
    capture = None
    machine = PoseStateMachine()
    controls = DesktopGestureControls(pyautogui)
    eye_clicks = EyeClickController(pyautogui)
    try:
        capture, info = open_camera(camera_index, fourcc=None)
        print(f"Camera {camera_index}: {info['width']}x{info['height']} [{info['backend']}]")
        print("Start: prayer. Scroll: one open palm up/down. Zoom: pinch both hands apart/together.")
        print("Click: wink right eye for right-click, left eye for left-click; both closed is ignored.")
        print("Stop: both open palms facing camera. Press Q or ESC to quit.")
        while True:
            ok, frame = capture.read()
            if not ok or frame is None:
                print("Camera stopped returning frames.")
                break
            # Match the mirrored preview and coordinates in hand_pose_capture.py.
            frame = cv2.flip(frame, 1)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            result = landmarker.detect_for_video(image, time.monotonic_ns() // 1_000_000)
            size = (frame.shape[1], frame.shape[0])
            prayer = detect_prayer_pose(result, size)
            stop = detect_stop_pose(result, size)
            event = machine.update(prayer, stop)
            if event:
                print(f"POSE_EVENT={event}")
            active = machine.state == "active" and event != "STOP"
            action = controls.update(result, size, active)
            eye_action = eye_clicks.update(result.face_landmarks, size, active)
            if eye_action:
                print(f"EYE_EVENT={eye_action}")
                action = eye_action

            if result.pose_landmarks:
                _draw_chain(frame, result.pose_landmarks, POSE_EDGES, (0, 220, 0))
            for side in ("left", "right"):
                hand = getattr(result, f"{side}_hand_landmarks")
                if hand:
                    _draw_chain(frame, hand, HAND_CONNECTIONS, (0, 180, 255))

            points = extract_body_points(result, size)
            left_detected = points["hands"]["left"] is not None
            right_detected = points["hands"]["right"] is not None
            status = (f"Hands: L={left_detected} R={right_detected} | "
                      f"arm joints: {len(points['arms'])} | state: {machine.state}")
            cv2.putText(frame, status, (12, 30), cv2.FONT_HERSHEY_SIMPLEX,
                        0.52, (0, 255, 0), 2)
            if result.face_landmarks and len(result.face_landmarks) > 387:
                left_ear = _eye_aspect_ratio(result.face_landmarks, EYE_LANDMARKS["left"],
                                             size[0], size[1])
                right_ear = _eye_aspect_ratio(result.face_landmarks, EYE_LANDMARKS["right"],
                                              size[0], size[1])
                eye_status = (f"Eyes: L={left_ear:.2f} R={right_ear:.2f} "
                              f"(closed < {EYE_CLOSED_EAR:.2f})")
                cv2.putText(frame, eye_status, (12, 58), cv2.FONT_HERSHEY_SIMPLEX,
                            0.52, (0, 255, 0), 2)
            if action:
                cv2.putText(frame, action, (12, 84), cv2.FONT_HERSHEY_SIMPLEX,
                            0.52, (0, 255, 0), 2)
            cv2.imshow("Pose gesture controls - camera 0", frame)
            if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
                break
    finally:
        controls.reset()
        eye_clicks.reset()
        landmarker.close()
        if capture is not None:
            capture.release()
        cv2.destroyAllWindows()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--camera", type=int, default=0, help="camera index (default: 0)")
    parser.add_argument("--confidence", type=float, default=0.55,
                        help="MediaPipe detection confidence")
    args = parser.parse_args()
    run(args.camera, args.confidence)


if __name__ == "__main__":
    main()



