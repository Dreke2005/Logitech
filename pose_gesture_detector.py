"""Repeated scroll/zoom modes and wink clicks using camera 0 hand gestures.

Run with ``uv run python pose_gesture_detector.py``. Prayer pose starts control.
Tilt one open hand to enter scroll_up/down; point the thumb left and curl the
middle, ring and pinky fingers to enter zoom_in/out with the index finger.
Each mode repeats its wheel action until both
hands leave the frame. A right/left wink clicks at the cursor; both eyes closed
does nothing. Press Q or ESC to quit.
"""

# Author: Victor Lancieux
# Date: Original date unknown; metadata recorded 2026-11-22.
# Purpose: Control zoom, scrolling, rotation, and wink clicks with camera 0 gestures.

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
        "facing_camera": palm_facing_score > 0.35,
        "pinching": pinch_ratio < 0.45,
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
    """Enter a gesture mode once, then return to active after hands leave view."""

    ACTION_STATES = {"zoom_in", "zoom_out", "scroll_up", "scroll_down",
                     "rotate"}

    def __init__(self, hold_frames: int = 6, gesture_frames: int = 3,
                 missing_frames: int = 6):
        self.state = "waiting"
        self.hold_frames = hold_frames
        self.gesture_frames = gesture_frames
        self.missing_frames = missing_frames
        self.consecutive = 0
        self.pending_action = None
        self.pending_action_frames = 0
        self.no_hands_frames = 0

    def update(self, prayer: bool, stop: bool, action: Optional[str],
               hands_present: bool) -> Optional[str]:
        if self.state in ("waiting", "stopped"):
            self.consecutive = self.consecutive + 1 if prayer else 0
            if self.consecutive >= self.hold_frames:
                self.state = "active"
                self.consecutive = 0
                return "START"
            return None

        if self.state in self.ACTION_STATES:
            self.consecutive = self.consecutive + 1 if stop else 0
            if self.consecutive >= self.hold_frames:
                self.state = "stopped"
                self.consecutive = 0
                self.no_hands_frames = 0
                return "STOP"
            self.no_hands_frames = 0 if hands_present else self.no_hands_frames + 1
            if self.no_hands_frames >= self.missing_frames:
                self.state = "active"
                self.no_hands_frames = 0
                self.pending_action = None
                self.pending_action_frames = 0
                return "RETURN_ACTIVE"
            return None

        if stop:
            self.consecutive += 1
            self.pending_action = None
            self.pending_action_frames = 0
            if self.consecutive >= self.hold_frames:
                self.state = "stopped"
                self.consecutive = 0
                return "STOP"
            return None
        self.consecutive = 0

        if action:
            if action == self.pending_action:
                self.pending_action_frames += 1
            else:
                self.pending_action = action
                self.pending_action_frames = 1
            if self.pending_action_frames >= self.gesture_frames:
                self.state = action
                self.pending_action = None
                self.pending_action_frames = 0
                self.no_hands_frames = 0
                return f"MODE:{action}"
        else:
            self.pending_action = None
            self.pending_action_frames = 0
        return None


class DesktopGestureControls:
    """Classify poses into modes and repeat wheel actions while a mode is active."""

    def __init__(self, pyautogui_module: Any):
        self.gui = pyautogui_module
        self.last_action_time = 0.0
        self.desired_rotation = 0
        self.applied_rotation = None
        self.last_rotation_attempt = None

    def reset(self) -> None:
        self.last_action_time = 0.0
        self.desired_rotation = 0
        self.applied_rotation = None
        self.last_rotation_attempt = None

    def _scroll_at_cursor(self, clicks: int) -> None:
        """Send wheel input to the exact screen position currently under the mouse."""
        x, y = self.gui.position()
        self.gui.scroll(clicks, x=x, y=y)

    def classify(self, results: Any, frame_size: tuple) -> Optional[str]:
        width, height = frame_size
        raw_hands = {
            side: getattr(results, f"{side}_hand_landmarks")
            for side in ("left", "right")
        }
        metrics = {
            side: _hand_metrics(hand, width, height)
            for side, hand in raw_hands.items() if hand
        }

        # The image is mirrored before tracking, so MediaPipe's "right" label
        # corresponds to the user's physical left hand in this preview.
        left_hand = metrics.get("right")
        if left_hand:
            fingers = left_hand["extended"]
            closed_except_index = (
                fingers["index"] and not fingers["middle"]
                and not fingers["ring"] and not fingers["pinky"]
                and not left_hand["thumb_extended"]
            )
            if closed_except_index:
                pts = left_hand["points"]
                index_vector = pts[8] - pts[5]
                angle = float(np.degrees(np.arctan2(index_vector[0], -index_vector[1])))
                # Windows' display orientations are numbered counterclockwise;
                # negate the image angle so the visible screen turns with the finger.
                self.desired_rotation = int(round((-angle % 360.0) / 90.0)) % 4
                return "rotate"

        # Zoom pose: one hand with middle/ring/pinky curled and thumb pointing
        # left in the mirrored preview. The index points up for zoom out, or is
        # curled for zoom in. Require one matching hand to avoid ambiguity.
        zoom_candidates = []
        for hand in metrics.values():
            pts = hand["points"]
            fingers = hand["extended"]
            thumb_points_left = (
                pts[4][0] - pts[2][0] < -max(6.0, hand["palm_width"] * 0.20)
            )
            other_fingers_curled = not fingers["middle"] and not fingers["ring"] and not fingers["pinky"]
            if not thumb_points_left or not other_fingers_curled:
                continue
            index_points_up = (
                fingers["index"]
                and pts[8][1] < pts[5][1] - 0.22 * hand["length"]
            )
            if index_points_up:
                zoom_candidates.append("zoom_out")
            elif not fingers["index"]:
                zoom_candidates.append("zoom_in")
        if len(zoom_candidates) == 1:
            return zoom_candidates[0]

        # One open, camera-facing hand; either tracker label is accepted because
        # the mirrored preview can invert MediaPipe's left/right label.
        open_hands = [hand for hand in metrics.values()
                      if hand["open"] and hand["facing_camera"]]
        if len(open_hands) != 1:
            return None
        hand = open_hands[0]
        vector = hand["points"][12] - hand["points"][0]
        tilt = float(np.degrees(np.arctan2(vector[0], -vector[1])))
        if tilt <= -12.0:
            return "scroll_down"
        if tilt >= 12.0:
            return "scroll_up"
        return None

    def apply_mode(self, mode: str, interval: float = 0.25) -> None:
        """Apply wheel pulses or the index-selected Windows display rotation."""
        if mode == "rotate":
            if (self.applied_rotation != self.desired_rotation
                    and self.last_rotation_attempt != self.desired_rotation):
                self.last_rotation_attempt = self.desired_rotation
                result = self._set_display_orientation(self.desired_rotation)
                if result == 0:
                    self.applied_rotation = self.desired_rotation
                    print(f"SCREEN_ROTATION={self.desired_rotation * 90} degrees")
                else:
                    print(f"SCREEN_ROTATION_FAILED=Windows display mode result {result}")
            return
        now = time.monotonic()
        if now - self.last_action_time < interval:
            return
        if mode == "scroll_down":
            self._scroll_at_cursor(-200)
        elif mode == "scroll_up":
            self._scroll_at_cursor(200)
        elif mode in ("zoom_in", "zoom_out"):
            clicks = 1 if mode == "zoom_in" else -1
            self.gui.keyDown("ctrl")
            try:
                self._scroll_at_cursor(clicks)
            finally:
                self.gui.keyUp("ctrl")
        else:
            return
        self.last_action_time = now

    @staticmethod
    def _set_display_orientation(orientation: int) -> int:
        """Set the primary Windows display orientation; return its Win32 result."""
        if not hasattr(__import__("ctypes"), "windll"):
            return -1
        import ctypes

        DWORD = ctypes.c_uint32
        LONG = ctypes.c_int32
        WORD = ctypes.c_uint16
        SHORT = ctypes.c_int16

        class POINTL(ctypes.Structure):
            _fields_ = [("x", LONG), ("y", LONG)]

        class DisplayPosition(ctypes.Structure):
            _fields_ = [("position", POINTL), ("orientation", DWORD),
                        ("fixed_output", DWORD)]

        class PrinterFields(ctypes.Structure):
            _fields_ = [("orientation", SHORT), ("paper_size", SHORT),
                        ("paper_length", SHORT), ("paper_width", SHORT),
                        ("scale", SHORT), ("copies", SHORT),
                        ("default_source", SHORT), ("print_quality", SHORT)]

        class ModeUnion(ctypes.Union):
            _fields_ = [("display", DisplayPosition), ("printer", PrinterFields)]

        class DisplayFlags(ctypes.Union):
            _fields_ = [("display_flags", DWORD), ("nup", DWORD)]

        class DEVMODEW(ctypes.Structure):
            _anonymous_ = ("mode", "flags")
            _fields_ = [
                ("device_name", ctypes.c_wchar * 32), ("spec_version", WORD),
                ("driver_version", WORD), ("size", WORD), ("driver_extra", WORD),
                ("fields", DWORD), ("mode", ModeUnion), ("color", SHORT),
                ("duplex", SHORT), ("y_resolution", SHORT), ("tt_option", SHORT),
                ("collate", SHORT), ("form_name", ctypes.c_wchar * 32),
                ("log_pixels", WORD), ("bits_per_pixel", DWORD),
                ("width", DWORD), ("height", DWORD), ("flags", DisplayFlags),
                ("frequency", DWORD), ("icm_method", DWORD), ("icm_intent", DWORD),
                ("media_type", DWORD), ("dither_type", DWORD), ("reserved1", DWORD),
                ("reserved2", DWORD), ("panning_width", DWORD),
                ("panning_height", DWORD),
            ]

        user32 = ctypes.windll.user32
        current = DEVMODEW()
        current.size = ctypes.sizeof(DEVMODEW)
        if not user32.EnumDisplaySettingsW(None, -1, ctypes.byref(current)):
            return -2

        old_orientation = int(current.display.orientation)
        orientation %= 4
        old_portrait = old_orientation in (1, 3)
        new_portrait = orientation in (1, 3)
        current.display.orientation = orientation
        current.fields |= 0x00000080  # DM_DISPLAYORIENTATION
        if old_portrait != new_portrait:
            current.width, current.height = current.height, current.width
            current.fields |= 0x00080000 | 0x00100000  # DM_PELSWIDTH | DM_PELSHEIGHT

        test_result = user32.ChangeDisplaySettingsW(ctypes.byref(current), 2)  # CDS_TEST
        if test_result != 0:
            return int(test_result)
        return int(user32.ChangeDisplaySettingsW(ctypes.byref(current), 0))

# MediaPipe Face Mesh eye landmarks: (outer-left, upper-left, upper-right,
# outer-right, lower-right, lower-left), expressed in image coordinates.
EYE_LANDMARKS = {
    "left": (362, 385, 387, 263, 373, 380),
    "right": (33, 160, 158, 133, 153, 144),
}
EYE_CLOSED_EAR = 0.20*0.8


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
        #Eye aspect ratio (EAR) is a measure of eye openness. A lower EAR indicates a closed eye.
        #swapping left and right because the camera view is mirrored.
        left_ear = _eye_aspect_ratio(face_landmarks, EYE_LANDMARKS["right"], width, height) 
        right_ear = _eye_aspect_ratio(face_landmarks, EYE_LANDMARKS["left"], width, height)
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
        print("Start: prayer. Tilt one open hand left/right for scroll.")
        print("Rotate: physical left hand, curl all but index; screen orientation follows index direction.")
        print("Index up sets the screen upright; remove both hands to leave rotate mode.")
        print("Wheel pulses repeat while in a mode; move both hands out of frame to return to active.")
        print("Zoom: curl middle/ring/pinky and point thumb left; index up=out, index curled=in.")
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
            hands_present = bool(result.left_hand_landmarks or result.right_hand_landmarks)
            candidate = (controls.classify(result, size)
                         if machine.state in ("active", "rotate") else None)
            event = machine.update(prayer, stop, candidate, hands_present)
            if event:
                print(f"POSE_EVENT={event}")
                if event in ("STOP", "RETURN_ACTIVE"):
                    controls.reset()
            if machine.state in PoseStateMachine.ACTION_STATES:
                controls.apply_mode(machine.state)
            active = machine.state == "active"
            eye_action = eye_clicks.update(result.face_landmarks, size, active)
            if eye_action:
                print(f"EYE_EVENT={eye_action}")

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
            state_prompts = {
                "waiting": "Hold prayer pose to start",
                "active": "Active: scroll tilt | zoom: thumb left, 3 fingers curled, index up/out or curled/in",
                "zoom_in": "ZOOM IN mode - remove both hands to return active",
                "zoom_out": "ZOOM OUT mode - remove both hands to return active",
                "scroll_up": "SCROLL UP mode - remove both hands to return active",
                "scroll_down": "SCROLL DOWN mode - remove both hands to return active",
                "rotate": (f"ROTATE mode: index sets {controls.desired_rotation * 90} degrees - "
                           "remove both hands to return active"),
                "stopped": "Stopped: hold prayer pose to restart",
            }
            cv2.putText(frame, state_prompts[machine.state], (12, 84),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.52, (0, 255, 0), 2)
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















