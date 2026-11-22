"""Detect key strokes using hand landmarks and a detected keyboard.

Run with ``uv run python keyboard_typing_detector.py``. The program first
tries to detect the keyboard automatically. Press C to select its corners
manually (TL, TR, BR, BL). Press Q or ESC to quit.
"""

# Author: Victor Lancieux
# Date: 2026-11-22 (hand-only tracking and automatic keyboard detection integrated)
# Purpose: Map finger taps to keyboard keys and optionally send key output.

import argparse
import ctypes
import os
import time
from collections import defaultdict, deque
from typing import Any, Dict, List, Tuple

import cv2
import numpy as np

from camera_utils import open_camera
from find_keyboard import find_keyboard_corners
from keyboard_tracker2 import beacon_centers, keyboard_corners_from_poses
from hand_only_capture import HAND_CONNECTIONS, ensure_hand_model
from keyboard_tracker import (
    KeyboardDetection,
    draw_keyboard_view,
    finger_pose_to_key,
)

from keyboard_layout import KEYS as LAYOUT_KEYS   # your AZERTY paper layout

BEACON = False            # True: paper keyboard located by beacons; False: old keyboard
FLIP_LAYOUT = True       # the camera image is mirrored; set False if the key map comes out mirrored

_kb = np.array([v for v in LAYOUT_KEYS.values()], np.float32)   # x0, y0, x1, y1
LX0, LY0, LX1, LY1 = _kb[:, 0].min(), _kb[:, 1].min(), _kb[:, 2].max(), _kb[:, 3].max()
LAYOUT_SIZE = (int(LX1) + 8, int(LY1) + 8)       # key-map window size in layout px
LAYOUT_MAX_DIST = 4.0                            # fingertip farther than this from a key = no key
HALF_KEY = 19.0 if BEACON else HALF_KEY_OFFSET_PX
KEY_SCALE = 1.0 if BEACON else OUTPUT_SIZE[0] / 950.0

ASCII_LABEL = {"²": "2s", "é": "e'", "è": "e`", "ç": "c,", "à": "a`", "ù": "u`", "µ": "mu"}


def layout_key_at(x: float, y: float):
    """Nearest key of the paper layout for a point in layout coordinates."""
    best, best_d = None, 1e9
    for name, (x0, y0, x1, y1) in LAYOUT_KEYS.items():
        d = np.hypot(max(x0 - x, 0, x - x1), max(y0 - y, 0, y - y1))
        if d < best_d:
            best, best_d = name, d
    return best if best_d <= LAYOUT_MAX_DIST else None


def draw_layout_view(detection, highlighted=None) -> np.ndarray:
    """Warped paper keyboard with the AZERTY key boxes drawn on top."""
    view = cv2.cvtColor(cv2.cvtColor(detection.warped, cv2.COLOR_BGR2GRAY),
                        cv2.COLOR_GRAY2BGR)
    highlighted = highlighted or set()
    for name, (x0, y0, x1, y1) in LAYOUT_KEYS.items():
        a, b = (int(x0), int(y0)), (int(x1), int(y1))
        if name in highlighted:
            cv2.rectangle(view, a, b, (0, 255, 0), -1)
        cv2.rectangle(view, a, b, (0, 160, 0), 1)
        label = ASCII_LABEL.get(name, name if name.isascii() else "")
        cv2.putText(view, label, (a[0] + 2, b[1] - 4), cv2.FONT_HERSHEY_SIMPLEX,
                    0.3, (0, 0, 255), 1, cv2.LINE_AA)
    return view

BEACON = True
OUTPUT_SIZE = (1000, 284)
HALF_KEY_OFFSET_PX = 20.0
HAND_CONTACT_OFFSETS = {
    # The preview is mirrored: physical left appears on the keyboard view's right.
    "left": (0, 0),
    "right": (0, 0),
}
KEY_PITCH_CANONICAL_PX = 41.5
TILT_THRESHOLD_DEGREES = 10.0
SIDE_OFFSET_KEY_FRACTION = 1.0 / 3.0
SAME_KEY_DEBOUNCE_SECONDS = 0.5
FINGER_TIPS = {"thumb": 4, "index": 8, "middle": 12, "ring": 16, "pinky": 20}
FINGER_JOINTS = {
    "thumb": ((1, 2, 3), (2, 3, 4)),
    "index": ((5, 6, 7), (6, 7, 8)),
    "middle": ((9, 10, 11), (10, 11, 12)),
    "ring": ((13, 14, 15), (14, 15, 16)),
    "pinky": ((17, 18, 19), (18, 19, 20)),
}

def _meter_exposure_on_keyboard(capture: Any, frame: np.ndarray,
                                corners: np.ndarray, state: Dict[str, Any]) -> None:
    """Adjust hardware exposure using the calibrated keyboard region as its meter."""
    now = time.monotonic()
    if now - state.get("last_exposure_adjustment", 0.0) < 0.5:
        return
    state["last_exposure_adjustment"] = now
    mask = np.zeros(frame.shape[:2], dtype=np.uint8)
    cv2.fillConvexPoly(mask, np.asarray(corners, dtype=np.int32), 255)
    luminance = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    keyboard_level = float(np.median(luminance[mask > 0]))
    state["keyboard_luminance"] = keyboard_level

    # OpenCV's DirectShow values are 0.25=manual and 0.75=auto. Leave the
    # camera's pixels untouched; only adjust its hardware exposure setting.
    if not state.get("manual_exposure_enabled"):
        if not capture.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.25):
            if not state.get("exposure_warning_shown"):
                print("Camera does not expose manual controls; keeping its automatic exposure.")
                state["exposure_warning_shown"] = True
                state["manual_exposure_enabled"] = True
            return
        state["manual_exposure_enabled"] = True
        state["exposure_value"] = float(capture.get(cv2.CAP_PROP_EXPOSURE))

    exposure = float(state.get("exposure_value", capture.get(cv2.CAP_PROP_EXPOSURE)))
    if keyboard_level > 150.0:
        exposure -= 0.5
    elif keyboard_level < 115.0:
        exposure += 0.5
    else:
        return
    if capture.set(cv2.CAP_PROP_EXPOSURE, exposure):
        state["exposure_value"] = exposure
    elif not state.get("exposure_warning_shown"):
        print("Camera ignored manual exposure changes; keeping its current exposure.")
        state["exposure_warning_shown"] = True
PYAUTOGUI_KEYS = {
    "Backspace": "backspace", "Tab": "tab", "Enter": "enter", "Space": "space",
    "Esc": "esc", "Insert": "insert", "Delete": "delete", "Home": "home",
    "End": "end", "PgUp": "pageup", "PgDn": "pagedown",
    "Left": "left", "Right": "right", "Up": "up", "Down": "down",
    "CapsLock": "capslock", "NumLock": "numlock", "PrtSc": "printscreen",
    "ScrLk": "scrolllock", "Pause": "pause", "LCtrl": "ctrl", "RCtrl": "ctrl",
    "LShift": "shift", "RShift": "shift", "LAlt": "alt", "LWin": "win",
    "RWin": "win", "Menu": "apps", "AltGr": "alt",
    "Special1": "browserhome", "Special2": "launchmail",
    "Special3": "volumemute", "Special4": "launchapp1",
}
PYAUTOGUI_KEYS.update({f"F{i}": f"f{i}" for i in range(1, 13)})
PYAUTOGUI_KEYS["KP_Enter"] = "enter"
NUMPAD_CHARACTERS = {
    **{f"KP_{digit}": digit for digit in "0123456789"},
    "KP_+": "+", "KP_-": "-", "KP_*": "*", "KP_/": "/", "KP_.": ".",
}


def send_key_to_focused_app(key: str, pyautogui_module: Any) -> bool:
    """Send a detected printable or named key to the currently focused app."""
    if key in NUMPAD_CHARACTERS:
        pyautogui_module.write(NUMPAD_CHARACTERS[key], interval=0)
        return True
    if len(key) == 1:
        # Sending unshifted letters lets Windows apply its real Caps Lock state.
        if key.isalpha():
            key = key.lower()
        pyautogui_module.write(key, interval=0)
        return True
    pyautogui_key = PYAUTOGUI_KEYS.get(key)
    if pyautogui_key is None:
        return False
    pyautogui_module.press(pyautogui_key)
    return True


def is_text_entry_key(key: str) -> bool:
    """Allow letters, numpad keys, and the requested editing/control keys."""
    return (
        (len(key) == 1 and key.upper() in "ABCDEFGHIJKLMNOPQRSTUVWXYZ")
        or key in {"CapsLock", "Backspace", "Space", "NumLock", "KP_Enter"}
        or key in NUMPAD_CHARACTERS
    )


def read_caps_lock_state() -> bool:
    """Read the active Windows Caps Lock toggle state."""
    try:
        import ctypes
        return bool(ctypes.windll.user32.GetKeyState(0x14) & 1)
    except (AttributeError, OSError):
        return False


def foreground_external_window() -> int | None:
    """Return the foreground window when it belongs to another process."""
    if os.name != "nt":
        return None
    try:
        user32 = ctypes.windll.user32
        hwnd = user32.GetForegroundWindow()
        if not hwnd:
            return None
        process_id = ctypes.c_ulong()
        user32.GetWindowThreadProcessId(hwnd, ctypes.byref(process_id))
        return int(hwnd) if process_id.value != os.getpid() else None
    except (AttributeError, OSError):
        return None


def restore_output_window(hwnd: Any) -> None:
    """Give typing focus back to the app that was active before a virtual click."""
    if os.name != "nt" or not hwnd:
        return
    try:
        ctypes.windll.user32.SetForegroundWindow(int(hwnd))
    except (AttributeError, OSError, TypeError, ValueError):
        pass


def apply_key_output(key: str, source: str, state: Dict[str, Any],
                     pyautogui_module: Any, send_output: bool) -> None:
    """Update the text preview and optionally send a key to the focused app."""
    if send_output:
        try:
            sent = send_key_to_focused_app(key, pyautogui_module)
            if not sent:
                print(f"KEY_OUTPUT_UNSUPPORTED={key}")
            state["caps_lock"] = read_caps_lock_state()
        except Exception as exc:
            print(f"KEY_OUTPUT_FAILED={key}: {exc}")
    elif key == "CapsLock":
        state["caps_lock"] = not state["caps_lock"]

    state["last_key"] = f"{key} ({source})"
    state["last_event_time"] = time.monotonic()
    if key == "Backspace":
        state["typed"] = state["typed"][:-1]
    elif key == "Space":
        state["typed"] += " "
    elif key in ("Enter", "KP_Enter"):
        state["typed"] += "\n"
    elif key in NUMPAD_CHARACTERS:
        state["typed"] += NUMPAD_CHARACTERS[key]
    elif len(key) == 1:
        state["typed"] += key.upper() if state["caps_lock"] else key.lower()
    print(f"KEY_OUTPUT={key} SOURCE={source}")


def _make_detection(frame: np.ndarray, corners: np.ndarray) -> KeyboardDetection:
    corners = np.asarray(corners, dtype=np.float32).reshape(4, 2)
    if BEACON:
        # corners are TL, TR, BR, BL on screen. In a mirrored image the physical
        # Esc corner is on the right, so swap left/right for the warp.
        src = corners[[1, 0, 3, 2]] if FLIP_LAYOUT else corners
        destination = np.array([[LX0, LY0], [LX1, LY0], [LX1, LY1], [LX0, LY1]], np.float32)
        size = LAYOUT_SIZE
    else:
        width, height = OUTPUT_SIZE
        src = corners
        destination = np.array([[0, 0], [width - 1, 0],
                                [width - 1, height - 1], [0, height - 1]], np.float32)
        size = OUTPUT_SIZE
    transform = cv2.getPerspectiveTransform(src, destination)
    inverse = cv2.getPerspectiveTransform(destination, src)
    warped = cv2.warpPerspective(frame, transform, size)
    return KeyboardDetection(corners, warped, transform, inverse, 1.0)


def _draw_black_white_keyboard(detection: KeyboardDetection,
                               highlighted: Any = None) -> np.ndarray:
    """Show a contrast-normalized black-and-white keyboard with key overlays."""
    gray = cv2.cvtColor(detection.warped, cv2.COLOR_BGR2GRAY)
    gray = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8)).apply(gray)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    binary = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY, 31, 7)
    base = cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)
    bw_detection = KeyboardDetection(
        detection.corners, base, detection.image_to_keyboard,
        detection.keyboard_to_image, detection.confidence,
    )
    return draw_keyboard_view(bw_detection, highlighted)


def fingertip_measurements(results: Any, detection: KeyboardDetection,
                           frame_size: Tuple[int, int], finger_mode: str = "all"
                           ) -> Dict[Tuple[str, str], Dict[str, Any]]:
    """Return each visible fingertip's key and its last two joint bend angles."""
    width, height = frame_size
    measurements = {}
    selected_fingers = ({"index": FINGER_TIPS["index"]}
                        if finger_mode == "indexes-only" else FINGER_TIPS)
    for side in ("left", "right"):
        hand = results.get(side)
        if not hand:
            continue
        # The hand-only tracker sees the mirrored selfie image and returns
        # user-relative Left/Right labels, so keep its handedness as reported.
        user_side = side
        for finger, tip_index in selected_fingers.items():
            tip = hand[tip_index]
            x, y = float(tip.x * width), float(tip.y * height)
            distal_joint = hand[FINGER_JOINTS[finger][1][1]]
            camera_points = np.array([[[x, y]],
                                      [[distal_joint.x * width,
                                        distal_joint.y * height]]], dtype=np.float32)
            keyboard_points = cv2.perspectiveTransform(
                camera_points, detection.image_to_keyboard).reshape(2, 2)
            tip_on_board, joint_on_board = keyboard_points
            finger_direction = tip_on_board - joint_on_board
            direction_length = float(np.linalg.norm(finger_direction))
            contact_point = tip_on_board.copy()
            if direction_length > 1e-5:
                unit_direction = finger_direction / direction_length
                contact_point += unit_direction * HALF_KEY
                tilt_degrees = float(np.degrees(np.arctan2(
                    finger_direction[0], max(abs(float(finger_direction[1])), 1e-5))))
                if abs(tilt_degrees) > TILT_THRESHOLD_DEGREES:
                    side_offset = -(KEY_PITCH_CANONICAL_PX * KEY_SCALE * SIDE_OFFSET_KEY_FRACTION)
                    contact_point[0] += np.sign(finger_direction[0]) * side_offset
            contact_point += np.asarray(HAND_CONTACT_OFFSETS[user_side], dtype=np.float32)
            if BEACON:
                key = layout_key_at(float(contact_point[0]), float(contact_point[1]))
            else:
                contact_camera = cv2.perspectiveTransform(
                    np.array([[contact_point]], dtype=np.float32),
                    detection.keyboard_to_image)[0, 0]
                key = finger_pose_to_key(tuple(contact_camera), detection)
            bends = tuple(_bend_angle(hand[a], hand[b], hand[c], width, height)
                          for a, b, c in FINGER_JOINTS[finger])
            measurements[(user_side, finger)] = {
                "xy": (x, y),
                "keyboard_xy": (float(contact_point[0]), float(contact_point[1])),
                "key": key,
                "bends": bends,
            }
    return measurements


def _bend_angle(a: Any, b: Any, c: Any, width: int, height: int) -> float:
    """Return how far the middle landmark's joint is bent from straight."""
    def xyz(landmark: Any) -> np.ndarray:
        return np.array([landmark.x * width, landmark.y * height,
                         landmark.z * width], dtype=np.float32)

    first = xyz(a) - xyz(b)
    second = xyz(c) - xyz(b)
    denominator = float(np.linalg.norm(first) * np.linalg.norm(second))
    if denominator < 1e-6:
        return 180.0
    cosine = float(np.clip(np.dot(first, second) / denominator, -1.0, 1.0))
    return 180.0 - float(np.degrees(np.arccos(cosine)))


class StraightFingerDetector:
    """Emit after a straight finger settles on a key; release when it moves/bends."""

    def __init__(self, straightness_degrees: float = 5.0,
                 rearm_degrees: float = 8.0, still_frames: int = 3,
                 still_radius: float = 6.0, release_movement: float = 12.0):
        self.straightness_degrees = straightness_degrees
        self.rearm_degrees = rearm_degrees
        self.still_frames = still_frames
        self.still_radius = still_radius
        self.release_movement = release_movement
        self.positions: Dict[Tuple[str, str], Any] = defaultdict(
            lambda: {"key": None, "points": deque(maxlen=still_frames)})
        self.straight: Dict[Tuple[str, str], bool] = defaultdict(bool)
        self.active_keys: Dict[Tuple[str, str], str] = {}
        self.last_pressed_position: Dict[Tuple[str, str], np.ndarray] = {}
        self.missing_frames: Dict[Tuple[str, str], int] = defaultdict(int)

    def update(self, measurements: Dict[Tuple[str, str], Dict[str, Any]]) -> List[Tuple[str, str, str]]:
        """Return key events only after a straight fingertip is still over a key."""
        events: List[Tuple[str, str, str]] = []
        visible = set(measurements)
        for finger_id in list(self.active_keys):
            if finger_id not in visible:
                self.missing_frames[finger_id] += 1
                if self.missing_frames[finger_id] >= 4:
                    self._release(finger_id)

        for finger_id, data in measurements.items():
            self.missing_frames[finger_id] = 0
            pip_bend, dip_bend = data["bends"]
            within_limit = (pip_bend <= self.straightness_degrees
                            and dip_bend <= self.straightness_degrees)
            key = data["key"]
            point = np.asarray(data["keyboard_xy"], dtype=np.float32)

            if self.straight[finger_id]:
                previous = self.last_pressed_position.get(finger_id, point)
                moved = float(np.linalg.norm(point - previous)) > self.release_movement
                bent = max(pip_bend, dip_bend) > self.rearm_degrees
                left_key = key != self.active_keys.get(finger_id)
                if bent or moved or left_key:
                    self._release(finger_id)
                else:
                    self.last_pressed_position[finger_id] = point
                    continue

            if not within_limit or key is None:
                self._clear_position_history(finger_id)
                continue

            history = self.positions[finger_id]
            if history["key"] != key:
                history["key"] = key
                history["points"].clear()
            history["points"].append(point)
            if len(history["points"]) < self.still_frames:
                continue

            points = np.asarray(history["points"])
            center = np.mean(points, axis=0)
            radius = float(np.max(np.linalg.norm(points - center, axis=1)))
            if radius <= self.still_radius:
                events.append((finger_id[0], finger_id[1], key))
                self.straight[finger_id] = True
                self.active_keys[finger_id] = key
                self.last_pressed_position[finger_id] = point
                self._clear_position_history(finger_id)
        return events

    def _release(self, finger_id: Tuple[str, str]) -> None:
        self.straight[finger_id] = False
        self.active_keys.pop(finger_id, None)
        self.last_pressed_position.pop(finger_id, None)
        self._clear_position_history(finger_id)

    def _clear_position_history(self, finger_id: Tuple[str, str]) -> None:
        history = self.positions[finger_id]
        history["key"] = None
        history["points"].clear()


def run(camera_index: int = 1, confidence: float = 0.55,
        straightness_degrees: float = 20.0, finger_mode: str = "all",
        send_output: bool = True, text_only: bool = False) -> None:
    try:
        import mediapipe as mp
        if send_output:
            import pyautogui
        else:
            pyautogui = None
    except ImportError as exc:
        raise RuntimeError("Required package missing. Run: uv add mediapipe pyautogui") from exc

    model_path = ensure_hand_model()
    vision = mp.tasks.vision
    options = vision.HandLandmarkerOptions(
        base_options=mp.tasks.BaseOptions(model_asset_path=str(model_path)),
        running_mode=vision.RunningMode.VIDEO,
        num_hands=2,
        min_hand_detection_confidence=confidence,
        min_hand_presence_confidence=confidence,
        min_tracking_confidence=confidence,
    )
    landmarker = vision.HandLandmarker.create_from_options(options)
    tap_detector = StraightFingerDetector(straightness_degrees=straightness_degrees)
    capture = None
    try:
        capture, info = open_camera(camera_index, fourcc=None)
        print(f"Camera {camera_index}: {info['width']}x{info['height']} [{info['backend']}]")
        print("Press C and click keyboard corners in order: TL, TR, BR, BL.")
        print(f"Finger mode: {finger_mode} | straightness threshold: "
              f"{straightness_degrees:.1f} degrees at both PIP and DIP joints.")
        print(f"Text-only filter: {text_only}")
        print("Hand-only tracking is active; no face or arm landmarks are computed.")
        print("The key-map window shows fingertip cursors.")
        print("Click any key in the key-map window to send it directly, bypassing finger filters.")
        if send_output:
            print("Key events go to the active text field; click it before typing.")
        else:
            print("System key output is disabled; detected text stays in this preview.")
        camera_window = "Keyboard typing detector - camera"
        keyboard_window = "Keyboard typing detector - key map"
        cv2.namedWindow(camera_window, cv2.WINDOW_NORMAL)
        cv2.namedWindow(keyboard_window, cv2.WINDOW_AUTOSIZE)

        state: Dict[str, Any] = {
            "frame": None,
            "points": [],
            "calibration": None,
            "selecting": False,
            "last_key": "",
            "last_event_time": 0.0,
            "typed": "",
            "caps_lock": read_caps_lock_state(),
            "detection": None,
            "mouse_key_queue": [],
            "mouse_pressed_key": None,
            "mouse_highlight_until": 0.0,
            "target_window": foreground_external_window(),
            "frame_count": 0,
            "auto_candidate": None,
            "auto_candidate_hits": 0,
            "drag_corner": None,
        }

        def camera_image_point(x: int, y: int) -> Tuple[int, int]:
            """Convert camera-window mouse coordinates into source-image pixels."""
            frame = state["frame"]
            if frame is None:
                return x, y
            image_h, image_w = frame.shape[:2]
            try:
                _left, _top, window_w, window_h = cv2.getWindowImageRect(camera_window)
                if window_w > 0 and window_h > 0:
                    x = round(x * image_w / window_w)
                    y = round(y * image_h / window_h)
            except cv2.error:
                pass
            return (max(0, min(image_w - 1, x)),
                    max(0, min(image_h - 1, y)))

        def on_mouse(event: int, x: int, y: int, _flags: int, _userdata: Any) -> None:
            if state["frame"] is None:
                return
            image_x, image_y = camera_image_point(x, y)
            if event == cv2.EVENT_LBUTTONDOWN:
                if state["selecting"]:
                    state["points"].append((image_x, image_y))
                    if len(state["points"]) == 4:
                        state["calibration"] = np.asarray(state["points"], dtype=np.float32)
                        state["selecting"] = False
                        print("Keyboard corners calibrated; drag a numbered corner to adjust it.")
                    return
                if state["calibration"] is not None:
                    distances = np.linalg.norm(
                        state["calibration"] - np.array([image_x, image_y]), axis=1)
                    nearest = int(np.argmin(distances))
                    if distances[nearest] <= 22:
                        state["drag_corner"] = nearest
                        tap_detector.straight.clear()
                        tap_detector.positions.clear()
                        tap_detector.active_keys.clear()
                        tap_detector.last_pressed_position.clear()
            elif event == cv2.EVENT_MOUSEMOVE and state["drag_corner"] is not None:
                state["calibration"][state["drag_corner"]] = (image_x, image_y)
                state["points"] = state["calibration"].tolist()
            elif event == cv2.EVENT_LBUTTONUP:
                state["drag_corner"] = None

        def on_keyboard_mouse(event: int, x: int, y: int,
                              _flags: int, _userdata: Any) -> None:
            detection = state["detection"]
            if detection is None:
                return
            if event == cv2.EVENT_LBUTTONDOWN:
                if BEACON:
                    key = layout_key_at(float(x), float(y))   # the view is already in layout coordinates
                else:
                    point = np.array([[[float(x), float(y)]]], dtype=np.float32)
                    camera_point = cv2.perspectiveTransform(point, detection.keyboard_to_image)[0, 0]
                    key = finger_pose_to_key(tuple(camera_point), detection)
                if key is not None:
                    state["mouse_key_queue"].append(key)
                    state["mouse_pressed_key"] = key
                    state["mouse_highlight_until"] = time.monotonic() + 0.2
            elif event == cv2.EVENT_LBUTTONUP:
                state["mouse_pressed_key"] = None

        cv2.setMouseCallback(camera_window, on_mouse)
        cv2.setMouseCallback(keyboard_window, on_keyboard_mouse)
        while True:
            # Remember the last external app the user focused. OpenCV's own
            # windows take focus when clicked, so the click callback cannot
            # discover the typing destination after the fact.
            external_window = foreground_external_window()
            if external_window:
                state["target_window"] = external_window
            ok, frame = capture.read()
            if not ok or frame is None:
                print("Camera stopped returning frames.")
                break
            # Mirror once at capture so corner clicks, hands, and key lookup agree.
            frame = cv2.flip(frame, 1)
            state["frame_count"] += 1
            if (state["calibration"] is None and not state["selecting"]
                    and state["frame_count"] % 8 == 0):
                if BEACON:
                    candidate = keyboard_corners_from_poses(beacon_centers(frame))
                    if candidate is not None:
                        state["auto_candidate"] = candidate
                        state["calibration"] = state["auto_candidate"]
                        state["points"] = state["calibration"].tolist()
                        state["auto_candidate"] = None
                        print("Keyboard detected automatically; mapped fingers are active.")
                else:
                    candidate = find_keyboard_corners(frame)
                    if candidate is not None:
                        candidate = np.array([candidate[i] for i in (3, 2, 1, 0)])  # BR, BL, TL, TR
                    previous_candidate = state["auto_candidate"]
                    if candidate is None:
                        state["auto_candidate"] = None
                        state["auto_candidate_hits"] = 0
                    elif (previous_candidate is not None
                        and np.max(np.linalg.norm(candidate - previous_candidate, axis=1)) < 20):
                        state["auto_candidate"] = candidate
                        state["auto_candidate_hits"] += 1
                    else:
                        state["auto_candidate"] = candidate
                        state["auto_candidate_hits"] = 1
                    if state["auto_candidate_hits"] >= 3:
                        state["calibration"] = state["auto_candidate"]
                        state["points"] = state["calibration"].tolist()
                        state["auto_candidate"] = None
                        state["auto_candidate_hits"] = 0
                        print("Keyboard detected automatically; mapped fingers are active.")
            # Meter the camera's actual exposure from the keyboard, without
            # applying any brightness or contrast filter to captured pixels.
            if state["calibration"] is not None:
                _meter_exposure_on_keyboard(capture, frame, state["calibration"], state)
            state["frame"] = frame.copy()
            width, height = frame.shape[1], frame.shape[0]

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            result = landmarker.detect_for_video(image, time.monotonic_ns() // 1_000_000)
            hands = {"left": None, "right": None}
            for hand_index, landmarks in enumerate(result.hand_landmarks):
                if hand_index >= len(result.handedness) or not result.handedness[hand_index]:
                    continue
                label = result.handedness[hand_index][0].category_name.lower()
                if label in hands:
                    hands[label] = landmarks

            detection = None
            if state["calibration"] is not None:
                detection = _make_detection(frame, state["calibration"])
            state["detection"] = detection
            measurements = (fingertip_measurements(hands, detection, (width, height),
                                                   finger_mode)
                            if detection else {})
            events = tap_detector.update(measurements) if detection else []
            accepted_events = []
            for side, finger, key in events:
                if text_only and not is_text_entry_key(key):
                    continue
                event_time = time.monotonic()
                if (key == state.get("last_typed_key")
                        and event_time - state.get("last_typed_time", 0.0)
                        < SAME_KEY_DEBOUNCE_SECONDS):
                    continue
                state["last_typed_key"] = key
                state["last_typed_time"] = event_time
                accepted_events.append((side, finger, key))
                apply_key_output(key, f"{side} {finger}", state, pyautogui, send_output)
            events = accepted_events

            for key in state["mouse_key_queue"]:
                # Mouse clicks deliberately bypass finger and text-only filters,
                # as well as the same-key finger debounce.
                if send_output:
                    restore_output_window(state.get("target_window"))
                apply_key_output(key, "mouse", state, pyautogui, send_output)
                state["last_mouse_key"] = key
            state["mouse_key_queue"].clear()

            view = frame.copy()
            if detection:
                cv2.polylines(view, [detection.corners.astype(np.int32).reshape(-1, 1, 2)],
                              True, (40, 220, 40), 2)
                selected = set(tap_detector.active_keys.values())
                if (state["mouse_pressed_key"] is not None
                        or time.monotonic() < state["mouse_highlight_until"]):
                    clicked_key = (state["mouse_pressed_key"]
                                   or state.get("last_mouse_key"))
                    if clicked_key:
                        selected.add(clicked_key)
                keyboard_view = (draw_layout_view(detection, selected) if BEACON
                 else _draw_black_white_keyboard(detection, selected))
                for finger_id, data in measurements.items():
                    kx, ky = data["keyboard_xy"]
                    if not (0 <= kx < keyboard_view.shape[1]
                            and 0 <= ky < keyboard_view.shape[0]):
                        continue
                    x, y = int(round(kx)), int(round(ky))
                    straight = tap_detector.straight.get(finger_id, False)
                    color = (0, 70, 255) if straight else (255, 180, 0)
                    cv2.drawMarker(keyboard_view, (x, y), color, cv2.MARKER_CROSS,
                                   18, 2, cv2.LINE_AA)
                    cv2.circle(keyboard_view, (x, y), 6, color, 2, cv2.LINE_AA)
                    short_name = f"{finger_id[0][0].upper()}-{finger_id[1][0]}"
                    cv2.putText(keyboard_view, f"{short_name}:{data['key'] or '-'}",
                                (x + 8, y - 7), cv2.FONT_HERSHEY_SIMPLEX,
                                0.32, color, 1, cv2.LINE_AA)
                cv2.imshow(keyboard_window, keyboard_view)

            for side in ("left", "right"):
                hand = hands[side]
                if not hand:
                    continue
                color = (0, 220, 0) if side == "left" else (0, 165, 255)
                for start, end in HAND_CONNECTIONS:
                    a, b = hand[start], hand[end]
                    p1 = (round(a.x * width), round(a.y * height))
                    p2 = (round(b.x * width), round(b.y * height))
                    cv2.line(view, p1, p2, color, 2, cv2.LINE_AA)
                for landmark in hand:
                    point = (round(landmark.x * width), round(landmark.y * height))
                    cv2.circle(view, point, 3, color, -1, cv2.LINE_AA)
            for finger_id, data in measurements.items():
                x, y = map(round, data["xy"])
                pip_bend, dip_bend = data["bends"]
                is_straight = tap_detector.straight.get(finger_id, False)
                color = (0, 0, 255) if is_straight else (0, 220, 0)
                cv2.circle(view, (x, y), 7, color, -1)
                label = f"{data['key'] or '-'} P{pip_bend:.0f} D{dip_bend:.0f}"
                cv2.putText(view, label, (x + 7, y - 7), cv2.FONT_HERSHEY_SIMPLEX,
                            0.38, color, 1, cv2.LINE_AA)

            if not detection:
                status = "Finding keyboard automatically | press C for manual corners"
            else:
                status = (f"Caps Lock: {'ON' if state['caps_lock'] else 'OFF'} | "
                          f"Straight + still (3 frames; cursor +1/2 key) | "
                          f"PIP+DIP <= {straightness_degrees:.0f} deg | "
                          f"mapped fingertips: {len(measurements)}")
            cv2.putText(view, status, (10, 25), cv2.FONT_HERSHEY_SIMPLEX,
                        0.58, (0, 255, 0), 2, cv2.LINE_AA)
            if detection and "keyboard_luminance" in state:
                cv2.putText(view,
                            f"Camera exposure metered on keyboard: "
                            f"{state['keyboard_luminance']:.0f}/255",
                            (10, 50), cv2.FONT_HERSHEY_SIMPLEX,
                            0.48, (0, 255, 255), 1, cv2.LINE_AA)
            if state["last_key"] and time.monotonic() - state["last_event_time"] < 2.0:
                cv2.putText(view, f"KEY TYPED: {state['last_key']}", (10, 78),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2, cv2.LINE_AA)
            cv2.putText(view, f"Text: {state['typed'][-55:]}", (10, height - 15),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)
            if state["selecting"]:
                cv2.putText(view, f"Click corner {len(state['points']) + 1}/4",
                            (10, height - 45), cv2.FONT_HERSHEY_SIMPLEX,
                            0.7, (255, 0, 255), 2, cv2.LINE_AA)
            elif state["calibration"] is None and state["auto_candidate"] is not None:
                cv2.polylines(view,
                              [state["auto_candidate"].astype(np.int32).reshape(-1, 1, 2)],
                              True, (255, 0, 255), 2)
                cv2.putText(view, "Checking keyboard corners...", (10, height - 45),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 2, cv2.LINE_AA)
            elif state["calibration"] is not None:
                cv2.putText(view, "Drag numbered corners to adjust | C: recalibrate",
                            (10, height - 45), cv2.FONT_HERSHEY_SIMPLEX,
                            0.52, (255, 0, 255), 2, cv2.LINE_AA)
            for i, (x, y) in enumerate(state["points"]):
                pixel = (int(round(x)), int(round(y)))
                color = (0, 255, 255) if i == state["drag_corner"] else (255, 0, 255)
                cv2.circle(view, pixel, 9, color, -1)
                cv2.circle(view, pixel, 14, color, 2, cv2.LINE_AA)
                cv2.putText(view, str(i + 1), (pixel[0] + 8, pixel[1] - 8),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            cv2.imshow(camera_window, view)

            pressed = cv2.waitKey(1) & 0xFF
            try:
                if cv2.getWindowProperty(camera_window, cv2.WND_PROP_VISIBLE) < 1:
                    break
            except cv2.error:
                break
            if pressed == ord("c"):
                state["points"] = []
                state["calibration"] = None
                state["selecting"] = True
                state["auto_candidate"] = None
                state["auto_candidate_hits"] = 0
                state["drag_corner"] = None
                tap_detector.straight.clear()
                tap_detector.positions.clear()
                tap_detector.active_keys.clear()
                tap_detector.last_pressed_position.clear()
                print("Click keyboard corners TL, TR, BR, BL.")
            elif pressed == ord("r"):
                tap_detector.straight.clear()
                tap_detector.positions.clear()
                tap_detector.active_keys.clear()
                tap_detector.last_pressed_position.clear()
                print("Straight-finger event states reset.")
            elif pressed in (ord("q"), 27):
                break
    finally:
        landmarker.close()
        if capture is not None:
            capture.release()
        cv2.destroyAllWindows()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--camera", type=int, default=1,
                        help="camera index that sees the keyboard and hands (default: 1)")
    parser.add_argument("--confidence", type=float, default=0.55,
                        help="MediaPipe detection confidence")
    parser.add_argument("--straightness-deg", type=float, default=20.0,
                        help="maximum bend at each of the PIP/DIP joints (default: 20 degrees)")
    parser.add_argument("--finger", choices=("all", "indexes-only"), default="all",
                        help="which fingertips to track (default: all)")
    parser.add_argument("--no-system-output", action="store_true",
                        help="show detected keys without sending them to the focused application")
    parser.add_argument("--text-only", "--filter-non-text", dest="text_only",
                        action="store_true",
                        help="ignore non-text keys in output; keep characters and Space")
    args = parser.parse_args()
    run(args.camera, args.confidence, args.straightness_deg, args.finger,
        not args.no_system_output, args.text_only)


if __name__ == "__main__":
    main()
