import time
import cv2
import numpy as np
import mediapipe as mp
from pynput.keyboard import Controller, Key
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# --- CONFIGURATION ---
CAM_LAPTOP_ID = 0       # Usually 0 for built-in webcam
CAM_LOGITECH_ID = 1     # Usually 1 for external USB webcam
FRAME_W, FRAME_H = 640, 480

# Canonical top-down warped plane dimensions (matches 2970x1650 template at scale S = 0.5)
WARP_W, WARP_H = 1485, 825

# Stereo touch detection tuning
TOUCH_DISPARITY_PX = 14.0   # Max pixel distance between Cam 1 & Cam 2 warped fingertip to count as touching
RELEASE_DISPARITY_PX = 24.0 # Hysteresis: finger must lift above this disparity before re-triggering
COOLDOWN_SEC = 0.25         # Minimum time between repeated taps on the same finger

# --- INITIALIZE HARDWARE & MODELS ---
keyboard = Controller()

def open_camera(idx):
    cap = cv2.VideoCapture(idx, cv2.CAP_DSHOW)  # Remove cv2.CAP_DSHOW on macOS/Linux
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_W)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_H)
    cap.set(cv2.CAP_PROP_FPS, 30)
    return cap

cap1 = open_camera(CAM_LAPTOP_ID)
cap2 = open_camera(CAM_LOGITECH_ID)

aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
aruco_params = cv2.aruco.DetectorParameters()
detector = cv2.aruco.ArucoDetector(aruco_dict, aruco_params)

base_options = python.BaseOptions(model_asset_path="hand_landmarker.task")
options = vision.HandLandmarkerOptions(
    base_options=base_options,
    running_mode=vision.RunningMode.VIDEO,
    num_hands=2,
    min_hand_detection_confidence=0.6,
    min_tracking_confidence=0.6,
)
hands1 = vision.HandLandmarker.create_from_options(options)
hands2 = vision.HandLandmarker.create_from_options(options)

# Destination outer corners of the 4 ArUco markers on the canonical plane
S = 0.5
MX, MY = int(120 * S), int(50 * S)
DST_CORNERS = {
    0: np.array([MX, MY], dtype=np.float32),                    # Top-Left outer corner of ID 0
    1: np.array([WARP_W - MX, MY], dtype=np.float32),           # Top-Right outer corner of ID 1
    2: np.array([WARP_W - MX, WARP_H - MY], dtype=np.float32),  # Bottom-Right outer corner of ID 2
    3: np.array([MX, WARP_H - MY], dtype=np.float32),           # Bottom-Left outer corner of ID 3
}

def build_key_zones():
    zones = []
    kb_x0 = 120 * S
    kb_y0 = (50 + 180 + 40) * S
    kb_w = (2970 - 2 * 120) * S
    kb_h = (1650 - 2 * (50 + 180 + 40)) * S

    layout = [
        [("ESC", 1.0), ("F1", 1.0), ("F2", 1.0), ("F3", 1.0), ("F4", 1.0), ("F5", 1.0), ("F6", 1.0),
         ("F7", 1.0), ("F8", 1.0), ("F9", 1.0), ("F10", 1.0), ("F11", 1.0), ("F12", 1.0), ("DEL", 2.0)],
        [("`", 1.0), ("1", 1.0), ("2", 1.0), ("3", 1.0), ("4", 1.0), ("5", 1.0), ("6", 1.0),
         ("7", 1.0), ("8", 1.0), ("9", 1.0), ("0", 1.0), ("-", 1.0), ("=", 1.0), ("BACK", 2.0)],
        [("TAB", 1.5), ("Q", 1.0), ("W", 1.0), ("E", 1.0), ("R", 1.0), ("T", 1.0), ("Y", 1.0),
         ("U", 1.0), ("I", 1.0), ("O", 1.0), ("P", 1.0), ("[", 1.0), ("]", 1.0), ("\\", 1.5)],
        [("CAPS", 1.75), ("A", 1.0), ("S", 1.0), ("D", 1.0), ("F", 1.0), ("G", 1.0), ("H", 1.0),
         ("J", 1.0), ("K", 1.0), ("L", 1.0), (";", 1.0), ("'", 1.0), ("ENTER", 2.25)],
        [("SHIFT", 2.25), ("Z", 1.0), ("X", 1.0), ("C", 1.0), ("V", 1.0), ("B", 1.0), ("N", 1.0),
         ("M", 1.0), (",", 1.0), (".", 1.0), ("/", 1.0), ("UP", 1.0), ("RSHIFT", 1.75)],
        [("CTRL", 1.25), ("WIN", 1.25), ("ALT", 1.25), ("SPACE", 6.25), ("RALT", 1.0),
         ("LEFT", 1.0), ("DOWN", 1.0), ("RIGHT", 1.0), ("RCTRL", 1.0)]
    ]

    row_h = kb_h / len(layout)
    unit_w = kb_w / 15.0
    pad = 10 * S

    for r_idx, row in enumerate(layout):
        y1 = kb_y0 + r_idx * row_h + pad
        y2 = kb_y0 + (r_idx + 1) * row_h - pad
        cur_x = kb_x0
        for label, w_mult in row:
            x1 = cur_x + pad
            x2 = cur_x + w_mult * unit_w - pad
            zones.append((label, (x1, y1, x2, y2)))
            cur_x += w_mult * unit_w
    return zones

KEY_ZONES = build_key_zones()

def get_key_at(x, y):
    for key, (x1, y1, x2, y2) in KEY_ZONES:
        if x1 <= x <= x2 and y1 <= y <= y2:
            return key
    return None

SPECIAL_KEYS = {
    "ESC": Key.esc, "TAB": Key.tab, "CAPS": Key.caps_lock,
    "SHIFT": Key.shift, "RSHIFT": Key.shift_r,
    "CTRL": Key.ctrl, "RCTRL": Key.ctrl_r,
    "ALT": Key.alt, "RALT": Key.alt_r, "WIN": Key.cmd,
    "SPACE": Key.space, "ENTER": Key.enter, "BACK": Key.backspace, "DEL": Key.delete,
    "UP": Key.up, "DOWN": Key.down, "LEFT": Key.left, "RIGHT": Key.right,
    "F1": Key.f1, "F2": Key.f2, "F3": Key.f3, "F4": Key.f4,
    "F5": Key.f5, "F6": Key.f6, "F7": Key.f7, "F8": Key.f8,
    "F9": Key.f9, "F10": Key.f10, "F11": Key.f11, "F12": Key.f12,
}

def send_keystroke(key_label):
    if key_label in SPECIAL_KEYS:
        k = SPECIAL_KEYS[key_label]
        keyboard.press(k)
        keyboard.release(k)
    else:
        keyboard.press(key_label.lower())
        keyboard.release(key_label.lower())

def compute_homography_and_preview(frame, prev_H, cam_label):
    preview = frame.copy()
    corners, ids, _ = detector.detectMarkers(frame)
    detected_map = {}

    if ids is not None:
        # Draw detected ArUco borders and IDs on the camera preview
        cv2.aruco.drawDetectedMarkers(preview, corners, ids)
        for marker_corners, marker_id in zip(corners, ids.flatten()):
            if marker_id in DST_CORNERS:
                detected_map[int(marker_id)] = marker_corners[0][marker_id]

    found_count = len(detected_map)
    if found_count == 4:
        # Order corners 0 -> 1 -> 2 -> 3 to draw the outer keyboard boundary box
        ordered_src = np.array([detected_map[i] for i in [0, 1, 2, 3]], dtype=np.float32)
        ordered_dst = np.array([DST_CORNERS[i] for i in [0, 1, 2, 3]], dtype=np.float32)
        H, _ = cv2.findHomography(ordered_src, ordered_dst)

        cv2.polylines(preview, [ordered_src.astype(np.int32)], isClosed=True, color=(0, 255, 0), thickness=2)
        cv2.putText(preview, f"{cam_label}: READY (4/4)", (15, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 220, 0), 2)
        return H, True, preview
    else:
        missing = [i for i in [0, 1, 2, 3] if i not in detected_map]
        cv2.putText(preview, f"{cam_label}: MISSING IDs {missing}", (15, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        return prev_H, False, preview

def get_fingertips(frame, preview, landmarker, H, timestamp_ms, dot_color):
    if H is None:
        return []
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    res = landmarker.detect_for_video(mp_image, timestamp_ms)

    tips = []
    if res.hand_landmarks:
        for hand_landmarks in res.hand_landmarks:
            # Landmark 8 = Index Fingertip
            lm = hand_landmarks[8]
            px, py = lm.x * FRAME_W, lm.y * FRAME_H
            # Draw fingertip dot on raw camera preview
            cv2.circle(preview, (int(px), int(py)), 6, dot_color, -1)

            pt = np.array([[[px, py]]], dtype=np.float32)
            warped_pt = cv2.perspectiveTransform(pt, H)[0][0]
            tips.append(warped_pt)
    return tips

H1, H2 = None, None
is_pressed = [False, False]
last_tap_time = [0.0, 0.0]
prev_ts = 0

print("Starting Stereo Paper Keyboard. Use the Camera Alignment Preview window to frame all 4 markers...")

while True:
    # 1. Synchronized capture
    cap1.grab()
    cap2.grab()
    ret1, frame1 = cap1.retrieve()
    ret2, frame2 = cap2.retrieve()
    if not ret1 or not ret2:
        print("Failed to read from one or both cameras. Check CAM_LAPTOP_ID and CAM_LOGITECH_ID.")
        break

    # 2. Update Homographies & build annotated raw camera previews
    H1, lock1, prev1 = compute_homography_and_preview(frame1, H1, "Cam 1 (Laptop)")
    H2, lock2, prev2 = compute_homography_and_preview(frame2, H2, "Cam 2 (Logitech)")

    # 3. Extract and warp index fingertips to canonical plane
    timestamp_ms = int(time.time() * 1000)
    if timestamp_ms <= prev_ts:
        timestamp_ms = prev_ts + 1
    prev_ts = timestamp_ms

    tips1 = get_fingertips(frame1, prev1, hands1, H1, timestamp_ms, (255, 0, 0))
    tips2 = get_fingertips(frame2, prev2, hands2, H2, timestamp_ms, (0, 0, 255))

    # Show side-by-side raw camera alignment window
    combined_preview = np.hstack((prev1, prev2))
    cv2.imshow("1. Camera Alignment Preview (Cam 1 | Cam 2)", combined_preview)

    # 4. Create top-down debug canvas showing the warped keyboard & stereo disparity
    canvas = np.ones((WARP_H, WARP_W, 3), dtype=np.uint8) * 240
    for key, (x1, y1, x2, y2) in KEY_ZONES:
        cv2.rectangle(canvas, (int(x1), int(y1)), (int(x2), int(y2)), (180, 180, 180), 2)
        cv2.putText(canvas, key, (int(x1 + 10), int(y2 - 15)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (80, 80, 80), 2)

    for p1 in tips1:
        cv2.circle(canvas, (int(p1[0]), int(p1[1])), 8, (255, 0, 0), -1)
    for p2 in tips2:
        cv2.circle(canvas, (int(p2[0]), int(p2[1])), 8, (0, 0, 255), -1)

    # 5. Match nearest fingertips across cameras & compute stereo disparity
    now = time.time()
for idx, p1 in enumerate(tips1[:2]):
    if not tips2:
        continue
    dists = [np.linalg.norm(p1 - p2) for p2 in tips2]
    min_idx = int(np.argmin(dists))
    disparity = dists[min_idx]
    p2 = tips2[min_idx]
    mid_pt = (p1 + p2) / 2.0
    mx, my = float(mid_pt[0]), float(mid_pt[1])

    color = (0, 200, 0) if disparity < TOUCH_DISPARITY_PX else (0, 165, 255)
    cv2.line(canvas, (int(p1[0]), int(p1[1])), (int(p2[0]), int(p2[1])), color, 2)
    cv2.putText(canvas, f"d={disparity:.1f}px", (int(mx + 10), int(my)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    if disparity < TOUCH_DISPARITY_PX and not is_pressed[idx] and (now - last_tap_time[idx] > COOLDOWN_SEC):
        pressed_key = get_key_at(mx, my)
        if pressed_key:
            print(f"TAP: {pressed_key} (disparity={disparity:.1f}px)")
            send_keystroke(pressed_key)
            is_pressed[idx] = True
            last_tap_time[idx] = now
            for k, (x1, y1, x2, y2) in KEY_ZONES:
                if k == pressed_key:
                    cv2.rectangle(canvas, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), -1)

    elif disparity > RELEASE_DISPARITY_PX:
        is_pressed[idx] = False

    status = f"Cam1 Lock: {lock1} | Cam2 Lock: {lock2}"
    cv2.putText(canvas, status, (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
    cv2.imshow("2. Stereo Paper Keyboard & Disparity", canvas)

    if cv2.waitKey(1) & 0xFF == 27:  # ESC to quit
        break

cap1.release()
cap2.release()
cv2.destroyAllWindows()