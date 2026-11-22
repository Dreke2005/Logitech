import time
from collections import deque

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision
from pynput.keyboard import Controller, Key

from keyboard_layout import KEYS

# ======================= CONFIG =======================
CAMERA_ID = 0                 # external camera; change if needed
FRAME_W, FRAME_H = 1280, 720
USE_MJPG = True               # set False if the image comes out gray/black

MARKER_MM = 40.0              # printed size of the BLACK square of each marker
GAP_MM = 12.0                 # gap between the key block and the markers
MIN_MARKERS = 2               # markers needed to update the plane
MAX_REPROJ_ERR = 6.0          # clear-image px (~3 mm); worse fits are ignored
H_SMOOTH = 0.25               # weight of the new homography (1 = no smoothing)

FINGERS = (8, 12, 16, 20)     # MediaPipe tips: index, middle, ring, pinky (4 = thumb)
MODE = "tap"                  # "tap" or "dwell" (press 'm' to switch live)
MAX_KEY_DIST = 6.0            # clear px; fingertip farther than this = no key
POS_ALPHA = 0.6               # fingertip position smoothing

# tap mode (speeds in mm/s measured on the keyboard plane)
MOVE_MM_S = 200               # must have moved at least this fast recently...
STILL_MM_S = 70               # ...then be nearly still (landing on the key)
REARM_MM_S = 120              # finger must move again above this to re-arm
STILL_FRAMES = 2
PEAK_FRAMES = 8
COOLDOWN_SEC = 0.15
# dwell mode
DWELL_SEC = 0.5
9
SEND_KEYS = False             # dry run at start; press 's' to enable typing

# ======================= GEOMETRY =======================
MM = 41.5 / 19.05             # clear px per mm (key pitch 41.5 px = 19.05 mm)
KEY_NAMES = list(KEYS)
ALL_QUADS = np.array([[[x0, y0], [x1, y0], [x1, y1], [x0, y1]]
                      for (x0, y0, x1, y1) in KEYS.values()], np.float32)
KB_X0, KB_Y0 = ALL_QUADS[:, :, 0].min(), ALL_QUADS[:, :, 1].min()
KB_X1, KB_Y1 = ALL_QUADS[:, :, 0].max(), ALL_QUADS[:, :, 1].max()

# Beacon centers on the paper, in mm (x right, y down), origin = center of beacon 2.
# Layout from your screenshot: 2 = top-left, 3 = top-right, 0 = bottom-right, 1 = bottom-left.
# MEASURE these once with a ruler (center to center).
BEACON_MM = {
    0: (0.0,   0.0),
    1: (300.0, 0.0),
    2: (300.0, 160.0),
    3: (0.0,   160.0),
}

# Keyboard corners in the same mm frame: TL (Esc), TR, BR, BL (Ctrl).
# MEASURE these too: distance from beacon 2's center to each keyboard corner.
MARGIN_LEFT,  MARGIN_RIGHT  = -12.0, -16.0
MARGIN_TOP,   MARGIN_BOTTOM =   6.0,  24.0

_xs = [p[0] for p in BEACON_MM.values()]
_ys = [p[1] for p in BEACON_MM.values()]
X0, X1 = min(_xs) + MARGIN_LEFT, max(_xs) - MARGIN_RIGHT
Y0, Y1 = min(_ys) + MARGIN_TOP,  max(_ys) - MARGIN_BOTTOM

KEYBOARD_MM = np.array([
    [X0, Y0],   # TL (Esc)
    [X1, Y0],   # TR
    [X1, Y1],   # BR
    [X0, Y1],   # BL
], dtype=np.float32)

_aruco = cv2.aruco.ArucoDetector(
    cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50),
    cv2.aruco.DetectorParameters())

def beacon_centers(frame):
    """frame is the MIRRORED frame. Returns {id: (x, y)} in that frame's pixels."""
    w = frame.shape[1]
    corners, ids, _ = _aruco.detectMarkers(cv2.flip(frame, 1))  # markers decode only unmirrored
    poses = {}
    if ids is not None:
        for c, i in zip(corners, ids.flatten()):
            if int(i) in BEACON_MM:
                x, y = c.reshape(4, 2).mean(axis=0)
                poses[int(i)] = (w - 1 - x, y)   # back into mirrored coordinates
    return poses


def keyboard_corners_from_poses(poses):
    """poses: {id: (x, y)}. Returns 4x2 (TL, TR, BR, BL) in image px, or None."""
    ids = [i for i in BEACON_MM if i in poses]
    if len(ids) < 4:
        return None
    centers = np.array([poses[i] for i in ids], np.float32)
    paper = np.array([BEACON_MM[i] for i in ids], np.float32)
    Hp = cv2.getPerspectiveTransform(paper, centers)     # paper mm -> image px
    return cv2.perspectiveTransform(KEYBOARD_MM[None], Hp)[0]

def keyboard_corners_from_beacons(cache):
    """cache: {id: (4x2 corners, last_seen)}. Returns 4x2 keyboard corners
    (TL, TR, BR, BL) in image pixels, or None if fewer than 4 beacons."""
    ids = [i for i in BEACON_MM if i in cache]
    if len(ids) < 4:
        return None

    centers = np.array([cache[i][0].mean(axis=0) for i in ids], np.float32)  # image px
    paper = np.array([BEACON_MM[i] for i in ids], np.float32)                # mm

    H = cv2.getPerspectiveTransform(paper, centers)   # paper mm -> image px
    return cv2.perspectiveTransform(KEYBOARD_MM[None], H)[0]

def build_markers():
    s, g = MARKER_MM * MM, GAP_MM * MM
    xc = (KB_X0 + KB_X1) / 2
    top, bot = KB_Y0 - g - s, KB_Y1 + g
    tl = {0: (KB_X0, top), 1: (xc - s / 2, top), 2: (KB_X1 - s, top),
          3: (KB_X0, bot), 4: (xc - s / 2, bot), 5: (KB_X1 - s, bot)}
    # corner order = ArUco order: TL, TR, BR, BL
    return {i: np.array([[x, y], [x + s, y], [x + s, y + s], [x, y + s]], np.float32)
            for i, (x, y) in tl.items()}

MARKERS = build_markers()

# rectified (top-down) view bounds
_pts = np.vstack([ALL_QUADS.reshape(-1, 2)] + list(MARKERS.values()))
OX, OY = _pts.min(0) - 20
VW, VH = (_pts.max(0) + 20) - np.array([OX, OY])
VW, VH = int(VW), int(VH)
VIEW_T = np.array([[1, 0, -OX], [0, 1, -OY], [0, 0, 1]], np.float64)

def closest_key(x, y):
    best, best_d = None, 1e9
    for name, (x0, y0, x1, y1) in KEYS.items():
        d = np.hypot(max(x0 - x, 0, x - x1), max(y0 - y, 0, y - y1))
        if d < best_d:
            best, best_d = name, d
    return best if best_d <= MAX_KEY_DIST else None

# ======================= ARUCO PLANE =======================
aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
aruco_params = cv2.aruco.DetectorParameters()
aruco_params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
detector = cv2.aruco.ArucoDetector(aruco_dict, aruco_params)

HOLD_SEC = 1.0   # keep a marker's last position this long after it disappears

class PlaneTracker:
    """Maintains H: camera image -> keyboard ('clear') coordinates."""
    def __init__(self):
        self.reset()

    def reset(self):
        self.H, self.err, self.seen = None, None, []
        self.cache = {}                      # id -> (corners 4x2, last_seen_time)

    def update(self, frame, now):
        corners, ids, _ = detector.detectMarkers(frame)
        if ids is not None:
            for c, i in zip(corners, ids.flatten()):
                if int(i) in MARKERS:
                    self.cache[int(i)] = (c.reshape(4, 2).copy(), now)
        # drop markers not seen for HOLD_SEC
        self.cache = {i: v for i, v in self.cache.items() if now - v[1] <= HOLD_SEC}
        self.seen = sorted(self.cache)

        if len(self.cache) >= MIN_MARKERS:
            src = np.array([p for i in self.seen for p in self.cache[i][0]], np.float32)
            dst = np.array([p for i in self.seen for p in MARKERS[i]], np.float32)
            H, _ = cv2.findHomography(src, dst, cv2.RANSAC, 3.0)
            if H is not None:
                proj = cv2.perspectiveTransform(src[None], H)[0]
                err = float(np.linalg.norm(proj - dst, axis=1).mean())
                if err < MAX_REPROJ_ERR:
                    H = H / H[2, 2]
                    self.H = H if self.H is None else H_SMOOTH * H + (1 - H_SMOOTH) * self.H
                    self.err = err
        else:
            self.H = None if not self.cache and self.H is not None and False else self.H
        return self.cache

# ======================= FINGER LOGIC =======================
class Finger:
    def __init__(self):
        self.p = None
        self.t = 0.0
        self.v = 0.0
        self.peaks = deque([0.0], maxlen=PEAK_FRAMES)
        self.armed = True
        self.still_n = 0
        self.last_fire = 0.0
        self.key, self.t0 = None, 0.0

    def update(self, x, y, now):
        new = np.array([x, y], np.float64)
        if self.p is None:
            self.p, self.t = new, now
            return
        q = POS_ALPHA * new + (1 - POS_ALPHA) * self.p
        inst = np.linalg.norm(q - self.p) / MM / max(now - self.t, 1e-3)
        self.v = 0.5 * inst + 0.5 * self.v
        self.peaks.append(self.v)
        self.p, self.t = q, now

    def should_fire(self, key, now):
        if MODE == "dwell":
            if key != self.key:
                self.key, self.t0, self.armed = key, now, True
                return False
            if key and self.armed and now - self.t0 >= DWELL_SEC:
                self.armed = False
                return True
            return False
        if self.v > REARM_MM_S:
            self.armed = True
        self.still_n = self.still_n + 1 if self.v < STILL_MM_S else 0
        if (self.armed and key and self.still_n >= STILL_FRAMES
                and max(self.peaks) > MOVE_MM_S and now - self.last_fire > COOLDOWN_SEC):
            self.armed, self.last_fire = False, now
            self.peaks.clear()
            self.peaks.append(0.0)
            return True
        return False

# ======================= KEYSTROKES =======================
kbd = Controller()
SPECIAL = {
    "Esc": Key.esc, "Tab": Key.tab, "CapsLock": Key.caps_lock, "Enter": Key.enter,
    "KP_Enter": Key.enter, "Backspace": Key.backspace, "Delete": Key.delete,
    "Insert": Key.insert, "Home": Key.home, "End": Key.end, "PgUp": Key.page_up,
    "PgDn": Key.page_down, "Up": Key.up, "Down": Key.down, "Left": Key.left,
    "Right": Key.right, "Space": Key.space, "PrtSc": Key.print_screen,
    "ScrLk": Key.scroll_lock, "Pause": Key.pause, "NumLock": Key.num_lock,
    "Menu": Key.menu, **{f"F{i}": getattr(Key, f"f{i}") for i in range(1, 13)},
}
MODS = {"LShift": Key.shift_l, "RShift": Key.shift_r, "LCtrl": Key.ctrl_l,
        "RCtrl": Key.ctrl_r, "LAlt": Key.alt_l, "AltGr": Key.alt_gr,
        "LWin": Key.cmd_l, "RWin": Key.cmd_r}
sticky = set()

def resolve(name):
    if name in SPECIAL:
        return SPECIAL[name]
    if name.startswith("KP_"):
        return name[3:].lower()
    if name.startswith("Special"):
        return None
    return name.lower() if len(name) == 1 else None

def send(name):
    if name in MODS:                       # modifiers are sticky: apply to next key
        sticky.symmetric_difference_update({name})
        return
    k = resolve(name)
    if k is None:
        return
    held = [MODS[m] for m in sticky]
    for m in held:
        kbd.press(m)
    kbd.press(k)
    kbd.release(k)
    for m in reversed(held):
        kbd.release(m)
    sticky.clear()

# ======================= DRAWING =======================
ASCII = {"²": "2s", "é": "e'", "è": "e`", "ç": "c,", "à": "a`", "ù": "u`", "µ": "mu"}

def draw_camera(frame, H, hover, flash, now):
    quads = cv2.perspectiveTransform(ALL_QUADS.reshape(1, -1, 2),
                                     np.linalg.inv(H))[0].reshape(-1, 4, 2).astype(np.int32)
    for name, q in zip(KEY_NAMES, quads):
        if now - flash.get(name, -9) < 0.25:
            cv2.fillPoly(frame, [q], (0, 255, 0))
        cv2.polylines(frame, [q], True, (0, 255, 255) if name in hover else (0, 160, 0), 1)

def draw_view(frame, H, tips, hover, flash, now):
    view = cv2.warpPerspective(frame, VIEW_T @ H, (VW, VH))
    for name, (x0, y0, x1, y1) in KEYS.items():
        a, b = (int(x0 - OX), int(y0 - OY)), (int(x1 - OX), int(y1 - OY))
        if now - flash.get(name, -9) < 0.25:
            cv2.rectangle(view, a, b, (0, 255, 0), -1)
        cv2.rectangle(view, a, b, (0, 255, 255) if name in hover else (0, 160, 0), 1)
        lab = ASCII.get(name, name if name.isascii() else "")
        cv2.putText(view, lab, (a[0] + 3, b[1] - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 255), 1)
    for (x, y) in tips:
        cv2.circle(view, (int(x - OX), int(y - OY)), 6, (255, 0, 0), -1)
    return view

# ======================= MAIN =======================
def open_camera(idx):
    cap = cv2.VideoCapture(idx, cv2.CAP_DSHOW)   # remove CAP_DSHOW on macOS/Linux
    if not cap.isOpened():
        raise RuntimeError(f"Camera {idx} failed to open")
    if USE_MJPG:
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_W)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_H)
    for _ in range(30):                          # discard gray warm-up frames
        cap.read()
    return cap

def print_markers():
    for i in MARKERS:
        img = cv2.aruco.generateImageMarker(aruco_dict, i, 400)
        img = cv2.copyMakeBorder(img, 50, 50, 50, 50, cv2.BORDER_CONSTANT, value=255)
        cv2.imwrite(f"marker_{i}.png", img)
    print(f"Wrote marker_0..{len(MARKERS) - 1}.png - print so the BLACK square is {MARKER_MM:.0f} mm.")

def main2():
    poses = []
    global MODE, SEND_KEYS
    import sys
    if "--print-markers" in sys.argv:
        print_markers()
        return

    cap = open_camera(CAMERA_ID)
    landmarker = vision.HandLandmarker.create_from_options(vision.HandLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_path="hand_landmarker.task"),
        running_mode=vision.RunningMode.VIDEO, num_hands=2,
        min_hand_detection_confidence=0.5, min_tracking_confidence=0.5))

    plane, fingers, flash = PlaneTracker(), {}, {}
    prev_ts, fps, t_prev = 0, 0.0, time.time()
    cv2.namedWindow("Camera", cv2.WINDOW_NORMAL); cv2.resizeWindow("Camera", 960, 540)
    cv2.namedWindow("Keyboard (rectified)", cv2.WINDOW_NORMAL); cv2.resizeWindow("Keyboard (rectified)", 800, int(800 * VH / VW))

    while True:
        ok, frame = cap.read()
        if not ok or frame is None:
            print("Camera read failed"); break
        now = time.time()
        fps = 0.9 * fps + 0.1 / max(now - t_prev, 1e-3); t_prev = now

        cache = plane.update(frame, now)
        for i, (c, _) in cache.items():
            cv2.polylines(frame, [c.astype(np.int32)], True, (0, 255, 0), 2)
            cv2.putText(frame, f"id={i}", tuple(c[0].astype(int)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
        if len(cache) == 4:
            return keyboard_corners_from_beacons(poses)  # hypothetical function to compute keyboard corners

        hover, tips_clear = set(), []
        if plane.H is not None:
            h, w = frame.shape[:2]
            ts = max(int(now * 1000), prev_ts + 1); prev_ts = ts
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            res = landmarker.detect_for_video(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb), ts)

            alive, used = set(), set()
            for hi, lms in enumerate(res.hand_landmarks):
                label = res.handedness[hi][0].category_name
                if label in used:
                    label += str(hi)
                used.add(label)
                for fid in FINGERS:
                    lm = lms[fid]
                    px, py = lm.x * w, lm.y * h
                    cx, cy = cv2.perspectiveTransform(
                        np.array([[[px, py]]], np.float32), plane.H)[0, 0]
                    fk = (label, fid)
                    alive.add(fk)
                    f = fingers.setdefault(fk, Finger())
                    f.update(cx, cy, now)
                    key = closest_key(*f.p)
                    tips_clear.append(tuple(f.p))
                    cv2.circle(frame, (int(px), int(py)), 6, (255, 0, 0), -1)
                    if key:
                        hover.add(key)
                    if f.should_fire(key, now):
                        flash[key] = now
                        print(f"TAP {key}  ({'sent' if SEND_KEYS else 'dry run'})")
                        if SEND_KEYS:
                            send(key)
            for fk in list(fingers):
                if fk not in alive:
                    del fingers[fk]

            draw_camera(frame, plane.H, hover, flash, now)
            cv2.imshow("Keyboard (rectified)", draw_view(frame, plane.H, tips_clear, hover, flash, now))

        status = (f"markers {sorted(plane.seen)}  err "
                  f"{'-' if plane.err is None else f'{plane.err:.1f}px'}  mode {MODE}  "
                  f"keys {'ON' if SEND_KEYS else 'OFF'}  {fps:.0f}fps")
        cv2.putText(frame, status, (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                    (0, 220, 0) if plane.H is not None else (0, 0, 255), 2)
        cv2.imshow("Camera", frame)

        k = cv2.waitKey(1) & 0xFF
        if k == 27:
            break
        elif k == ord("s"):
            SEND_KEYS = not SEND_KEYS
        elif k == ord("m"):
            MODE = "dwell" if MODE == "tap" else "tap"
        elif k == ord("r"):
            plane.reset()

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main2()