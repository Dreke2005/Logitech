import numpy as np
import cv2

# ---------------------------------------------------------------
# 1. Key rectangles in the CLEAR image (x0, y0, x1, y1)
#    (estimated by eye from the 950px-wide clear image; tweak if needed)
# ---------------------------------------------------------------
KEYS = {}

F_Y, N_Y, R2_Y, R3_Y, R4_Y, R5_Y = (10, 38), (70, 102), (112, 150), (163, 197), (211, 245), (258, 292)

def box(name, x0, x1, ylim):
    KEYS[name] = (x0, ylim[0], x1, ylim[1])

def span(names, x_first, pitch, ylim, w=38):
    for i, n in enumerate(names):
        c = x_first + i * pitch
        box(n, c - w / 2, c + w / 2, ylim)

# --- Function row ---
box("Esc", 8, 45, F_Y)
for grp, centers in (("F1 F2 F3 F4", (104, 144, 184, 227)),
                     ("F5 F6 F7 F8", (290, 330, 372, 412)),
                     ("F9 F10 F11 F12", (476, 517, 559, 601))):
    for n, c in zip(grp.split(), centers):
        box(n, c - 14, c + 14, F_Y)
box("PrtSc", 645, 683, F_Y); box("ScrLk", 687, 725, F_Y); box("Pause", 728, 758, F_Y)
for i, x in enumerate((783, 826, 868, 912), 1):
    box(f"Special{i}", x, x + 32, F_Y)

# --- Number row ---
span(list("²&é\"'(-è_çà)="), 27, 41.7, N_Y)
box("Backspace", 546, 620, N_Y)
box("Insert", 645, 683, N_Y); box("Home", 687, 725, N_Y); box("PgUp", 728, 758, N_Y)
box("NumLock", 785, 815, N_Y); box("KP_/", 828, 858, N_Y)
box("KP_*", 869, 897, N_Y);    box("KP_-", 912, 944, N_Y)

# --- AZERTY row ---
box("Tab", 8, 58, R2_Y)
span(list("AZERTYUIOP") + ["^", "$"], 81, 41.5, R2_Y)
box("µ", 560, 620, R2_Y)
box("Delete", 645, 683, R2_Y); box("End", 687, 725, R2_Y); box("PgDn", 728, 758, R2_Y)
box("KP_7", 785, 815, R2_Y); box("KP_8", 828, 858, R2_Y); box("KP_9", 869, 897, R2_Y)
box("KP_+", 912, 944, (112, 197))                       # tall key

# --- QSDF row ---
box("CapsLock", 8, 70, R3_Y)
span(list("QSDFGHJKLM") + ["ù"], 94, 41.5, R3_Y)
box("Enter", 537, 620, R3_Y)
box("KP_4", 785, 815, R3_Y); box("KP_5", 828, 858, R3_Y); box("KP_6", 869, 897, R3_Y)

# --- WXCV row ---
box("LShift", 8, 56, R4_Y)
span(["<"] + list("WXCVBN") + [",", ";", ":", "!"], 75, 41.5, R4_Y)
box("RShift", 515, 620, R4_Y)
box("Up", 685, 715, R4_Y)
box("KP_1", 785, 815, R4_Y); box("KP_2", 828, 858, R4_Y); box("KP_3", 869, 897, R4_Y)
box("KP_Enter", 912, 944, (211, 292))                   # tall key

# --- Bottom row ---
box("LCtrl", 8, 52, R5_Y); box("LWin", 60, 98, R5_Y); box("LAlt", 108, 146, R5_Y)
box("Space", 160, 413, R5_Y)
box("AltGr", 425, 463, R5_Y); box("RWin", 475, 515, R5_Y)
box("Menu", 527, 567, R5_Y);  box("RCtrl", 580, 620, R5_Y)
box("Left", 645, 675, R5_Y);  box("Down", 685, 715, R5_Y); box("Right", 725, 755, R5_Y)
box("KP_0", 785, 858, R5_Y);  box("KP_.", 869, 897, R5_Y)

def center(name):
    x0, y0, x1, y1 = KEYS[name]
    return ((x0 + x1) / 2, (y0 + y1) / 2)

# ---------------------------------------------------------------
# 2. Calibration: green circles (blurry image) <-> key centers (clear image)
# ---------------------------------------------------------------
CALIB_KEYS = (list("²&é\"'(-è_çà)=")
              + list("AZERTYUIOP") + ["^", "$"]
              + list("QSDFGHJKLM") + ["ù"]
              + list("WXCVBN") + [",", ";", ":", "!"])

BLUR_PTS = (
    [(x, 72)  for x in (8, 41, 76, 111, 145, 180, 215, 250, 284, 318, 353, 388, 423)] +
    [(x, 110) for x in (60, 93, 127, 162, 197, 232, 266, 301, 335, 369, 403, 437)] +
    [(x, 155) for x in (68, 102, 136, 171, 206, 241, 275, 310, 344, 379, 413)] +
    [(x, 198) for x in (84, 119, 153, 188, 222, 257, 291, 325, 360, 395)]
)

src = np.array([center(k) for k in CALIB_KEYS], dtype=np.float32)   # clear
dst = np.array(BLUR_PTS, dtype=np.float32)                          # blurry
H, _ = cv2.findHomography(src, dst, 0)      # least squares (use cv2.RANSAC if you have outliers)
H_inv = np.linalg.inv(H)

# ---------------------------------------------------------------
# 3. Lookup
# ---------------------------------------------------------------
def to_clear(x, y):
    p = H_inv @ np.array([x, y, 1.0])
    return p[0] / p[2], p[1] / p[2]

def closest_key(x, y, max_dist=None):
    """Return (key_name, distance) for a point (x, y) in the BLURRY image.
    distance is in clear-image pixels (0 if the point is inside the key).
    If max_dist is set and the nearest key is farther, returns (None, distance)."""
    px, py = to_clear(x, y)
    best, best_d = None, float("inf")
    for name, (x0, y0, x1, y1) in KEYS.items():
        dx = max(x0 - px, 0, px - x1)
        dy = max(y0 - py, 0, py - y1)
        d = np.hypot(dx, dy)
        if d < best_d:
            best, best_d = name, d
    if max_dist is not None and best_d > max_dist:
        return None, float(best_d)
    return best, float(best_d)

# ---------------------------------------------------------------
# 4. Visual check: draw all key boxes on the blurry image
# ---------------------------------------------------------------
def draw_keys(img_path="image.png", out="check.png"):
    img = cv2.imread(img_path)
    for name, (x0, y0, x1, y1) in KEYS.items():
        pts = np.array([[x0, y0], [x1, y0], [x1, y1], [x0, y1]], dtype=np.float32)[None]
        proj = cv2.perspectiveTransform(pts, H)[0].astype(np.int32)
        cv2.polylines(img, [proj], True, (0, 255, 0), 1)
        cx, cy = proj.mean(axis=0).astype(int)
        cv2.putText(img, name, (cx - 6, cy + 3), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 0, 255), 1)
    cv2.imwrite(out, img)

if __name__ == "__main__":
    print(closest_key(205, 155))        # a home-row key
    print(closest_key(520, 228))        # Space
    print(closest_key(700, 240))        # numpad / arrow area
    print(closest_key(400, 20, max_dist=30))