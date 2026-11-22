import cv2
import numpy as np

# Canvas size in pixels (300 DPI A4 landscape aspect ratio)
W, H = 2970, 1650
img = np.ones((H, W, 3), dtype=np.uint8) * 255

# 1. Draw 4 ArUco markers ABOVE and BELOW the keyboard (IDs 0, 1, 2, 3)
aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
marker_size = 180
margin_x = 120
margin_y = 50

corners_pos = {
    0: (margin_x, margin_y),                                      # Top-Left (Above keyboard)
    1: (W - margin_x - marker_size, margin_y),                    # Top-Right (Above keyboard)
    2: (W - margin_x - marker_size, H - margin_y - marker_size),  # Bottom-Right (Below keyboard)
    3: (margin_x, H - margin_y - marker_size),                    # Bottom-Left (Below keyboard)
}

for marker_id, (x, y) in corners_pos.items():
    marker_img = cv2.aruco.generateImageMarker(aruco_dict, marker_id, marker_size)
    marker_bgr = cv2.cvtColor(marker_img, cv2.COLOR_GRAY2BGR)
    img[y:y+marker_size, x:x+marker_size] = marker_bgr

# 2. Define Full 78-Key Layout: (Label, Width_Multiplier)
KEYBOARD_LAYOUT = [
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

# Keyboard uses full width between left/right page margins, positioned between top & bottom beacons
kb_x0 = margin_x
kb_y0 = margin_y + marker_size + 40
kb_w = W - 2 * margin_x
kb_h = H - 2 * (margin_y + marker_size + 40)

row_h = kb_h / len(KEYBOARD_LAYOUT)
unit_w = kb_w / 15.0
pad = 10

for r_idx, row in enumerate(KEYBOARD_LAYOUT):
    y1 = int(kb_y0 + r_idx * row_h + pad)
    y2 = int(kb_y0 + (r_idx + 1) * row_h - pad)
    cur_x = kb_x0

    for label, w_mult in row:
        x1 = int(cur_x + pad)
        x2 = int(cur_x + w_mult * unit_w - pad)

        is_mod = len(label) > 1 and label != "SPACE"
        if is_mod:
            cv2.rectangle(img, (x1, y1), (x2, y2), (235, 235, 235), -1)

        cv2.rectangle(img, (x1, y1), (x2, y2), (40, 40, 40), 4)

        font_scale = 1.55 if len(label) == 1 else (1.0 if len(label) <= 4 else 0.85)
        thickness = 3 if len(label) == 1 else 2
        text_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)[0]

        tx = x1 + (x2 - x1 - text_size[0]) // 2
        ty = y1 + (y2 - y1 + text_size[1]) // 2
        cv2.putText(img, label, (tx, ty), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 0), thickness)

        cur_x += w_mult * unit_w

cv2.imwrite("paper_keyboard_full_aruco.png", img)
print("Saved 'paper_keyboard_full_aruco.png' (2970x1650). Print at 100% scale on A4/Letter.")
