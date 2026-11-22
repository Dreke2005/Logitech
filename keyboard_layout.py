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
