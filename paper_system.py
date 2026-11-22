"""The keypad's stock keys on the start page: media, volume, emoji, screenshot, Explorer, Options+, Ask ChatGPT.

The keypad plugin performs these itself (they must work while the workbench is closed); this
module does the same for the workbench window's buttons and for ``--slot``.
"""

import ctypes
import os
import subprocess
import time
import webbrowser
from pathlib import Path
from threading import Thread

KEYUP, UNICODE = 0x0002, 0x0004
VK = {"win": 0x5B, "shift": 0x10, "ctrl": 0x11, "enter": 0x0D, "c": 0x43, "v": 0x56, "s": 0x53, "period": 0xBE,
      "next": 0xB0, "playpause": 0xB3, "mute": 0xAD}
OPTIONS_PLUS = Path(os.environ.get("ProgramFiles", r"C:\Program Files")) / "LogiOptionsPlus" / "logioptionsplus.exe"
CHATGPT_PROMPT = "Please write a draft reply to the prompt below:"  # The stock Ask ChatGPT macro's text.


def _key(code, up=False):
    ctypes.windll.user32.keybd_event(code, 0, KEYUP if up else 0, 0)


def press(*names):
    """Press a key combination, e.g. press("win", "shift", "s")."""
    codes = [VK[name] for name in names]
    for code in codes:
        _key(code)
    for code in reversed(codes):
        _key(code, up=True)


def type_text(text):
    """Type text as Unicode key events, whatever the keyboard layout."""

    class KeyInput(ctypes.Structure):
        _fields_ = [("wVk", ctypes.c_ushort), ("wScan", ctypes.c_ushort), ("dwFlags", ctypes.c_ulong),
                    ("time", ctypes.c_ulong), ("dwExtraInfo", ctypes.c_size_t)]

    class Input(ctypes.Structure):
        class _Union(ctypes.Union):
            _fields_ = [("ki", KeyInput), ("padding", ctypes.c_byte * 32)]
        _anonymous_ = ("u",)
        _fields_ = [("type", ctypes.c_ulong), ("u", _Union)]

    for char in text:
        for flags in (UNICODE, UNICODE | KEYUP):
            event = Input(type=1)
            event.ki = KeyInput(0, ord(char), flags, 0, 0)
            ctypes.windll.user32.SendInput(1, ctypes.byref(event), ctypes.sizeof(Input))


def ask_chatgpt():
    """Copy the selection, open ChatGPT, and paste it after the stock prompt, as the stock macro does."""
    press("ctrl", "c")
    time.sleep(0.15)
    webbrowser.open("https://chatgpt.com")
    time.sleep(1.5)
    type_text(CHATGPT_PROMPT)
    press("shift", "enter")
    press("ctrl", "v")
    press("enter")


ACTIONS = {
    "media_next": lambda: press("next"),
    "media_playpause": lambda: press("playpause"),
    "volume_mute": lambda: press("mute"),
    "emoji": lambda: press("win", "period"),
    "screenshot": lambda: press("win", "shift", "s"),
    "explorer": lambda: subprocess.Popen(["explorer.exe"]),
    "optionsplus": lambda: subprocess.Popen([str(OPTIONS_PLUS)], cwd=OPTIONS_PLUS.parent),
    "ask_chatgpt": lambda: Thread(target=ask_chatgpt, daemon=True).start(),
}


def run_stock(action):
    """Perform a stock key. Returns False for an unknown key."""
    handler = ACTIONS.get(action)
    if handler is None:
        return False
    handler()
    return True
