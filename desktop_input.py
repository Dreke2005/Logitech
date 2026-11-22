"""Windows mouse and shortcut output for gaze and navigation gestures."""

import ctypes as ct
import time
from ctypes import wintypes as wt

MOVE, WHEEL, HWHEEL, ABSOLUTE = 0x0001, 0x0800, 0x1000, 0x8000
BUTTONS = {"left": (0x0002, 0x0004)}
KEYS = {"tab": 0x09, "shift": 0x10, "ctrl": 0x11, "alt": 0x12, "a": 0x41}
MODIFIERS = {"shift", "ctrl", "alt"}
KEYUP = 0x0002
WHEEL_DELTA = 120


class _Mouse(ct.Structure):
    _fields_ = [("dx", wt.LONG), ("dy", wt.LONG), ("data", wt.DWORD),
                ("flags", wt.DWORD), ("time", wt.DWORD), ("extra", ct.c_size_t)]


class _Keyboard(ct.Structure):
    _fields_ = [("key", wt.WORD), ("scan", wt.WORD), ("flags", wt.DWORD),
                ("time", wt.DWORD), ("extra", ct.c_size_t)]


class _Payload(ct.Union):
    # These are the largest two members of Win32's INPUT union.
    _fields_ = [("mouse", _Mouse), ("keyboard", _Keyboard)]


class _Input(ct.Structure):
    _fields_ = [("kind", wt.DWORD), ("payload", _Payload)]


def _mouse(flags, dx=0, dy=0, data=0):
    return _Input(0, _Payload(mouse=_Mouse(dx, dy, data & 0xFFFFFFFF, flags, 0, 0)))


def _key(code, release=False):
    return _Input(1, _Payload(keyboard=_Keyboard(code, 0, KEYUP if release else 0, 0, 0)))


class WindowsInput:
    """Send mouse and key events to the current Windows desktop."""

    def __init__(self, user32=None):
        self._user32 = user32 if user32 is not None else ct.WinDLL("user32", use_last_error=True)
        self._user32.SendInput.argtypes = (wt.UINT, ct.POINTER(_Input), ct.c_int)
        self._user32.SendInput.restype = wt.UINT
        self._user32.GetAsyncKeyState.argtypes = (ct.c_int,)
        self._user32.GetAsyncKeyState.restype = ct.c_short
        self._shortcuts = {}
        self._navigation = None

    def _send(self, events, recovery=()):
        if not events:
            return
        events_array = (_Input * len(events))(*events)
        sent = self._user32.SendInput(len(events_array), events_array, ct.sizeof(_Input))
        if sent != len(events_array):
            # A partly inserted shortcut can leave a key down.
            if sent and recovery:
                recovery_array = (_Input * len(recovery))(*recovery)
                self._user32.SendInput(len(recovery), recovery_array, ct.sizeof(_Input))
            raise OSError("Windows did not accept the desktop input. The target may be elevated.")

    def _is_held(self, code):
        return bool(self._user32.GetAsyncKeyState(code) & 0x8000)

    def poll_pause_shortcut(self):
        """Return (just pressed, currently held) for global Ctrl+Alt+F8."""
        return self._poll_shortcut(0x77)

    def poll_mode_shortcut(self):
        """Return (just pressed, still releasing) for global Ctrl+Alt+F9."""
        return self._poll_shortcut(0x78)

    def _poll_shortcut(self, function_key):
        keys_down = [self._is_held(code) for code in (KEYS["ctrl"], KEYS["alt"], function_key)]
        held = all(keys_down)
        was_held, releasing = self._shortcuts.get(function_key, (False, False))
        pressed = held and not was_held
        # Wait for the physical modifiers to come up before sending an event.
        releasing = held or (releasing and any(keys_down))
        self._shortcuts[function_key] = held, releasing
        return pressed, releasing

    def move_to(self, x, y):
        x, y = (round(65535 * min(1, max(0, value))) for value in (x, y))
        self._send([_mouse(MOVE | ABSOLUTE, x, y)])

    def screen_size(self):
        """Size of the primary display used by absolute pointer coordinates."""
        return self._user32.GetSystemMetrics(0), self._user32.GetSystemMetrics(1)

    def click(self, button="left", count=1):
        down, up = (_mouse(flag) for flag in BUTTONS[button])
        self._send([down, up] * count, recovery=[up])

    def _remember_navigation(self, action):
        self._navigation = (time.monotonic(), f"Sent: {action}")

    def navigation_feedback(self):
        if self._navigation is not None and time.monotonic() - self._navigation[0] < 1.25:
            return self._navigation[1]
        return None

    def scroll(self, vertical=0, horizontal=0):
        axes = [(round(notches * WHEEL_DELTA), flag)
                for notches, flag in ((vertical, WHEEL), (horizontal, HWHEEL))]
        events = [_mouse(flag, data=delta) for delta, flag in axes if delta]
        if events:
            self._send(events)
            direction = ("up" if vertical > 0 else "down") if vertical else ("right" if horizontal > 0 else "left")
            self._remember_navigation(f"scroll {direction}")

    def zoom(self, notches):
        if notches:
            code = KEYS["ctrl"]
            wheel = _mouse(WHEEL, data=round(notches * WHEEL_DELTA))
            if self._is_held(code):
                self._send([wheel])
            else:
                release = _key(code, True)
                self._send([_key(code), wheel, release], recovery=[release])
            self._remember_navigation("zoom in" if notches > 0 else "zoom out")

    def hotkey(self, *keys):
        codes = []
        for name in keys:
            code = KEYS[name]
            if name not in MODIFIERS or not self._is_held(code):
                codes.append(code)
        release = [_key(code, True) for code in reversed(codes)]
        self._send([_key(code) for code in codes] + release, recovery=release)
        if keys in (("ctrl", "tab"), ("ctrl", "shift", "tab")):
            self._remember_navigation("previous tab" if "shift" in keys else "next tab")
        elif keys == ("ctrl", "shift", "a"):
            self._remember_navigation("tab overview")
