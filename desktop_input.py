"""Windows mouse and shortcut output shared by the two desktop modes."""

import ctypes as ct
from ctypes import wintypes as wt

MOVE, WHEEL, HWHEEL, ABSOLUTE = 0x0001, 0x0800, 0x1000, 0x8000
BUTTONS = {"left": (0x0002, 0x0004), "right": (0x0008, 0x0010),
           "middle": (0x0020, 0x0040)}
KEYS = {"tab": 0x09, "shift": 0x10, "ctrl": 0x11, "alt": 0x12, "win": 0x5B,
        "left": 0x25, "right": 0x27, "up": 0x26, "down": 0x28,
        "plus": 0xBB, "minus": 0xBD}
MODIFIERS = {"shift", "ctrl", "alt", "win"}
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
        self._buttons = set()
        self._move_remainder = (0.0, 0.0)
        self._shortcut_was_held = False
        self._shortcut_releasing = False

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
        keys_down = [self._is_held(code) for code in (KEYS["ctrl"], KEYS["alt"], 0x77)]
        held = all(keys_down)
        pressed = held and not self._shortcut_was_held
        self._shortcut_was_held = held
        # Wait for the physical modifiers to come up before sending an event.
        self._shortcut_releasing = held or (self._shortcut_releasing and any(keys_down))
        return pressed, self._shortcut_releasing

    def move(self, dx, dy):
        dx += self._move_remainder[0]
        dy += self._move_remainder[1]
        x, y = round(dx), round(dy)
        self._move_remainder = dx - x, dy - y
        if x or y:
            self._send([_mouse(MOVE, x, y)])

    def move_to(self, x, y):
        x, y = (round(65535 * min(1, max(0, value))) for value in (x, y))
        self._send([_mouse(MOVE | ABSOLUTE, x, y)])

    def click(self, button="left", count=1):
        if button in self._buttons:
            return
        down, up = (_mouse(flag) for flag in BUTTONS[button])
        self._send([down, up] * count, recovery=[up])

    def button_down(self, button="left"):
        if button not in self._buttons:
            self._send([_mouse(BUTTONS[button][0])])
            self._buttons.add(button)

    def button_up(self, button="left"):
        if button in self._buttons:
            self._send([_mouse(BUTTONS[button][1])])
            self._buttons.remove(button)

    def scroll(self, vertical=0, horizontal=0):
        self._send([_mouse(flag, data=round(notches * WHEEL_DELTA))
                    for notches, flag in ((vertical, WHEEL), (horizontal, HWHEEL)) if notches])

    def zoom(self, notches):
        if notches:
            code = KEYS["ctrl"]
            wheel = _mouse(WHEEL, data=round(notches * WHEEL_DELTA))
            if self._is_held(code):
                self._send([wheel])
            else:
                release = _key(code, True)
                self._send([_key(code), wheel, release], recovery=[release])

    def hotkey(self, *keys):
        codes = []
        for name in keys:
            code = KEYS[name] if name in KEYS else ord(name.upper())
            if name not in MODIFIERS or not self._is_held(code):
                codes.append(code)
        release = [_key(code, True) for code in reversed(codes)]
        self._send([_key(code) for code in codes] + release, recovery=release)

    def release_buttons(self):
        for button in tuple(self._buttons):
            self.button_up(button)
        self._move_remainder = (0.0, 0.0)
