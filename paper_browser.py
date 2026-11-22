"""Hand text or a picture to a web app that has no local interface: Overleaf and ChatGPT in the browser.

The route is the Windows clipboard plus one paste keystroke into the browser tab. Keystrokes go only
to the window they are meant for: the window is brought to the front, and right before the keystroke
the foreground window is checked again. When it is not the expected app, nothing is typed and the
content stays on the clipboard for the user to paste.
"""

import ctypes
import os
import struct
import time
from pathlib import Path

import paper_project
import paper_system

CF_UNICODETEXT, CF_DIB, GMEM_MOVEABLE = 13, 8, 0x0002
OVERLEAF_PROJECTS = "https://www.overleaf.com/project"
CHATGPT = "https://chatgpt.com/"
# Overleaf pages that are not a project's editor: there is no cursor in a document to paste at.
NOT_AN_EDITOR = ("your projects", "projects -", "log in", "login", "sign up", "register", "overleaf,", "overleaf -")


_libraries = {}


def _dll(name):
    """This module's own handle on a Windows library, so the types set here do not affect other modules."""
    if name not in _libraries:
        _libraries[name] = ctypes.WinDLL(name)
    return _libraries[name]


def is_overleaf_editor(process, title):
    """A browser window whose active tab is an Overleaf project (the editor), not the project list."""
    title = (title or "").strip().lower()
    return (process.lower() in paper_project.BROWSERS and "overleaf" in title
            and not title.startswith(NOT_AN_EDITOR))


def is_chatgpt(process, title):
    """A window whose active tab (or app) is ChatGPT."""
    return paper_project.app_for_window(process, title) == "chatgpt"


def _describe(window):
    """(process name, title) of a window handle, or ("", "")."""
    user32, kernel32 = _dll("user32"), _dll("kernel32")
    pid = ctypes.c_ulong()
    user32.GetWindowThreadProcessId(ctypes.c_void_p(window), ctypes.byref(pid))
    handle = kernel32.OpenProcess(0x1000, False, pid.value)  # PROCESS_QUERY_LIMITED_INFORMATION
    if not handle:
        return "", ""
    try:
        buffer, size = ctypes.create_unicode_buffer(1024), ctypes.c_ulong(1024)
        if not kernel32.QueryFullProcessImageNameW(handle, 0, buffer, ctypes.byref(size)):
            return "", ""
    finally:
        kernel32.CloseHandle(handle)
    title = ctypes.create_unicode_buffer(512)
    user32.GetWindowTextW(ctypes.c_void_p(window), title, 512)
    return Path(buffer.value).name, title.value


def windows():
    """[(handle, process name, title)] of the visible top-level windows with a title, front-most first."""
    if os.name != "nt":
        return []
    user32 = _dll("user32")
    found = []

    @ctypes.WINFUNCTYPE(ctypes.c_bool, ctypes.c_void_p, ctypes.c_void_p)
    def visit(handle, _):
        if user32.IsWindowVisible(ctypes.c_void_p(handle)) and user32.GetWindowTextLengthW(ctypes.c_void_p(handle)):
            process, title = _describe(handle)
            if process:
                found.append((handle, process, title))
        return True

    user32.EnumWindows(visit, 0)
    return found


def find_window(matches):
    """The front-most window for which ``matches(process, title)`` holds, or None."""
    return next((handle for handle, process, title in windows() if matches(process, title)), None)


def foreground():
    """(handle, process name, title) of the window in front, or (None, "", "")."""
    if os.name != "nt":
        return None, "", ""
    user32 = _dll("user32")
    user32.GetForegroundWindow.restype = ctypes.c_void_p
    window = user32.GetForegroundWindow()
    if not window:
        return None, "", ""
    return (window, *_describe(window))


def bring_to_front(window):
    """Show a window in front. Windows lets a background program do that only right after input, so a
    harmless Alt key press comes first."""
    user32 = _dll("user32")
    if user32.IsIconic(ctypes.c_void_p(window)):
        user32.ShowWindow(ctypes.c_void_p(window), 9)  # SW_RESTORE
    user32.keybd_event(0x12, 0, 0, 0)  # Alt down, up
    user32.keybd_event(0x12, 0, 0x0002, 0)
    user32.SetForegroundWindow(ctypes.c_void_p(window))


def paste_into(matches, window=None, wait=0.0, settle=0.35, enter_after=None):
    """Press Ctrl+V in the window for which ``matches(process, title)`` holds. Returns True when pressed.

    ``window`` is brought to the front first; without one, up to ``wait`` seconds are given for such a
    window to come to the front by itself (a page that is still opening). The foreground window is
    checked right before the keystroke, so nothing is ever typed into another program. With
    ``enter_after`` seconds, Enter follows after that pause under the same check."""
    if os.name != "nt":
        return False
    if window is not None:
        bring_to_front(window)
    deadline = time.monotonic() + max(wait, 1.5 if window is not None else 0.0)
    while True:
        time.sleep(0.15)
        front, process, title = foreground()
        if front is not None and matches(process, title) and (window is None or front == window):
            break
        if time.monotonic() >= deadline:
            return False
    time.sleep(settle)
    front, process, title = foreground()
    if front is None or not matches(process, title):
        return False
    paper_system.press("ctrl", "v")
    if enter_after is not None:
        time.sleep(enter_after)
        front, process, title = foreground()
        if front is not None and matches(process, title):
            paper_system.press("enter")
    return True


def _set_clipboard(clipboard_format, data):
    """Put one block of data on the Windows clipboard, available at once to every program."""
    if os.name != "nt":
        raise OSError("The clipboard route needs Windows.")
    user32, kernel32 = _dll("user32"), _dll("kernel32")
    kernel32.GlobalAlloc.restype = kernel32.GlobalLock.restype = ctypes.c_void_p
    kernel32.GlobalLock.argtypes = kernel32.GlobalUnlock.argtypes = kernel32.GlobalFree.argtypes = [ctypes.c_void_p]
    user32.SetClipboardData.argtypes = [ctypes.c_uint, ctypes.c_void_p]
    user32.SetClipboardData.restype = ctypes.c_void_p
    for _ in range(10):  # Another program may hold the clipboard for a moment.
        if user32.OpenClipboard(None):
            break
        time.sleep(0.05)
    else:
        raise OSError("The clipboard is in use by another program.")
    try:
        user32.EmptyClipboard()
        block = kernel32.GlobalAlloc(GMEM_MOVEABLE, len(data))
        if not block:
            raise OSError("No memory for the clipboard.")
        target = kernel32.GlobalLock(block)
        ctypes.memmove(target, data, len(data))
        kernel32.GlobalUnlock(block)
        if not user32.SetClipboardData(clipboard_format, block):
            kernel32.GlobalFree(block)
            raise OSError("The clipboard did not accept the content.")
    finally:
        user32.CloseClipboard()


def copy_text(text):
    _set_clipboard(CF_UNICODETEXT, (text + "\0").encode("utf-16-le"))


def dib(image):
    """A BGR picture (numpy array, height x width x 3) as a device-independent bitmap for the clipboard."""
    height, width = image.shape[:2]
    padding = (-width * 3) % 4  # Bitmap rows are bottom-up and padded to four bytes.
    rows = image[::-1, :, :3]
    body = b"".join(row.tobytes() + bytes(padding) for row in rows) if padding else rows.tobytes()
    header = struct.pack("<IiiHHIIiiII", 40, width, height, 1, 24, 0, len(body), 2835, 2835, 0, 0)
    return header + body


def copy_image(image):
    """Put a picture on the clipboard as an image, as a screenshot would be."""
    _set_clipboard(CF_DIB, dib(image))
