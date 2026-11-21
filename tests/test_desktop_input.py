"""Verify Windows event encoding without sending keyboard or mouse input."""

import ctypes
import sys
import unittest
from unittest.mock import Mock

from desktop_input import WindowsInput, _Input


class WindowsInputTests(unittest.TestCase):
    def setUp(self):
        self.events = []
        self.held = set()
        api = Mock()
        api.SendInput.side_effect = self.record_events
        api.GetAsyncKeyState.side_effect = lambda key: 0x8000 if key in self.held else 0
        self.output = WindowsInput(api)

    def record_events(self, count, inputs, size):
        self.assertEqual(size, ctypes.sizeof(_Input))
        if sys.platform == "win32":
            self.assertEqual(size, 40 if ctypes.sizeof(ctypes.c_void_p) == 8 else 28)
        for event in inputs:
            if event.kind == 1:
                key = event.payload.keyboard
                self.events.append(("key", key.key, key.flags))
            else:
                mouse = event.payload.mouse
                self.events.append(("mouse", mouse.flags, mouse.dx, mouse.dy, mouse.data))
        return count

    def test_zoom_sends_ctrl_wheel_and_releases_modifier(self):
        self.output.zoom(-1)
        self.assertEqual(self.events, [
            ("key", 0x11, 0),
            ("mouse", 0x0800, 0, 0, (-120) & 0xFFFFFFFF),
            ("key", 0x11, 2),
        ])

    def test_pause_releases_held_mouse_button_once(self):
        self.output.button_down()
        self.output.release_buttons()
        self.output.release_buttons()
        self.assertEqual(self.events, [
            ("mouse", 0x0002, 0, 0, 0),
            ("mouse", 0x0004, 0, 0, 0),
        ])

    def test_pause_chord_edges_and_physical_modifier_release(self):
        self.assertEqual(self.output.poll_pause_shortcut(), (False, False))
        self.held = {0x11, 0x12, 0x77}
        self.assertEqual(self.output.poll_pause_shortcut(), (True, True))
        self.assertEqual(self.output.poll_pause_shortcut(), (False, True))
        self.held.remove(0x77)
        self.assertEqual(self.output.poll_pause_shortcut(), (False, True))
        self.held.clear()
        self.assertEqual(self.output.poll_pause_shortcut(), (False, False))
