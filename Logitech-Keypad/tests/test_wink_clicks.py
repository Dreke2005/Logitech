"""Verify eye winks are converted into left/right mouse clicks."""

import ctypes
import sys
import unittest
from pathlib import Path
from unittest.mock import Mock

PROJECT_DIR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_DIR / "Logitech-Keypad"))
sys.path.insert(0, str(PROJECT_DIR))

from desktop_input import WindowsInput, _Input
from wink_clicks import WinkClickController


class TrackingOutput:
    def __init__(self):
        self.held = set()
        self.clicks = []

    def button_down(self, button):
        self.held.add(button)

    def button_up(self, button):
        self.held.discard(button)

    def click(self, button="left", count=1):
        self.clicks.extend([button] * count)


class WinkClickTests(unittest.TestCase):
    def test_left_wink_clicks_once_and_rearms_after_both_eyes_open(self):
        output = TrackingOutput()
        controller = WinkClickController(output, hold_frames=2)
        controller.update({"ear": (0.3, 0.3)})
        sample = {"ear": (0.1, 0.3)}

        self.assertIsNone(controller.update(sample))
        self.assertEqual(output.held, set())
        self.assertEqual(controller.update(sample), "Left eye closed: left button held")
        self.assertEqual(output.held, {"left"})
        self.assertIsNone(controller.update(sample))
        self.assertEqual(output.held, {"left"})
        self.assertEqual(controller.update({"ear": (0.3, 0.3)}),
                         "Left eye open: left button released")
        self.assertEqual(output.held, set())
        controller.update(sample)
        controller.update(sample)
        self.assertEqual(output.held, {"left"})
        controller.update({"ear": (0.3, 0.3)})
        self.assertEqual(output.held, set())

    def test_right_wink_selects_right_mouse_button(self):
        output = Mock()
        controller = WinkClickController(output, hold_frames=1)
        controller.update({"ear": (0.3, 0.3)})

        self.assertEqual(controller.update({"ear": (0.3, 0.1)}),
                         "Right eye wink: right click")
        output.click.assert_called_once_with(button="right")

    def test_both_eyes_closed_never_clicks(self):
        output = Mock()
        controller = WinkClickController(output, hold_frames=1)
        controller.update({"ear": (0.3, 0.3)})

        self.assertEqual(controller.update({"ear": (0.1, 0.1)}, enabled=True),
                         "Both eyes closed; ignored")
        output.click.assert_not_called()

    def test_both_open_releases_left_wink_and_disabled_tracking_releases_it_too(self):
        output = TrackingOutput()
        controller = WinkClickController(output, hold_frames=1)
        controller.update({"ear": (0.3, 0.3)})

        controller.update({"ear": (0.1, 0.3)})
        self.assertEqual(output.held, {"left"})
        controller.update({"ear": (0.3, 0.3)})
        self.assertEqual(output.held, set())

        controller.update({"ear": (0.1, 0.3)})
        self.assertEqual(output.held, {"left"})
        controller.update(None)
        self.assertEqual(output.held, set())

    def test_right_mouse_button_emits_down_and_up_events(self):
        events = []
        api = Mock()

        def record(count, inputs, size):
            self.assertEqual(size, ctypes.sizeof(_Input))
            events.extend(event.payload.mouse.flags for event in inputs)
            return count

        api.SendInput.side_effect = record
        desktop = WindowsInput(api)
        desktop.click(button="right")

        self.assertEqual(events, [0x0008, 0x0010])

    def test_left_wink_releases_a_held_drag_button_when_eye_reopens(self):
        events = []
        api = Mock()

        def record(count, inputs, size):
            events.extend(event.payload.mouse.flags for event in inputs)
            return count

        api.SendInput.side_effect = record
        desktop = WindowsInput(api)
        desktop.button_down("left")
        controller = WinkClickController(desktop, hold_frames=1)
        controller.update({"ear": (0.3, 0.3)})
        controller.update({"ear": (0.1, 0.3)})
        self.assertIn("left", desktop._buttons_held)

        controller.update({"ear": (0.3, 0.3)})

        self.assertEqual(events, [0x0002, 0x0004])
        self.assertNotIn("left", desktop._buttons_held)


if __name__ == "__main__":
    unittest.main()
