"""Verify Windows event encoding without sending keyboard or mouse input."""

import ctypes
import sys
import unittest
from contextlib import ExitStack
from pathlib import Path
from unittest.mock import Mock, patch

import cv2
import numpy as np

import rgb_preview
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
        self.assertEqual(self.output.navigation_feedback(), "Sent: zoom out")

    def test_scroll_reports_direction_only_after_sending_a_wheel_event(self):
        self.output.scroll(vertical=0, horizontal=0)
        self.assertIsNone(self.output.navigation_feedback())
        self.output.scroll(vertical=-1)
        self.assertEqual(self.events, [("mouse", 0x0800, 0, 0, (-120) & 0xFFFFFFFF)])
        self.assertEqual(self.output.navigation_feedback(), "Sent: scroll down")

    def test_tab_overview_sends_chrome_shortcut_and_releases_keys(self):
        self.output.hotkey("ctrl", "shift", "a")
        self.assertEqual(self.events, [
            ("key", 0x11, 0), ("key", 0x10, 0), ("key", 0x41, 0),
            ("key", 0x41, 2), ("key", 0x10, 2), ("key", 0x11, 2),
        ])
        self.assertEqual(self.output.navigation_feedback(), "Sent: tab overview")

    def test_eye_pointer_moves_absolutely_and_a_click_includes_release(self):
        self.output.move_to(0.0, 1.0)
        self.output.click()
        self.assertEqual(self.events, [
            ("mouse", 0x8001, 0, 65535, 0),
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


class DesktopPreviewTests(unittest.TestCase):
    def test_capture_runs_both_models_on_the_same_mirrored_frame(self):
        image = np.zeros((4, 8, 3), dtype=np.uint8)
        image[0, 0] = [1, 2, 3]
        camera = Mock()
        camera.read.return_value = True, image
        stop = Mock()
        stop.is_set.side_effect = [False, True]
        output = Mock()
        with ExitStack() as stack:
            stack.enter_context(patch("camera_utils.open_camera", return_value=(camera, {})))
            hands = stack.enter_context(patch("fingertip_tracker.FingertipTracker"))
            eyes = stack.enter_context(patch("eye_pointer.EyeLandmarker"))
            rgb_preview._capture(0, "DSHOW", 8, 4, output, stop, "hand.task", True, "eye.task")
        kind, (frame, hand_sample, eye_sample) = output.put_nowait.call_args.args[0]
        self.assertEqual(kind, "frame")
        self.assertEqual(frame[0, -1].tolist(), [1, 2, 3])
        self.assertIs(hands.return_value.process.call_args.args[0], frame)
        self.assertIs(eyes.return_value.process.call_args.args[0], frame)
        self.assertIs(hand_sample, hands.return_value.process.return_value)
        self.assertIs(eye_sample, eyes.return_value.process.return_value)
        for close in (camera.release, hands.return_value.close, eyes.return_value.close):
            close.assert_called_once_with()

    def test_combined_mode_accepts_gestures_even_when_gaze_is_not_calibrated(self):
        hand_samples = [{"hands": [], "timestamp_ms": i} for i in range(7)]
        eye_samples = [{"gaze": None, "timestamp_ms": i} for i in range(7)]
        frames = [("frame", (np.zeros((32, 64, 3), dtype=np.uint8), hands, eyes))
                  for hands, eyes in zip(hand_samples, eye_samples)]
        model = Path(__file__).resolve()
        args = ["rgb_preview.py", "--mode", "desktop", "--frames", "7",
                "--model", str(model), "--eye-model", str(model)]
        with ExitStack() as stack:
            enter = stack.enter_context
            enter(patch.object(sys, "argv", args))
            graph = enter(patch("pygrabber.dshow_graph.FilterGraph"))
            graph.return_value.get_input_devices.return_value = ["LVNext Camera"]
            ctx = enter(patch.object(rgb_preview.mp, "get_context")).return_value
            ctx.Queue.return_value.get.side_effect = frames
            adapter = enter(patch("desktop_input.WindowsInput")).return_value
            adapter.navigation_feedback.return_value = None
            adapter.screen_size.return_value = (400, 220)
            adapter.poll_pause_shortcut.side_effect = [
                (False, False), (True, True), (False, False), (False, False),
                (False, False), (True, True), (False, False),
            ]
            gestures = enter(patch("gesture_touchpad.TouchpadController")).return_value
            gestures.update.return_value = "Scroll"
            gaze = enter(patch("eye_pointer.EyePointer")).return_value
            gaze.calibrating = False
            gaze.target = None
            gaze.target_progress = 0.5
            gaze.update.return_value = "Gaze"

            def begin_calibration():
                gaze.calibrating = True
                gaze.target = (0.5, 0.16)
                return "Look at the dot"

            def fail_calibration():
                gaze.calibrating = False
                gaze.target = None
                return "Eye calibration unclear"

            gaze.begin_calibration.side_effect = begin_calibration
            gaze.capture_target.side_effect = fail_calibration
            display = enter(patch.object(cv2, "imshow"))
            enter(patch.object(cv2, "namedWindow"))
            enter(patch.object(cv2, "moveWindow"))
            enter(patch.object(cv2, "setWindowProperty"))
            close_calibration = enter(patch.object(cv2, "destroyWindow"))
            enter(patch.object(cv2, "waitKey", side_effect=[ord("c"), -1, ord("c"), ord(" "), -1, -1]))
            enter(patch.object(cv2, "getWindowProperty", return_value=1))
            enter(patch.object(cv2, "destroyAllWindows"))
            self.assertEqual(rgb_preview.main(), 0)
        self.assertEqual(ctx.Process.call_args.kwargs["args"][-3:], (model, False, model))
        gaze.begin_calibration.assert_called_once_with()
        gaze.capture_target.assert_called_once_with()
        self.assertEqual([call.args[0] for call in gaze.update.call_args_list], eye_samples[2:5])
        self.assertEqual([call.args[0] for call in gestures.update.call_args_list],
                         [hand_samples[2], hand_samples[4]])
        targets = [call.args[1] for call in display.call_args_list
                   if call.args[0] == rgb_preview.CALIBRATION_WINDOW]
        self.assertEqual(len(targets), 1)
        self.assertEqual(tuple(targets[0][35, 200]), (248, 248, 250))
        close_calibration.assert_called_once_with(rgb_preview.CALIBRATION_WINDOW)
