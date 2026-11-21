"""Hardware-free tests using overhead-style hand landmarks in image pixels."""

import math
import sys
import unittest
from contextlib import ExitStack
from pathlib import Path
from unittest.mock import Mock, patch

import cv2
import numpy as np

import rgb_preview
from typing_calibration import PoseCalibration
from virtual_keyboard import VirtualKeyboard

FRAME_SIZE = (640, 480)
FINGERTIPS = {"thumb": 4, "index": 8, "middle": 12, "ring": 16, "pinky": 20}


def hand(screen_left, *, move=(0, 0), tap=0, digit=0, curled=False, pinch=False, thumb_x=307):
    """Two hands hover with wrist, palm, PIP and DIP visible to the camera."""
    fingers = [
        ((225, 251), (242, 228), (248, 217), (254, 210)),
        ((195, 253), (209, 227), (215, 217), (218, 209)),
        ((174, 253), (181, 229), (183, 220), (184, 213)),
        ((155, 256), (153, 235), (152, 225), (152, 218)),
    ]
    points = [(195, 280), (220, 271), (250, 265), (280, 262), (thumb_x, 260)]
    for finger, (mcp, pip, dip, tip) in enumerate(fingers):
        if curled:
            pip = (mcp[0], 255)
            dip = (mcp[0], 263)
            tip = (mcp[0], 273)
        elif finger == digit:
            tip = (tip[0], tip[1] + tap)
        points += [mcp, pip, dip, tip]
    if pinch:
        points[4] = points[8]

    dx, dy = move
    landmarks = []
    for x, y in points:
        x = (x if screen_left else FRAME_SIZE[0] - x) + dx
        y += dy
        landmarks.append({"pixel": [x, y], "x": x / FRAME_SIZE[0], "y": y / FRAME_SIZE[1], "z": 0})
    return {
        "handedness": "Left" if screen_left else "Right",
        "score": 0.97,
        "landmarks": landmarks,
        "fingertips": {name: landmarks[index] for name, index in FINGERTIPS.items()},
    }


def frame(timestamp, *, tap=0, digit=0, right_tap=False, move=(0, 0), one_hand=False,
          curled=False, reverse=False, pinch=False, thumb_x=307, angle=0):
    hands = [hand(True, tap=0 if right_tap else tap, digit=digit, move=move,
                  curled=curled, pinch=pinch, thumb_x=thumb_x)]
    if not one_hand:
        hands.append(hand(False, tap=tap if right_tap else 0, digit=digit, move=move,
                          curled=curled, thumb_x=thumb_x))
    if angle:
        cos, sin = math.cos(angle), math.sin(angle)
        width, height = FRAME_SIZE
        for pose in hands:
            for point in pose["landmarks"]:
                x, y = point["pixel"][0] - width / 2, point["pixel"][1] - height / 2
                point["pixel"] = [round(x * cos - y * sin + width / 2),
                                  round(x * sin + y * cos + height / 2)]
                point["x"] = point["pixel"][0] / width
                point["y"] = point["pixel"][1] / height
    return {
        "frame_size": list(FRAME_SIZE),
        "timestamp_ms": timestamp,
        "hands": list(reversed(hands)) if reverse else hands,
    }


class TypingKeyboardTests(unittest.TestCase):
    def activate(self, keyboard, start=0):
        for timestamp in range(start, start + 900, 50):
            keyboard.update(frame(timestamp, reverse=(timestamp // 50) % 2 == 0))
        self.assertTrue(keyboard.keys)
        return start + 900

    def tap_f(self, keyboard, start):
        keyboard.update(frame(start))
        keyboard.update(frame(start + 50, tap=9))
        keyboard.update(frame(start + 100))

    def test_brief_pose_single_hand_and_grab_do_not_arm(self):
        keyboard = VirtualKeyboard()
        for timestamp in range(0, 1100, 50):
            keyboard.update(frame(timestamp, curled=True))
        self.assertFalse(keyboard.keys)
        for timestamp in range(1100, 2100, 50):
            keyboard.update(frame(timestamp, one_hand=True))
        self.assertFalse(keyboard.keys)
        for timestamp in range(2100, 2450, 50):
            keyboard.update(frame(timestamp))
        self.assertFalse(keyboard.keys)
        self.assertEqual(keyboard.text, "")

    def test_thumb_proximity_is_required_and_midpoint_anchors_space(self):
        calibration = PoseCalibration()
        for timestamp in range(0, 800, 50):
            self.assertIsNone(calibration.update(frame(timestamp, thumb_x=260)))
        self.assertIn("thumbs a little closer", calibration.hint)
        for timestamp in range(800, 1600, 50):
            reference = calibration.update(frame(timestamp, thumb_x=320))
        self.assertIsNotNone(reference)
        self.assertEqual(reference.thumb_anchor, (320, 260))
        self.assertEqual(reference.key_at(*reference.thumb_anchor), "Space")

        calibration.reset()
        for timestamp in range(0, 800, 50):
            tracking = frame(timestamp)
            for pose in tracking["hands"]:
                pose["fingertips"]["thumb"]["pixel"][0] += 20
            shifted = calibration.update(tracking)
        self.assertIsNotNone(shifted)
        space = next(key for key in shifted.keys if key.label == "Space")
        self.assertAlmostEqual(sum(x for x, _ in space.corners) / 4, 340)

    def test_compact_pose_places_nonoverlapping_space_at_the_thumb_midpoint(self):
        calibration = PoseCalibration()
        for timestamp in range(0, 800, 50):
            tracking = frame(timestamp)
            for pose in tracking["hands"]:
                pose["fingertips"]["thumb"]["pixel"][1] = 218
            reference = calibration.update(tracking)
        self.assertIsNotNone(reference)
        self.assertEqual(reference.thumb_anchor, (320, 218))
        self.assertEqual([key.label for key in reference.keys if key.contains(320, 218)], ["Space"])
        space = next(key for key in reference.keys if key.label == "Space")
        self.assertAlmostEqual(sum(x for x, _ in space.corners) / 4, 320)
        self.assertAlmostEqual(sum(y for _, y in space.corners) / 4, 218)

    def test_calibration_hold_tolerates_a_missed_frame_and_thumb_jitter(self):
        calibration = PoseCalibration()
        calibration.update(frame(0))
        calibration.update(frame(50))
        calibration.update(frame(100, one_hand=True))
        self.assertAlmostEqual(calibration.progress, 50 / calibration.hold_ms)
        for timestamp in range(150, 850, 50):
            reference = calibration.update(frame(timestamp, thumb_x=301))
        self.assertIsNotNone(reference)

    def test_typing_works_with_a_quarter_turn_and_reordered_detections(self):
        keyboard = VirtualKeyboard()
        for timestamp in range(0, 900, 50):
            keyboard.update(frame(timestamp, angle=math.pi / 2, reverse=timestamp % 100 == 0))
        self.assertTrue(keyboard.keys)
        keyboard.update(frame(900, angle=math.pi / 2, tap=9))
        keyboard.update(frame(950, angle=math.pi / 2))
        self.assertEqual(keyboard.text, "f")

    def test_different_hand_angles_can_share_one_keyboard(self):
        calibration = PoseCalibration()
        for timestamp in range(0, 800, 50):
            tracking = frame(timestamp)
            for pose, angle in zip(tracking["hands"], (0.16, -0.26)):
                anchor_x, anchor_y = pose["fingertips"]["thumb"]["pixel"]
                for point in pose["landmarks"]:
                    x, y = point["pixel"][0] - anchor_x, point["pixel"][1] - anchor_y
                    point["pixel"] = [round(anchor_x + x * math.cos(angle) - y * math.sin(angle)),
                                      round(anchor_y + x * math.sin(angle) + y * math.cos(angle))]
            reference = calibration.update(tracking)
        self.assertIsNotNone(reference)
        for pose, label in zip(tracking["hands"], ("F", "J")):
            self.assertEqual(reference.key_at(*pose["fingertips"]["index"]["pixel"]), label)

    def test_depth_tap_is_relative_to_the_knuckle(self):
        keyboard = VirtualKeyboard()
        start = self.activate(keyboard)
        whole_hand = frame(start)
        for point in whole_hand["hands"][0]["landmarks"]:
            point["z"] = 0.02
        keyboard.update(whole_hand)
        keyboard.update(frame(start + 50))
        self.assertEqual(keyboard.text, "")

        fingertip_only = frame(start + 100)
        fingertip_only["hands"][0]["fingertips"]["index"]["z"] = 0.02
        keyboard.update(fingertip_only)
        self.assertEqual(keyboard.text, "")
        keyboard.update(frame(start + 150))
        self.assertEqual(keyboard.text, "f")

    def test_generated_layout_places_home_keys_under_index_fingers(self):
        calibration = PoseCalibration()
        for timestamp in range(0, 750, 50):
            reference = calibration.update(frame(timestamp, reverse=timestamp % 100 == 0))
        self.assertIsNotNone(reference)
        for pose, keys in zip((hand(True), hand(False)), ("FDSQ", "JKLM")):
            for label, finger in zip(keys, ("index", "middle", "ring", "pinky")):
                point = pose["fingertips"][finger]["pixel"]
                self.assertEqual(reference.key_at(*point), label)
        self.assertIsNone(reference.key_at(10, 10))

        calibration.reset()
        for timestamp in range(0, 750, 50):
            shifted = calibration.update(frame(timestamp, move=(25, 20)))
        self.assertIsNotNone(shifted)
        self.assertEqual(shifted.key_at(279, 230), "F")
        self.assertEqual(shifted.key_at(411, 230), "J")
        self.assertEqual(shifted.pitch, reference.pitch)

    def test_setup_feedback_distinguishes_missing_hands_and_a_clipped_layout(self):
        calibration = PoseCalibration()
        self.assertIsNone(calibration.update(frame(0, one_hand=True)))
        self.assertIn("Show both hands", calibration.hint)
        self.assertIsNone(calibration.update(frame(50, move=(0, -250))))
        self.assertIn("both wrists in the frame", calibration.hint)
        self.assertIsNone(calibration.update(frame(100, move=(0, -170))))
        self.assertIn("move the camera back", calibration.hint)
        self.assertEqual(calibration.progress, 0)

        self.assertIsNone(calibration.update(frame(150)))
        self.assertIn("Hold both hands still", calibration.hint)
        self.assertIsNone(calibration.update(frame(200)))
        self.assertGreater(calibration.progress, 0)

    def test_tap_commits_only_on_release_and_once(self):
        keyboard = VirtualKeyboard()
        start = self.activate(keyboard)
        keyboard.update(frame(start, tap=9))
        keyboard.update(frame(start + 50, tap=9))
        self.assertEqual(keyboard.text, "")
        keyboard.update(frame(start + 100))
        keyboard.update(frame(start + 150))
        self.assertEqual(keyboard.text, "f")
        self.tap_f(keyboard, start + 200)
        self.assertEqual(keyboard.text, "ff")

    def test_whole_hand_motion_cannot_type_before_return_to_reference(self):
        keyboard = VirtualKeyboard()
        start = self.activate(keyboard)
        keyboard.update(frame(start, move=(0, 9)))
        keyboard.update(frame(start + 50))
        self.assertEqual(keyboard.text, "")
        self.tap_f(keyboard, start + 100)
        self.assertEqual(keyboard.text, "f")

    def test_brief_grab_pinch_or_tracking_loss_pauses_without_moving_layout(self):
        for reason in ("grab", "missing hand", "pinch"):
            with self.subTest(reason=reason):
                keyboard = VirtualKeyboard()
                start = self.activate(keyboard)
                reference = keyboard.reference
                self.tap_f(keyboard, start)
                keyboard.update(frame(start + 150, tap=9))
                keyboard.update(frame(start + 200, curled=reason == "grab",
                                      one_hand=reason == "missing hand", pinch=reason == "pinch"))
                self.assertIs(keyboard.reference, reference)
                self.assertEqual(keyboard.text, "f")
                for offset in (250, 310, 370):
                    keyboard.update(frame(start + offset, thumb_x=280))
                keyboard.update(frame(start + 420, tap=9, thumb_x=280))
                keyboard.update(frame(start + 470, thumb_x=280))
                self.assertIs(keyboard.reference, reference)
                self.assertEqual(keyboard.text, "ff")

    def test_thumb_separation_and_small_hand_shift_preserve_typing_layout(self):
        keyboard = VirtualKeyboard()
        start = self.activate(keyboard)
        reference = keyboard.reference
        keyboard.update(frame(start, move=(5, 6), thumb_x=283))
        keyboard.update(frame(start + 50, move=(6, 6), tap=9, thumb_x=286))
        keyboard.update(frame(start + 100, move=(5, 6), thumb_x=283))
        self.assertIs(keyboard.reference, reference)
        self.assertEqual(keyboard.text, "f")

    def test_sustained_departure_needs_deliberate_rearm(self):
        keyboard = VirtualKeyboard()
        start = self.activate(keyboard)
        reference = keyboard.reference
        self.tap_f(keyboard, start)
        for offset in range(150, 1750, 100):
            keyboard.update(frame(start + offset, move=(0, 66), thumb_x=280))
            self.assertIs(keyboard.reference, reference)
        for offset in range(1750, 2650, 100):
            keyboard.update(frame(start + offset, move=(0, 66), thumb_x=280))
            self.assertIsNone(keyboard.reference)
        for offset in range(2650, 3450, 50):
            keyboard.update(frame(start + offset, move=(0, 66)))
        self.assertIsNotNone(keyboard.reference)
        self.assertEqual(keyboard.reference.thumb_anchor, (320, 326))
        self.assertEqual(keyboard.text, "f")

    def test_repeated_dropouts_do_not_accumulate_as_a_departure(self):
        keyboard = VirtualKeyboard()
        start = self.activate(keyboard)
        reference = keyboard.reference
        for offset in range(0, 2200, 50):
            keyboard.update(frame(start + offset, one_hand=offset % 100 == 0))
        self.assertIs(keyboard.reference, reference)
        self.assertEqual(keyboard.text, "")

    def test_right_middle_finger_taps_its_own_key(self):
        keyboard = VirtualKeyboard()
        start = self.activate(keyboard)
        keyboard.update(frame(start, digit=1, tap=9, right_tap=True, reverse=True))
        keyboard.update(frame(start + 50))
        self.assertEqual(keyboard.text, "k")

    def test_reaching_to_another_key_does_not_type_until_the_tap(self):
        keyboard = VirtualKeyboard()
        start = self.activate(keyboard)

        def reach(timestamp, tap=0):
            tracking = frame(timestamp, tap=tap)
            points = tracking["hands"][0]["landmarks"]
            for index, dx in zip(range(5, 9), (12, 21, 33, 44)):
                points[index]["pixel"][0] += dx
                points[index]["x"] = points[index]["pixel"][0] / FRAME_SIZE[0]
            return tracking

        keyboard.update(reach(start))
        self.assertEqual(keyboard.text, "")
        keyboard.update(reach(start + 50, tap=9))
        keyboard.update(reach(start + 100))
        self.assertEqual(keyboard.text, "g")

    def test_manual_recalibration_preserves_text_and_clears_keys(self):
        keyboard = VirtualKeyboard()
        self.tap_f(keyboard, self.activate(keyboard))
        keyboard.reset_calibration()
        self.assertFalse(keyboard.keys)
        self.assertEqual(keyboard.text, "f")

    def test_preview_renders_calibrated_keys_and_committed_text(self):
        tracks = [frame(timestamp) for timestamp in range(0, 900, 50)]
        tracks += [frame(900, tap=9), frame(950)]
        context = Mock()
        context.Queue.return_value.get.side_effect = [
            ("frame", (np.zeros((480, 640, 3), dtype=np.uint8), tracking))
            for tracking in tracks
        ]

        with ExitStack() as stack:
            stack.enter_context(patch.object(sys, "argv", ["rgb_preview.py", "--frames", str(len(tracks))]))
            device_list = stack.enter_context(patch("pygrabber.dshow_graph.FilterGraph"))
            stack.enter_context(patch("fingertip_tracker.ensure_model", return_value=Path(__file__)))
            stack.enter_context(patch.object(rgb_preview.mp, "get_context", return_value=context))
            display = stack.enter_context(patch.object(cv2, "imshow"))
            stack.enter_context(patch.object(cv2, "waitKey", return_value=-1))
            stack.enter_context(patch.object(cv2, "getWindowProperty", return_value=1))
            stack.enter_context(patch.object(cv2, "destroyAllWindows"))
            draw_text = stack.enter_context(patch.object(cv2, "putText", wraps=cv2.putText))
            device_list.return_value.get_input_devices.return_value = ["LVNext Camera"]
            context.Process.return_value.is_alive.return_value = False
            self.assertEqual(rgb_preview.main(), 0)

        self.assertEqual(display.call_count, len(tracks))
        blank, active = [call.args[1] for call in (display.call_args_list[0], display.call_args)]
        self.assertFalse(blank[150, 300].any())
        self.assertTrue(active[150, 300].any())
        self.assertIn("f|", [call.args[1] for call in draw_text.call_args_list])


if __name__ == "__main__":
    unittest.main()
