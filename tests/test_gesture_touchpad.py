import unittest
from unittest.mock import Mock, call

from gesture_touchpad import TouchpadController


def sample(timestamp, pose="point", offset=(0, 0), contact=None, thumb=None):
    tracking = {"timestamp_ms": timestamp, "frame_size": [640, 480], "hands": []}
    if pose == "lost":
        return tracking
    points = [(310, 300), (290, 286), (272, 275), (265, 260), (255, 254)]
    fingers = {
        "point": (True, False, False, False),
        "scroll": (True, True, False, False),
        "zoom": (True, True, True, True),
        "swipe": (True, True, True, False),
        "fist": (False, False, False, False),
    }[pose]
    for x, raised in zip((284, 310, 334, 355), fingers):
        points.extend(((x, 250), (x, 228), (x, 214), (x, 201 if raised else 255)))
    if contact in ("left", "right"):
        x, y = points[8 if contact == "left" else 12]
        points[4] = (x - 5, y + 2)
    if thumb is not None:
        points[4] = thumb
    dx, dy = offset
    tracking["hands"] = [{"landmarks": [{"pixel": [x + dx, y + dy]} for x, y in points]}]
    return tracking


class TouchpadTests(unittest.TestCase):
    def setUp(self):
        self.output = Mock()
        self.controller = TouchpadController(self.output)

    def start(self, pose="point"):
        for timestamp in (0, 50, 100):
            self.controller.update(sample(timestamp, pose))

    def test_point_translates_and_a_short_loss_rebases(self):
        self.start()
        self.controller.update(sample(150, offset=(12, 6)))
        dx, dy = self.output.move.call_args.args
        self.assertGreater(dx, 0)
        self.assertGreater(dy, 0)
        self.output.reset_mock()
        self.controller.update(sample(200, "lost"))
        self.controller.update(sample(250, offset=(-30, 0)))
        self.controller.update(sample(300, offset=(-30, 0)))
        self.controller.update(sample(350, offset=(-30, 0)))
        self.assertEqual(self.output.method_calls, [])

    def test_short_pinches_click_once_each_for_native_double_click(self):
        self.start()
        for first in (150, 320):
            self.controller.update(sample(first, contact="left"))
            self.controller.update(sample(first + 45, contact="left"))
            self.controller.update(sample(first + 100))
            self.controller.update(sample(first + 125))
        self.assertEqual(self.output.method_calls, [call.click("left"), call.click("left")])

    def test_middle_finger_pinch_right_clicks_without_scroll(self):
        self.start()
        self.controller.update(sample(150, "scroll"))
        self.controller.update(sample(200, "scroll", contact="right"))
        self.controller.update(sample(250, "scroll", contact="right"))
        self.controller.update(sample(300, "scroll"))
        self.assertEqual(self.output.method_calls, [call.click("right")])

    def test_pinching_before_point_pose_is_established_cannot_click(self):
        for timestamp in (0, 50, 100, 150):
            self.controller.update(sample(timestamp, contact="left"))
        self.controller.update(sample(200))
        self.assertEqual(self.output.method_calls, [])

    def test_drag_ends_on_loss_and_never_leaves_button_pressed(self):
        self.start()
        for timestamp in range(150, 601, 50):
            self.controller.update(sample(timestamp, contact="left"))
        self.controller.update(sample(650, offset=(15, 0), contact="left"))
        self.controller.update(sample(700, "lost"))
        self.controller.reset()
        self.assertEqual([action[0] for action in self.output.method_calls],
                         ["button_down", "move", "button_up"])

    def test_reset_releases_a_drag_when_pausing_or_closing_preview(self):
        self.start()
        for timestamp in range(150, 601, 50):
            self.controller.update(sample(timestamp, contact="left"))
        self.controller.reset()
        self.controller.reset()
        self.assertEqual(self.output.method_calls, [call.button_down("left"), call.button_up("left")])

    def test_scroll_both_axes_and_zoom_as_a_separate_mode(self):
        self.start("scroll")
        self.controller.update(sample(150, "scroll", offset=(28, -28)))
        self.assertEqual(self.output.method_calls, [call.scroll(vertical=1, horizontal=1)])
        self.output.reset_mock()
        for timestamp in (200, 250, 300):
            self.controller.update(sample(timestamp, "zoom"))
        self.controller.update(sample(350, "zoom", thumb=(264, 212)))
        self.assertTrue(all(action[0] == "zoom" for action in self.output.method_calls))
        self.assertLess(self.output.zoom.call_args.args[0], 0)
        self.controller.update(sample(400, "zoom"))
        self.assertGreater(self.output.zoom.call_args.args[0], 0)

    def test_swipe_selects_one_action_on_dominant_axis(self):
        variants = {
            (1, 0): ("ctrl", "tab"),
            (-1, 0): ("ctrl", "shift", "tab"),
            (0, -1): ("win", "tab"),
            (0, 1): ("win", "d"),
            (1, 1): None,
        }
        for (dx, dy), keys in variants.items():
            with self.subTest(dx=dx, dy=dy):
                self.output.reset_mock()
                self.controller.reset()
                self.start("swipe")
                for i in range(1, 8):
                    self.controller.update(sample(100 + i * 45, "swipe", offset=(i * dx * 13, i * dy * 13)))
                self.assertEqual(self.output.method_calls, [] if keys is None else [call.hotkey(*keys)])

    def test_unrecognized_pose_provides_hint_without_moving(self):
        self.start()
        status = self.controller.update(sample(150, "fist"))
        self.assertIn("Raise your index", status)
        self.assertEqual(self.output.method_calls, [])


if __name__ == "__main__":
    unittest.main()
