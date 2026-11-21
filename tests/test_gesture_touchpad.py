import unittest
from unittest.mock import Mock, call

from gesture_touchpad import TouchpadController


def sample(timestamp, pose="point", offset=(0, 0), thumb=None):
    tracking = {"timestamp_ms": timestamp, "frame_size": [640, 480], "hands": []}
    if pose == "lost":
        return tracking
    points = [(310, 300), (290, 286), (272, 275), (265, 260), (255, 254)]
    if pose == "point":
        points[4] = (278, 270)
    fingers = {
        "point": (True, False, False, False),
        "scroll": (True, True, True, True),
        "flat": (True, True, True, True),
        "gathered": (True, True, True, True),
        "gathering": (True, True, True, True),
        "two": (True, True, False, False),
        "zoom": (True, False, False, False),
        "swipe": (True, True, True, False),
        "fist": (False, False, False, False),
    }[pose]
    for x, raised in zip((284, 310, 334, 355), fingers):
        points.extend(((x, 250), (x, 228), (x, 214), (x, 201 if raised else 255)))
    if pose == "flat":
        for start, spread in zip((5, 9, 13, 17), (-27, -8, 8, 27)):
            for index, weight in zip(range(start + 1, start + 4), (0.4, 0.7, 1.0)):
                x, y = points[index]
                points[index] = x + round(spread * weight), y
    if pose in ("gathered", "gathering"):
        for base, target in zip((5, 9, 13, 17), ((314, 211), (318, 203), (324, 205), (328, 213))):
            base_x, base_y = points[base]
            for index, weight in zip(range(base + 1, base + 4), (0.4, 0.7, 1.0)):
                points[index] = (round(base_x + (target[0] - base_x) * weight),
                                 round(base_y + (target[1] - base_y) * weight))
        if pose == "gathered":
            points[4] = (310, 215)
    if thumb is not None:
        points[4] = thumb
    dx, dy = offset
    tracking["hands"] = [{"landmarks": [{"pixel": [x + dx, y + dy]} for x, y in points]}]
    return tracking


class TouchpadTests(unittest.TestCase):
    def setUp(self):
        self.output = Mock(spec=["scroll", "zoom", "hotkey"])
        self.controller = TouchpadController(self.output)

    def start(self, pose="scroll"):
        for timestamp in (0, 50, 100):
            self.controller.update(sample(timestamp, pose))

    def test_tracking_loss_rebases_scroll_before_emitting(self):
        self.start()
        self.controller.update(sample(150, "scroll", offset=(28, 28)))
        self.assertEqual(self.output.method_calls, [call.scroll(vertical=-1, horizontal=1)])
        self.output.reset_mock()
        self.controller.update(sample(200, "lost"))
        for timestamp in (250, 300, 350):
            self.controller.update(sample(timestamp, "scroll", offset=(-30, 0)))
        self.assertEqual(self.output.method_calls, [])

    def test_idle_second_hand_does_not_block_scroll(self):
        for timestamp in (0, 50, 100, 150):
            tracking = sample(timestamp, "scroll", offset=(0, 30 if timestamp == 150 else 0))
            tracking["hands"].insert(0, sample(timestamp, "fist", offset=(100, 0))["hands"][0])
            self.controller.update(tracking)
        self.output.scroll.assert_called_once_with(vertical=-1, horizontal=0)

    def test_slow_whole_hand_movement_accumulates_into_scroll(self):
        self.start()
        for i in range(1, 7):
            self.controller.update(sample(100 + i * 40, "scroll", offset=(0, 3 * i)))
        self.output.scroll.assert_called_once_with(vertical=-1, horizontal=0)

    def test_either_reset_stops_scroll_until_open_hand_sets_a_new_baseline(self):
        for pose in ("fist", "flat"):
            with self.subTest(pose=pose):
                self.controller.reset()
                self.output.reset_mock()
                self.start()
                self.controller.update(sample(150, "scroll", offset=(0, 26)))
                self.output.scroll.assert_called_once_with(vertical=-1, horizontal=0)
                self.output.reset_mock()
                for i, timestamp in enumerate(range(200, 400, 50)):
                    self.controller.update(sample(timestamp, pose, offset=(0, 30 + 13 * i)))
                for timestamp in (400, 450, 500):
                    self.controller.update(sample(timestamp, "scroll", offset=(0, 70)))
                self.output.scroll.assert_not_called()
                self.controller.update(sample(550, "scroll", offset=(0, 50)))
                self.output.scroll.assert_called_once_with(vertical=1, horizontal=0)

    def test_coordinated_fingertips_scroll_even_with_still_palm(self):
        self.start()
        tracking = sample(150, "scroll")
        points = tracking["hands"][0]["landmarks"]
        # A single swinging finger is not a scroll gesture.
        points[8]["pixel"][1] += 20
        self.controller.update(tracking)
        self.assertEqual(self.output.method_calls, [])
        tracking = sample(200, "scroll")
        points = tracking["hands"][0]["landmarks"]
        for tip in (8, 12, 16):
            points[tip]["pixel"][1] += 20
        self.controller.update(tracking)
        self.output.scroll.assert_called_once_with(vertical=-1, horizontal=0)

    def test_open_hand_tolerates_a_foreshortened_pinky_without_switching_tabs(self):
        for i in range(8):
            tracking = sample(50 * i, "scroll", offset=(i * 12, 0))
            points = tracking["hands"][0]["landmarks"]
            joint_x, joint_y = points[18]["pixel"]
            # The pinky projects sideways rather than along the other fingers
            # and briefly crosses its joint in the image without fully tucking.
            points[20]["pixel"] = [joint_x + 8, joint_y - 1 if i < 4 else joint_y + 1]
            self.controller.update(tracking)
        self.assertGreater(sum(action.kwargs["horizontal"] for action in self.output.scroll.call_args_list), 0)
        self.output.hotkey.assert_not_called()

    def test_pointing_and_two_finger_poses_have_no_desktop_effect(self):
        self.start("point")
        for i, timestamp in enumerate(range(150, 550, 50)):
            self.controller.update(sample(timestamp, offset=(3 * i, 3 * i)))
        for i, timestamp in enumerate(range(550, 950, 50)):
            self.controller.update(sample(timestamp, "two", offset=(3 * i, 3 * i)))
        self.assertEqual(self.output.method_calls, [])

    def test_pause_discards_zoom_baseline(self):
        self.start("zoom")
        self.controller.reset()
        for timestamp in (150, 200, 250):
            self.controller.update(sample(timestamp, "zoom", thumb=(264, 212)))
        self.assertEqual(self.output.method_calls, [])

    def test_scroll_both_axes_and_zoom_as_a_separate_mode(self):
        self.start("scroll")
        self.controller.update(sample(150, "scroll", offset=(28, -28)))
        self.assertEqual(self.output.method_calls, [call.scroll(vertical=1, horizontal=1)])
        self.controller.update(sample(200, "scroll", offset=(28, 0)))
        self.assertEqual(self.output.method_calls[-1], call.scroll(vertical=-1, horizontal=0))
        self.output.reset_mock()
        for timestamp in (250, 300, 350):
            self.controller.update(sample(timestamp, "zoom"))
        self.controller.update(sample(400, "zoom", thumb=(264, 212)))
        self.assertTrue(all(action[0] == "zoom" for action in self.output.method_calls))
        self.assertLess(self.output.zoom.call_args.args[0], 0)

    def test_closing_after_zoom_in_does_not_zoom_back_out(self):
        close = (278, 210)
        for timestamp in (0, 50, 100):
            self.controller.update(sample(timestamp, "zoom", thumb=close))
        self.controller.update(sample(150, "zoom"))
        self.assertGreater(self.output.zoom.call_args.args[0], 0)
        self.output.reset_mock()
        self.controller.update(sample(200, "zoom", thumb=(265, 212)))
        self.controller.update(sample(250, "zoom", thumb=close))
        for timestamp in (300, 350, 400, 450, 500, 550):
            self.controller.update(sample(timestamp, "fist"))
        self.assertEqual(self.output.method_calls, [])

        for timestamp in (600, 650, 700):
            self.controller.update(sample(timestamp, "zoom"))
        self.controller.update(sample(750, "zoom", thumb=close))
        self.assertLess(self.output.zoom.call_args.args[0], 0)

    def test_small_pinch_strokes_can_repeat_without_changing_pose(self):
        close, spread = (278, 210), (278, 217)
        for timestamp, thumb in (
            (0, close), (50, close), (100, spread), (150, spread),
            (200, close), (250, spread), (300, spread),
        ):
            self.controller.update(sample(timestamp, "zoom", thumb=thumb))
        self.assertEqual(self.output.method_calls, [call.zoom(1), call.zoom(1)])

    def test_index_point_with_thumb_resting_on_curled_fingers_does_not_zoom(self):
        # The reach alone looks like a wide pinch. The thumb remains beside
        # the middle fingertip even as their projected aperture changes.
        for timestamp, y in ((0, 250), (50, 250), (100, 250),
                             (150, 267), (200, 267), (250, 250)):
            self.controller.update(sample(timestamp, "zoom", thumb=(309, y)))
        self.assertEqual(self.output.method_calls, [])

    def test_thumbs_up_does_not_reset_latched_zoom(self):
        def thumbs_up(timestamp, y):
            tracking = sample(timestamp, "fist", thumb=(241, y))
            tracking["hands"][0]["landmarks"][8]["pixel"][1] = 271
            return tracking

        for timestamp, y in ((0, 205), (50, 205), (100, 205), (150, 182), (200, 182)):
            self.controller.update(thumbs_up(timestamp, y))
        self.assertEqual(self.output.method_calls, [])
        self.controller.reset()
        self.start("zoom")
        self.controller.update(sample(150, "zoom", thumb=(278, 210)))
        self.output.zoom.assert_called_once()
        self.output.reset_mock()
        for timestamp in (200, 250, 300, 350):
            self.controller.update(thumbs_up(timestamp, 205))
        for timestamp, offset in ((400, 0), (450, 0), (500, 0), (550, 35)):
            self.controller.update(sample(timestamp, "scroll", offset=(0, offset)))
        self.assertEqual(self.output.method_calls, [])

    def test_either_reset_finishes_zoom_in_without_scrolling(self):
        for pose in ("flat", "fist"):
            with self.subTest(pose=pose):
                self.controller.reset()
                self.output.reset_mock()
                close = (278, 210)
                for timestamp in (0, 50, 100):
                    self.controller.update(sample(timestamp, "zoom", thumb=close))
                self.controller.update(sample(150, "zoom"))
                self.assertGreater(self.output.zoom.call_args.args[0], 0)
                self.output.reset_mock()
                for i, timestamp in enumerate(range(200, 500, 50)):
                    self.controller.update(sample(timestamp, pose, offset=(0, i * 12)))
                self.assertEqual(self.output.method_calls, [])

                for timestamp in (500, 550, 600):
                    self.controller.update(sample(timestamp, "zoom"))
                self.controller.update(sample(650, "zoom", thumb=close))
                self.assertLess(self.output.zoom.call_args.args[0], 0)

    def test_scroll_pose_and_tracking_loss_do_not_clear_zoom_out(self):
        close = (278, 210)
        self.start("zoom")
        self.controller.update(sample(150, "zoom", thumb=close))
        self.assertLess(self.output.zoom.call_args.args[0], 0)
        self.output.reset_mock()
        self.controller.tracking_lost()
        for i, timestamp in enumerate(range(200, 500, 50)):
            status = self.controller.update(sample(timestamp, "scroll", offset=(0, i * -10)))
            self.assertIn("fist", status)
        for timestamp in (500, 550, 600):
            self.controller.update(sample(timestamp, "zoom", thumb=close))
        self.controller.update(sample(650, "zoom"))
        self.assertEqual(self.output.method_calls, [])

        for timestamp in (700, 750, 800, 850):
            self.controller.update(sample(timestamp, "flat"))
        for timestamp in (900, 950, 1000):
            self.controller.update(sample(timestamp, "scroll"))
        self.controller.update(sample(1050, "scroll", offset=(0, 24)))
        self.output.scroll.assert_called_once_with(vertical=-1, horizontal=0)

    def test_swipe_only_switches_tabs_on_dominant_horizontal_axis(self):
        variants = {
            (1, 0): ("ctrl", "tab"),
            (-1, 0): ("ctrl", "shift", "tab"),
            (0, -1): None,
            (0, 1): None,
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

    def test_gathered_fingertips_open_overview_once_until_visibly_released(self):
        self.start()
        self.controller.update(sample(150, "scroll", offset=(0, 26)))
        self.output.scroll.assert_called_once()
        self.output.reset_mock()

        for timestamp in (200, 250, 300, 350):
            status = self.controller.update(sample(timestamp, "gathered", offset=(0, 55)))
            self.assertIn("Hold all five", status)
        self.assertEqual(self.output.method_calls, [])
        for timestamp in (400, 450, 500, 550):
            self.controller.update(sample(timestamp, "gathered", offset=(0, 55)))
        self.assertEqual(self.output.method_calls, [call.hotkey("ctrl", "shift", "a")])

        self.controller.update(sample(600, "lost"))
        # A single open pose between held gathered poses cannot rearm.
        self.controller.update(sample(650, "scroll", offset=(0, 50)))
        for timestamp in (700, 750, 800, 850, 900):
            self.controller.update(sample(timestamp, "gathered", offset=(0, 55)))
        self.assertEqual(len(self.output.method_calls), 1)

        for i, timestamp in enumerate(range(950, 1150, 50)):
            self.controller.update(sample(timestamp, "scroll", offset=(0, 55 - i * 18)))
        for timestamp in range(1150, 1400, 50):
            self.controller.update(sample(timestamp, "gathered"))
        self.assertEqual(self.output.method_calls, [call.hotkey("ctrl", "shift", "a")] * 2)

    def test_four_gathered_fingers_without_thumb_never_open_overview_or_scroll(self):
        self.start()
        for i, timestamp in enumerate(range(150, 450, 50)):
            status = self.controller.update(sample(timestamp, "gathering", offset=(0, 11 * i)))
            self.assertIn("thumb", status)
        self.assertEqual(self.output.method_calls, [])

    def test_two_finger_pinch_and_tips_near_the_palm_are_not_overview(self):
        for timestamp in range(0, 300, 50):
            self.controller.update(sample(timestamp, "zoom", thumb=(280, 204)))
        for timestamp in range(300, 600, 50):
            tracking = sample(timestamp, "fist")
            points = tracking["hands"][0]["landmarks"]
            for index, point in zip((4, 8, 12, 16, 20), ((309, 249), (310, 251), (318, 249), (320, 254), (325, 249))):
                points[index]["pixel"] = point
            self.controller.update(tracking)
        self.assertEqual(self.output.method_calls, [])

    def test_moving_gather_and_missing_frames_do_not_satisfy_dwell(self):
        for i, timestamp in enumerate(range(0, 400, 50)):
            self.controller.update(sample(timestamp, "gathered", offset=(i * 24, 0)))
        self.assertEqual(self.output.method_calls, [])
        for timestamp in (400, 450, 500):
            self.controller.update(sample(timestamp, "gathered", offset=(168, 0)))
        self.controller.update(sample(550, "lost"))
        for timestamp in (600, 650, 700, 750):
            self.controller.update(sample(timestamp, "gathered", offset=(168, 0)))
        self.assertEqual(self.output.method_calls, [])
        self.controller.update(sample(800, "gathered", offset=(168, 0)))
        self.output.hotkey.assert_called_once_with("ctrl", "shift", "a")

    def test_competing_navigation_hand_does_not_dispatch_tab_overview_or_scroll(self):
        for i, timestamp in enumerate(range(0, 350, 50)):
            tracking = sample(timestamp, "gathered")
            tracking["hands"].extend(sample(timestamp, "scroll", offset=(130, 10 * i))["hands"])
            self.controller.update(tracking)
        self.assertEqual(self.output.method_calls, [])

    def test_swipe_requires_horizontal_fingertips_to_follow_the_palm(self):
        self.start("swipe")
        for i, timestamp in enumerate(range(145, 460, 45), start=1):
            tracking = sample(timestamp, "swipe", offset=(13 * i, 0))
            for index in (8, 12, 16):
                tracking["hands"][0]["landmarks"][index]["pixel"][0] -= 7 * i
            status = self.controller.update(tracking)
        self.assertIn("all three fingertips", status)
        self.assertEqual(self.output.method_calls, [])

    def test_swipe_rejects_isolated_pose_shift_and_rearms_only_after_visible_release(self):
        self.start("swipe")
        for timestamp in (150, 200, 250):
            self.controller.update(sample(timestamp, "swipe", offset=(78, 0)))
        self.assertEqual(self.output.method_calls, [])

        self.controller.reset()
        self.start("swipe")
        for i in range(1, 8):
            self.controller.update(sample(100 + 45 * i, "swipe", offset=(13 * i, 0)))
        self.controller.update(sample(450, "lost"))
        for i in range(8):
            self.controller.update(sample(500 + 45 * i, "swipe", offset=(-13 * i, 0)))
        self.output.hotkey.assert_called_once_with("ctrl", "tab")

        for timestamp in (850, 900, 950):
            self.controller.update(sample(timestamp, "point", offset=(30, 0)))
        for i in range(9):
            self.controller.update(sample(1000 + 45 * i, "swipe", offset=(30 - 13 * i, 0)))
        self.assertEqual(self.output.method_calls, [call.hotkey("ctrl", "tab"), call.hotkey("ctrl", "shift", "tab")])

    def test_unrecognized_pose_provides_hint_without_moving(self):
        self.start()
        status = self.controller.update(sample(150, "point"))
        self.assertIn("Open hand to scroll", status)
        self.assertEqual(self.output.method_calls, [])


if __name__ == "__main__":
    unittest.main()
