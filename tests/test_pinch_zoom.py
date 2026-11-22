import hermetic  # noqa: F401
import unittest

from pinch_zoom import MAX_NOTCHES, PinchZoom


class PinchZoomTests(unittest.TestCase):
    def test_small_strokes_are_sensitive_but_a_single_threshold_spike_is_ignored(self):
        zoom = PinchZoom()
        apertures = [0.60, 0.695, 0.62, 0.587, 0.687, 0.692, 0.691]
        outputs = [zoom.update(distance, frame * 35) for frame, distance in enumerate(apertures)]
        self.assertEqual(outputs, [0, 0, 0, 0, 0, 1, 0])
        self.assertEqual(zoom.direction, 1)

    def test_short_repeat_strokes_never_undo_previous_zoom(self):
        for direction in (1, -1):
            with self.subTest(direction=direction):
                zoom = PinchZoom()
                # Return past the previous stroke start, then immediately go
                # in the intended direction again without releasing the pose.
                offsets = (0, 0.105, 0.107, 0.026, -0.03, 0.06, 0.062, -0.018, 0.09, 0.091)
                outputs = [
                    zoom.update(0.7 + direction * offset, frame * 32)
                    for frame, offset in enumerate(offsets)
                ]
                self.assertEqual(outputs, [0, 0, direction, 0, 0, 0, direction, 0, 0, direction])

    def test_flat_hand_completes_zoom_in_and_next_pinch_rebases(self):
        zoom = PinchZoom()
        zoom.update(0.3, 0)
        self.assertGreater(zoom.update(0.5, 40), 0)
        self.assertTrue(zoom.complete("flat"))
        self.assertEqual(zoom.direction, 0)
        self.assertEqual(zoom.update(0.78, 305), 0)
        self.assertLess(zoom.update(0.64, 345), 0)

    def test_either_pose_resets_zoom_out(self):
        for pose in ("flat", "fist"):
            with self.subTest(pose=pose):
                zoom = PinchZoom()
                zoom.update(0.78, 0)
                self.assertLess(zoom.update(0.64, 40), 0)
                self.assertTrue(zoom.complete(pose))
                self.assertEqual(zoom.direction, 0)
                self.assertEqual(zoom.update(0.26, 120), 0)
                self.assertGreater(zoom.update(0.48, 160), 0)

    def test_fist_clears_pending_motion_before_first_notch(self):
        zoom = PinchZoom()
        zoom.update(0.45, 0)
        self.assertEqual(zoom.update(0.555, 35), 0)
        self.assertEqual(zoom.direction, 0)
        self.assertTrue(zoom.complete("fist"))
        self.assertEqual(zoom.update(0.555, 70), 0)
        self.assertEqual(zoom.update(0.555, 105), 0)

    def test_nonzoom_and_tracking_loss_never_unlock_direction(self):
        zoom = PinchZoom()
        zoom.update(0.3, 0)
        self.assertGreater(zoom.update(0.52, 40), 0)
        for now in (80, 120, 165, 240, 310):
            zoom.inactive(now)
        self.assertFalse(zoom.complete("unrecognized"))
        self.assertEqual(zoom.direction, 1)
        self.assertEqual(zoom.update(0.64, 345), 0)
        self.assertEqual(zoom.update(0.32, 385), 0)
        self.assertGreater(zoom.update(0.47, 425), 0)

    def test_tracking_gap_and_jump_rebase_without_emitting_a_zoom_command(self):
        zoom = PinchZoom()
        zoom.update(0.3, 0)
        zoom.update(0.48, 40)
        self.assertEqual(zoom.update(0.19, 340), 0)
        self.assertEqual(zoom.update(1.25, 380), 0)
        self.assertEqual(zoom.update(0.19, 420), 0)
        self.assertEqual(zoom.direction, 1)
        self.assertGreater(zoom.update(0.35, 460), 0)

    def test_fast_strokes_are_capped_and_do_not_leave_a_backlog(self):
        zoom = PinchZoom()
        self.assertEqual(zoom.update(0.24, 0), 0)
        self.assertEqual(zoom.update(0.84, 33), MAX_NOTCHES)
        self.assertEqual(zoom.update(0.84, 66), 0)
        self.assertEqual(zoom.update(0.84, 99), 0)


if __name__ == "__main__":
    unittest.main()
