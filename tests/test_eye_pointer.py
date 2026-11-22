"""Exercise calibration and eye gestures without opening a camera or the desktop."""

import hermetic  # noqa: F401
import unittest
from types import SimpleNamespace

import numpy as np

from eye_pointer import (
    POSE_DWELL_MS,
    POSE_WAIT_MS,
    SCORE_NAMES,
    TARGET_DWELL_MS,
    TARGETS,
    EyePointer,
    extract_eye_sample,
)
from gaze_pose import POSE_DIRECTIONS, orientation_from_matrix

TARGET_INTERVAL_MS = TARGET_DWELL_MS + 150


class FakeOutput:
    def __init__(self):
        self.positions = []
        self.clicks = []

    def move_to(self, x, y):
        self.positions.append((x, y))

    def click(self, button="left", count=1):
        self.clicks.append((button, count))


def sample(ms, *, gaze=(0, 0), blink=(0.1, 0.1), squint=(0.1, 0.1),
           head=(0.5, 0.4, 0.2), orientation=None):
    return {"timestamp_ms": ms, "gaze": gaze, "head": head, "orientation": orientation,
            "blink": blink, "squint": squint}


def linear_gaze(x, y):
    return (x - 0.5) * 0.38, (y - 0.5) * 0.3


def curved_gaze(x, y):
    """Synthetic iris readings with compression near one edge and axis coupling."""
    u = (-0.47 + np.sqrt(0.47**2 + 0.36 * (x - 0.5))) / 0.18
    v = (y - 0.5 - 0.065 * u * u) / (0.43 + 0.07 * u)
    return 0.17 * u, 0.12 * v


def feed_target(pointer, start, gaze, noise=0.0, orientation=None, dwell=TARGET_DWELL_MS):
    for step, dt in enumerate(range(0, dwell + 1, 50)):
        drift = noise * (-1) ** step
        status = pointer.update(sample(start + dt, gaze=(gaze[0] + drift, gaze[1] - drift),
                                       orientation=orientation))
    return status


class EyePointerTests(unittest.TestCase):
    def calibrated(self, gaze_at=linear_gaze, noise=0.0):
        output = FakeOutput()
        pointer = EyePointer(output)
        self.assertFalse(pointer.ready)
        self.assertFalse(pointer.calibrating)
        self.assertIn(f"1/{len(TARGETS)}", pointer.begin_calibration())
        self.assertFalse(pointer.ready)
        self.assertTrue(pointer.calibrating)
        for i, (x, y) in enumerate(TARGETS):
            self.assertEqual(pointer.target, (x, y))
            status = feed_target(pointer, i * TARGET_INTERVAL_MS, gaze_at(x, y), noise)
        self.assertIn("Eye ready", status)
        self.assertTrue(pointer.ready)
        self.assertFalse(pointer.calibrating)
        self.assertFalse(output.positions)
        self.assertFalse(output.clicks)
        return pointer, output

    def test_calibration_improves_held_out_curved_positions_and_keeps_affine_for_linear_gaze(self):
        self.assertFalse(self.calibrated()[0]._fit.quadratic)
        pointer, _ = self.calibrated(curved_gaze, noise=0.003)
        self.assertTrue(pointer._fit.quadratic)
        calibration_gaze = np.array([(*curved_gaze(x, y), 1) for x, y in TARGETS])
        affine = np.linalg.lstsq(calibration_gaze, TARGETS, rcond=None)[0]
        # Positions strictly between calibration points, including edge-adjacent
        # areas where interpolation is harder than at a center fixation.
        targets = np.array([(x, y) for x in (0.15, 0.31, 0.68, 0.85) for y in (0.15, 0.31, 0.68, 0.85)])
        gaze = np.array([curved_gaze(x, y) for x, y in targets])
        curved_error = np.sqrt(np.mean(np.sum((pointer._fit.predict(gaze) - targets) ** 2, axis=1)))
        affine_error = np.sqrt(np.mean(np.sum((np.column_stack((gaze, np.ones(len(gaze)))) @ affine
                                             - targets) ** 2, axis=1)))
        self.assertLess(curved_error, 0.015)
        self.assertLess(curved_error, affine_error * 0.6)

    def test_curved_calibration_holds_still_with_noisy_fixation_near_an_edge(self):
        pointer, output = self.calibrated(curved_gaze, noise=0.004)
        gaze = curved_gaze(0.85, 0.20)
        for ms in (8000, 8040, 8080, 8120, 8160):
            pointer.update(sample(ms, gaze=gaze))
        self.assertEqual(len(output.positions), 1)
        np.testing.assert_allclose(output.positions[0], (0.85, 0.20), atol=0.015)
        for i in range(20):
            jitter = 0.004 * (-1) ** i
            point = (gaze[0] + jitter, gaze[1] - jitter)
            if i == 8:
                point = curved_gaze(0.55, 0.50)
            pointer.update(sample(8200 + i * 35, gaze=point))
        self.assertEqual(len(output.positions), 1)

    def test_calibrated_cursor_and_intentional_squint_freezes_then_clicks_once(self):
        pointer, output = self.calibrated()
        for ms in range(2000, 2230, 55):
            pointer.update(sample(ms))
        cursor = output.positions[-1]
        self.assertAlmostEqual(cursor[0], 0.5)
        self.assertAlmostEqual(cursor[1], 0.5)
        count = len(output.positions)
        for ms in range(2280, 2630, 55):
            pointer.update(sample(ms, gaze=(0.12, -0.10), squint=(0.86, 0.13)))
        self.assertEqual(len(output.positions), count)
        self.assertFalse(output.clicks)
        pointer.update(sample(2670))
        pointer.update(sample(2730))
        self.assertEqual(output.clicks, [("left", 1)])

    def test_quick_one_eye_blink_clicks_on_reopening_and_freezes_pointer(self):
        pointer, output = self.calibrated()
        for ms in range(2000, 2230, 35):
            pointer.update(sample(ms))
        count = len(output.positions)
        pointer.update(sample(2245, blink=(0.9, 0.10)))
        pointer.update(sample(2280, blink=(0.75, 0.10)))
        pointer.update(sample(2310, blink=(0.39, 0.10)))  # Partly reopened.
        self.assertFalse(output.clicks)
        self.assertEqual(len(output.positions), count)
        pointer.update(sample(2340))
        self.assertEqual(output.clicks, [("left", 1)])
        self.assertEqual(len(output.positions), count)

        for ms in range(2360, 2540, 35):
            pointer.update(sample(ms))
        pointer.update(sample(2550, blink=(0.1, 0.9)))
        pointer.update(sample(2585, blink=(0.1, 0.9)))
        pointer.update(sample(2620))
        self.assertEqual(output.clicks, [("left", 1)])  # Within the post-click cooldown.

    def test_fixation_holds_through_noise_and_spikes_but_follows_a_new_target(self):
        pointer, output = self.calibrated()
        for ms in range(2000, 2180, 33):
            pointer.update(sample(ms))
        self.assertEqual(len(output.positions), 1)
        for i in range(18):
            gaze = (0.12, -0.12) if i == 8 else ((-1) ** i * 0.003, 0.002)
            pointer.update(sample(2200 + i * 33, gaze=gaze))
        self.assertEqual(len(output.positions), 1)
        for i in range(13):
            pointer.update(sample(2800 + i * 33, gaze=(0.095, -0.09)))
        self.assertGreater(output.positions[-1][0], 0.72)
        self.assertLess(output.positions[-1][1], 0.23)

    def test_pose_loss_does_not_jump_to_the_first_returning_frame(self):
        pointer, output = self.calibrated()
        for ms in (2000, 2040, 2080):
            pointer.update(sample(ms))
        self.assertEqual(len(output.positions), 1)
        pointer.update(sample(2120, gaze=None))
        pointer.update(sample(2160, gaze=(0.14, 0.12)))
        pointer.update(sample(2200))
        pointer.update(sample(2240))
        self.assertEqual(len(output.positions), 1)

    def test_ordinary_blinks_and_a_single_false_closure_frame_do_not_click(self):
        pointer, output = self.calibrated()
        for ms in range(2000, 2240, 60):
            pointer.update(sample(ms))
        for ms in range(2240, 2640, 80):
            pointer.update(sample(ms, blink=(0.94, 0.94)))
        for ms in range(2640, 2880, 60):
            pointer.update(sample(ms))
        pointer.update(sample(2880, blink=(0.92, 0.14)))
        pointer.update(sample(2970))
        self.assertFalse(output.clicks)

    def test_staggered_bilateral_blink_does_not_become_a_click(self):
        pointer, output = self.calibrated()
        for ms in range(2000, 2240, 60):
            pointer.update(sample(ms))
        pointer.update(sample(2240, blink=(0.85, 0.14)))
        pointer.update(sample(2275, blink=(0.93, 0.28)))
        pointer.update(sample(2310, blink=(0.91, 0.72)))
        pointer.update(sample(2345, blink=(0.15, 0.77)))
        pointer.update(sample(2380))
        self.assertFalse(output.clicks)

    def test_face_loss_and_pausing_cancel_pending_click_but_keep_calibration(self):
        for interrupted in ("missing face", "camera stall", "controls paused"):
            with self.subTest(interrupted=interrupted):
                pointer, output = self.calibrated()
                for ms in range(2000, 2240, 60):
                    pointer.update(sample(ms))
                for ms in range(2240, 2590, 60):
                    pointer.update(sample(ms, blink=(0.91, 0.12)))
                if interrupted == "missing face":
                    pointer.update(sample(2600, gaze=None))
                elif interrupted == "controls paused":
                    pointer.pause()
                reopen = 2900 if interrupted == "camera stall" else 2640
                self.assertIn("Eye ready", pointer.update(sample(reopen)))
                self.assertFalse(output.clicks)

    def test_invalid_calibration_does_not_move_or_click_after_target_sequence(self):
        output = FakeOutput()
        pointer = EyePointer(output)
        pointer.begin_calibration()
        for i in range(len(TARGETS)):
            status = feed_target(pointer, i * TARGET_INTERVAL_MS, (0, 0))
        self.assertIn("calibration unclear", status)
        self.assertFalse(output.positions)
        pointer.update(sample(2500))
        self.assertFalse(output.positions)
        self.assertFalse(output.clicks)

    def test_visible_targets_advance_with_weak_vertical_motion_and_occasional_blinks(self):
        output = FakeOutput()
        pointer = EyePointer(output)
        pointer.begin_calibration()
        for index, (x, y) in enumerate(TARGETS):
            self.assertEqual(pointer.target, (x, y))
            self.assertEqual(pointer.target_progress, 0)
            for step in range(20):
                blink = (0.9, 0.9) if step in (7, 8, 14) else (0.14, 0.14)
                pointer.update(sample(TARGET_INTERVAL_MS * index + 100 * step, blink=blink,
                                      squint=(0.53, 0.3),
                                      gaze=((x - 0.5) * 0.33, (y - 0.5) * 0.028)))
        self.assertTrue(pointer.ready)
        self.assertIsNone(pointer.target)
        self.assertFalse(output.positions)

    def test_lost_tracking_and_shaky_samples_hold_the_same_target(self):
        pointer = EyePointer(FakeOutput())
        pointer.begin_calibration()
        for timestamp in range(0, 1800, 50):
            pointer.update(sample(timestamp))
        self.assertEqual(pointer.target, TARGETS[0])
        self.assertGreater(pointer.target_progress, 0.8)
        for timestamp in range(1800, 2400, 50):
            pointer.update(sample(timestamp, gaze=None))
        self.assertEqual(pointer.target, TARGETS[0])
        self.assertEqual(pointer.target_progress, 0)
        for timestamp in range(2400, 5000, 50):
            status = pointer.update(sample(timestamp, gaze=(0.07 * (-1) ** (timestamp // 50), 0)))
        self.assertEqual(pointer.target, TARGETS[0])
        self.assertIn("Hold your gaze steady.", status)
        for timestamp in range(5000, 7200, 50):
            pointer.update(sample(timestamp))
            if pointer.target != TARGETS[0]:
                break
        self.assertEqual(pointer.target, TARGETS[1])

    def test_calibration_rechecks_an_individual_missed_dot(self):
        output = FakeOutput()
        pointer = EyePointer(output)
        pointer.begin_calibration()
        missed = 13
        for i, target in enumerate(TARGETS):
            gaze = curved_gaze(*(TARGETS[0] if i == missed else target))
            status = feed_target(pointer, i * TARGET_INTERVAL_MS, gaze, noise=0.002)
        self.assertIn("recheck dot", status)
        self.assertEqual(pointer.target, TARGETS[missed])
        self.assertFalse(pointer.ready)
        status = feed_target(pointer, len(TARGETS) * TARGET_INTERVAL_MS,
                             curved_gaze(*TARGETS[missed]), noise=0.002)
        self.assertIn("Eye ready", status)
        self.assertTrue(pointer.ready)
        self.assertFalse(output.positions)
        self.assertFalse(output.clicks)

    def test_extracts_iris_relative_to_eye_and_ignores_multiple_faces(self):
        points = [SimpleNamespace(x=0.5, y=0.5) for _ in range(478)]
        for i, x, y in ((33, 0.2, 0.3), (133, 0.3, 0.3), (468, 0.26, 0.32),
                        (362, 0.6, 0.3), (263, 0.7, 0.3), (473, 0.66, 0.32)):
            points[i] = SimpleNamespace(x=x, y=y)
        scores = [SimpleNamespace(category_name=name, score=0.2) for name in SCORE_NAMES]
        result = SimpleNamespace(face_landmarks=[points], face_blendshapes=[scores])
        observation = extract_eye_sample(result, 10, (640, 480))
        self.assertAlmostEqual(observation["gaze"][0], 0.10)
        self.assertAlmostEqual(observation["gaze"][1], 0.15)
        self.assertEqual(observation["squint"], (0.2, 0.2))
        result.face_landmarks.append(points)
        self.assertIsNone(extract_eye_sample(result, 11, (640, 480))["gaze"])

    def test_head_matrix_orientation_ignores_translation_and_scale(self):
        yaw = 0.17
        cos, sin = np.cos(yaw), np.sin(yaw)
        matrix = np.eye(4)
        matrix[:3, :3] = 1.08 * np.array(((cos, 0, sin), (0, 1, 0), (-sin, 0, cos)))
        matrix[:3, 3] = (1.4, 3.0, -15.0)
        np.testing.assert_allclose(orientation_from_matrix(matrix), (yaw, 0, 0), atol=1e-9)
        self.assertIsNone(orientation_from_matrix(np.zeros((4, 4))))
        self.assertIsNone(orientation_from_matrix(np.diag((-1, 1, 1, 1))))

    def test_guided_head_calibration_reduces_held_out_rotation_error_and_jitter(self):
        output = FakeOutput()
        pointer = EyePointer(output)
        start = self._begin_head_calibration(pointer)
        drift = np.array(((0.28, 0.045), (-0.03, 0.24)))
        orientations = ((-0.19, 0.018, 0.04), (0.21, -0.020, 0.02),
                        (0.015, 0.19, 0.025), (-0.025, -0.21, -0.025))
        for step, (orientation, (prompt, _)) in enumerate(zip(orientations, POSE_DIRECTIONS)):
            self.assertEqual(pointer.target, TARGETS[0])
            self.assertIn(prompt.capitalize(), pointer._target_hint())
            gaze = np.array(orientation[:2]) @ drift
            status = feed_target(pointer, start + step * 2500, gaze, noise=0.001,
                                 orientation=orientation, dwell=POSE_DWELL_MS)
        self.assertIn("head assist", status)
        self.assertTrue(pointer.ready)
        self.assertEqual(output.positions, [])
        # Three untrained screen points with head angles combining both axes.
        targets = np.array(((0.23, 0.34), (0.81, 0.26), (0.62, 0.67)))
        angles = np.array(((0.14, 0.12, 0.06), (-0.10, -0.13, -0.04), (0.06, -0.11, 0.07)))
        readings = np.array([linear_gaze(*point) for point in targets]) + angles[:, :2] @ drift
        unassisted = np.linalg.norm(pointer._fit.predict(readings) - targets, axis=1)
        predicted = pointer._fit.predict(pointer._head_assist.correct(readings, angles))
        assisted = np.linalg.norm(predicted - targets, axis=1)
        self.assertLess(assisted.max(), 0.018)
        self.assertLess(assisted.mean(), unassisted.mean() * 0.15)

        # Fixed gaze stays in the same spot through small physical head shifts.
        for index in range(20):
            perturbation = np.array((0.014 * (-1) ** index, 0.01 * (-1) ** (index // 2), 0))
            angle = angles[0] + perturbation
            reading = np.array(linear_gaze(*targets[0])) + angle[:2] @ drift
            pointer.update(sample(80000 + 40 * index, gaze=reading, orientation=angle))
        self.assertEqual(len(output.positions), 1)
        np.testing.assert_allclose(output.positions[0], targets[0], atol=0.018)

        status = pointer.update(sample(80900, gaze=(0.12, 0.14), orientation=(0.55, 0, 0)))
        self.assertIn("calibration angles", status)
        self.assertEqual(len(output.positions), 1)

    def test_unavailable_optional_turns_finish_with_existing_eye_only_calibration(self):
        output = FakeOutput()
        pointer = EyePointer(output)
        start = self._begin_head_calibration(pointer)
        for step in range(len(POSE_DIRECTIONS)):
            timestamp = start + step * (POSE_WAIT_MS + 50)
            pointer.update(sample(timestamp, gaze=None))
            status = pointer.update(sample(timestamp + POSE_WAIT_MS, gaze=None))
        self.assertTrue(pointer.ready)
        self.assertIn("Eye ready", status)
        self.assertIsNone(pointer._head_assist)
        self.assertEqual(output.positions, [])

    def _begin_head_calibration(self, pointer):
        pointer.begin_calibration()
        for index, (x, y) in enumerate(TARGETS):
            status = feed_target(pointer, index * TARGET_INTERVAL_MS, linear_gaze(x, y),
                                 orientation=(0, 0, 0))
        self.assertIn("Head assist 1/4", status)
        self.assertTrue(pointer.calibrating)
        self.assertFalse(pointer.ready)
        return len(TARGETS) * TARGET_INTERVAL_MS


if __name__ == "__main__":
    unittest.main()
