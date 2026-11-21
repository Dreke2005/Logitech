"""Exercise calibration and eye gestures without opening a camera or the desktop."""

import unittest
from types import SimpleNamespace

from eye_pointer import SCORE_NAMES, TARGETS, EyePointer, extract_eye_sample


class FakeOutput:
    def __init__(self):
        self.positions = []
        self.clicks = []

    def move_to(self, x, y):
        self.positions.append((x, y))

    def click(self, button="left", count=1):
        self.clicks.append((button, count))


def sample(ms, *, gaze=(0, 0), blink=(0.1, 0.1), squint=(0.1, 0.1), head=(0.5, 0.4, 0.2)):
    return {"timestamp_ms": ms, "gaze": gaze, "head": head, "blink": blink, "squint": squint}


class EyePointerTests(unittest.TestCase):
    def calibrated(self):
        output = FakeOutput()
        pointer = EyePointer(output)
        self.assertIn("1/5", pointer.begin_calibration())
        for i, (x, y) in enumerate(TARGETS):
            gaze = ((x - 0.5) * 0.38, (y - 0.5) * 0.3)
            for dt in range(0, 365, 60):
                pointer.update(sample(i * 420 + dt, gaze=gaze))
            status = pointer.capture_target()
        self.assertIn("Eye ready", status)
        self.assertEqual(output.positions, list(TARGETS))
        self.assertFalse(output.clicks)
        return pointer, output

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

    def test_ordinary_blinks_and_a_short_one_eye_closure_do_not_click(self):
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
            for dt in range(0, 365, 60):
                pointer.update(sample(i * 420 + dt))
            status = pointer.capture_target()
        self.assertIn("calibration unclear", status)
        self.assertEqual(output.positions, list(TARGETS))
        pointer.update(sample(2500))
        self.assertEqual(output.positions, list(TARGETS))
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


if __name__ == "__main__":
    unittest.main()
