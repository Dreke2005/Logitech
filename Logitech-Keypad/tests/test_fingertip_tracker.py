import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np

from fingertip_tracker import FingertipTracker, draw_hands, extract_hands


class FingertipTests(unittest.TestCase):
    def test_coordinates_use_actual_frame_size_for_both_hands(self):
        points = [SimpleNamespace(x=0.25, y=0.75, z=-0.02) for _ in range(21)]
        result = SimpleNamespace(
            hand_landmarks=[points, points],
            handedness=[[SimpleNamespace(category_name=side, score=0.9)]
                        for side in ("Left", "Right")],
        )
        data = extract_hands(result, 1280, 720, 123)
        self.assertEqual(data["frame_size"], [1280, 720])
        self.assertEqual(len(data["hands"]), 2)
        for hand in data["hands"]:
            self.assertEqual(len(hand["fingertips"]), 5)
            self.assertEqual(hand["fingertips"]["index"]["pixel"], [320, 540])
        frame = np.zeros((720, 1280, 3), dtype=np.uint8)
        draw_hands(frame, data)
        self.assertTrue(frame[540, 320].any())

    def test_empty_detection_has_no_previous_fingertips(self):
        result = SimpleNamespace(hand_landmarks=[], handedness=[])
        self.assertEqual(extract_hands(result, 640, 480, 1)["hands"], [])

    @patch("fingertip_tracker.time.monotonic_ns", return_value=1_000_000)
    def test_same_clock_tick_still_produces_increasing_model_timestamps(self, _clock):
        tracker = object.__new__(FingertipTracker)
        tracker.mp = Mock()
        tracker.detector = Mock()
        tracker.detector.detect_for_video.return_value = SimpleNamespace(hand_landmarks=[], handedness=[])
        tracker.last_timestamp = -1
        frame = np.zeros((48, 64, 3), dtype=np.uint8)
        first = tracker.process(frame)
        second = tracker.process(frame)
        self.assertEqual(first["timestamp_ms"], 1)
        self.assertEqual(second["timestamp_ms"], 2)


if __name__ == "__main__":
    unittest.main()
