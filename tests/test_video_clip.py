import hermetic  # noqa: F401
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

from video_clip import FPS, VideoClip


class VideoClipTests(unittest.TestCase):
    def test_user_started_clip_keeps_frame_timing_without_a_tracking_freeze(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "test.mp4"
            clip = VideoClip(path)
            frame = np.full((64, 96, 3), 90, dtype=np.uint8)
            clip.add_frame(frame, 50)
            self.assertFalse(path.exists())
            clip.toggle()
            try:
                for timestamp in (51, 51.04, 51.12, 53):
                    clip.add_frame(frame, timestamp)
            finally:
                clip.finish()
            capture = cv2.VideoCapture(str(path))
            try:
                self.assertTrue(capture.isOpened())
                self.assertAlmostEqual(capture.get(cv2.CAP_PROP_FPS), FPS)
                self.assertEqual(int(capture.get(cv2.CAP_PROP_FRAME_COUNT)), 5)
                success, stored = capture.read()
                self.assertTrue(success)
                self.assertEqual(stored.shape, frame.shape)
            finally:
                capture.release()

    def test_recording_never_overwrites_an_existing_path(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "test.mp4"
            path.write_bytes(b"existing data")
            clip = VideoClip(path)
            clip.toggle()
            with self.assertRaises(FileExistsError):
                clip.add_frame(np.zeros((32, 32, 3), dtype=np.uint8), 10)
            self.assertEqual(path.read_bytes(), b"existing data")
