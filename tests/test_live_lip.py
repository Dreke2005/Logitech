import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from live_lip import LiveLipReader
from silent_auto_avsr import MIN_FRAMES, TARGET_FPS


class LiveLipReaderTests(unittest.TestCase):
    def setUp(self):
        with patch.object(LiveLipReader, "_run"):
            self.reader = LiveLipReader("model", "python.exe", "visual.pth")
        self.addCleanup(self.reader.close)

    def ready(self):
        self.reader._events.put(("ready", ""))
        self.assertIsNone(self.reader.poll())
        self.assertEqual(self.reader.phase, "ready")

    def test_press_to_capture_returns_a_draft_after_the_background_result(self):
        self.assertFalse(self.reader.start())
        self.ready()
        self.assertTrue(self.reader.start())
        frame = np.ones((8, 16, 3), dtype=np.uint8)
        self.reader.add(frame, 40)
        frame[:] = 7
        self.reader.add(frame, 40.01)
        self.reader.add(frame, 40.12)
        self.assertEqual(self.reader.frame_count, 4)
        for index in range(4, MIN_FRAMES):
            self.reader.add(frame, 40 + index / TARGET_FPS)
        self.assertTrue(self.reader.stop())
        self.assertEqual(self.reader.phase, "transcribing")
        self.assertFalse(self.reader.start())
        self.assertIsNone(self.reader.poll())
        buffered = self.reader._commands.get_nowait()
        self.assertEqual([int(image[0, 0, 0]) for image in buffered[:4]], [1, 1, 1, 7])
        self.reader._events.put(("transcript", "One short phrase"))
        self.assertEqual(self.reader.poll(), "One short phrase")
        self.assertTrue(self.reader.start())

    def test_short_phrase_is_discarded_and_valid_phrase_autostops_at_the_limit(self):
        self.ready()
        self.assertTrue(self.reader.start())
        frame = np.zeros((8, 8, 3), dtype=np.uint8)
        self.reader.add(frame, 0)
        self.assertFalse(self.reader.stop())
        self.assertEqual(self.reader.phase, "ready")
        self.assertTrue(self.reader._commands.empty())
        self.assertTrue(self.reader.start())
        with patch("live_lip.MAX_LIVE_FRAMES", MIN_FRAMES):
            for index in range(MIN_FRAMES):
                self.reader.add(frame, index / TARGET_FPS)
        self.assertEqual(self.reader.frame_count, MIN_FRAMES)
        self.assertEqual(self.reader.phase, "transcribing")
        self.reader._events.put(("retry", "Could not track a face."))
        self.assertIsNone(self.reader.poll())
        self.assertEqual(self.reader.phase, "ready")
        self.assertEqual(self.reader.status, "Could not track a face.")

    def test_missing_backend_reports_error_without_touching_capture(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            reader = LiveLipReader(root, root / "missing.exe", root / "missing.pth")
            try:
                reader._thread.join(timeout=1)
                self.assertIsNone(reader.poll())
                self.assertEqual(reader.phase, "error")
                self.assertIn("missing.exe", reader.status)
            finally:
                reader.close()
            self.assertEqual(reader.phase, "closed")


if __name__ == "__main__":
    unittest.main()
