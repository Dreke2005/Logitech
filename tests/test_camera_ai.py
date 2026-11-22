"""The camera's own face detection, read through the SDK, tells where the writer sits."""
import hermetic  # noqa: F401
import os
import time
import unittest
from unittest.mock import patch

import paper_camera_ai
from paper_camera_ai import CameraAI, writer_turn


class FakeClient:
    results = []

    def __enter__(self):
        return self

    def __exit__(self, *_):
        return False

    def get_firmware_version(self):
        return "VERSIONMAGIC<9.0> hackathon-1.5.2 rgb abc"

    def set_ai_config(self, **flags):
        FakeClient.flags = flags
        return True

    def get_latest_ai_data(self):
        return FakeClient.results.pop(0) if FakeClient.results else None


class WriterSideTests(unittest.TestCase):
    def test_a_visible_face_is_a_writer_across_the_desk(self):
        size = (1920, 1080)
        self.assertEqual(writer_turn((400, 200, 380, 320), size), 180)  # upper left, as seen live
        self.assertEqual(writer_turn((900, 0, 150, 150), size), 180)
        self.assertEqual(writer_turn((1500, 100, 200, 200), size), 180)
        self.assertEqual(writer_turn((0, 480, 150, 150), size), 270)  # at the left edge, mid height
        self.assertEqual(writer_turn((1770, 480, 150, 150), size), 90)

    def test_results_are_polled_in_the_background(self):
        FakeClient.results = [None, {"plugin_loaded": False, "faces": []},
                              {"plugin_loaded": True, "faces": [{"bbox": [400, 200, 380, 320], "confidence": 0.9}]},
                              {"plugin_loaded": True, "faces": []}]
        with patch.dict(os.environ, {"LOGIWEBCAM_USER": "user", "LOGIWEBCAM_PASSWORD": "secret"}), \
                patch("logiwebcam.LogiWebcamClient", FakeClient):
            ai = CameraAI(interval=0.01)
            self.assertTrue(ai.start())
            deadline = time.monotonic() + 3
            while ai.face is None and time.monotonic() < deadline:
                time.sleep(0.01)
            ai.stop()
        self.assertEqual(ai.face, (400.0, 200.0, 380.0, 320.0))
        self.assertEqual(ai.writer_rotation((1920, 1080)), 180)
        self.assertEqual(FakeClient.flags, {"enable_face_detection": True, "enable_face_mesh": False})
        self.assertIn("1.5.2", ai.status())
        # A face seen long ago says nothing about now; a doubtful one is not used either.
        self.assertIsNone(ai.writer_rotation((1920, 1080), within=-1))
        self.assertIsNone(ai.writer_rotation((1920, 1080), confidence=0.95))

    def test_without_credentials_nothing_is_started(self):
        with patch.dict(os.environ, {}, clear=False), patch("winreg.OpenKey", side_effect=OSError):
            os.environ.pop("LOGIWEBCAM_USER", None)
            os.environ.pop("LOGIWEBCAM_PASSWORD", None)
            ai = CameraAI()
            self.assertFalse(ai.start())
            self.assertIn("not set", ai.status())
            self.assertIsNone(ai.writer_rotation((1920, 1080)))
        self.assertIsNotNone(paper_camera_ai)


if __name__ == "__main__":
    unittest.main()
