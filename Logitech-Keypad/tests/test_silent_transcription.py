import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import numpy as np

from silent_auto_avsr import read_rgb_clip
from silent_transcription import transcribe


class SilentTranscriptionTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        path = Path(directory.name)
        self.repo = path / "model repo"
        self.clip = path / "mouthed phrase.mp4"
        self.checkpoint = path / "visual.pth"
        self.python = path / "isolated Python.exe"
        for file in (self.clip, self.checkpoint, self.python):
            file.touch()
        for name in ("lightning.py", "datamodule/transforms.py", "spm/unigram/unigram5000.model",
                     "preparation/detectors/mediapipe/video_process.py"):
            file = self.repo / name
            file.parent.mkdir(parents=True, exist_ok=True)
            file.touch()

    def test_worker_receives_absolute_paths_and_returns_only_draft_text(self):
        def run(command, **options):
            self.assertEqual(command[0], str(self.python.resolve()))
            self.assertEqual(command[1], "-I")
            self.assertEqual(options["cwd"], self.repo.resolve())
            self.assertEqual(command[4:7], [str(self.clip.resolve()), str(self.checkpoint.resolve()), "cpu"])
            Path(command[-1]).write_text("  test transcript  ", encoding="utf-8")
            return subprocess.CompletedProcess(command, 0, stdout="model log", stderr="")

        with patch("silent_transcription.subprocess.run", side_effect=run):
            self.assertEqual(transcribe(self.clip, self.repo, self.python, self.checkpoint), "test transcript")

    def test_failed_backend_reports_actionable_error_instead_of_text(self):
        failure = subprocess.CompletedProcess([], 1, stdout="", stderr="ModuleNotFoundError: No module named 'torch'")
        with patch("silent_transcription.subprocess.run", return_value=failure):
            with self.assertRaisesRegex(RuntimeError, "No module named 'torch'"):
                transcribe(self.clip, self.repo, self.python, self.checkpoint)

    def test_decode_resamples_and_uses_only_bgr_frames_from_a_file(self):
        capture = Mock()
        capture.isOpened.return_value = True
        capture.get.return_value = 30
        bgr = np.full((4, 6, 3), (8, 32, 127), dtype=np.uint8)
        capture.read.side_effect = [(True, bgr)] * 30 + [(False, None)]
        with patch("cv2.VideoCapture", return_value=capture) as open_clip:
            frames = read_rgb_clip(self.clip)
        self.assertEqual(len(frames), 25)
        self.assertEqual(open_clip.call_args.args[0], str(self.clip))
        self.assertEqual(tuple(frames[0][0, 0]), (127, 32, 8))
        capture.release.assert_called_once_with()


if __name__ == "__main__":
    unittest.main()
