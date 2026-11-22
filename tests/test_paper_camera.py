import hermetic  # noqa: F401
import unittest

from paper_camera_check import choose_camera


class CameraChoiceTests(unittest.TestCase):
    def test_logiinsight_rgb_is_preferred_and_a_webcam_is_the_fallback(self):
        names = ["Integrated Camera", "LVNext Camera Hello", "LVNext Camera CPD", "LVNext Camera"]
        self.assertEqual(choose_camera(names), (3, True))
        self.assertEqual(choose_camera(["Integrated IR Camera", "Integrated Camera"]), (1, False))
        self.assertEqual(choose_camera(["OBS Virtual Camera"]), (None, False))
        self.assertEqual(choose_camera(names, 0), (0, False))
        self.assertEqual(choose_camera(names, 9), (None, False))


if False:
    unittest.main()


class MotionBoundsTests(unittest.TestCase):
    def test_region_where_the_hand_wrote_is_used_when_no_ink_is_found(self):
        import numpy as np

        from paper_writing import TipSample, WritingSession

        page = np.full((720, 1280, 3), 240, np.uint8)
        session = WritingSession(page)
        self.assertIsNone(session.motion_bounds((1280, 720)))
        from collections import deque
        session.tips["Right"] = deque(TipSample(1 + 0.05 * i, (400 + 6 * i, 300 + (i % 5))) for i in range(40))
        for i in range(40):
            session.path.append((1 + 0.05 * i, "Right", (420 + 6 * i, 330), 40))
        left, top, right, bottom = session.motion_bounds((1280, 720))
        self.assertTrue(360 < left < 400 and 640 < right < 680, (left, right))
        self.assertTrue(bottom - top < 120)
        self.assertEqual(session.summary(), (40, 0))
        self.assertFalse(session.hand_over_page(10.0))


class CutOffReplyTests(unittest.TestCase):
    def test_a_cut_off_reply_is_retried_once_with_a_larger_budget(self):
        from unittest.mock import patch

        import local_model

        calls = []

        def fake(model, system, content, schema, image, evidence, tokens, timeout, penalty):
            calls.append((tokens, penalty, content))
            if len(calls) == 1:
                raise local_model.ModelOutputLimit("cut off")
            return {"source": "x"}

        with patch.object(local_model, "_chat", side_effect=fake):
            self.assertEqual(local_model.chat("m", "s", "Read.", {}, tokens=400), {"source": "x"})
        self.assertEqual([(tokens, penalty) for tokens, penalty, _ in calls], [(400, 1.1), (800, 1.3)])
        self.assertIn("briefly", calls[1][2])


class LiveReadingTests(unittest.TestCase):
    def test_a_reading_made_while_writing_is_reused_for_the_same_strokes(self):
        from types import SimpleNamespace
        from unittest.mock import patch

        import numpy as np

        import paper_workbench

        strokes = (np.zeros((6, 2), int), np.ones((7, 2), int))
        selection = SimpleNamespace(strokes=strokes)
        key = ((2, 13), "LaTeX", "equation")
        cached = paper_workbench.read_recording.__globals__["Reading"]("y = x", (), "trace")
        bench = SimpleNamespace(live={"key": key, "reading": cached, "pending": None, "thread": None},
                                git_repo=None, saved=None, report_folder=None)
        with patch.object(paper_workbench, "read_recording") as reader:
            self.assertIs(paper_workbench._read_tracked(bench, selection, "LaTeX", "Python", "equation", "m"), cached)
            reader.assert_not_called()
            more = SimpleNamespace(strokes=(*strokes, np.ones((3, 2), int)))
            paper_workbench._read_tracked(bench, more, "LaTeX", "Python", "equation", "m")
            reader.assert_called_once()
