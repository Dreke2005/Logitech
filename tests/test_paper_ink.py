"""The ink written during a recording is registered live, independent of the pen-tip estimate."""
import hermetic  # noqa: F401
import unittest
from unittest.mock import patch

import cv2
import numpy as np

from paper_conversion import ink_usable, read_recording
from paper_writing import WritingSession
from test_paper_writing import glyph_strokes, hand_for_tip, with_hand


def write(session, page, strokes, offset=(10, -10), rest=8):
    """Write ``strokes`` with a tracked tip that is ``offset`` away from the real nib, then keep
    the hand on the last glyph. Returns the final frame."""
    now, frame, point = 1.0, None, None
    for stroke in strokes:
        for index, point in enumerate(stroke):
            if index:
                cv2.line(page, stroke[index - 1], point, (95, 95, 95), 2, cv2.LINE_AA)
            hand = hand_for_tip((point[0] + offset[0], point[1] + offset[1]), scale=110.0)
            frame = with_hand(page, hand)
            session.observe(frame, {"hands": [hand]}, now)
            now += 0.07
    hand = hand_for_tip((point[0] + offset[0], point[1] + offset[1]), scale=110.0)
    for _ in range(rest):
        frame = with_hand(page, hand)
        session.observe(frame, {"hands": [hand]}, now)
        now += 0.07
    return frame


class LiveInkTests(unittest.TestCase):
    def setUp(self):
        self.page = np.full((720, 1280, 3), 205, np.uint8)
        cv2.putText(self.page, "old notes", (820, 560), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (60, 60, 60), 2)
        self.session = WritingSession(self.page.copy())
        self.strokes = [stroke for x in (300, 420, 540) for stroke in glyph_strokes(x, top=300, height=34)]

    def test_ink_is_registered_with_an_off_target_tip_and_the_hand_left_on_the_page(self):
        frame = write(self.session, self.page, self.strokes)
        selection = self.session.select(frame, prefer_recent=True)
        self.assertIsNotNone(selection)
        self.assertTrue(ink_usable(selection))
        left, top, right, bottom = selection.page.bounds
        # Tight around the new writing (x 300..622, y 300..334), and nowhere near the old notes.
        self.assertTrue(270 <= left <= 300 and 600 <= right <= 660, (left, right))
        self.assertTrue(275 <= top <= 300 and 334 <= bottom <= 365, (top, bottom))
        ink = selection.ink[:, :, 0]
        self.assertEqual(ink.shape, (bottom - top, right - left))
        # The first glyph, which the hand uncovered long ago, is in the ink picture; clean paper is white.
        first = ink[:, 300 - left - 3:300 - left + 4]
        self.assertGreater(int((first < 120).sum()), 20)
        self.assertGreater(float((ink > 240).mean()), 0.7)

    def test_reading_uses_the_ink_picture_with_the_fast_reader(self):
        frame = write(self.session, self.page, self.strokes)
        selection = self.session.select(frame, prefer_recent=True)
        with patch("paper_reader._ask", return_value="1+2 1+2 1+2") as reader, \
                patch("paper_conversion.chat") as vision:
            reading = read_recording(selection, "ChatGPT", "Python", "idea", "local")
        self.assertEqual((reading.path, reading.source), ("ink", "1+2 1+2 1+2"))
        self.assertEqual((reader.call_count, vision.call_count), (1, 0))
        model, prompt, picture = reader.call_args.args[:3]
        self.assertEqual((model, prompt), ("glm-ocr:q8_0", "Text Recognition:"))
        self.assertGreater(picture.shape[0], 60)  # Scaled up for the reader.

    def test_a_rejected_reading_is_read_again_in_the_other_mode(self):
        frame = write(self.session, self.page, self.strokes)
        selection = self.session.select(frame, prefer_recent=True)
        with patch("paper_reader._ask", side_effect=["y = (x + 1", "y = (x + 1)"]) as reader:
            reading = read_recording(selection, "LaTeX", "Python", "equation", "local")
        self.assertEqual((reading.source, reader.call_count), ("y = (x + 1)", 2))
        # The second look uses the same fast model in its other recognition mode.
        self.assertEqual([call.args[:2] for call in reader.call_args_list],
                         [("glm-ocr:q8_0", "Formula Recognition:"), ("glm-ocr:q8_0", "Text Recognition:")])

    def test_keys_with_boxes_and_arrows_go_to_the_vision_model(self):
        frame = write(self.session, self.page, self.strokes)
        selection = self.session.select(frame, prefer_recent=True)
        reply = {"source": "branch a from main", "notes": [], "kind": "diagram", "clear": True}
        with patch("paper_reader._ask") as reader, patch("paper_conversion.chat", return_value=reply) as vision:
            reading = read_recording(selection, "Git", "Python", "diagram", "local")
        self.assertEqual((reader.call_count, vision.call_count, reading.source), (0, 1, "branch a from main"))

    def test_version_control_recording_commits_writing_and_reads_drawings_as_diagrams(self):
        frame = write(self.session, self.page, self.strokes)
        selection = self.session.select(frame, prefer_recent=True)
        with patch("paper_reader._ask", return_value="Add login form") as reader, \
                patch("paper_conversion.chat") as vision:
            reading = read_recording(selection, "Git", "Python", "git_auto", "local")
        self.assertEqual(reading.source, 'commit "Add login form"')
        self.assertEqual((reader.call_count, vision.call_count), (1, 0))

        # A branch line under a label: a drawing, read by the vision model as git commands.
        session = WritingSession(self.page.copy())
        line = [[(x, 360) for x in range(300, 600, 8)]]
        frame = write(session, self.page.copy(), self.strokes[:3] + line)
        selection = session.select(frame, prefer_recent=True)
        reply = {"source": "branch login from main", "notes": [], "kind": "diagram", "clear": True}
        with patch("paper_reader._ask") as reader, patch("paper_conversion.chat", return_value=reply) as vision:
            reading = read_recording(selection, "Git", "Python", "git_auto", "local")
        self.assertEqual((reader.call_count, vision.call_count, reading.source), (0, 1, "branch login from main"))

    def test_a_shadow_that_rests_and_leaves_is_not_ink(self):
        page = self.page.copy()
        session = WritingSession(page.copy())
        shadowed = page.copy()
        cv2.line(shadowed, (700, 420), (760, 470), (120, 120, 120), 3)  # a finger's shadow, far from the writing
        now = 1.0
        for _ in range(14):  # It rests for a second, while the hand tracker sees no hand.
            session.observe(shadowed, {"hands": []}, now)
            now += 0.08
        frame = write(session, page, self.strokes)
        selection = session.select(frame, prefer_recent=True)
        self.assertIsNotNone(selection)
        x0, y0, x1, y1 = selection.page.bounds
        self.assertLess(x1, 680, "the shadow's spot must not stretch the frame")

    def test_drawn_lines_are_told_from_handwriting(self):
        from paper_ink import looks_drawn, straight_stroke

        text = np.full((80, 400), 255, np.uint8)
        for index, letter in enumerate("add login page"):
            cv2.putText(text, letter, (10 + 27 * index, 50), cv2.FONT_HERSHEY_SCRIPT_SIMPLEX, 1.1, 0, 2, cv2.LINE_AA)
        self.assertFalse(looks_drawn(text))
        drawing = text.copy()
        cv2.line(drawing, (10, 70), (390, 66), 0, 2, cv2.LINE_AA)
        self.assertTrue(looks_drawn(drawing))
        self.assertTrue(straight_stroke([[(0, 0), (60, 2), (120, 1), (200, 3)]], 20))
        self.assertFalse(straight_stroke([[(0, 0), (20, 30), (40, 0), (60, 30), (80, 0)]], 20))

    def test_nothing_written_registers_no_ink(self):
        hovering = [[(x, 300) for x in range(300, 420, 6)]]
        now = 1.0
        for point in hovering[0]:
            hand = hand_for_tip(point, scale=110.0)
            frame = with_hand(self.page, hand)
            self.session.observe(frame, {"hands": [hand]}, now)
            now += 0.07
        selection = self.session.select(frame, prefer_recent=True)
        self.assertTrue(selection is None or not ink_usable(selection))


if __name__ == "__main__":
    unittest.main()
