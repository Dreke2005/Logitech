"""The fast reader: ink pictures from photos, tidy readings, validated repair, and direct results."""
import hermetic  # noqa: F401
import unittest
from pathlib import Path
from unittest.mock import patch

import cv2
import numpy as np

import local_model
import paper_bench
import paper_reader
from paper_context import check_code, validate_reading, writing_context
from paper_conversion import _direct_result, convert_transcription, read_static, written_outline
from paper_ink import extract_ink, normalize
from paper_math import display_latex
from paper_scaffold import parse_sketch, sketch_scaffold

FONT = Path("C:/Windows/Fonts/Inkfree.ttf")


@unittest.skipUnless(FONT.is_file(), "needs the Ink Free font")
class InkExtractionTests(unittest.TestCase):
    def setUp(self):
        self.frame, self.region = paper_bench.on_page(paper_bench.write("y = x^3 + 3x + 2", FONT), 20, 1)

    def test_writing_is_lifted_out_of_a_photo(self):
        picture = extract_ink(self.frame, self.region)
        self.assertIsNotNone(picture)
        left, top, right, bottom = picture.bounds
        self.assertTrue(self.region[0] <= left and right <= self.region[2])
        self.assertTrue(12 <= picture.text_height <= 40, picture.text_height)
        gray = picture.image[:, :, 0]
        self.assertGreater(float((gray > 235).mean()), 0.75)  # clean paper is white
        self.assertGreater(int((gray < 110).sum()), 60)  # ink is dark
        self.assertEqual(picture.lines, 1)

    def test_whole_page_keeps_writing_and_drops_the_page_edge(self):
        picture = extract_ink(self.frame)
        self.assertIsNotNone(picture)
        left, top, right, bottom = picture.bounds
        # The page border rectangle (40 px from the frame edge) is not part of the picture.
        self.assertTrue(left > 60 and top > 60 and right < 1220 and bottom < 660, picture.bounds)

    def test_blank_paper_has_no_ink(self):
        blank = np.full((300, 400, 3), 205, np.uint8)
        self.assertIsNone(extract_ink(blank))

    def test_normalize_scales_small_writing_up(self):
        picture = extract_ink(self.frame, self.region)
        prepared = normalize(picture.image, picture.text_height, glyph=64)
        self.assertGreater(prepared.shape[0], 2 * picture.image.shape[0])
        self.assertTrue((prepared[0] == 255).all())

    def test_snapshot_reading_uses_the_fast_reader(self):
        with patch("paper_reader._ask", return_value="$y=x^{3}+3x+2$") as reader, patch("paper_conversion.chat") as vision:
            reading = read_static(self.frame[self.region[1]:self.region[3], self.region[0]:self.region[2]],
                                  "VS Code", "Python", "default")
        self.assertEqual((reading.source, reading.path, vision.call_count), ("y = x^3 + 3x + 2", "ink", 0))
        self.assertEqual(reader.call_args.args[:2], ("glm-ocr:q8_0", "Text Recognition:"))

    def test_keys_for_drawings_are_left_to_the_vision_model(self):
        self.assertIsNone(read_static(self.frame, "Git", "Python", "diagram"))
        self.assertEqual(writing_context("LTspice", "default").reader, "vision")
        self.assertEqual(writing_context("LaTeX", "equation").reader, "ocr")


class TidyTests(unittest.TestCase):
    def test_latex_from_the_formula_reader_becomes_plain_math(self):
        self.assertEqual(paper_reader.tidy(r"$$ y = \frac{x^{2}}{2} + \sqrt{x_{1}} $$", "equation"),
                         "y = (x^2)/(2) + sqrt(x_1)")
        self.assertEqual(paper_reader.tidy("f (x) = 2 x ^2-5 x+7.", "equation"), "f(x) = 2x^2 - 5x + 7")
        self.assertEqual(paper_reader.tidy_math("z = 7x - 1 2"), "z = 7x - 12")
        self.assertEqual(paper_reader.tidy_math("y = - x^2 + 3"), "y = -x^2 + 3")
        self.assertEqual(paper_reader.tidy(r"\[ y = \frac{a}{b} \]", "latex"), r"y = \frac{a}{b}")

    def test_table_output_in_any_markup_becomes_rows(self):
        html = "<table><tr><th>name</th><th>mass</th></tr><tr><td>e</td><td>0.511</td></tr></table>"
        self.assertEqual(paper_reader.tidy(html, "table"), "name | mass\ne | 0.511")
        self.assertEqual(paper_reader.tidy("| a | b |\n|---|---|\n| 1 | 2 |", "table"), "a | b\n1 | 2")

    def test_look_alike_repair_needs_exactly_one_valid_candidate(self):
        valid = lambda text: validate_reading("LaTeX", "equation", text)[0]  # noqa: E731
        self.assertEqual(paper_reader.repair_math("v = u + a +", valid), "v = u + a t")
        self.assertEqual(paper_reader.repair_math("v = u + +", valid), "v = u + t")
        self.assertIsNone(paper_reader.repair_math("v = u + a )", valid))  # no swap helps: left alone
        self.assertIsNone(paper_reader.repair_math("", valid))

    def test_rejected_equation_is_repaired_before_a_second_look(self):
        picture = np.full((40, 200, 3), 255, np.uint8)
        with patch("paper_reader._ask", return_value="v = u + a +") as reader:
            text, notes, valid = paper_reader.decipher(picture, 20.0, "LaTeX", "equation")
        self.assertEqual((text, valid, reader.call_count), ("v = u + a t", True, 1))
        self.assertIn("look-alike", notes[0])

    def test_copy_reading_gets_its_kind(self):
        picture = np.full((40, 200, 3), 255, np.uint8)
        with patch("paper_reader._ask", return_value="y = x^2 + 1"):
            text, _, valid = paper_reader.decipher(picture, 20.0, "LaTeX", "copy")
        self.assertTrue(valid)
        self.assertEqual(text.splitlines()[0], "KIND: equation")


class DirectResultTests(unittest.TestCase):
    def test_equations_and_valid_code_need_no_model(self):
        with patch("paper_conversion.chat") as model:
            latex = convert_transcription("f(x) = 2x^2 - 5x + 7", (), "LaTeX", "Python", "local", "equation")
            python = convert_transcription("E = m c^2", (), "VS Code", "Python", "local", "default")
            loop = convert_transcription("for i in range(10):", (), "VS Code", "Python", "local", "default")
            matrix = convert_transcription("A = [ 1 2 ; 3 4 ]", (), "VS Code", "Python", "local", "default")
            matlab = convert_transcription("A = [1 2; 3 4]", (), "MATLAB", "Python", "local", "default")
        model.assert_not_called()
        self.assertEqual(latex.content, "\\[\nf(x) = 2x^{2} - 5x + 7\n\\]")
        self.assertIn("def compute_E(m, c)", python.content)
        self.assertEqual(loop.content, "for i in range(10):\n    pass")
        self.assertIn("np.array([[1, 2], [3, 4]])", matrix.content)
        self.assertIn("A = [1 2; 3 4]", matlab.content)

    def test_pseudocode_still_goes_to_the_model(self):
        self.assertIsNone(_direct_result("repeat until converged", "VS Code", "Python"))
        with patch("paper_conversion.chat", return_value={"content": "while True:\n    pass"}) as model:
            draft = convert_transcription("repeat until converged", (), "VS Code", "Python", "local", "default")
        self.assertEqual((model.call_count, draft.content), (1, "while True:\n    pass"))

    def test_slow_generator_keeps_the_reading(self):
        with patch("paper_conversion.chat", side_effect=RuntimeError("timed out")):
            draft = convert_transcription("repeat until converged", (), "VS Code", "Python", "local", "default")
        self.assertEqual(draft.content, "# repeat until converged")
        self.assertTrue(any("did not answer in time" in note for note in draft.notes))

    def test_block_headers_are_valid_code(self):
        self.assertIsNone(check_code("for i in range(10):", "Python"))
        self.assertIsNone(check_code("if x > 0:\n    y = 1\nelse:", "Python"))
        self.assertIsNotNone(check_code("for i in range(10:", "Python"))

    def test_display_latex(self):
        self.assertEqual(display_latex("y = sin(x) + sqrt(x)/2 + alpha"), "\\[\ny = \\sin(x) + \\sqrt{x}/2 + \\alpha\n\\]")
        self.assertIn("\\begin{aligned}", display_latex("y = x + 1\nz = y^2"))

    def test_report_outline_is_read_from_the_lines(self):
        self.assertEqual(written_outline("Thermal model report\n1. Introduction\n- Method\nResults"),
                         {"title": "Thermal model report", "sections": ["Introduction", "Method", "Results"]})
        self.assertIsNone(written_outline("One line only"))


class ScaffoldTests(unittest.TestCase):
    SKETCH = "class Sensor read() calibrate()\nrate: float\nLogger\nwrite(value), flush()\nLogger -> Sensor"

    def test_class_sketch_becomes_files_in_every_language(self):
        classes = parse_sketch(self.SKETCH)
        self.assertEqual(list(classes), ["Sensor", "Logger"])
        self.assertEqual(classes["Logger"]["uses"], ["Sensor"])
        from paper_conversion import architecture_files

        for language, expected in (("Python", ["sensor.py", "logger.py", "main.py"]),
                                   ("TypeScript", ["Sensor.ts", "Logger.ts", "main.ts"]),
                                   ("C++", ["Sensor.hpp", "Logger.hpp", "main.cpp"]),
                                   ("MATLAB", ["Sensor.m", "Logger.m", "main.m"])):
            with self.subTest(language=language):
                files = sketch_scaffold(self.SKETCH, language)
                checked = architecture_files([{"path": name, "content": text} for name, text in files], language)
                self.assertEqual([name for name, _ in checked], expected)
        logger = dict(sketch_scaffold(self.SKETCH, "Python"))["logger.py"]
        self.assertIn("from sensor import Sensor", logger)
        self.assertIn("def write(self, value):", logger)

    def test_prose_is_not_a_sketch(self):
        self.assertIsNone(parse_sketch("Build a logger that reads sensors"))
        self.assertIsNone(sketch_scaffold("A tool for plotting", "Python"))

    def test_project_key_writes_the_sketch_without_a_model(self):
        with patch("paper_conversion.chat") as model:
            draft = convert_transcription(self.SKETCH, (), "VS Code", "Python", "local", "project")
        model.assert_not_called()
        self.assertEqual([name for name, _ in draft.files], ["sensor.py", "logger.py", "main.py"])


class GpuSharingTests(unittest.TestCase):
    def test_context_window_follows_the_request(self):
        self.assertEqual(local_model.context_window("s", "short prompt", 1100), local_model.SMALL_CONTEXT)
        self.assertEqual(local_model.context_window("s", "short prompt", 2600), local_model.LARGE_CONTEXT)
        self.assertEqual(local_model.context_window("s", "p", 1600, image=b"x"), local_model.LARGE_CONTEXT)

    def test_large_requests_unload_other_models(self):
        import io
        import json

        calls = []

        class Opener:
            def open(self, request, timeout=None):
                url = request if isinstance(request, str) else request.full_url
                calls.append((url, None if isinstance(request, str) else json.loads(request.data)))
                loaded = {"models": [{"name": "glm-ocr:q8_0", "context_length": 2048},
                                     {"name": "big", "context_length": 4096}]}
                return io.BytesIO(json.dumps(loaded).encode())

        local_model.make_room("qwen", local_model.SMALL_CONTEXT, Opener())
        self.assertEqual([body["model"] for _, body in calls if body], ["big"])  # only the large-context one
        calls.clear()
        local_model.make_room("qwen", local_model.LARGE_CONTEXT, Opener())
        self.assertEqual(sorted(body["model"] for _, body in calls if body), ["big", "glm-ocr:q8_0"])


@unittest.skipUnless(FONT.is_file(), "needs the Ink Free font")
class TipCalibrationTests(unittest.TestCase):
    def test_tip_offset_is_learned_from_the_ink(self):
        selection, _, session = paper_bench.live_recording("total = total + x[i]", FONT, 20, 1, offset=(14, -10))
        tips, down = session.summary()
        self.assertGreater(down, 0.75 * tips)  # without calibration about a third counted as pen-down
        self.assertLess(session.tip_offset[0], -5)
        self.assertGreater(session.tip_offset[1], 4)
        self.assertIsNotNone(selection)
        self.assertGreater(int((selection.ink[:, :, 0] < 120).sum()), 150)

    def test_a_tip_on_the_nib_learns_no_offset(self):
        # Ink next to the newest samples is still under the hand; that must not pull the tip back.
        _, _, session = paper_bench.live_recording("total = total + x[i]", FONT, 20, 1, offset=(0, 0))
        self.assertLessEqual(float(np.hypot(*session.tip_offset)), 3)

    def test_the_trace_lies_on_the_ink_and_covers_it(self):
        for hard in (False, True):
            with self.subTest(hard=hard):
                selection, _, _, truth = paper_bench.live_recording("if n > 0: return n", FONT, 20, 5, hard=hard,
                                                                    truth=True)
                precision, recall, _ = paper_bench.trace_scores(selection.strokes, truth)
                self.assertGreater(precision, 0.97)  # the tip-landmark trace reached about 0.9
                self.assertGreater(recall, 0.9)  # and about 0.85


class InkTraceTests(unittest.TestCase):
    def test_pieces_follow_the_tip_in_writing_order(self):
        from paper_trace import ink_strokes

        mask = np.zeros((60, 200), np.uint8)
        cv2.line(mask, (20, 20), (80, 20), 1, 2)  # written first, left to right
        cv2.line(mask, (180, 40), (120, 40), 1, 2)  # then right to left
        samples = [(0.1 * i, (20 + 6 * i, 22)) for i in range(11)] + [(2 + 0.1 * i, (180 - 6 * i, 42))
                                                                        for i in range(11)]
        strokes = ink_strokes(mask, samples, reach=10)
        self.assertEqual(len(strokes), 2)
        self.assertLess(strokes[0][0, 0], strokes[0][-1, 0])
        self.assertGreater(strokes[1][0, 0], strokes[1][-1, 0])
        self.assertTrue(all(mask[stroke[:, 1], stroke[:, 0]].all() for stroke in strokes))


if __name__ == "__main__":
    unittest.main()
