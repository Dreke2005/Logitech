import json
import tempfile
import unittest
from io import BytesIO
from pathlib import Path
from unittest.mock import patch
from zipfile import ZipFile

import cv2
import numpy as np

import paper_keypad
from lip_correction import polish
from local_model import ModelOutputLimit, chat
from paper_conversion import (
    architecture_files,
    build_schematic,
    convert_page,
    read_page_source,
    read_recording,
)
from paper_context import (
    APPS,
    CONTEXTS,
    GENERIC,
    KEY_WORKFLOWS,
    canonical_workflow,
    end_circle_enabled,
    validate_reading,
    writing_context,
)
from paper_handoff import package_latex, save_generated, save_project
from paper_pen import render_trace
from paper_selection import PaperSelection
from paper_transfer import graph, import_equation, import_figure
from paper_writing import WritingSelection


class LipCorrectionTests(unittest.TestCase):
    def test_shows_grounded_topic_substitution(self):
        with patch("lip_correction.chat", return_value={"text": "Use gradient descent."}) as chat:
            self.assertEqual(polish("USE GRADIENT DISSENT", "local", "machine learning"),
                             "Use gradient descent.")
        self.assertIn("machine learning", chat.call_args.args[2])

    def test_keeps_original_when_model_completes_or_changes_numbers(self):
        draft = "I WANT TWO APPLES AND 9 PEARS"
        for candidate in ("Buy fruit and groceries now.", "I want two apples and 8 pears.",
                          "I want two apples and 9 pears with breakfast."):
            with self.subTest(candidate=candidate), patch("lip_correction.chat", return_value={"text": candidate}):
                self.assertEqual(polish(draft, "local"), draft)


class PaperConversionTests(unittest.TestCase):
    def test_complete_json_is_usable_at_the_output_limit(self):
        reply = {"done_reason": "length", "message": {"content": '{"source":"y=x^2+2x+1"}'}}
        with patch("local_model._online", return_value=True), patch("local_model.build_opener") as opener:
            opener.return_value.open.return_value = BytesIO(json.dumps(reply).encode())
            self.assertEqual(chat("local", "", "", {})["source"], "y=x^2+2x+1")

    def test_incomplete_dual_image_reply_retries_with_the_final_rgb_image(self):
        image = np.full((60, 100, 3), 225, dtype=np.uint8)
        with patch("paper_conversion.chat", side_effect=[
            ModelOutputLimit("limit"), {"source": "y=x^2+2x+1", "notes": []},
        ]) as model:
            source, _ = read_page_source(image, "local", "Read the equation.", evidence=image)
        self.assertEqual(source, "y=x^2+2x+1")
        self.assertIn("evidence", model.call_args_list[0].kwargs)
        self.assertNotIn("evidence", model.call_args_list[1].kwargs)

    def test_recording_sends_the_crop_and_aligned_pen_strokes_together(self):
        page = np.full((60, 100, 3), 225, dtype=np.uint8)
        trace = np.full((60, 100, 3), 255, dtype=np.uint8)
        trace[10:25, 10:30] = 6
        selection = WritingSelection(PaperSelection(page, (5, 5, 105, 65), "manual"), trace=trace,
                                     trace_reliable=True)
        reply = {"source": "y=x^2+2x+1", "notes": [], "kind": "equation", "clear": True}
        with patch("paper_conversion.chat", return_value=reply) as model:
            source, notes = read_recording(selection, "VS Code", "Python", "default", "local")
        self.assertEqual(source, reply["source"])
        self.assertIn("pen strokes", notes[-1])
        self.assertEqual(model.call_count, 1)
        self.assertIn("evidence", model.call_args.kwargs)
        prompt = model.call_args.args[2]
        self.assertIn("Python code", prompt)
        self.assertIn("same scale and position", prompt)
        self.assertIn("already on the page", prompt)
        image = cv2.imdecode(np.frombuffer(model.call_args.kwargs["image"], np.uint8), cv2.IMREAD_COLOR)
        strokes = cv2.imdecode(np.frombuffer(model.call_args.kwargs["evidence"], np.uint8), cv2.IMREAD_COLOR)
        self.assertEqual(image.shape, strokes.shape)
        self.assertGreater(image.shape[1], 100)  # Small crops are enlarged for the model.

    def test_recording_without_strokes_uses_the_new_ink_with_project_context(self):
        page = np.full((60, 100, 3), 225, dtype=np.uint8)
        ink = np.full((60, 100, 3), 255, dtype=np.uint8)
        ink[20:30, 20:60] = 10
        selection = WritingSelection(PaperSelection(page, None, "full"), evidence=ink)
        reply = {"source": "class TaskList", "notes": [], "kind": "diagram", "clear": True}
        with patch("paper_conversion.chat", return_value=reply) as model:
            source, notes = read_recording(selection, "VS Code", "Python", "project", "local")
        self.assertEqual(source, "class TaskList")
        self.assertIn("class boxes", model.call_args.args[2])
        self.assertIn("appeared while writing", model.call_args.args[2])
        self.assertIn("evidence", model.call_args.kwargs)

    def test_gradient_draft_passes_photo_and_keeps_source_separate(self):
        frame = np.zeros((40, 60, 3), dtype=np.uint8)
        source = "Gradient descent\nfor t = 0, ..., T-1:\nw[t+1] = w[t] - lr * grad_L(w[t])"
        with patch("paper_conversion.chat", return_value={"source": source, "notes": []}) as model:
            result = convert_page(frame, "VS Code", "Python", "local")
        self.assertEqual((result.source, result.extension), (source, ".py"))
        test_scope = {}
        exec(result.content, test_scope)
        self.assertAlmostEqual(test_scope["gradient_descent"](2.0, lambda w: 2 * w, 0.1, 3), 1.024)
        self.assertTrue(model.call_args.kwargs["image"].startswith(b"\xff\xd8"))

    def test_schematic_links_parts_by_shared_label_and_rejects_invalid_net(self):
        parts = [{"reference": "R1", "value": "1k", "nets": ["IN", "OUT"]},
                 {"reference": "C1", "value": "100n", "nets": ["OUT", "GND"]}]
        schematic = build_schematic(parts)
        self.assertIn("L Device:R R1", schematic)
        self.assertIn("L Device:C C1", schematic)
        self.assertEqual(schematic.count("\nOUT\n"), 2)
        parts[0]["nets"][0] = "unclear wire\n$EndComp"
        with self.assertRaisesRegex(ValueError, "net labels"):
            build_schematic(parts)

    def test_ltspice_schematic_connects_parts_by_net_flags(self):
        from paper_ltspice import build_asc, netlist_text

        parts = [{"reference": "V1", "value": "SINE(0 1 1k)", "nets": ["in", "GND"]},
                 {"reference": "R1", "value": "1k", "nets": ["in", "out"]},
                 {"reference": "C1", "value": "100n", "nets": ["out", "gnd"]}]
        schematic = build_asc(parts)
        self.assertTrue(schematic.startswith("Version 4\nSHEET 1 "))
        for line in ("SYMBOL voltage", "SYMBOL res", "SYMBOL cap", "SYMATTR InstName C1", "SYMATTR Value 100n",
                     "!.tran 10m"):
            self.assertIn(line, schematic)
        self.assertEqual(sum(line.endswith(" out") for line in schematic.splitlines() if line.startswith("FLAG")), 2)
        self.assertEqual(sum(line.endswith(" 0") for line in schematic.splitlines() if line.startswith("FLAG")), 2)
        self.assertIn("!.ac dec 10 1 100k", build_asc(parts, ".ac dec 10 1 100k"))
        self.assertIn("R1 in out 1k", netlist_text(parts))
        with self.assertRaisesRegex(ValueError, "ground"):
            build_asc(parts[1:2])
        with self.assertRaisesRegex(ValueError, "net labels"):
            build_asc([{"reference": "R1", "value": "1k", "nets": ["in\nSYMBOL x", "GND"]}])

    def test_circuit_sketch_becomes_an_ltspice_draft(self):
        frame = np.zeros((40, 60, 3), dtype=np.uint8)
        reply = {"source": "V1 - R1 - GND", "notes": [], "directive": "", "components": [
            {"reference": "V1", "value": "5", "nets": ["in", "GND"]},
            {"reference": "R1", "value": "1k", "nets": ["in", "GND"]}]}
        with patch("paper_conversion.chat", return_value=reply):
            result = convert_page(frame, "LTspice", None, "local")
        self.assertEqual(result.extension, ".asc")
        self.assertIn("!.op", result.content)
        self.assertTrue(any("R1 in 0 1k" in note for note in result.notes))
        with patch("paper_conversion.chat", return_value={**reply, "components": []}):
            self.assertEqual(convert_page(frame, "LTspice", None, "local").content, "")

    def test_uncertain_circuit_has_no_export(self):
        frame = np.zeros((40, 60, 3), dtype=np.uint8)
        reply = {"source": "R1", "components": [], "notes": ["Unclear junction"]}
        with patch("paper_conversion.chat", return_value=reply):
            result = convert_page(frame, "KiCad", None, "local")
        self.assertEqual((result.content, result.notes), ("", ("Unclear junction",)))

    def test_matlab_solver_uses_the_transcribed_equation_and_selected_task(self):
        frame = np.zeros((40, 60, 3), dtype=np.uint8)
        code = "function x = paper_draft(A, b)\n    x = A\\b;\nend"
        with patch("paper_conversion.chat", return_value={"source": "A*x = b\nfind x", "notes": []}):
            result = convert_page(frame, "MATLAB", None, "local", "solve")
        self.assertEqual(result.content, code)
        with patch("paper_conversion.chat", side_effect=[
            {"source": "sin(x) = 1/2, find x", "notes": []},
            {"content": "syms x; x = solve(sin(x) == 1/2, x);"},
        ]) as model:
            other = convert_page(frame, "MATLAB", None, "local", "solve")
        self.assertIn("solve(sin(x)", other.content)
        self.assertIn("sin(x) = 1/2", model.call_args.args[2])
        self.assertIn("finding the unknown", model.call_args.args[2])

    def test_architecture_generates_a_project_with_safe_validated_paths(self):
        frame = np.zeros((40, 60, 3), dtype=np.uint8)
        files = [{"path": "task.py", "content": "class Task:\n    pass"},
                 {"path": "task_list.py", "content": "class TaskList:\n    def add(self, task: Task):\n        pass"}]
        with patch("paper_conversion.chat", side_effect=[
            {"source": "TaskList contains Task", "notes": []}, {"files": files},
        ]):
            result = convert_page(frame, "VS Code", "Python", "local", "architecture")
        with tempfile.TemporaryDirectory() as temp:
            project = save_project(result.files, "Python", Path(temp))
            self.assertEqual({path.name for path in project.iterdir()}, {"task.py", "task_list.py"})
            self.assertIn("from task import Task", (project / "task_list.py").read_text())
        for path in ("../escape.py", "C:/file.py", "package/CON.py", "task.PY", "Task.py", "task.py/nested.py"):
            with self.subTest(path=path), self.assertRaises(ValueError):
                architecture_files([files[0], {"path": path, "content": "pass"}], "Python")

    def test_general_scalar_equation_becomes_a_function_with_explicit_inputs(self):
        frame = np.zeros((40, 60, 3), dtype=np.uint8)
        with patch("paper_conversion.chat", return_value={"source": "c = sqrt(a^2+b^2)", "notes": []}):
            result = convert_page(frame, "VS Code", "Python", "local", "equation")
        scope = {}
        exec(result.content, scope)
        self.assertEqual(scope["compute_c"](3, 4), 5)

    def test_visual_check_also_passes_visible_ink_near_tracking(self):
        frame = np.full((60, 100, 3), 225, dtype=np.uint8)
        ink = frame.copy()
        ink[20:24, 30:55] = 0
        with patch("paper_conversion.chat", return_value={"source": "y=x^2+2*x+1", "notes": []}) as model:
            draft = convert_page(frame, "VS Code", "Python", "local", evidence=ink)
        self.assertNotEqual(model.call_args.kwargs["image"], model.call_args.kwargs["evidence"])
        self.assertIn("only glyphs supported by the final image", model.call_args.args[2])
        with tempfile.TemporaryDirectory() as temp:
            path = save_generated(draft, "Python", Path(temp))
            self.assertEqual(path.name, "paper_draft.py")
            self.assertIn("2 * x", path.read_text(encoding="utf-8"))

    def test_equation_import_and_overleaf_archive_keep_relative_paths(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            project = root / "project"
            project.mkdir()
            main = project / "main.tex"
            main.write_bytes(b"\\documentclass{article}\r\n\\begin{document}\r\n\\end{document}\r\n")
            (project / "private.py").write_text("excluded")
            snippet, reference, entry = import_equation(r"\[y=x^2+2x+1\]", project)
            self.assertEqual(entry, main)
            self.assertIn(reference.encode("ascii") + b"\r\n\\end{document}", main.read_bytes())
            bundle = package_latex(project, root / "overleaf.zip")
            with ZipFile(bundle) as archive:
                name = snippet.relative_to(project).as_posix()
                self.assertCountEqual(archive.namelist(), ["main.tex", name])
                self.assertIn(b"2x+1", archive.read(name))

    def test_latex_report_keeps_section_order_and_escapes_paper_headings(self):
        frame = np.zeros((40, 60, 3), dtype=np.uint8)
        with patch("paper_conversion.chat", side_effect=[
            {"source": "Test & plots\nMethod_1\nResults", "notes": []},
            {"title": "Test & plots", "sections": ["Method_1", "Results"]},
        ]):
            draft = convert_page(frame, "LaTeX", "Python", "local", "project")
        with tempfile.TemporaryDirectory() as temp:
            project = save_project(draft.files, "LaTeX", Path(temp))
            source = (project / "main.tex").read_text()
            self.assertIn(r"\title{Test \& plots}", source)
            self.assertLess(source.index(r"\section{Method\_1}"), source.index(r"\section{Results}"))

    def test_matlab_plot_uses_x_range_and_exports_relative_latex_figure(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            path, domain = graph("x = linspace(-2, 3, 100);\ny = x.^2;\nplot(x,y);", root)
            self.assertEqual(domain, (-2, 3))
            snippet = import_figure(path, root)
            figure, = (root / "figures").iterdir()
            self.assertEqual(figure.read_bytes(), path.read_bytes())
            self.assertIn(f"{{figures/{figure.name}}}", snippet)

    def test_plot_rejects_non_math_code(self):
        with tempfile.TemporaryDirectory() as temp:
            with self.assertRaises(ValueError):
                graph("y = globals()", Path(temp))
            self.assertEqual(list(Path(temp).iterdir()), [])


class KeypadTests(unittest.TestCase):
    def test_ignores_press_from_preceding_menu_revision(self):
        with tempfile.TemporaryDirectory() as location:
            folder = Path(location)
            with patch.object(paper_keypad, "STATE", folder / "state.json"), \
                    patch.object(paper_keypad, "COMMANDS", folder / "commands"):
                keypad = paper_keypad.KeypadSession()
                keypad.publish(["VS Code"])
                paper_keypad.send_slot(paper_keypad.current_state(), 2)
                keypad.revision += 1
                keypad.publish(["Python"])
                paper_keypad.send_slot(paper_keypad.current_state(), 0)
                self.assertEqual(list(keypad.presses()), [0])
                keypad.close()
                self.assertIsNone(paper_keypad.current_state())

    def test_native_folder_command_has_a_stable_app_route(self):
        with tempfile.TemporaryDirectory() as location:
            folder = Path(location)
            with patch.object(paper_keypad, "STATE", folder / "state.json"), \
                    patch.object(paper_keypad, "COMMANDS", folder / "commands"):
                keypad = paper_keypad.KeypadSession()
                keypad.publish(["VS Code"], "vscode")
                paper_keypad._write_json(folder / "commands" / "1.json",
                                         {"session": keypad.id, "menu": "matlab", "action": "code"})
                paper_keypad._write_json(folder / "commands" / "2.json",
                                         {"session": "old", "menu": "matlab", "action": "read"})
                self.assertEqual(list(keypad.presses()), [("matlab", "code")])
                self.assertEqual(paper_keypad.current_state()["menu"], "vscode")


def _equation_strokes():
    """Hand-made pen-down strokes for "y = x^2 + 1" in frame coordinates."""
    def seg(a, b, n=6):
        return [(round(a[0] + (b[0] - a[0]) * k / (n - 1)), round(a[1] + (b[1] - a[1]) * k / (n - 1)))
                for k in range(n)]
    return [seg((300, 200), (310, 220)), seg((320, 200), (305, 240)), seg((330, 215), (348, 215)),
            seg((330, 225), (348, 225)), seg((358, 210), (376, 238)), seg((376, 210), (358, 238)),
            seg((404, 224), (424, 224)), seg((414, 214), (414, 234)), seg((440, 202), (440, 240))]


def _recorded(strokes=(), rotation=0):
    page = np.full((60, 160, 3), 225, dtype=np.uint8)
    ink = np.full((60, 160, 3), 255, dtype=np.uint8)
    ink[20:30, 20:120] = 10
    return WritingSelection(PaperSelection(page, (290, 190, 450, 250), "manual"), evidence=ink,
                            strokes=tuple(np.asarray(stroke) for stroke in strokes), rotation=rotation,
                            text_height=38.0)


class TraceFirstReadingTests(unittest.TestCase):
    def reply(self, source, clear=True):
        return {"source": source, "notes": [], "kind": "equation", "clear": clear}

    def test_trace_reading_is_confirmed_by_the_photo_of_the_new_ink(self):
        with patch("paper_conversion.chat", return_value=self.reply("y = x^{2} + 1")) as model:
            reading = read_recording(_recorded(_equation_strokes()), "LaTeX", "Python", "equation", "local")
        self.assertEqual(model.call_count, 2)
        self.assertEqual((reading.path, reading[0]), ("trace+photo", "y = x^{2} + 1"))
        self.assertTrue(any("agrees" in note for note in reading[1]))
        self.assertIn("evidence", model.call_args_list[1].kwargs)  # Photo crop plus the new-ink map.
        self.assertIn("'y = x^{2} + 1'", model.call_args_list[1].args[2])
        model.call_args = model.call_args_list[0]
        self.assertNotIn("evidence", model.call_args.kwargs)  # Only the rendered strokes.
        prompt = model.call_args.args[2]
        self.assertIn("9 pen-down strokes", prompt)
        self.assertIn("LaTeX math", prompt)
        image = cv2.imdecode(np.frombuffer(model.call_args.kwargs["image"], np.uint8), cv2.IMREAD_GRAYSCALE)
        dark = np.argwhere(image < 128)
        self.assertGreater(len(dark), 50)
        glyph = np.ptp(dark[:, 0])
        self.assertTrue(60 <= glyph <= 130, glyph)  # Scaled so the writing is about 80 px tall.
        self.assertLess(image.shape[1], 600)  # Tight: no page around the strokes.

    def test_invalid_trace_reading_is_retried_with_the_reason_then_falls_back_to_the_photo(self):
        with patch("paper_conversion.chat", side_effect=[
            self.reply("y = (x^{2} + 1"), self.reply("y = (x^{2} + 1"), self.reply("y = x^{2} + 1"),
        ]) as model:
            reading = read_recording(_recorded(_equation_strokes()), "LaTeX", "Python", "equation", "local")
        self.assertEqual(model.call_count, 3)
        self.assertIn("unclosed '('", model.call_args_list[1].args[2])
        self.assertNotIn("evidence", model.call_args_list[1].kwargs)
        self.assertIn("evidence", model.call_args_list[2].kwargs)  # Photo crop plus the new-ink map.
        self.assertEqual((reading.path, reading[0]), ("visual", "y = x^{2} + 1"))
        self.assertTrue(any("pen-trace reading was rejected" in note for note in reading[1]))

    def test_retry_with_feedback_can_fix_the_trace_reading(self):
        with patch("paper_conversion.chat", side_effect=[self.reply("y = = x^2"), self.reply("y = x^2"),
                                                          self.reply("y = x^2")]) as model:
            reading = read_recording(_recorded(_equation_strokes()), "LaTeX", "Python", "equation", "local")
        self.assertEqual(model.call_count, 3)
        self.assertEqual((reading.path, reading[0]), ("trace+photo", "y = x^2"))

    def test_photo_corrects_a_symbol_in_a_valid_but_wrong_trace_reading(self):
        with patch("paper_conversion.chat", side_effect=[self.reply("y = x^{2} + 7"), self.reply("y = x^{2} + 1")]):
            reading = read_recording(_recorded(_equation_strokes()), "LaTeX", "Python", "equation", "local")
        self.assertEqual((reading.path, reading[0]), ("trace+photo", "y = x^{2} + 1"))
        self.assertTrue(any("corrected" in note for note in reading[1]))

    def test_photo_that_reads_something_else_only_flags_the_trace_reading(self):
        with patch("paper_conversion.chat", side_effect=[self.reply("y = x^{2} + 1"), self.reply("a b c d e f g h")]):
            reading = read_recording(_recorded(_equation_strokes()), "LaTeX", "Python", "equation", "local")
        self.assertEqual(reading[0], "y = x^{2} + 1")
        self.assertTrue(any("Check this reading" in note for note in reading[1]))

    def test_trace_without_photo_evidence_is_read_alone(self):
        selection = WritingSelection(PaperSelection(np.full((60, 160, 3), 225, np.uint8), None, "full"),
                                     strokes=tuple(np.asarray(stroke) for stroke in _equation_strokes()),
                                     text_height=38.0)
        with patch("paper_conversion.chat", return_value=self.reply("y = x^{2} + 1")) as model:
            reading = read_recording(selection, "LaTeX", "Python", "equation", "local")
        self.assertEqual((model.call_count, reading.path), (1, "trace"))

    def test_sparse_trace_goes_straight_to_the_photo(self):
        with patch("paper_conversion.chat", return_value=self.reply("y = x^2")) as model:
            reading = read_recording(_recorded(_equation_strokes()[:1]), "LaTeX", "Python", "equation", "local")
        self.assertEqual((model.call_count, reading.path), (1, "visual"))
        self.assertIn("evidence", model.call_args.kwargs)

    def test_upside_down_writing_is_turned_upright(self):
        strokes = [np.c_[1000 - stroke[:, 0], 800 - stroke[:, 1]]
                   for stroke in map(np.asarray, _equation_strokes())]
        upright = render_trace(_equation_strokes(), 0, 38.0)
        turned = render_trace(strokes, 180, 38.0)
        self.assertEqual(upright.shape, turned.shape)
        self.assertLess(np.mean(np.abs(upright.astype(int) - turned.astype(int))), 3)

    def test_every_writing_key_has_a_context(self):
        menus = json.loads((Path(__file__).parents[1] / "paper_menus.json").read_text(encoding="utf-8"))
        checked = 0
        for menu, app in APPS.items():
            for entry in menus.get(menu, ()):
                if entry[0] in KEY_WORKFLOWS:
                    with self.subTest(app=app, key=entry[0]):
                        workflow = canonical_workflow(app, KEY_WORKFLOWS[entry[0]])
                        self.assertTrue((app, workflow) in CONTEXTS or ("*", workflow) in CONTEXTS)
                        checked += 1
        self.assertGreaterEqual(checked, 14)
        # The workbench's internal workflow names resolve to specific contexts too.
        for app, workflow in (("VS Code", "default"), ("MATLAB", "graph"), ("MATLAB", "project"),
                              ("LaTeX", "equation"), ("LaTeX", "project"), ("LaTeX", "table"),
                              ("ChatGPT", "idea"), ("KiCad", "default"), ("LTspice", "default"), ("Git", "git_repo"),
                              ("Git", "git_branch"), ("Git", "git_commit"), ("Git", "diagram"), ("MATLAB", "copy")):
            with self.subTest(app=app, workflow=workflow):
                context = writing_context(app, workflow, "Python")
                self.assertNotEqual(context.writing, GENERIC.writing)
                self.assertNotIn("{language}", context.writing + context.output)

    def test_end_circle_is_off_where_circles_are_content(self):
        self.assertFalse(end_circle_enabled("diagram", "Git"))
        self.assertFalse(end_circle_enabled("git_diagram", "Git"))
        self.assertFalse(end_circle_enabled("copy", "LaTeX"))
        self.assertTrue(end_circle_enabled("git_commit", "Git"))
        self.assertTrue(end_circle_enabled("equation", "LaTeX"))

    def test_validators_follow_the_key(self):
        valid = [("LaTeX", "equation", r"E = \frac{1}{2} m v^{2}"), ("LaTeX", "equation", r"\sum_{i=1}^{n} i = 0"),
                 ("VS Code", "default", "for i in range(3):\n    print(i)"), ("VS Code", "default", "y = 2x + 1"),
                 ("MATLAB", "default", "A = [1 2; 3 4];\nfor k = 1:3\n  x = A*k;\nend"),
                 ("MATLAB", "graph", "y = x^2 + 1"), ("LaTeX", "table", "a | b | c\n1 | 2 | 3"),
                 ("Git", "git_branch", "add login page"), ("Git", "git_commit", "Fix the parser"),
                 ("Git", "diagram", 'branch feature from main\ncommit feature "add login"\n'
                                    "merge feature into main\npush main"),
                 ("MATLAB", "copy", "KIND: code\nx = 1;"), ("ChatGPT", "copy", "KIND: todo\n- [ ] buy milk")]
        invalid = [("LaTeX", "equation", "y = (x + 1"), ("LaTeX", "equation", "please read this sentence"),
                   ("VS Code", "default", "def f(x:\n  return"), ("MATLAB", "default", "for k = 1:3\n x = k;"),
                   ("VS Code", "graph", "hello world"), ("LaTeX", "table", "a | b\n1 | 2 | 3"),
                   ("Git", "git_branch", ".."), ("Git", "diagram", "create branch feature"),
                   ("LaTeX", "copy", "y = x^2"), ("LaTeX", "equation", "")]
        for app, workflow, text in valid:
            with self.subTest(text=text):
                self.assertEqual(validate_reading(app, workflow, text, "Python"), (True, ""))
        for app, workflow, text in invalid:
            with self.subTest(text=text):
                ok, reason = validate_reading(app, workflow, text, "Python")
                self.assertFalse(ok)
                self.assertTrue(reason)
