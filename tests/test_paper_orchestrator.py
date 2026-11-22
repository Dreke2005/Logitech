"""The orchestrated key press, the circuit apps, and the hand-over to Overleaf and ChatGPT."""
import hermetic  # noqa: F401
import json
import queue
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np

import paper_browser
import paper_cloud
import paper_ltspice
import paper_orchestrator
from paper_conversion import Draft, build_schematic, circuit_draft, convert_page


def answer(**fields):
    reply = {"transcription": "y = x^2 + 1", "kind": "equation", "summary": "A function computing y from x.",
             "content": "def compute_y(x):\n    return x ** 2 + 1\n", "actions": ["save", "open_in_app"], "notes": []}
    return json.dumps({**reply, **fields})


def cloud(*answers):
    return patch.multiple(paper_cloud, enabled=lambda: True, ask=Mock(side_effect=list(answers)))


class PlanTests(unittest.TestCase):
    def setUp(self):
        self.key = paper_orchestrator.key_for("VS Code", "default", "Python")

    def test_one_request_reads_and_writes(self):
        with cloud("\x60\x60\x60json\n" + answer() + "\n\x60\x60\x60"):
            plan = paper_orchestrator.plan(self.key, [b"\x89PNG-picture"], hints=("gradient_step",))
            request = paper_cloud.ask.call_args
        self.assertEqual((plan.transcription, plan.kind, plan.actions), ("y = x^2 + 1", "equation",
                                                                          ("save", "open_in_app")))
        self.assertIn("def compute_y", plan.content)
        self.assertEqual(request.args[2], [b"\x89PNG-picture"])
        self.assertIn("Code key on the VS Code page", request.args[1])
        self.assertIn("gradient_step", request.args[1])
        self.assertEqual(request.kwargs["model"], paper_cloud.READER)

    def test_a_corrected_reading_is_a_text_only_request_to_the_fast_model(self):
        with cloud(answer(transcription="something else")):
            plan = paper_orchestrator.plan(self.key, text="y = x^2 + 7")
            request = paper_cloud.ask.call_args
        self.assertEqual(plan.transcription, "y = x^2 + 7")  # The user's text stands, whatever comes back.
        self.assertEqual((request.args[2], request.kwargs["model"]), ([], paper_cloud.FAST))
        self.assertIn("y = x^2 + 7", request.args[1])

    def test_anything_but_the_expected_object_is_rejected(self):
        rejected = [
            "no json", answer(kind="virus"), answer(actions=["run_shell"]), answer(actions=["run_python"]),
            answer(actions=[]), answer(content=["a"]), answer(extra="x"), answer(actions=["save", 3]),
            json.dumps({"transcription": "x"}), answer(content="x" * 13000),
            answer(command="del /q *"), answer(path="C:/Windows/system32/x.py"),
        ]
        for text in rejected:
            with self.subTest(text=text[:60]), cloud(text):
                self.assertIsNone(paper_orchestrator.plan(self.key, [b"\x89PNG"]))
        with cloud(paper_cloud.CloudUnavailable("offline")):
            self.assertIsNone(paper_orchestrator.plan(self.key, [b"\x89PNG"]))
        with patch.object(paper_cloud, "ask") as ask:  # The local setting: nothing is asked.
            self.assertIsNone(paper_orchestrator.plan(self.key, [b"\x89PNG"]))
            self.assertEqual(ask.call_count, 0)

    def test_only_code_plot_and_equation_keys_are_orchestrated(self):
        covered = [("VS Code", "default", "C++"), ("MATLAB", "default", "Python"), ("VS Code", "graph", "Python"),
                   ("MATLAB", "graph", "Python"), ("LaTeX", "equation", "Python")]
        for target, workflow, language in covered:
            key = paper_orchestrator.key_for(target, workflow, language)
            self.assertTrue(set(key.allowed) <= set(paper_orchestrator.ACTIONS))
            self.assertRegex(key.filename, r"^[a-z_]+\.(py|m|tex|cpp)$")
        self.assertEqual(paper_orchestrator.key_for("VS Code", "graph", "C++").allowed,
                         ("save", "run_python", "open_in_app", "stage"))
        for target, workflow in (("LaTeX", "table"), ("ChatGPT", "idea"), ("Plan", "gantt"), ("VS Code", "project")):
            self.assertIsNone(paper_orchestrator.key_for(target, workflow, "Python"))


class OrchestratedKeyTests(unittest.TestCase):
    def bench(self, target="VS Code", menu="vscode", edit=None):
        import paper_workbench

        bench = SimpleNamespace(results=queue.SimpleQueue(), args=SimpleNamespace(model="local"), menu=menu,
                                git_repo=None, saved=None, report_folder=None, shown=[],
                                CONFIRMED=paper_workbench.PaperWorkbench.CONFIRMED,
                                _upright_turn=lambda image, turn, *_: turn)

        def confirm(source, workflow, again=None, summary=None):
            bench.shown.append((source, summary() if summary else None))
            return edit or source

        bench.confirm_reading = confirm
        bench.generate = lambda workflow, language="Python": paper_workbench.PaperWorkbench._generate(
            bench, SimpleNamespace(page=SimpleNamespace(image=np.full((40, 120, 3), 230, np.uint8)), rotation=0,
                                   evidence=None), target, language, workflow)
        return bench

    def test_the_code_key_is_one_request_and_the_result_is_checked(self):
        bench = self.bench()
        with cloud(answer()):
            bench.generate("default")
            self.assertEqual(paper_cloud.ask.call_count, 1)
        draft, figure, workflow = bench.results.get_nowait()
        self.assertEqual((draft.source, workflow, figure), ("y = x^2 + 1", "default", None))
        self.assertIn("def compute_y(x):", draft.content)
        self.assertTrue(draft.content.startswith("# From paper:"))
        self.assertEqual(bench.shown, [("y = x^2 + 1", "A function computing y from x.")])
        self.assertEqual(bench.plan_actions, ("save", "open_in_app"))
        self.assertIn("one cloud request", bench.engines)

    def test_code_that_does_not_compile_goes_back_once(self):
        bench = self.bench()
        with cloud(answer(content="def compute_y(x)\n    return x ** 2 + 1\n"), answer()):
            bench.generate("default")
            self.assertEqual(paper_cloud.ask.call_count, 2)
            self.assertIn("Python syntax error", paper_cloud.ask.call_args.args[1])
        draft, _, _ = bench.results.get_nowait()
        compile(draft.content, "paper_draft.py", "exec")
        self.assertTrue(any("failed a check" in note for note in draft.notes))

    def test_an_edited_reading_is_converted_again_from_the_text(self):
        bench = self.bench(edit="y = x^2 + 7")
        with cloud(answer()):
            bench.generate("default")
            self.assertEqual(paper_cloud.ask.call_count, 1)  # y = x^2 + 7 converts directly: exact and instant.
        draft, _, _ = bench.results.get_nowait()
        self.assertEqual(draft.source, "y = x^2 + 7")
        self.assertIn("x ** 2 + 7", draft.content)

    def test_the_plot_key_runs_the_program_that_came_with_the_reading(self):
        import paper_workbench

        program = ("from pathlib import Path\nimport matplotlib.pyplot as plt\nimport numpy as np\n"
                   "x = np.linspace(-5, 5, 200)\nfig, ax = plt.subplots()\nax.plot(x, x ** 2 + 1)\n"
                   "folder = Path(__file__).resolve().parent\nfig.savefig(folder / 'paper_plot.png')\n"
                   "fig.savefig(folder / 'paper_plot.pdf')\nplt.show()\n")
        bench = self.bench()
        with tempfile.TemporaryDirectory() as folder, patch.object(paper_workbench, "ROOT", Path(folder)), \
                cloud(answer(kind="plot", content=program, actions=["save", "run_python", "open_in_app"])):
            bench.generate("graph", "TypeScript")
            draft, figure, workflow = bench.results.get_nowait()
            self.assertEqual((workflow, bench.plot_script.name), ("graph", "paper_plot.py"))
            self.assertTrue(figure.is_file() and figure.parent == bench.plot_script.parent)
            self.assertEqual(bench.plot_script.read_text(encoding="utf-8"), program)

    def test_the_equation_key_gets_display_math(self):
        bench = self.bench("LaTeX", "latex")
        with cloud(answer(transcription="a^2 + b^2 = c^2", content="$a^{2} + b^{2} = c^{2}$")):
            bench.generate("equation")
        draft, _, _ = bench.results.get_nowait()
        self.assertEqual(draft.content, "\\[\na^{2} + b^{2} = c^{2}\n\\]")
        self.assertEqual(draft.extension, ".tex")

    def test_without_the_cloud_the_step_by_step_pipeline_runs(self):
        bench = self.bench()
        bench._checked = lambda source, *_: source
        with patch("paper_conversion.read_static", return_value=None), \
                patch("paper_conversion.chat", return_value={"source": "y = x^2 + 1", "notes": []}):
            bench.generate("default")
        draft, _, _ = bench.results.get_nowait()
        self.assertIn("x ** 2 + 1", draft.content)
        self.assertIsNone(bench.plan_actions)


class CircuitTests(unittest.TestCase):
    reply = {"source": "V1 - R1 - C1 - GND", "notes": ["R1 has no written value; 1k assumed."], "directive": "",
             "components": [{"reference": "V1", "value": "SINE(0 1 1k)", "nets": ["in", "GND"]},
                            {"reference": "R1", "value": "1k", "nets": ["in", "out"]},
                            {"reference": "C1", "value": "100n", "nets": ["out", "GND"]},
                            {"reference": "D1", "value": "D", "nets": ["out", "GND"]}]}

    def test_kicad_gets_sources_and_ground_symbols(self):
        frame = np.zeros((40, 60, 3), dtype=np.uint8)
        with patch("paper_conversion.chat", return_value=self.reply) as model:
            draft = convert_page(frame, "KiCad", None, "local")
        self.assertTrue(model.call_args.kwargs["large"])
        self.assertEqual(draft.extension, ".sch")
        for symbol in ("L Simulation_SPICE:VSIN V1", "L Device:R R1", "L Device:C C1", "L Device:D D1"):
            self.assertIn(symbol, draft.content)
        self.assertEqual(draft.content.count("L power:GND #PWR"), 3)
        self.assertIn("\t2100 1800 1850 1800", draft.content)  # The source's pins end 200 from its centre.
        # The reading gives a diode's anode first; KiCad's diode has the cathode on pin 1.
        diode = draft.content[draft.content.index("L Device:D D1"):]
        self.assertLess(diode.index("L power:GND"), diode.index("\nout\n"))
        self.assertTrue(any("1k assumed" in note for note in draft.notes))
        self.assertTrue(any("R1 in out 1k" in note for note in draft.notes))
        self.assertIn("L Simulation_SPICE:IDC I1", build_schematic(
            [{"reference": "I1", "value": "1m", "nets": ["a", "GND"]}]))

    def test_a_circuit_that_fails_in_the_simulator_is_read_once_more(self):
        wrong = {**self.reply, "components": [{"reference": "V1", "value": "5", "nets": ["a", "GND"]},
                                               {"reference": "V2", "value": "3", "nets": ["a", "GND"]}]}
        checks = iter([(False, ["Two voltage sources sit across the same two nets."]), (True, ["V(out) = 1 V"])])
        with patch("paper_conversion.chat", side_effect=[wrong, self.reply]) as model:
            draft = circuit_draft(b"picture", "local", "LTspice", check=lambda content: next(checks))
        self.assertEqual(model.call_count, 2)
        self.assertIn("The simulator rejected it", model.call_args.args[2])
        self.assertIn("SYMATTR InstName C1", draft.content)
        self.assertTrue(any("read again" in note for note in draft.notes))
        self.assertTrue(any(note.startswith("Checked: LTspice simulated") for note in draft.notes))

    def test_simulation_failures_say_what_to_check(self):
        lines = paper_ltspice.explain(['C:\\work\\x.cir(3): Undefined model "abc".', "in: R1 a 0 abc"])
        self.assertTrue(lines[0].startswith("A part's value is not something LTspice knows"))
        self.assertEqual(lines[1], 'Netlist line 3: Undefined model "abc".')
        lines = paper_ltspice.explain(["Voltage source V2 and voltage source V1 are paralleled making an "
                                       "over-defined circuit matrix."])
        self.assertTrue(lines[0].startswith("Two voltage sources"))


class HandOverTests(unittest.TestCase):
    def bench(self, **values):
        statuses = []
        defaults = dict(target="LaTeX", menu="latex", language="Python", workflow="equation", saved=None,
                        draft=Draft("y = x", "\\[\ny = x\n\\]", (), ".tex"), paper_clipboard=None, key_result=None,
                        status=SimpleNamespace(set=statuses.append, get=lambda: statuses[-1] if statuses else ""),
                        statuses=statuses, window=SimpleNamespace(after=lambda delay, function, *args: function(*args),
                                                                  clipboard_get=lambda: ""))
        bench = SimpleNamespace(**{**defaults, **values})
        bench.result = SimpleNamespace(get=lambda *_: bench.draft.content if bench.draft else "")
        return bench

    def run_threads(self):
        return patch("paper_workbench.Thread", lambda target, daemon=True: SimpleNamespace(start=target))

    def test_overleaf_pastes_the_equation_into_the_open_project_only(self):
        import paper_workbench

        bench = self.bench()
        bench.overleaf_fragment = lambda: paper_workbench.PaperWorkbench.overleaf_fragment(bench)
        with patch.object(paper_browser, "copy_text") as copied, self.run_threads(), \
                patch.object(paper_browser, "find_window", return_value=77), \
                patch.object(paper_browser, "paste_into", return_value=True) as pasted, \
                patch.object(paper_workbench.webbrowser, "open") as browser:
            paper_workbench.PaperWorkbench.open_overleaf(bench)
        self.assertEqual(copied.call_args.args[0], "\\begin{equation}\n  y = x\n\\end{equation}")
        self.assertEqual(pasted.call_args.args, (paper_browser.is_overleaf_editor, 77))
        self.assertEqual(browser.call_count, 0)
        self.assertIn("Pasted the LaTeX at the cursor", bench.statuses[-1])

    def test_overleaf_opens_the_project_list_when_no_project_is_open(self):
        import paper_workbench

        bench = self.bench()
        bench.overleaf_fragment = lambda: paper_workbench.PaperWorkbench.overleaf_fragment(bench)
        with patch.object(paper_browser, "copy_text"), patch.object(paper_browser, "find_window", return_value=None), \
                patch.object(paper_browser, "paste_into") as pasted, \
                patch.object(paper_workbench.webbrowser, "open") as browser:
            paper_workbench.PaperWorkbench.open_overleaf(bench)
            self.assertEqual((browser.call_args.args[0], pasted.call_count), (paper_browser.OVERLEAF_PROJECTS, 0))
            self.assertIn("press Overleaf again", bench.statuses[-1])
            # Nothing drafted or copied: the ZIP upload of the local project is offered instead.
            bench.draft, uploads = None, []
            bench.upload_overleaf = lambda: uploads.append(True)
            with patch.object(paper_workbench.paper_clipboard, "load", return_value=None):
                paper_workbench.PaperWorkbench.open_overleaf(bench)
            self.assertEqual(uploads, [True])

    def test_keys_are_pressed_only_in_the_expected_window(self):
        import itertools

        clock = itertools.count(0.0, 0.5)
        with patch.object(paper_browser, "foreground", return_value=(5, "chrome.exe", "News - Google Chrome")), \
                patch.object(paper_browser, "bring_to_front"), patch.object(paper_browser.time, "sleep"), \
                patch.object(paper_browser.time, "monotonic", side_effect=lambda: next(clock)), \
                patch.object(paper_browser.paper_system, "press") as press, patch.object(paper_browser.os, "name", "nt"):
            self.assertFalse(paper_browser.paste_into(paper_browser.is_overleaf_editor, 5, wait=0.0))
            self.assertEqual(press.call_count, 0)
        editor = (5, "chrome.exe", "Thesis - Online LaTeX Editor Overleaf - Google Chrome")
        with patch.object(paper_browser, "foreground", return_value=editor), \
                patch.object(paper_browser, "bring_to_front"), patch.object(paper_browser.time, "sleep"), \
                patch.object(paper_browser.paper_system, "press") as press, patch.object(paper_browser.os, "name", "nt"):
            self.assertTrue(paper_browser.paste_into(paper_browser.is_overleaf_editor, 5))
            self.assertEqual(press.call_args.args, ("ctrl", "v"))
        self.assertFalse(paper_browser.is_overleaf_editor("chrome.exe", "Your Projects - Overleaf, Online LaTeX Editor"))
        self.assertFalse(paper_browser.is_overleaf_editor("Code.exe", "overleaf.tex - Visual Studio Code"))
        self.assertTrue(paper_browser.is_chatgpt("msedge.exe", "ChatGPT - Microsoft Edge"))

    def test_the_image_key_hands_the_drawing_to_chatgpt(self):
        import paper_workbench

        sketch = np.full((160, 200, 3), 240, np.uint8)
        with tempfile.TemporaryDirectory() as folder, patch.object(paper_workbench, "ROOT", Path(folder)), \
                self.run_threads(), patch.object(paper_workbench.time, "sleep"), \
                patch.object(paper_browser, "copy_image") as copied, \
                patch.object(paper_browser, "paste_into", return_value=True) as pasted, \
                patch.object(paper_browser, "foreground", return_value=(1, "chrome.exe", "ChatGPT - Google Chrome")), \
                patch.object(paper_workbench.paper_system, "type_text") as typed, \
                patch.object(paper_workbench.paper_system, "press") as press, \
                patch.object(paper_workbench.webbrowser, "open") as browser:
            bench = self.bench(target="ChatGPT", menu="chatgpt", sketch=sketch)
            paper_workbench.PaperWorkbench.finish_image(bench, Draft("", paper_workbench.IMAGE_REQUEST, (), ".md"))
            self.assertTrue(bench.saved.is_file() and bench.saved.name == "sketch.png")
            self.assertGreaterEqual(min(copied.call_args.args[0].shape[:2]), 512)  # Small drawings are enlarged.
            self.assertEqual(browser.call_args.args[0], paper_browser.CHATGPT)
            self.assertEqual(pasted.call_args.args[0], paper_browser.is_chatgpt)
            self.assertEqual(typed.call_args.args[0], paper_workbench.IMAGE_REQUEST)
            self.assertEqual(press.call_args.args, ("enter",))
            # ChatGPT not in front: nothing is typed, and the status says to paste.
            typed.reset_mock()
            pasted.return_value = False
            paper_workbench.PaperWorkbench.finish_image(bench, Draft("", paper_workbench.IMAGE_REQUEST, (), ".md"))
            self.assertEqual(typed.call_count, 0)
            self.assertIn("nothing was typed", bench.statuses[-1])

    def test_a_drawing_is_taken_without_reading_it(self):
        import paper_workbench

        bench = SimpleNamespace(results=queue.SimpleQueue(), args=SimpleNamespace(model="local"), menu="chatgpt")
        selected = SimpleNamespace(page=SimpleNamespace(image=np.zeros((30, 50, 3), np.uint8)), rotation=90,
                                   evidence=None)
        with patch.object(paper_workbench, "read_recording") as reader, patch("paper_conversion.chat") as model:
            paper_workbench.PaperWorkbench._generate(bench, selected, "ChatGPT", "Python", "image", True)
        draft, figure, workflow = bench.results.get_nowait()
        self.assertEqual((workflow, draft.content, reader.call_count, model.call_count),
                         ("image", paper_workbench.IMAGE_REQUEST, 0, 0))
        self.assertEqual(bench.sketch.shape[:2], (50, 30))  # Turned upright for the writer.

    def test_a_bitmap_for_the_clipboard_has_padded_bottom_up_rows(self):
        image = np.arange(3 * 5 * 3, dtype=np.uint8).reshape(3, 5, 3)
        data = paper_browser.dib(image)
        self.assertEqual(len(data), 40 + 3 * 16)
        self.assertEqual(data[40:55], image[2].tobytes())  # The last row comes first.


if __name__ == "__main__":
    unittest.main()
