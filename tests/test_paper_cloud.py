"""Cloud reading and generation with the local fallback, and plots drawn by a generated program."""
import hermetic  # noqa: F401
import json
import subprocess
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

import paper_cloud
import paper_conversion
import paper_transfer
from paper_context import validate_reading
from paper_conversion import make_plot, read_recording, read_static


def cloud(answer):
    """Turn the cloud setting on for one test, with Claude answering ``answer`` (text, or an exception)."""
    effect = {"side_effect": answer} if isinstance(answer, (Exception, list)) else {"return_value": answer}
    return patch.multiple(paper_cloud, enabled=lambda: True, ask=unittest.mock.Mock(**effect))


def page(height=60, width=240):
    image = np.full((height, width, 3), 235, np.uint8)
    image[20:24, 30:200] = 40
    return image


class CloudCallTests(unittest.TestCase):
    def test_only_the_request_and_the_picture_are_sent(self):
        result = json.dumps({"type": "result", "is_error": False, "result": "y = x^2"})
        done = SimpleNamespace(stdout=result + "\n", stderr="", returncode=0)
        with patch.object(paper_cloud, "binary", return_value="claude"), \
                patch.object(subprocess, "run", return_value=done) as run:
            answer = paper_cloud._ask_claude("system text", "read this", [b"\x89PNG-picture"])
        self.assertEqual(answer, "y = x^2")
        command, options = run.call_args.args[0], run.call_args.kwargs
        # No tools, MCP servers, settings, skills, or saved session; a folder that names no user or project.
        for flag in ("--tools", "--strict-mcp-config", "--disable-slash-commands", "--no-session-persistence",
                     "--setting-sources"):
            self.assertIn(flag, command)
        self.assertEqual(command[command.index("--tools") + 1], "")
        self.assertNotIn(str(Path.home()), options["cwd"])
        self.assertNotIn(str(Path.cwd()), " ".join(command) + options["input"])
        message = json.loads(options["input"])["message"]["content"]
        self.assertEqual([block["type"] for block in message], ["image", "text"])
        self.assertEqual(message[1]["text"], "read this")

    def test_a_failure_turns_the_cloud_off_for_a_while(self):
        paper_cloud.reset()
        with patch.object(paper_cloud, "binary", return_value="claude"), patch.object(paper_cloud, "MODE", "cloud"), \
                patch.object(subprocess, "run", side_effect=subprocess.TimeoutExpired("claude", 5)):
            self.assertTrue(paper_cloud.enabled())
            with self.assertRaises(paper_cloud.CloudUnavailable):
                paper_cloud._ask_claude("s", "c", timeout=5)
            self.assertFalse(paper_cloud.enabled())
            self.assertIn("local models for now", paper_cloud.status())
        paper_cloud.reset()

    def test_chat_uses_the_cloud_and_falls_back_to_the_local_model(self):
        schema = {"type": "object", "properties": {"content": {"type": "string"}}, "required": ["content"]}
        paper_cloud.take_trail()
        with cloud("\x60\x60\x60json\n{\"content\": \"x = 1\"}\n\x60\x60\x60"), \
                patch("local_model.chat") as local:
            self.assertEqual(paper_cloud.chat("qwen", "s", "c", schema), {"content": "x = 1"})
            self.assertEqual(local.call_count, 0)
        for answer in (paper_cloud.CloudUnavailable("offline"), "no json here"):
            with cloud(answer), patch("local_model.chat", return_value={"content": "local"}) as local:
                self.assertEqual(paper_cloud.chat("qwen", "s", "c", schema, tokens=700), {"content": "local"})
                self.assertEqual(local.call_args.args[0], "qwen")
        steps = paper_cloud.take_trail()
        self.assertEqual([engine for _, engine, _ in steps], ["cloud", "local", "local"])
        self.assertIn("in the cloud and by the local model", paper_cloud.describe(steps))

    def test_the_local_setting_never_asks_the_cloud(self):
        with patch.object(paper_cloud, "ask") as ask, patch("local_model.chat", return_value={"content": "x"}):
            paper_cloud.chat("qwen", "s", "c", {"required": ["content"]})
        self.assertEqual(ask.call_count, 0)


class CloudReadingTests(unittest.TestCase):
    def test_a_snapshot_is_read_in_the_cloud(self):
        paper_conversion._STATIC_CACHE.clear()
        with cloud("$y = x^{3} + e^{x}$"), patch("paper_reader._ask") as local:
            reading = read_static(page(), "VS Code", "Python", "graph")
            again = read_static(page(), "VS Code", "Python", "graph")  # The same picture is not read twice.
            self.assertEqual(paper_cloud.ask.call_count, 1)
        self.assertEqual((reading.source, reading.path, local.call_count), ("y = x^3 + e^x", "cloud", 0))
        self.assertIs(again, reading)
        self.assertTrue(validate_reading("VS Code", "graph", reading.source, "Python")[0])
        paper_conversion._STATIC_CACHE.clear()

    def test_the_local_reader_takes_over_when_the_cloud_fails(self):
        paper_conversion._STATIC_CACHE.clear()
        with cloud(paper_cloud.CloudUnavailable("offline")), \
                patch("paper_conversion.extract_ink", return_value=SimpleNamespace(image=page(), text_height=20.0)), \
                patch("paper_reader._ask", return_value="y = x^2 + 1") as local:
            reading = read_static(page(), "VS Code", "Python", "graph")
        self.assertEqual((reading.source, reading.path, local.call_count), ("y = x^2 + 1", "ink", 1))
        paper_conversion._STATIC_CACHE.clear()

    def test_a_recording_sends_the_photo_and_the_ink_and_previews_stay_local(self):
        ink = np.full((60, 240, 3), 255, np.uint8)
        ink[20:24, 30:200] = 0
        selection = SimpleNamespace(page=SimpleNamespace(image=page()), ink=ink, strokes=(), rotation=180,
                                    text_height=20.0, evidence=None, trace=None, trace_reliable=False)
        with patch.multiple(paper_cloud, enabled=lambda: True), \
                patch.object(paper_cloud, "ask", return_value="y = x + 2") as ask, \
                patch("paper_reader._ask", return_value="y = x + 7") as local:
            final = read_recording(selection, "VS Code", "Python", "graph", "local", hints=("gradient_step",))
            self.assertEqual((final.source, final.path, local.call_count), ("y = x + 2", "cloud", 0))
            self.assertEqual(len(ask.call_args.args[2]), 2)  # The photo crop and the ink picture.
            self.assertIn("gradient_step", ask.call_args.args[1])
            preview = read_recording(selection, "VS Code", "Python", "graph", "local", cloud=False)
            self.assertEqual((preview.source, preview.path, ask.call_count), ("y = x + 7", "ink", 1))


class PlotProgramTests(unittest.TestCase):
    def test_plain_functions_become_programs_without_a_model(self):
        program, domain = paper_transfer.plot_script("y = x^3 + e^x\nf(x) = 2x - 1\n-2 <= x <= 2", "Python")
        self.assertIn("y = x**3 + np.exp(x)", program)
        self.assertIn("f = 2 * x - 1", program)
        self.assertIn("np.linspace(-2, 2, 601)", program)
        self.assertEqual(domain, (-2.0, 2.0))
        compile(program, "paper_plot.py", "exec")
        self.assertTrue(paper_transfer.script_is_safe(program))
        script, _ = paper_transfer.plot_script("y = sin(2*pi*x)/x", "MATLAB")
        self.assertIn("y = sin(2 .* pi .* x) ./ x;", script)
        self.assertIn("exportgraphics(fig", script)
        for unreadable in ("y = globals()", "hello world", "y = __import__('os')"):
            with self.assertRaises(ValueError):
                paper_transfer.plot_script(unreadable, "Python")

    def test_unsafe_programs_are_recognized(self):
        safe = "import numpy as np\nimport matplotlib.pyplot as plt\nx = np.linspace(0, 1)\nplt.plot(x, x)\n"
        self.assertTrue(paper_transfer.script_is_safe(safe))
        for line in ("import os", "import subprocess", "open('a.txt', 'w')", "__import__('os')",
                     "from pathlib import Path\nPath('a').unlink()", "eval('1')", "np.load('x.npy')"):
            self.assertFalse(paper_transfer.script_is_safe(f"{safe}{line}\n"), line)

    def test_the_plot_key_writes_and_runs_a_python_program(self):
        with tempfile.TemporaryDirectory() as folder, patch("paper_conversion.chat") as model:
            draft, figure, script = make_plot("y = x^2 - 4", "VS Code", "TypeScript", "local", Path(folder))
            self.assertEqual((model.call_count, script.name, draft.extension), (0, "paper_plot.py", ".py"))
            self.assertTrue(figure.is_file() and figure.with_suffix(".png").is_file())
            self.assertEqual(figure.parent, script.parent)  # The figure is the one the program saved.
            self.assertIn("y = x**2 - 4", script.read_text(encoding="utf-8"))
            self.assertTrue(any("ran in" in note for note in draft.notes))
            self.assertTrue(any("TypeScript" in note for note in draft.notes))

    def test_matlab_gets_a_script_and_a_preview_without_starting_matlab(self):
        with tempfile.TemporaryDirectory() as folder, patch("paper_transfer.run_matlab_script") as batch:
            draft, figure, script = make_plot("y = x^2 - 4", "MATLAB", "Python", "local", Path(folder))
            self.assertEqual((batch.call_count, script.name), (0, "paper_plot.m"))
            self.assertIn("y = x.^2 - 4;", draft.content)
            self.assertTrue(figure.is_file() and figure.with_suffix(".png").is_file())

    def test_a_model_written_program_is_checked_run_and_repaired(self):
        broken = "import matplotlib.pyplot as plt\nplt.plot([0, 1], [0, undefined_name])\n"
        fixed = ("from pathlib import Path\nimport matplotlib.pyplot as plt\nimport numpy as np\n"
                 "t = np.linspace(0, 6.28, 200)\nfig, ax = plt.subplots()\nax.plot(np.cos(t), np.sin(t))\n"
                 "fig.savefig(Path(__file__).resolve().parent / 'paper_plot.png')\n"
                 "fig.savefig(Path(__file__).resolve().parent / 'paper_plot.pdf')\nplt.show()\n")
        with tempfile.TemporaryDirectory() as folder, patch(
                "paper_conversion.chat", side_effect=[{"content": broken}, {"content": fixed}]) as model:
            draft, figure, script = make_plot("circle: x = cos t, y = sin t", "VS Code", "Python", "local",
                                              Path(folder))
            self.assertEqual(model.call_count, 2)
            self.assertIn("undefined_name", model.call_args.args[2])  # The error went back with the program.
            self.assertTrue(figure is not None and figure.is_file())
            self.assertEqual(script.read_text(encoding="utf-8"), fixed)
        unsafe = "import os\nos.system('echo hi')\n"
        with tempfile.TemporaryDirectory() as folder, patch("paper_conversion.chat", return_value={"content": unsafe}), \
                patch("paper_transfer.run_plot_script") as run:
            draft, figure, script = make_plot("circle: x = cos t, y = sin t", "VS Code", "Python", "local",
                                              Path(folder))
            self.assertEqual((run.call_count, figure), (0, None))
            self.assertTrue(any("was not run" in note for note in draft.notes))

    def test_what_cannot_be_plotted_keeps_the_plotter_message_when_no_model_answers(self):
        with tempfile.TemporaryDirectory() as folder, patch("paper_conversion.chat", side_effect=RuntimeError("down")):
            with self.assertRaises(ValueError):
                make_plot("hello world", "VS Code", "Python", "local", Path(folder))


class WorkbenchTests(unittest.TestCase):
    def bench(self, **values):
        statuses = []
        defaults = dict(target="VS Code", menu="vscode", language="Python", saved=None, figure=None, project_of={},
                        status=SimpleNamespace(set=statuses.append, get=lambda: statuses[-1] if statuses else ""),
                        statuses=statuses, plot_preview=SimpleNamespace(pack=lambda **_: None), key_result=None)
        return SimpleNamespace(**{**defaults, **values})

    def test_the_plot_program_and_its_figure_open_in_vs_code(self):
        import paper_workbench

        with tempfile.TemporaryDirectory() as folder:
            script, figure = Path(folder) / "paper_plot.py", Path(folder) / "paper_plot.pdf"
            bench = self.bench(plot_script=script)
            bench.open_path = lambda path: paper_workbench.PaperWorkbench.open_path(bench, path)
            with patch.object(paper_workbench, "open_code") as opened:
                paper_workbench.PaperWorkbench.finish_plot(bench, paper_conversion.Draft("y = x", "", (), ".py"), figure)
            self.assertEqual([call.args[0] for call in opened.call_args_list], [script, figure.with_suffix(".png")])
            self.assertEqual(bench.saved, script)
            self.assertIn("ran and drew paper_plot.png", bench.statuses[-1])

    def test_a_started_matlab_runs_the_plot_script(self):
        import paper_workbench

        bench = self.bench(target="MATLAB", menu="matlab", plot_script=Path("paper_plot.m"))
        with patch.object(paper_workbench, "open_matlab") as opened, \
                patch.object(paper_workbench, "matlab_running", return_value=False):
            paper_workbench.PaperWorkbench.finish_plot(bench, paper_conversion.Draft("y = x", "", (), ".m"), None)
        self.assertEqual((opened.call_args.args, opened.call_args.kwargs), ((Path("paper_plot.m"),), {"run": True}))
        self.assertIn("MATLAB is starting and runs paper_plot.m", bench.statuses[-1])

    def test_chatgpt_opens_with_the_brief_asked(self):
        import paper_workbench

        brief = "Help me develop this idea.\n\n## What I wrote\n\nrobot arm & gripper"
        bench = self.bench(target="ChatGPT", copy_draft=lambda capture=True: True,
                           result=SimpleNamespace(get=lambda *_: brief))
        with patch.object(paper_workbench.webbrowser, "open", return_value=True) as browser:
            paper_workbench.PaperWorkbench.chatgpt(bench)
            self.assertTrue(browser.call_args.args[0].startswith("https://chatgpt.com/?q=Help%20me%20develop"))
            self.assertIn("%26%20gripper", browser.call_args.args[0])
            bench.result = SimpleNamespace(get=lambda *_: "x" * 7000)  # Too long for a link: clipboard only.
            paper_workbench.PaperWorkbench.chatgpt(bench)
            self.assertEqual(browser.call_args.args[0], "https://chatgpt.com/")
        self.assertIn("clipboard", bench.statuses[-1])

    def test_the_setting_is_read_from_the_settings_file(self):
        import paper_workbench

        with tempfile.TemporaryDirectory() as folder, patch.object(paper_workbench, "ROOT", Path(folder)), \
                patch.object(paper_cloud, "MODE", "local"):
            (Path(folder) / "settings.json").write_text(json.dumps({"generation": "cloud"}), encoding="utf-8")
            paper_workbench.PaperWorkbench.load_settings(SimpleNamespace())
            self.assertEqual(paper_cloud.MODE, "cloud")
            self.assertEqual(paper_cloud.status(), "cloud (Claude)" if paper_cloud.binary() else
                             "local models (the Claude CLI is not installed)")


if __name__ == "__main__":
    unittest.main()
