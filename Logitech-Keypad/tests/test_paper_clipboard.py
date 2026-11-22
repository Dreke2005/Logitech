import shutil
import tempfile
import unittest
from functools import partial
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import paper_clipboard
from paper_clipboard import Clip, parse_reading
from paper_conversion import Draft


class KindTests(unittest.TestCase):
    def test_kind_line_is_read_in_any_spelling(self):
        cases = {
            "KIND: equation\n\\[ y = \\frac{x^2}{2} \\]": ("equation", r"y = \frac{x^2}{2}"),
            "**Kind:** Graph\ny = sin(x)\n-2 <= x <= 3": ("plot", "y = sin(x)\n-2 <= x <= 3"),
            "kind = TODO\n* buy milk\n2. [x] call": ("todo", "- [ ] buy milk\n- [x] call"),
            "KIND: table\n| a | b |\n|---|---|\n| 1 | 2 |": ("table", "a | b\n1 | 2"),
            "```\nKIND: code\ndef f(x):\n    return x\n```": ("code", "def f(x):\n    return x"),
            "KIND: text Hello there": ("text", "Hello there"),
        }
        for reading, expected in cases.items():
            with self.subTest(reading=reading):
                self.assertEqual(parse_reading(reading), expected)

    def test_heuristics_without_kind_line(self):
        cases = {
            r"E = \frac{1}{2} m v^2": "equation",
            "y = 3x + 1": "equation",
            "def area(r):\n    return 3.14 * r * r": "code",
            "for i in range(3):\n    print(i)": "code",
            "x = 1;\ny = 2;": "code",
            "name | age\nAda | 36": "table",
            "- [ ] write report\n- [ ] send it": "todo",
            "plot(x, x.^2)": "plot",
            "Remember to ask about the deadline.": "text",
            "camera -> model -> keypad": "diagram",
        }
        for text, kind in cases.items():
            with self.subTest(text=text):
                self.assertEqual(paper_clipboard.classify(text), kind)
                self.assertEqual(parse_reading(text)[0], kind)

    def test_clip_round_trip_and_currency(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "clipboard.json"
            figure = Path(folder) / "plot.pdf"
            figure.write_bytes(b"%PDF-1.4")
            clip = Clip("plot", "y = x^2", "MATLAB", files=[figure])
            paper_clipboard.save(clip, path)
            loaded = paper_clipboard.load(path)
            self.assertEqual((loaded.kind, loaded.content, loaded.source_app, loaded.files),
                             ("plot", "y = x^2", "MATLAB", [str(figure)]))
            self.assertTrue(loaded.created)
            self.assertEqual(loaded.image(), figure)
            self.assertTrue(loaded.matches(f"  {figure}\r\n"))
            self.assertFalse(loaded.matches("something else"))
            self.assertFalse(loaded.matches(""))
            path.write_text("{not json", encoding="utf-8")
            self.assertIsNone(paper_clipboard.load(path))
            with self.assertRaises(ValueError):
                Clip("poem", "x")


class TableTests(unittest.TestCase):
    def test_booktabs_table_escapes_and_aligns_numbers(self):
        table = paper_clipboard.latex_table("Item | Cost | Share\nA_1 & B | 1,200 | 10%\n#2 {x} | −3.5 | 4%",
                                            label="tab:table_1")
        self.assertIn(r"\begin{tabular}{lrr}", table)
        self.assertIn(r"\textbf{Item} & \textbf{Cost} & \textbf{Share} \\", table)
        self.assertIn(r"A\_1 \& B & 1,200 & 10\% \\", table)
        self.assertIn(r"\#2 \{x\} & -3.5 & 4\% \\", table)
        self.assertLess(table.index(r"\toprule"), table.index(r"\midrule"))
        self.assertLess(table.index(r"\midrule"), table.index(r"\bottomrule"))
        self.assertIn(r"\label{tab:table_1}", table)
        self.assertTrue(table.startswith(r"\begin{table}[htbp]"))

    def test_table_without_header_and_ragged_rows(self):
        table = paper_clipboard.latex_table("1 | 2 | 3\n4 | 5")
        self.assertNotIn(r"\midrule", table)
        self.assertIn(r"4 & 5 &  \\", table)
        self.assertIn("{rrr}", table)
        self.assertEqual(paper_clipboard.latex_escape("~"), r"\textasciitilde{}")
        self.assertEqual(paper_clipboard.latex_escape("a\\b^c"), r"a\textbackslash{}b\textasciicircum{}c")
        with self.assertRaises(ValueError):
            paper_clipboard.latex_table("no pipes here")

    def test_csv_and_matlab_tables(self):
        self.assertEqual(paper_clipboard.csv_text("a | b, c\n1 | 2"), 'a,"b, c"\n1,2\n')
        numeric = paper_clipboard.matlab_table("x | y\n1 | 2\n3 | −4")
        self.assertIn("M = [1 2;\n     3 -4];", numeric)
        self.assertIn("array2table(M, 'VariableNames', {'x', 'y'})", numeric)
        mixed = paper_clipboard.matlab_table("Name | Score\nAda | 3.5\nBo \"B\" | 2")
        self.assertIn('table(["Ada"; "Bo ""B"""], [3.5; 2]', mixed)


class RenderTests(unittest.TestCase):
    def test_equation_becomes_code_in_each_language(self):
        latex = r"E = \frac{1}{2} m v^{2} + \sin(\pi t)"
        python = paper_clipboard.function_stub(latex, "Python")
        namespace = {}
        exec(compile(python, "equation.py", "exec"), namespace)
        self.assertAlmostEqual(namespace["compute_E"](2, 3, 0.5), 10.0)
        self.assertIn("torch.sin", paper_clipboard.function_stub(latex, "PyTorch"))
        self.assertIn("Math.sin(Math.PI * t)", paper_clipboard.function_stub(latex, "JavaScript"))
        self.assertIn("m: number", paper_clipboard.function_stub(latex, "TypeScript"))
        cpp = paper_clipboard.function_stub(latex, "C++")
        self.assertIn("std::pow(v, 2.0)", cpp)
        self.assertIn("1.0 / 2.0", cpp)
        stub = paper_clipboard.function_stub(r"\int_0^1 f(x)\,dx", "Python")
        self.assertIn("NotImplementedError", stub)
        compile(stub, "stub.py", "exec")

    def test_latex_renderings(self):
        render = partial(paper_clipboard.render, app="LaTeX")
        equation = render(Clip("equation", r"\[ y = x^2 \]"))
        self.assertEqual(equation.content, "\\begin{equation}\n  y = x^2\n\\end{equation}")
        table = render(Clip("table", "a | b\n1 | 2"))
        self.assertEqual((table.folder, table.packages), ("tables", ("booktabs",)))
        code = render(Clip("code", "print(1)"), language="Python")
        self.assertIn(r"\begin{lstlisting}[language=Python]", code.content)
        todo = render(Clip("todo", "- [ ] fix 100% & more"))
        self.assertIn(r"\item fix 100\% \& more", todo.content)
        self.assertIn(r"\begin{verbatim}", render(Clip("diagram", "a -> b")).content)
        self.assertEqual(render(Clip("text", "Cost is $5_x")).content, r"Cost is \$5\_x")
        document = paper_clipboard.latex_draft_document(table)
        self.assertIn(r"\usepackage{amsmath,booktabs}", document)

    def test_vscode_matlab_and_chatgpt_renderings(self):
        vscode = partial(paper_clipboard.render, app="VS Code", language="JavaScript")
        self.assertEqual(vscode(Clip("equation", "y = x^2")).filename, "equation.js")
        self.assertEqual(vscode(Clip("code", "let a = 1;")).filename, "pasted_code.js")
        self.assertEqual(vscode(Clip("table", "a | b\n1 | 2")).content, "a,b\n1,2\n")
        self.assertEqual(vscode(Clip("todo", "- [ ] one\n- [x] two")).content, "// TODO: one\n// DONE: two\n")
        plot = vscode(Clip("plot", "y = sin(x)\n0 <= x <= 3"))
        self.assertIn("x = np.linspace(0, 3, 601)\ny = np.sin(x)", plot.content)
        compile(plot.content, "plot.py", "exec")

        matlab = partial(paper_clipboard.render, app="MATLAB")
        equation = matlab(Clip("equation", r"y = x^2 \cdot e^{-x}")).content
        self.assertIn("y = @(x) x.^2 .* exp(-x);", equation)
        self.assertIn("fplot(y, [-5 5]);", equation)
        self.assertIn("fplot(y, [-1 2]);", matlab(Clip("plot", "y = x^3\n-1 <= x <= 2")).content)
        self.assertEqual(matlab(Clip("code", "a = 1;")).filename, "paper_code.m")
        self.assertIn("array2table", matlab(Clip("table", "a | b\n1 | 2")).content)

        prompt = paper_clipboard.render(Clip("code", "x = 1"), "ChatGPT").content
        self.assertTrue(prompt.startswith("Review this code"))
        self.assertIn("```\nx = 1\n```", prompt)
        with self.assertRaises(ValueError):
            paper_clipboard.render(Clip("text", "x"), "KiCad")


class WorkbenchPasteTests(unittest.TestCase):
    def setUp(self):
        self.folder = Path(tempfile.mkdtemp(prefix="paper-clipboard-test-"))
        self.addCleanup(shutil.rmtree, self.folder, ignore_errors=True)

    def bench(self, target, menu, **values):
        import paper_workbench

        statuses, shown, copied = [], [], []
        defaults = dict(target=target, menu=menu, language="Python", report_folder=None, saved=None,
                        git_repo=None, figure=None, draft=None, workflow="default", paper_clipboard=None,
                        status=SimpleNamespace(set=statuses.append, get=lambda: statuses[-1] if statuses else ""),
                        statuses=statuses, shown=shown, copied=copied, pasted=False,
                        show_draft=shown.append, copy=copied.append, choose_report=lambda: None)
        bench = SimpleNamespace(**{**defaults, **values})
        for name in ("paste_clip", "place_latex", "open_path", "finish_table", "finish_copy", "copy_clip",
                     "draft_clip"):
            setattr(bench, name, partial(getattr(paper_workbench.PaperWorkbench, name), bench))
        return bench

    def project(self):
        project = self.folder / "report"
        project.mkdir()
        (project / "main.tex").write_text("\\documentclass{article}\n\\usepackage{amsmath}\n\\begin{document}\n"
                                          "Text.\n\\end{document}\n", encoding="utf-8")
        return project

    def test_latex_table_key_saves_numbered_tables_and_inputs_them(self):
        import paper_workbench

        project = self.project()
        bench = self.bench("LaTeX", "latex", report_folder=project)
        with patch.object(paper_workbench, "open_code") as opened:
            bench.finish_table(Draft("Item | Cost\nPen & ink | 2.50", "", (), ".tex"))
            bench.finish_table(Draft("a | b\n1 | 2", "", (), ".tex"))
        first, second = project / "tables" / "table_1.tex", project / "tables" / "table_2.tex"
        self.assertIn(r"Pen \& ink & 2.50 \\", first.read_text(encoding="utf-8"))
        self.assertIn(r"\label{tab:table_2}", second.read_text(encoding="utf-8"))
        main = (project / "main.tex").read_text(encoding="utf-8")
        self.assertEqual(main.count(r"\usepackage{booktabs}"), 1)
        self.assertLess(main.index(r"\usepackage{booktabs}"), main.index(r"\begin{document}"))
        self.assertLess(main.index(r"\input{tables/table_1.tex}"), main.index(r"\input{tables/table_2.tex}"))
        self.assertLess(main.index(r"\input{tables/table_2.tex}"), main.index(r"\end{document}"))
        opened.assert_called_with(project / "main.tex")
        self.assertIn("tables/table_2.tex", bench.statuses[-1])

    def test_latex_without_project_saves_a_standalone_draft(self):
        import paper_workbench

        bench = self.bench("LaTeX", "latex")
        with patch.object(paper_workbench, "ROOT", self.folder), patch.object(paper_workbench, "open_code") as opened:
            bench.finish_table(Draft("a | b\n1 | 2", "", (), ".tex"))
            bench.paste_clip(Clip("todo", "- [ ] one", "VS Code"))
        drafts = [path.read_text(encoding="utf-8") for path in (self.folder / "drafts" / "latex").rglob("*.tex")]
        self.assertEqual(len(drafts), 2)
        self.assertTrue(any(r"\usepackage{amsmath,booktabs}" in text and r"\begin{tabular}" in text
                            for text in drafts))
        self.assertEqual(opened.call_count, 2)
        self.assertIn("Pasted todo from VS Code as an itemize list in a standalone draft", bench.statuses[-1])

    def test_latex_equation_and_plot_paste(self):
        import paper_workbench

        project = self.project()
        figure = self.folder / "plot.png"
        figure.write_bytes(b"\x89PNG\r\n\x1a\n" + b"0" * 16)
        bench = self.bench("LaTeX", "latex", report_folder=project)
        bench.stage_figure = lambda path: bench.statuses.append(f"staged {path.name}")
        with patch.object(paper_workbench, "open_code"):
            bench.paste_clip(Clip("equation", r"y = \frac{1}{x}", "MATLAB"))
            bench.paste_clip(Clip("plot", "y = x", "MATLAB", files=[figure]))
        fragment, = (project / "paperbridge").glob("equation-*.tex")
        self.assertEqual(fragment.read_text(encoding="utf-8"), "\\begin{equation}\n  y = \\frac{1}{x}\n\\end{equation}\n")
        self.assertIn("Pasted equation from MATLAB as an equation environment", bench.statuses[-3])
        self.assertTrue(bench.statuses[-1].startswith("Pasted plot from MATLAB as a figure. staged plot.png"))

    def test_vscode_and_matlab_paste_create_and_open_files(self):
        import paper_workbench

        bench = self.bench("VS Code", "vscode")
        with patch.object(paper_workbench, "ROOT", self.folder), patch.object(paper_workbench, "open_code") as opened:
            bench.paste_clip(Clip("equation", r"A = \pi r^2", "LaTeX"))
        path = opened.call_args.args[0]
        self.assertEqual(path.name, "equation.py")
        self.assertTrue(path.is_relative_to(self.folder / "drafts" / "vscode"))
        self.assertIn("def compute_A(r):", path.read_text(encoding="utf-8"))
        self.assertIn("Pasted equation from LaTeX as a Python function: equation.py (opened).", bench.statuses[-1])

        project = self.folder / "project"
        project.mkdir()
        (project / "pasted_code.py").write_text("old\n", encoding="utf-8")
        bench.saved = project
        with patch.object(paper_workbench, "open_code"):
            bench.paste_clip(Clip("code", "print('new')"))
        self.assertEqual((project / "pasted_code_2.py").read_text(encoding="utf-8"), "print('new')\n")

        bench = self.bench("MATLAB", "matlab")
        with patch.object(paper_workbench, "ROOT", self.folder), \
                patch.object(paper_workbench, "open_matlab", side_effect=lambda path: path) as opened:
            bench.paste_clip(Clip("table", "t | v\n0 | 1\n1 | 3"))
        script = opened.call_args.args[0]
        self.assertEqual(script.name, "paper_table.m")
        self.assertIn("array2table", script.read_text(encoding="utf-8"))

    def test_chatgpt_paste_stages_a_review_prompt(self):
        bench = self.bench("ChatGPT", "chatgpt")
        bench.paste_clip(Clip("equation", "y = mx + b"))
        self.assertTrue(bench.shown[-1].content.startswith("Explain this equation"))
        self.assertTrue(bench.pasted)
        self.assertIn("Pasted equation as a review prompt", bench.statuses[-1])

    def test_copy_reading_saves_typed_clipboard(self):
        clipboard = self.folder / "clipboard.json"
        bench = self.bench("MATLAB", "matlab", plot_preview=SimpleNamespace(pack=lambda **_: None))
        with patch.object(paper_clipboard, "CLIPBOARD", clipboard):
            bench.finish_copy(Draft("KIND: table\nx | y\n1 | 2", "", (), ".txt"), None)
            clip = paper_clipboard.load()
        self.assertEqual((clip.kind, clip.content, clip.source_app), ("table", "x | y\n1 | 2", "MATLAB"))
        self.assertEqual(bench.copied, ["x | y\n1 | 2"])
        self.assertTrue(bench.statuses[-1].startswith("Copied table from paper."))

        bench.workflow, bench.draft = "default", Draft("x", "disp(1)", (), ".m")
        self.assertEqual(bench.draft_clip("disp(1)").kind, "code")
        bench.figure = self.folder / "plot.pdf"
        clip = bench.draft_clip("")
        self.assertEqual((clip.kind, clip.text), ("plot", str(self.folder / "plot.pdf")))

    def test_reading_workflows_call_the_shared_reader(self):
        import queue

        import paper_workbench

        bench = SimpleNamespace(results=queue.SimpleQueue(), args=SimpleNamespace(model="local"))
        readings = {"copy": "KIND: plot\ny = sin(x)\n0 <= x <= 2", "table": "a | b\n1 | 2",
                    "diagram": "branch x from main"}
        with patch.object(paper_workbench, "ROOT", self.folder), \
                patch.object(paper_workbench, "read_recording",
                             side_effect=lambda _s, _t, _l, workflow, _m: (readings[workflow], ())) as reader:
            for workflow, target in (("copy", "MATLAB"), ("table", "LaTeX"), ("diagram", "Git")):
                paper_workbench.PaperWorkbench._generate(bench, object(), target, "Python", workflow, True)
                draft, figure, done = bench.results.get_nowait()
                self.assertEqual((done, draft.source), (workflow, readings[workflow]))
                self.assertEqual(reader.call_args.args[1:4], (target, "Python", workflow))
                if workflow == "copy":
                    self.assertEqual(draft.content, "y = sin(x)\n0 <= x <= 2")
                    self.assertTrue(figure.is_file() and figure.with_suffix(".png").is_file())
                    self.assertTrue(figure.is_relative_to(self.folder / "figures"))
                else:
                    self.assertIsNone(figure)


if __name__ == "__main__":
    unittest.main()
