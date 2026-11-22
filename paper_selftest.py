"""End-to-end self-test: drive the real workbench through the cross-app flows without a camera.

    uv run paper_selftest.py

Synthetic photos of handwriting stand in for the camera. The real models read them, the
real workbench generates the results, and every step is checked on what ends up on disk
or on the clipboard: code in VS Code, a plot, copy and paste between apps, LaTeX
equations and tables, a Gantt chart, and git. Nothing is opened on screen and nothing is
written outside a temporary folder; the Windows clipboard text is restored afterwards.
Prints one line per step with the time it took and exits non-zero if a step failed.
"""

import os
import shutil
import subprocess
import sys
import tempfile
import time
import tkinter as tk
import webbrowser
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np

import paper_bench

FONT = Path("C:/Windows/Fonts/Inkfree.ttf")


def photo(lines, text_height=22, seed=1, oval=False):
    """A close photo of one or more handwritten lines; ``oval`` draws a dashed selection oval around them."""
    if isinstance(lines, str):
        lines = [lines]
    rendered = [paper_bench.write(line, FONT) for line in lines]
    width = max(image.shape[1] for image in rendered)
    gap = np.full((rendered[0].shape[0] // 2, width), 255, np.uint8)
    rows = []
    for image in rendered:
        rows += [np.pad(image, ((0, 0), (0, width - image.shape[1])), constant_values=255), gap]
    stacked = np.vstack(rows[:-1])
    scale = stacked.shape[0] / rendered[0].shape[0]
    frame, (left, top, right, bottom) = paper_bench.on_page(stacked, round(text_height * scale), seed)
    if oval:
        center = ((left + right) // 2, (top + bottom) // 2)
        axes = ((right - left) // 2 + 26, (bottom - top) // 2 + 22)
        for start in range(0, 360, 24):
            cv2.ellipse(frame, center, axes, 0, start, start + 14, (70, 70, 70), 2, cv2.LINE_AA)
        return frame[max(0, top - 70):bottom + 70, max(0, left - 90):right + 90]
    return frame[top:bottom, left:right]


class Run:
    def __init__(self, root, local=False):
        import paper_clipboard
        import paper_cloud
        import paper_keypad
        import paper_ltspice
        import paper_workbench

        self.root, self.opened, self.rows = root, [], []
        # Keep this instance apart from a running workbench and from the user's data.
        paper_keypad.ROOT = root
        paper_keypad.STATE, paper_keypad.COMMANDS = root / "state.json", root / "commands"
        paper_workbench.ROOT = root
        paper_clipboard.CLIPBOARD = root / "clipboard.json"
        paper_workbench.open_code = lambda path: self.opened.append(("code", Path(path)))
        paper_workbench.open_matlab = lambda path, run=False: self.opened.append(("matlab", Path(path))) or Path(path)
        paper_workbench.matlab_running = lambda: False
        # The check of the reading ("Is this right?") is answered by the step that tests it; elsewhere it is off.
        paper_workbench.PaperWorkbench.confirm_enabled = False

        class Hidden(tk.Toplevel):
            """The check window exists and takes key presses, but never appears or takes the focus."""

            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                self.withdraw()

            def lift(self, *args):
                pass

            def focus_force(self):
                pass

        paper_workbench.tk.Toplevel = Hidden
        paper_ltspice.open_ltspice = lambda path, run=False: self.opened.append(("ltspice", Path(path)))
        os.startfile = lambda path, *args: self.opened.append(("file", Path(path)))
        webbrowser.open = lambda url, *args, **kwargs: self.opened.append(("web", url)) or True
        for name in ("askdirectory", "askopenfilename", "asksaveasfilename"):
            setattr(paper_workbench.filedialog, name, self._dialog)
        self.window = tk.Tk()
        self.window.withdraw()
        self.app = paper_workbench.PaperWorkbench(self.window, SimpleNamespace(camera=None, model=paper_workbench.MODEL))
        if local:
            paper_cloud.MODE = "local"
        self.app.follow_front_app = lambda now: None
        self.app.announce = lambda: None
        self.dialogs = 0

    def _dialog(self, *args, **kwargs):
        self.dialogs += 1
        return ""

    def pump(self, seconds=120):
        end = time.monotonic() + seconds
        self.window.update()
        while (self.app.busy or self.app.pending_action is not None) and time.monotonic() < end:
            self.window.update()
            time.sleep(0.02)
        for _ in range(4):
            self.window.update()
            time.sleep(0.02)

    def press(self, menu, action, page=None):
        """Press a key, optionally with a photo as the page, and wait for the result."""
        if self.app.menu != menu:
            self.app.select(menu)
        if page is not None:
            self.app.set_page(page)
        start = time.perf_counter()
        self.app.dispatch(menu, action)
        self.pump()
        return time.perf_counter() - start

    def step(self, name, seconds, ok, detail=""):
        self.rows.append((name, seconds, bool(ok), detail))
        print(f"{'PASS' if ok else 'FAIL'}  {seconds:5.2f} s  {name}" + (f"  [{detail}]" if detail else ""), flush=True)

    def text(self):
        return self.app.result.get("1.0", "end-1c")

    def status(self):
        return self.app.status.get()

    def close(self):
        self.app.close()


def computes(code, x, expected):
    """True when the generated Python file compiles and its function gives ``expected`` for ``x``,
    whatever the function is called."""
    scope = {}
    try:
        exec(compile(code, "generated.py", "exec"), scope)
        functions = [value for name, value in scope.items() if callable(value) and not name.startswith("_")
                     and getattr(value, "__module__", None) is None]
        return any(abs(function(x) - expected) < 1e-9 for function in functions)
    except Exception:  # noqa: BLE001 - any failure of generated code fails the step
        return False


def git(repo, *args):
    return subprocess.run(["git", "-C", str(repo), *args], check=True, capture_output=True, text=True).stdout.strip()


def main():
    root = Path(tempfile.mkdtemp(prefix="paperbridge-selftest-"))
    saved_clipboard = None
    import paper_cloud

    run = Run(root, local="--local" in sys.argv)
    app = run.app
    print(f"Reading and generation: {paper_cloud.status()}", flush=True)
    try:
        try:
            saved_clipboard = run.window.clipboard_get()
        except tk.TclError:
            pass

        # 1. Handwritten equation -> Python function opened in VS Code.
        seconds = run.press("vscode", "code", photo("y = x^3 + 3x + 2"))
        opened = [path for kind, path in run.opened if kind == "code"]
        code = opened[-1].read_text(encoding="utf-8") if opened and opened[-1].is_file() else ""
        run.step("VS Code: equation on paper -> function file opened", seconds,
                 computes(code, 2, 16) and "# From paper:" in code,
                 run.status()[-60:])

        # 1b. The same key with the check of the reading on: the window opens, and pressing the key again
        # (as on the keypad) confirms it. Nothing is generated before that.
        app.confirm_enabled = True
        before = len(run.opened)
        app.select("vscode")
        app.set_page(photo("y = 2x^2 + 1"))
        # The worker thread asks through window.after, which needs Tk's main loop; this test pumps events
        # itself, so calls from the worker are queued and run here.
        import queue
        import threading
        calls, after = queue.SimpleQueue(), run.window.after
        run.window.after = lambda delay, function=None, *args: (
            after(delay, function, *args) if threading.current_thread() is threading.main_thread()
            else calls.put((function, args)))
        start = time.perf_counter()
        app.dispatch("vscode", "code")
        end = time.monotonic() + 60
        while app.confirm_state is None and app.busy and time.monotonic() < end:
            run.window.update()
            while not calls.empty():
                function, args = calls.get()
                function(*args)
            time.sleep(0.02)
        run.window.after = after
        asked, nothing_yet = app.confirm_state is not None, len(run.opened) == before
        shown = app.confirm_state[2].get("1.0", "end-1c") if asked else ""
        read_seconds = time.perf_counter() - start
        app.dispatch("vscode", "code")
        run.pump()
        app.confirm_enabled = False
        new = [path for kind, path in run.opened[before:] if kind == "code"]
        confirmed = new[-1].read_text(encoding="utf-8") if new and new[-1].is_file() else ""
        run.step("VS Code: reading is shown for a check; the key confirms it", time.perf_counter() - start,
                 asked and nothing_yet and computes(confirmed, 2, 9),
                 f"asked after {read_seconds:.1f} s: {shown[:30]!r}")

        # 1c. Plot in VS Code: a matplotlib program is written, run, and opened with the figure it drew.
        before = len(run.opened)
        seconds = run.press("vscode", "graph", photo("y = x^3 - 2x"))
        new = [path for kind, path in run.opened[before:] if kind == "code"]
        program = next((path for path in new if path.suffix == ".py"), None)
        drawn = next((path for path in new if path.suffix == ".png"), None)
        source = program.read_text(encoding="utf-8") if program and program.is_file() else ""
        run.step("VS Code: plot on paper -> program written, run, and opened with its figure", seconds,
                 "savefig" in source and "x" in source and drawn is not None and drawn.is_file()
                 and drawn.parent == program.parent and app.figure is not None, run.status()[:70])

        # 1d. Pseudocode has to be written by a model: the result must compile.
        before = len(run.opened)
        seconds = run.press("vscode", "code", photo(["total = 0", "for each v in values:", "add v squared to total",
                                                    "return total / count"]))
        new = [path for kind, path in run.opened[before:] if kind == "code"]
        generated = new[-1].read_text(encoding="utf-8") if new and new[-1].is_file() else ""
        try:
            compile(generated, "generated.py", "exec")
            compiled = "def " in generated
        except SyntaxError:
            compiled = False
        run.step("VS Code: pseudocode on paper -> generated function that compiles", seconds, compiled,
                 run.status()[-90:])

        # 2. Copy that code, paste it into ChatGPT (review prompt) and MATLAB (.m file).
        start = time.perf_counter()
        app.dispatch("vscode", "copy")
        run.pump()
        import paper_clipboard
        clip = paper_clipboard.load()
        seconds = time.perf_counter() - start
        run.step("VS Code: Copy records what was copied", seconds, clip is not None and clip.kind in ("code", "equation"),
                 f"kind={getattr(clip, 'kind', None)}")
        seconds = run.press("chatgpt", "paste")
        run.step("ChatGPT: Paste stages a review prompt with the code", seconds,
                 "def " in run.text() and "Review this code" in run.text() and app.pasted, run.status()[:60])
        app.copy_clip(clip)
        seconds = run.press("matlab", "paste")
        matlab = [path for kind, path in run.opened if kind == "matlab"]
        run.step("MATLAB: Paste writes and opens a .m file", seconds,
                 bool(matlab) and matlab[-1].suffix == ".m" and matlab[-1].is_file(), run.status()[:70])

        # 3. Plot from paper in MATLAB, copy it, paste into a LaTeX project: equation + figure.
        seconds = run.press("matlab", "graph", photo("y = x^2 - 4"))
        figure = app.figure
        scripts = [path for kind, path in run.opened if kind == "matlab" and path.name.startswith("paper_plot")]
        script = scripts[-1].read_text(encoding="utf-8") if scripts and scripts[-1].is_file() else ""
        run.step("MATLAB: plot on paper -> MATLAB script opened, with a preview figure", seconds,
                 figure is not None and Path(figure).is_file() and "x.^2" in script and "exportgraphics" in script,
                 run.status()[:50])
        start = time.perf_counter()
        app.dispatch("matlab", "copy")
        run.pump()
        clip = paper_clipboard.load()
        run.step("MATLAB: Copy takes the plot with its equation", time.perf_counter() - start,
                 clip is not None and clip.kind == "plot" and clip.image() is not None and "x^2" in clip.content,
                 f"kind={getattr(clip, 'kind', None)} content={getattr(clip, 'content', '')[:20]!r}")
        project = root / "report"
        project.mkdir()
        (project / "main.tex").write_text("\\documentclass{article}\n\\usepackage{amsmath}\n\\begin{document}\n"
                                          "Text.\n\\end{document}\n", encoding="utf-8")
        app.report_folder = project
        seconds = run.press("latex", "paste")
        fragments = [path.read_text(encoding="utf-8") for path in (project / "paperbridge").glob("*.tex")] \
            if (project / "paperbridge").is_dir() else []
        both = any("begin{equation}" in text and "includegraphics" in text for text in fragments)
        figures = list((project / "figures").glob("paper-*")) if (project / "figures").is_dir() else []
        main_tex = (project / "main.tex").read_text(encoding="utf-8")
        run.step("LaTeX: Paste inserts the equation and the figure", seconds,
                 both and figures and "\\input{paperbridge/" in main_tex and "graphicx" in main_tex, run.status()[:70])

        # 4. LaTeX equation and table keys.
        seconds = run.press("latex", "equation", photo("a^2 + b^2 = c^2"))
        typeset = "\n".join(path.read_text(encoding="utf-8") for path in (project / "paperbridge").glob("*.tex"))
        included = (project / "main.tex").read_text(encoding="utf-8").count("\\input{paperbridge/")
        run.step("LaTeX: equation on paper -> equation in the project document", seconds,
                 "a^2 + b^2 = c^2" in typeset.replace("{", "").replace("}", "") and "begin{equation}" in typeset
                 and included >= 2,
                 run.status()[:60])
        seconds = run.press("latex", "table", photo(["name | mass | charge", "e | 0.511 | -1", "p | 938 | 1"]))
        tables = list((project / "tables").glob("table_*.tex")) if (project / "tables").is_dir() else []
        table = tables[0].read_text(encoding="utf-8") if tables else ""
        run.step("LaTeX: drawn table -> booktabs table in the project", seconds,
                 "\\toprule" in table and "0.511" in table and "booktabs" in (project / "main.tex").read_text("utf-8"),
                 run.status()[:60])

        # 5. Dashed oval on paper -> Copy -> paste as a function in VS Code.
        seconds = run.press("latex", "copy", photo("v = 3x + 12", oval=True))
        clip = paper_clipboard.load()
        run.step("Copy: dashed oval around an equation on paper", seconds,
                 clip is not None and clip.kind == "equation" and "3x" in clip.content.replace(" ", ""),
                 f"kind={getattr(clip, 'kind', None)} content={getattr(clip, 'content', '')[:24]!r}")
        before = len(run.opened)
        seconds = run.press("vscode", "paste")
        new = [path for kind, path in run.opened[before:] if kind == "code"]
        pasted = new[-1].read_text(encoding="utf-8") if new and new[-1].is_file() else ""
        run.step("VS Code: Paste turns the copied equation into a function", seconds, "def compute_v(x)" in pasted,
                 run.status()[:60])

        # 6. Plan: written pipeline -> Gantt chart and Mermaid source.
        seconds = run.press("plan", "gantt", photo(["Literature review 2w", "Prototype 3w", "Demo milestone",
                                                    "Report 10d"]))
        charts = list(root.glob("drafts/plan/**/gantt.png"))
        plans = list(root.glob("drafts/plan/**/project_plan.md"))
        plan = plans[0].read_text(encoding="utf-8") if plans else ""
        run.step("Plan: written pipeline -> Gantt chart", seconds,
                 bool(charts) and "gantt" in plan and "Prototype" in plan and ":milestone" in plan, run.status()[:70])

        # 6b. Larger tasks from a snapshot: a class sketch -> project files, an outline -> report skeleton.
        before = len(run.opened)
        seconds = run.press("vscode", "project", photo(["class Sensor", "read() calibrate()", "class Logger",
                                                        "write(value)"]))
        folders = [path for kind, path in run.opened[before:] if kind == "code" and path.is_dir()]
        sources = list(folders[-1].rglob("*.py")) if folders else []
        joined = "\n".join(path.read_text(encoding="utf-8") for path in sources)
        run.step("VS Code: class sketch -> project folder with source files", seconds,
                 bool(sources) and "Sensor" in joined and "Logger" in joined,
                 f"{len(sources)} files; {run.status()[:40]}")
        before = len(run.opened)
        seconds = run.press("latex", "project", photo(["Thermal model report", "Introduction", "Method", "Results"]))
        reports = [path for kind, path in run.opened[before:] if kind == "code"]
        report = reports[-1] if reports else None
        report = (report if report and report.is_file() else (report / "main.tex" if report else None))
        skeleton = report.read_text(encoding="utf-8") if report and report.is_file() else ""
        run.step("LaTeX: written outline -> report skeleton", seconds,
                 "\\section{Method}" in skeleton and "\\title{Thermal model report}" in skeleton, run.status()[:50])
        app.report_folder = project

        # 7. Git: branch from writing, then a copied note pasted as a commit.
        repo = root / "repo"
        repo.mkdir()
        git(repo, "init", "-b", "main")
        git(repo, "-c", "user.name=PaperBridge", "-c", "user.email=paper@example.com", "commit", "--allow-empty", "-m",
            "start")
        git(repo, "config", "user.name", "PaperBridge")
        git(repo, "config", "user.email", "paper@example.com")
        app.repo_pin = app.git_repo = repo
        seconds = run.press("git", "branch", photo("add login page"))
        run.step("Git: branch name on paper -> branch created and checked out", seconds,
                 git(repo, "branch", "--show-current") == "add-login-page", run.status()[:60])
        (repo / "notes.txt").write_text("hello\n", encoding="utf-8")
        app.copy_clip(paper_clipboard.Clip("text", "Add notes file", "VS Code"))
        seconds = run.press("git", "paste")
        run.step("Git: copied text pasted as a commit", seconds, git(repo, "log", "-1", "--format=%s") == "Add notes file",
                 run.status()[:60])
        run.step("No folder or file dialog was needed", 0.0, run.dialogs == 0, f"dialogs={run.dialogs}")
    finally:
        try:
            run.window.clipboard_clear()
            if saved_clipboard:
                run.window.clipboard_append(saved_clipboard)
            run.window.update()
        except tk.TclError:
            pass
        run.close()
        shutil.rmtree(root, ignore_errors=True)
    failed = [row for row in run.rows if not row[2]]
    total = sum(row[1] for row in run.rows)
    print(f"\n{len(run.rows) - len(failed)}/{len(run.rows)} steps passed, {total:.1f} s in total")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
