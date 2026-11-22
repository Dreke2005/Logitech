"""Portable figure and text handoffs for PaperBridge."""

import ast
import keyword
import operator
import re
import shutil
from pathlib import Path
from tempfile import NamedTemporaryFile
from uuid import uuid4

import numpy as np

from paper_math import normalize_scalar

FUNCTIONS = {name: getattr(np, name) for name in ("sin", "cos", "tan", "sqrt", "exp", "log", "log10", "abs")}
OPERATORS = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul,
             ast.Div: operator.truediv, ast.Pow: operator.pow}


def _evaluate(node, x):
    if isinstance(node, ast.Constant) and type(node.value) in (float, int) and abs(node.value) <= 1e12:
        return np.float64(node.value)
    if isinstance(node, ast.Name) and node.id in ("x", "pi", "e"):
        return {"x": x, "pi": np.pi, "e": np.e}[node.id]
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
        value = _evaluate(node.operand, x)
        return -value if isinstance(node.op, ast.USub) else value
    if isinstance(node, ast.BinOp) and type(node.op) in OPERATORS:
        return OPERATORS[type(node.op)](_evaluate(node.left, x), _evaluate(node.right, x))
    if (isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
            and node.func.id in FUNCTIONS and len(node.args) == 1 and not node.keywords):
        return FUNCTIONS[node.func.id](_evaluate(node.args[0], x))
    raise ValueError("Use a function of x with ordinary arithmetic and functions such as sin or sqrt.")


class PlotSpec:
    """What a plot reading asks for: up to four curves of x and one x range."""

    def __init__(self, curves, bounds):
        self.curves, self.bounds = curves, bounds  # curves: [(name, label, expression AST)]


def plot_spec(source):
    """Parse 'y = f(x)' lines, an optional x range, or a simple plot(x, expression). Raises ValueError."""
    text = source.strip().replace("−", "-")
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if not 1 <= len(lines) <= 6 or len(text) > 512:
        raise ValueError("Select a scalar equation or import a figure already exported by your editor.")
    bounds = None
    number = r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)"
    domain = re.compile(rf"({number})\s*(?:<=|≤|<)\s*x\s*(?:<=|≤|<)\s*({number})")
    linspace = re.compile(rf"x\s*=\s*(?:(?:np|numpy)\.)?linspace\(\s*({number})\s*,"
                          rf"\s*({number})(?:\s*,\s*\d+)?\s*\)\s*;?")
    written = []
    for line in lines:
        match = domain.fullmatch(line) or linspace.fullmatch(line)
        if match:
            if bounds is not None:
                raise ValueError("Choose a single x range for this plot.")
            bounds = tuple(map(float, match.groups()))
        elif (match := re.fullmatch(r"(?:plt\.)?plot\(\s*x\s*,\s*(.+)\)\s*;?", line)):
            if not any(match[1].strip() == name for name, _, _ in written):  # plot(x, y) repeats a curve above.
                written.append((None, line, match[1]))
        elif (match := re.fullmatch(r"([A-Za-z][A-Za-z0-9_]*)\s*(?:\(\s*x\s*\))?\s*=\s*(.+?);?", line)) and match[1] != "x":
            written.append((match[1], line.rstrip(";"), match[2]))
        elif "=" not in line and re.search(r"\bx\b|\d", line):
            written.append((None, line.rstrip(";"), line.rstrip(";")))  # A bare expression: y = <expression>.
        else:
            raise ValueError("For more complex editor code, run it in your editor and import the exported figure.")
    if not written:
        raise ValueError("Write a scalar equation such as y = sin(x).")
    if len(written) > 4:
        raise ValueError("Plot at most four curves at once.")
    curves, names = [], set()
    for index, (name, label, expression) in enumerate(written):
        expression = re.sub(r"\b(?:np|numpy|math)\.", "", expression)
        expression = normalize_scalar(expression.replace(".^", "^").replace(".*", "*").replace("./", "/"))
        try:
            tree = ast.parse(expression, mode="eval")
        except SyntaxError as error:
            raise ValueError("Write a scalar equation such as y = sin(x).") from error
        if sum(1 for _ in ast.walk(tree)) > 64:
            raise ValueError("The equation is too complex for the built-in plotter.")
        with np.errstate(all="ignore"):
            _evaluate(tree.body, np.float64(1.0))  # Raises ValueError for anything but arithmetic in x.
        if not name or name in names or name in RESERVED or keyword.iskeyword(name):
            name = f"y{index + 1}" if len(written) > 1 else "y"
        names.add(name)
        curves.append((name, label, tree.body))
    lo, hi = bounds or (-5.0, 5.0)
    if not -10000 <= lo < hi <= 10000:
        raise ValueError("Choose a finite, increasing x range within −10,000 to 10,000.")
    return PlotSpec(curves, (lo, hi))


def graph(source, folder):
    """Render scalar equations or a simple plot(x, expression) without executing code."""
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure

    spec = plot_spec(source)
    lo, hi = spec.bounds
    x = np.linspace(lo, hi, 601)
    values = []
    for _, label, tree in spec.curves:
        with np.errstate(all="ignore"):
            y = np.broadcast_to(_evaluate(tree, x), x.shape).copy()
        y[~np.isfinite(y)] = np.nan
        if np.isfinite(y).sum() < 2:
            raise ValueError("This function has no readable values in the chosen range.")
        values.append((label, y))
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / f"paper-plot-{uuid4().hex[:10]}.pdf"
    figure = Figure(figsize=(6, 3.5), layout="constrained")
    FigureCanvasAgg(figure)
    axes = figure.subplots()
    for label, y in values:
        axes.plot(x, y, label=label)
    axes.set(xlabel="x", ylabel="y")
    axes.grid(alpha=0.25)
    if len(values) > 1:
        axes.legend()
    figure.savefig(path)
    figure.savefig(path.with_suffix(".png"), dpi=180)
    return path, (lo, hi)


# Plot scripts: the figure is made by a program the user can open, change, and run again ------------

PLOT_NAME = "paper_plot"
RESERVED = {"x", "np", "plt", "fig", "ax", "folder", "Path", "pi", "e", "i", "j"}


def _emit(node, matlab):
    """An expression AST (the subset _evaluate accepts) as NumPy or MATLAB source, elementwise in x."""
    precedence = {ast.Add: 1, ast.Sub: 1, ast.Mult: 2, ast.Div: 2, ast.Pow: 4}

    def wrapped(child, parent, right=False):
        text = _emit(child, matlab)
        if isinstance(child, ast.BinOp):
            inner, outer = precedence[type(child.op)], precedence[type(parent)]
            if inner < outer or (inner == outer and (right or outer == 4)):
                return f"({text})"
        if isinstance(child, ast.UnaryOp) and isinstance(parent, (ast.Pow, ast.Mult, ast.Div)):
            return f"({text})"
        return text

    if isinstance(node, ast.Constant):
        return repr(node.value)
    if isinstance(node, ast.Name):
        return {"x": "x", "pi": "pi" if matlab else "np.pi", "e": "exp(1)" if matlab else "np.e"}[node.id]
    if isinstance(node, ast.UnaryOp):
        return ("-" if isinstance(node.op, ast.USub) else "") + wrapped(node.operand, ast.Mult())
    if isinstance(node, ast.Call):
        return f"{'' if matlab else 'np.'}{node.func.id}({_emit(node.args[0], matlab)})"
    if isinstance(node.op, ast.Pow) and isinstance(node.left, ast.Name) and node.left.id == "e":
        return f"{'' if matlab else 'np.'}exp({_emit(node.right, matlab)})"
    symbol = {ast.Add: " + ", ast.Sub: " - ", ast.Mult: " * ", ast.Div: " / ", ast.Pow: "**"}[type(node.op)]
    if matlab:
        symbol = {" * ": " .* ", " / ": " ./ ", "**": ".^"}.get(symbol, symbol)
    return f"{wrapped(node.left, node.op)}{symbol}{wrapped(node.right, node.op, right=True)}"


def plot_script(source, language):
    """A runnable plotting program for the reading, written without a model: matplotlib for
    ``language`` "Python", a MATLAB script for "MATLAB". It saves paper_plot.png (Python: and
    paper_plot.pdf) next to itself and shows the figure. Returns (program text, (low, high)). Raises ValueError when
    the reading is not plain functions of x (then a model writes the program)."""
    spec = plot_spec(source)
    lo, hi = spec.bounds
    matlab = language == "MATLAB"
    comment = "%" if matlab else "#"
    header = [f"{comment} Plot from paper:"] + [f"{comment}   {label}" for _, label, _ in spec.curves]
    several = len(spec.curves) > 1
    if matlab:
        lines = [*header, "", f"x = linspace({lo:g}, {hi:g}, 601);"]
        lines += [f"{name} = {_emit(tree, True)};" for name, _, tree in spec.curves]
        lines += ["", "fig = figure;", "plot(" + ", ".join(f"x, {name}" for name, _, _ in spec.curves) + ");",
                  "grid on", "xlabel('x')", "ylabel('y')"]
        if several:
            labels = ", ".join("'" + label.replace("'", "''") + "'" for _, label, _ in spec.curves)
            lines.append(f"legend({labels}, 'Interpreter', 'none', 'Location', 'best')")
        lines += ["", "% Saved next to this script, for the report.",
                  "folder = fileparts(mfilename('fullpath'));",
                  f"exportgraphics(fig, fullfile(folder, '{PLOT_NAME}.png'), 'Resolution', 180);"]
        return "\n".join(lines) + "\n", (lo, hi)
    lines = [*header, "from pathlib import Path", "", "import matplotlib.pyplot as plt", "import numpy as np", "",
             'np.seterr(all="ignore")', f"x = np.linspace({lo:g}, {hi:g}, 601)"]
    lines += [f"{name} = {_emit(tree, False)}" for name, _, tree in spec.curves]
    lines += ["", 'fig, ax = plt.subplots(figsize=(6, 3.5), layout="constrained")']
    lines += [f"ax.plot(x, np.broadcast_to({name}, x.shape), label={label!r})" for name, label, _ in spec.curves]
    lines += ['ax.set(xlabel="x", ylabel="y")', "ax.grid(alpha=0.25)"]
    if several:
        lines.append("ax.legend()")
    lines += ["", "# Saved next to this script, for the report.", "folder = Path(__file__).resolve().parent",
              f'fig.savefig(folder / "{PLOT_NAME}.png", dpi=180)', f'fig.savefig(folder / "{PLOT_NAME}.pdf")',
              "plt.show()"]
    return "\n".join(lines) + "\n", (lo, hi)


SAFE_MODULES = {"numpy", "matplotlib", "math", "cmath", "scipy", "sympy", "pandas", "pathlib", "itertools",
                "functools", "random", "statistics", "fractions", "decimal", "collections", "mpl_toolkits"}
UNSAFE_NAMES = {"open", "exec", "eval", "compile", "__import__", "input", "breakpoint", "globals", "locals",
                "getattr", "setattr", "delattr", "vars", "exit", "quit"}
UNSAFE_ATTRIBUTES = {"unlink", "rmdir", "rmtree", "remove", "rename", "replace", "write_text", "write_bytes",
                     "system", "popen", "run", "Popen", "chmod", "mkdir", "touch", "symlink_to", "read_text",
                     "read_bytes", "iterdir", "glob", "rglob", "home", "load", "loadtxt", "genfromtxt", "fromfile",
                     "tofile", "save", "savez", "savetxt", "to_csv", "read_csv", "to_pickle", "read_pickle"}


def script_is_safe(code):
    """True when a model-written plotting program only computes and draws: imports from a short list of
    numeric and plotting modules, and no file, process, or introspection calls. Such a program may be
    run without the user reading it first; anything else is opened for review instead."""
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return False
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            if any(alias.name.split(".")[0] not in SAFE_MODULES for alias in node.names):
                return False
        elif isinstance(node, ast.ImportFrom):
            if node.level or (node.module or "").split(".")[0] not in SAFE_MODULES:
                return False
        elif isinstance(node, ast.Name) and (node.id in UNSAFE_NAMES or node.id.startswith("__") and node.id != "__file__"):
            return False
        elif isinstance(node, ast.Attribute) and (node.attr in UNSAFE_ATTRIBUTES or node.attr.startswith("__")):
            return False
    return True


def run_plot_script(path, timeout=40):
    """Run a Python plotting program without a window and return its figure (the PDF, with the PNG beside it).

    Raises ValueError with the program's last error line when it fails or draws nothing."""
    import os
    import subprocess
    import sys

    path = Path(path)
    figure = path.with_name(PLOT_NAME + ".pdf")
    for old in (figure, figure.with_suffix(".png")):
        old.unlink(missing_ok=True)
    environment = {**os.environ, "MPLBACKEND": "Agg", "PYTHONIOENCODING": "utf-8", "PYTHONWARNINGS": "ignore"}
    try:
        done = subprocess.run([sys.executable, str(path)], cwd=path.parent, env=environment, capture_output=True,
                              text=True, encoding="utf-8", errors="replace", timeout=timeout,
                              stdin=subprocess.DEVNULL, creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
    except subprocess.TimeoutExpired:
        raise ValueError(f"The plot script did not finish within {timeout} s.") from None
    except OSError as error:
        raise ValueError(f"The plot script could not be started: {error}") from error
    if done.returncode != 0:
        problem = [line for line in done.stderr.strip().splitlines() if line.strip()]
        raise ValueError("The plot script failed: " + (problem[-1].strip()[:240] if problem else "unknown error"))
    if not figure.with_suffix(".png").is_file():
        raise ValueError("The plot script ran but saved no figure.")
    if not figure.is_file():
        return figure.with_suffix(".png")
    return figure


def find_matlab():
    import os
    import shutil

    installed = sorted(Path(os.environ.get("ProgramFiles", "C:/Program Files")).glob("MATLAB/*/bin/matlab.exe"),
                       reverse=True)
    return shutil.which("matlab.exe") or (str(installed[0]) if installed else None)


def run_matlab_script(path, timeout=150):
    """Run a MATLAB plotting script in batch mode (starting MATLAB takes most of a minute) and return its figure.

    MATLAB's output goes to a log file next to the script instead of a pipe, and a run that exceeds
    ``timeout`` is ended together with the MATLAB it started, so a stuck script cannot block the key."""
    import subprocess

    path = Path(path)
    matlab = find_matlab()
    if not matlab:
        raise ValueError("MATLAB was not found, so the script was not run.")
    figure = path.with_name(PLOT_NAME + ".png")
    figure.unlink(missing_ok=True)
    name = path.name.replace("'", "''")
    log = path.with_suffix(".log")
    hidden = getattr(subprocess, "CREATE_NO_WINDOW", 0)
    try:
        with log.open("w", encoding="utf-8", errors="replace") as output:
            # Figures stay hidden in this run only; the script itself shows its figure when run in MATLAB.
            command = f"set(0, 'DefaultFigureVisible', 'off'); run('{name}'); close all force; exit"
            worker = subprocess.Popen([matlab, "-batch", command], cwd=path.parent, stdout=output,
                                      stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL, creationflags=hidden)
            try:
                code = worker.wait(timeout)
            except subprocess.TimeoutExpired:
                subprocess.run(["taskkill", "/T", "/F", "/PID", str(worker.pid)], capture_output=True,
                               creationflags=hidden, check=False)
                worker.kill()
                if figure.is_file():
                    return figure  # The script drew its figure; only MATLAB's exit was slow.
                raise ValueError(f"MATLAB did not finish the script within {timeout} s.") from None
    except OSError as error:
        raise ValueError(f"MATLAB could not be started: {error}") from error
    try:
        problem = [line for line in log.read_text(encoding="utf-8", errors="replace").splitlines() if line.strip()]
        log.unlink(missing_ok=True)
    except OSError:
        problem = []
    if code != 0 or not figure.is_file():
        raise ValueError("MATLAB reported: " + (" ".join(problem[-2:])[:240] if problem else "no figure was saved"))
    return figure


def import_figure(source, destination):
    source, destination = Path(source), Path(destination).resolve()
    suffix = source.suffix.lower()
    signatures = {".pdf": b"%PDF-", ".png": b"\x89PNG\r\n\x1a\n",
                  ".jpg": b"\xff\xd8\xff", ".jpeg": b"\xff\xd8\xff"}
    if suffix not in signatures or not source.is_file() or source.stat().st_size > 32 * 1024**2:
        raise ValueError("Choose an existing PDF, PNG, or JPEG figure under 32 MB.")
    with source.open("rb") as stream:
        if not stream.read(8).startswith(signatures[suffix]):
            raise ValueError("The figure does not match its file extension.")
    folder = destination / "figures"
    if folder.resolve().parent != destination:
        raise ValueError("The figures folder must be inside the selected LaTeX project.")
    folder.mkdir(exist_ok=True)
    name = f"paper-{uuid4().hex[:10]}{suffix}"
    with source.open("rb") as original, (folder / name).open("xb") as copied:
        shutil.copyfileobj(original, copied)
    return (r"\begin{figure}[htbp]" "\n  " r"\centering" "\n  "
            rf"\includegraphics[width=.9\linewidth]{{figures/{name}}}" "\n  "
            r"\caption{Add a caption.}" "\n" r"\end{figure}")


def fenced_code(text):
    fence = "\x60" * max(3, 1 + max((len(run) for run in re.findall(r"\x60+", text)), default=0))
    return f"{fence}\n{text}\n{fence}"


def import_equation(equation, project):
    """Write a reusable fragment and insert it if the root document is unambiguous."""
    if not equation.strip():
        raise ValueError("No equation is available to import.")
    return import_snippet(equation, project)


def next_numbered(folder, stem, suffix=".tex"):
    """Return folder/<stem>_<n><suffix> with n one past the highest existing number."""
    numbers = [int(match[1]) for path in Path(folder).glob(f"{stem}_*{suffix}")
               if (match := re.fullmatch(rf"{re.escape(stem)}_(\d+){re.escape(suffix)}", path.name))]
    return f"{stem}_{max(numbers, default=0) + 1}{suffix}"


def import_snippet(content, project, folder="paperbridge", name=None, packages=()):
    r"""Save a LaTeX fragment in project/folder and \input it before \end{document} of the root file.

    The root file is main.tex or the only complete document. Missing packages are added to its
    preamble. Returns (fragment path, input command, root file or None)."""
    project = Path(project).resolve()
    if not content.strip():
        raise ValueError("Nothing is available to import.")
    entry, original, insertion = None, None, None
    end = re.compile(rb"(?m)^[ \t]*\\end\{document\}[ \t]*(?:%[^\r\n]*)?\r?$")
    candidates = []
    for path in sorted(project.glob("*.tex")):
        if not path.is_file() or path.is_symlink() or path.stat().st_size > 2 * 1024**2:
            continue
        data = path.read_bytes()
        matches = tuple(end.finditer(data))
        if len(matches) == 1 and re.search(rb"(?m)^[ \t]*\\documentclass\b", data):
            candidates.append((path, data, matches[0].start()))
    main = next((item for item in candidates if item[0].name.lower() == "main.tex"), None)
    if main or len(candidates) == 1:
        entry, original, insertion = main or candidates[0]

    directory = project / folder
    if (not re.fullmatch(r"[A-Za-z0-9_-]+", folder) or directory.resolve().parent != project
            or directory.is_symlink()):
        raise ValueError("The fragment folder must be inside the selected LaTeX project.")
    directory.mkdir(exist_ok=True)
    name = name or f"{'equation' if folder == 'paperbridge' else folder}-{uuid4().hex[:10]}.tex"
    if not re.fullmatch(r"[A-Za-z0-9_.-]+\.tex", name):
        raise ValueError(f"Invalid fragment name: {name}")
    snippet = directory / name
    with snippet.open("x", encoding="utf-8", newline="\n") as output:
        output.write(content.strip() + "\n")
    reference = rf"\input{{{folder}/{snippet.name}}}"
    if entry is None:
        return snippet, reference, None

    newline = b"\r\n" if b"\r\n" in original else b"\n"
    updated = original[:insertion] + reference.encode("ascii") + newline + original[insertion:]
    begin = re.search(rb"(?m)^[ \t]*\\begin\{document\}", updated)
    missing = [package for package in packages if not re.search(
        rb"\\usepackage(?:\[[^\]]*\])?\{[^}]*\b" + re.escape(package.encode("ascii")) + rb"\b", original)]
    if missing and begin:
        lines = b"".join(rb"\usepackage{" + package.encode("ascii") + b"}" + newline for package in missing)
        updated = updated[:begin.start()] + lines + updated[begin.start():]
    with NamedTemporaryFile(dir=project, suffix=".tmp", delete=False) as temporary:
        staged = Path(temporary.name)
        temporary.write(updated)
    try:
        if not entry.is_symlink() and entry.read_bytes() == original:
            staged.replace(entry)
        else:
            entry = None
    except OSError:
        entry = None
    finally:
        staged.unlink(missing_ok=True)
    return snippet, reference, entry
