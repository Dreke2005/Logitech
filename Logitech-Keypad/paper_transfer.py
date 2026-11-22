"""Portable figure and text handoffs for PaperBridge."""

import ast
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


def graph(source, folder):
    """Render a scalar equation or a simple plot(x, expression) without executing code."""
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure

    text = source.strip().replace("−", "-")
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if not 1 <= len(lines) <= 3 or len(text) > 512:
        raise ValueError("Select a scalar equation or import a figure already exported by your editor.")
    bounds = None
    number = r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)"
    domain = re.compile(rf"({number})\s*(?:<=|≤)\s*x\s*(?:<=|≤)\s*({number})")
    linspace = re.compile(rf"x\s*=\s*(?:(?:np|numpy)\.)?linspace\(\s*({number})\s*,"
                          rf"\s*({number})(?:\s*,\s*\d+)?\s*\)\s*;?")
    expressions, plots = [], []
    for line in lines:
        match = domain.fullmatch(line) or linspace.fullmatch(line)
        if match:
            if bounds is not None:
                raise ValueError("Choose a single x range for this plot.")
            bounds = tuple(map(float, match.groups()))
        elif (match := re.fullmatch(r"y\s*=\s*(.+?);?", line)):
            expressions.append(match[1])
        elif (match := re.fullmatch(r"(?:plt\.)?plot\(\s*x\s*,\s*(.+)\)\s*;?", line)):
            plots.append(match[1])
        else:
            raise ValueError("For more complex editor code, run it in your editor and import the exported figure.")
    if len(expressions) > 1 or len(plots) > 1 or (expressions and plots and plots[0] != "y"):
        raise ValueError("Select one equation or plot.")
    expression = (expressions or plots or [""])[0]
    expression = re.sub(r"\b(?:np|numpy|math)\.", "", expression)
    expression = normalize_scalar(expression.replace(".^", "^").replace(".*", "*").replace("./", "/"))
    lo, hi = bounds or (-5.0, 5.0)
    if not -10000 <= lo < hi <= 10000:
        raise ValueError("Choose a finite, increasing x range within −10,000 to 10,000.")
    try:
        tree = ast.parse(expression, mode="eval")
    except SyntaxError as error:
        raise ValueError("Write a scalar equation such as y = sin(x).") from error
    if sum(1 for _ in ast.walk(tree)) > 64:
        raise ValueError("The equation is too complex for the built-in plotter.")
    x = np.linspace(lo, hi, 601)
    with np.errstate(all="ignore"):
        y = np.broadcast_to(_evaluate(tree.body, x), x.shape).copy()
    y[~np.isfinite(y)] = np.nan
    if np.isfinite(y).sum() < 2:
        raise ValueError("This function has no readable values in the chosen range.")
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / f"paper-plot-{uuid4().hex[:10]}.pdf"
    figure = Figure(figsize=(6, 3.5), layout="constrained")
    FigureCanvasAgg(figure)
    axes = figure.subplots()
    axes.plot(x, y)
    axes.set(xlabel="x", ylabel="y")
    axes.grid(alpha=0.25)
    figure.savefig(path)
    figure.savefig(path.with_suffix(".png"), dpi=180)
    return path, (lo, hi)


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
