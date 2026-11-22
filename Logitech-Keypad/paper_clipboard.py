"""A typed clipboard for PaperBridge: Copy remembers what it copied, Paste renders it for the target app."""

import ast
import csv
import io
import json
import os
import re
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from uuid import uuid4

from paper_keypad import ROOT
from paper_math import FUNCTIONS as MATH_FUNCTIONS, normalize_scalar
from paper_transfer import fenced_code

CLIPBOARD = ROOT / "clipboard.json"
KINDS = ("equation", "code", "table", "plot", "diagram", "todo", "text")
ALIASES = {"graph": "plot", "figure": "plot", "chart": "plot", "math": "equation", "formula": "equation",
           "latex": "equation", "program": "code", "pseudocode": "code", "script": "code", "matrix": "table",
           "list": "todo", "checklist": "todo", "todos": "todo", "tasks": "todo", "task": "todo",
           "note": "text", "notes": "text", "prose": "text", "paragraph": "text", "sketch": "diagram"}
LANGUAGE_EXTENSIONS = {"Python": ".py", "PyTorch": ".py", "JavaScript": ".js", "TypeScript": ".ts", "C++": ".cpp"}
LISTINGS_LANGUAGES = {"Python": "Python", "PyTorch": "Python", "JavaScript": "Java", "TypeScript": "Java",
                      "C++": "C++", "MATLAB": "Matlab"}


@dataclass
class Clip:
    kind: str
    content: str
    source_app: str = ""
    created: str = field(default_factory=lambda: time.strftime("%Y-%m-%dT%H:%M:%S"))
    files: list = field(default_factory=list)
    text: str = ""  # What went onto the Windows clipboard, used to tell whether this clip is still current.

    def __post_init__(self):
        if self.kind not in KINDS:
            raise ValueError(f"Unknown clipboard kind: {self.kind}")
        self.files = [str(path) for path in self.files]
        if not self.text:
            self.text = self.files[0] if self.kind == "plot" and self.files else self.content

    def image(self, suffixes=(".pdf", ".png", ".jpg", ".jpeg")):
        return next((Path(path) for path in self.files if Path(path).suffix.lower() in suffixes
                     and Path(path).is_file()), None)

    def matches(self, clipboard_text):
        """True when the Windows clipboard still holds what this clip put there."""
        return bool(clipboard_text.strip()) and " ".join(clipboard_text.split()) == " ".join(self.text.split())


def save(clip, path=None):
    path = Path(path or CLIPBOARD)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.stem}-{uuid4().hex}.tmp")
    try:
        temporary.write_text(json.dumps(asdict(clip), indent=2), encoding="utf-8")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)
    return path


def load(path=None):
    try:
        value = json.loads(Path(path or CLIPBOARD).read_text(encoding="utf-8"))
        if not isinstance(value.get("content"), str) or not isinstance(value.get("files", []), list):
            return None
        return Clip(value["kind"], value["content"], str(value.get("source_app", "")),
                    str(value.get("created", "")), value.get("files", []), str(value.get("text", "")))
    except (OSError, ValueError, TypeError, AttributeError, KeyError):
        return None


def write_new_file(folder, name, content, unique_folder=True):
    """Write content without overwriting: in a fresh subfolder, or as name, name_2, ... inside folder."""
    folder = Path(folder)
    if unique_folder:
        folder = folder / f"paste-{time.strftime('%Y%m%d-%H%M%S')}-{uuid4().hex[:6]}"
    folder.mkdir(parents=True, exist_ok=True)
    stem, suffix = Path(name).stem, Path(name).suffix
    for index in range(1, 1000):
        path = folder / (name if index == 1 else f"{stem}_{index}{suffix}")
        try:
            with path.open("x", encoding="utf-8", newline="\n") as output:
                output.write(content)
            return path
        except FileExistsError:
            continue
    raise OSError(f"Too many files named {name} in {folder}.")


# Reading what was copied ----------------------------------------------------------------------------------

_KIND_LINE = re.compile(r"^[\W_]*kind[\W_]*?\s*[:=\-–]\s*[*_`]*\s*([A-Za-z]+)[*_`.]*\s*(.*)$", re.I)
_FENCE = re.compile(r"\A\x60{3,}[^\n]*\n(.*?)\n?\x60{3,}\s*\Z", re.S)
_TODO = re.compile(r"^\s*(?:[-*•]\s*)?(?:\[\s*[xX✓✔]?\s*\]|☐|☑|✓|✔)")
_CODE = re.compile(r"^\s*(?:def |class |for\b|while\b|if\b.*[:{]|return\b|import |from \S+ import|function\b|"
                   r"#include|int |double |let |const |var |end\s*$|elif\b|else\s*[:{]?)", re.M)
_LATEX_MATH = re.compile(r"\\(?:frac|sqrt|int|sum|prod|lim|partial|alpha|beta|gamma|theta|lambda|sigma|mu|pi|"
                         r"cdot|times|infty|left|right|begin\{(?:equation|align|bmatrix|pmatrix))")


def strip_fence(text):
    text = text.strip()
    match = _FENCE.match(text)
    return match[1].strip() if match else text


def table_rows(text):
    """Split ' | ' rows into cells; drops Markdown separator rows and outer pipes."""
    rows = []
    for line in str(text).splitlines():
        line = line.strip()
        if "|" not in line:
            continue
        if line.startswith("|"):
            line = line[1:]
        if line.endswith("|"):
            line = line[:-1]
        cells = [cell.strip() for cell in line.split("|")]
        if all(re.fullmatch(r":?-{2,}:?", cell) or not cell for cell in cells):
            continue
        rows.append(cells)
    width = max((len(row) for row in rows), default=0)
    return [row + [""] * (width - len(row)) for row in rows]


def classify(text):
    """Guess the kind of plain text when the reading has no KIND line."""
    text = strip_fence(text)
    lines = [line for line in text.splitlines() if line.strip()]
    if not lines:
        return "text"
    if sum("|" in line for line in lines) >= 2 and len(table_rows(text)) >= 2 and len(table_rows(text)[0]) >= 2:
        return "table"
    if sum(bool(_TODO.match(line)) for line in lines) >= max(1, len(lines) // 2):
        return "todo"
    if re.search(r"^\s*(?:plt\.)?plot\s*\(", text, re.M) or re.search(r"\bfplot\s*\(", text):
        return "plot"
    if _CODE.search(text) or sum(line.rstrip().endswith((";", "{", "}", ":")) for line in lines) >= max(1, len(lines) // 2):
        return "code"
    if _LATEX_MATH.search(text) or (len(lines) <= 3 and "=" in text and re.fullmatch(
            r"[\w\s=+\-−*/^().,\\{}\[\]<>≤≥|'²³√π]+", text) and len(re.findall(r"[A-Za-z]{4,}", text)) <= 2):
        return "equation"
    if re.search(r"(?:->|→|=>|<-)", text):
        return "diagram"
    return "text"


def equation_body(latex):
    """Drop display delimiters so the equation can be re-wrapped."""
    text = strip_fence(latex).strip()
    for left, right in ((r"\[", r"\]"), ("$$", "$$"), (r"\(", r"\)"), ("$", "$")):
        if text.startswith(left) and text.endswith(right) and len(text) > len(left) + len(right):
            text = text[len(left):-len(right)].strip()
            break
    match = re.fullmatch(r"\\begin\{(equation\*?|align\*?|displaymath)\}(.*)\\end\{\1\}", text, re.S)
    return match[2].strip() if match else text


def todo_items(text):
    items = []
    for line in str(text).splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        done = bool(re.match(r"^(?:[-*•]|\d+[.)])?\s*(?:\[\s*[xX✓✔]\s*\]|☑|✓|✔)", stripped))
        item = re.sub(r"^(?:[-*•]|\d+[.)])?\s*(?:\[\s*[xX✓✔]?\s*\]|☐|☑|✓|✔)?\s*", "", stripped).strip()
        if item:
            items.append((item, done))
    return items


def parse_reading(reading):
    """Split a copy reading into (kind, content) using its KIND line or, failing that, heuristics."""
    text = strip_fence(str(reading).replace("\r\n", "\n"))
    lines = text.split("\n")
    first = next((index for index, line in enumerate(lines) if line.strip()), None)
    kind = None
    if first is not None and (match := _KIND_LINE.match(lines[first].strip())):
        word = match[1].lower()
        kind = word if word in KINDS else ALIASES.get(word)
        if kind is not None:
            rest = [match[2]] if match[2].strip() else []
            text = "\n".join(rest + lines[first + 1:])
    content = strip_fence(text)
    if kind is None:
        kind = classify(content)
    if kind == "equation":
        content = equation_body(content)
    elif kind == "todo":
        items = todo_items(content)
        content = "\n".join(f"- [{'x' if done else ' '}] {item}" for item, done in items) or content
    elif kind == "table":
        rows = table_rows(content)
        content = "\n".join(" | ".join(row) for row in rows) if rows else content
    return kind, content.strip()


# Math helpers -------------------------------------------------------------------------------------------

_GREEK = ("alpha", "beta", "gamma", "delta", "epsilon", "theta", "lambda", "mu", "sigma", "omega", "phi", "tau",
          "rho", "eta", "kappa", "nu", "xi", "zeta", "psi", "chi")


def _braced(text, start):
    """Return (inside, end) of a {...} group starting at text[start] == '{'."""
    depth = 0
    for index in range(start, len(text)):
        depth += {"{": 1, "}": -1}.get(text[index], 0)
        if depth == 0:
            return text[start + 1:index], index + 1
    raise ValueError("Unbalanced braces in the equation.")


def latex_to_plain(latex):
    """Turn simple LaTeX math into a plain expression such as 'y = (x^2)/(2) + sqrt(x)'."""
    text = equation_body(latex)
    if re.search(r"\\(?:int|iint|oint|sum|prod|lim|partial|nabla|begin)\b|\bd/d|\\mathrm\{d\}", text):
        raise ValueError("Integrals, sums, limits, derivatives, and matrices need a hand-written implementation.")
    text = re.sub(r"\\(?:left|right|displaystyle|,|;|!|quad|qquad)\b|\\[,;!]", "", text)
    text = text.replace(r"\cdot", "*").replace(r"\times", "*").replace("−", "-").replace(r"\pi", " pi ")
    text = text.replace(r"\ln", "log").replace(r"\exp", "exp")
    for name in _GREEK:
        text = re.sub(rf"\\{name}\b", f" {name} ", text)
    for _ in range(12):
        match = re.search(r"\\[dt]?frac\s*\{", text) or re.search(r"\\sqrt\s*\{", text)
        if match is None:
            break
        first, end = _braced(text, text.index("{", match.start()))
        if "frac" in match[0]:
            if end >= len(text) or text[end] != "{":
                raise ValueError("Unsupported fraction.")
            second, end = _braced(text, end)
            text = f"{text[:match.start()]}(({first})/({second})){text[end:]}"
        else:
            text = f"{text[:match.start()]}sqrt({first}){text[end:]}"
    text = re.sub(r"\\(sin|cos|tan|arcsin|arccos|arctan|log|exp|sqrt)\b", lambda m: m[1].replace("arc", "a"), text)
    text = text.replace("{", "(").replace("}", ")")
    text = re.sub(r"\\[A-Za-z]+", "", text)
    text = re.sub(r"_\(?([A-Za-z0-9]+)\)?", r"_\1", text)
    return " ".join(text.split())


_ALLOWED = (ast.Expression, ast.BinOp, ast.UnaryOp, ast.Add, ast.Sub, ast.Mult, ast.Div, ast.Pow, ast.USub,
            ast.UAdd, ast.Load, ast.Name, ast.Constant, ast.Call)


def split_equation(latex):
    """Return (output name, Python expression AST, ordered inputs) for 'name = expression' math."""
    plain = latex_to_plain(latex)
    match = re.fullmatch(r"\s*([A-Za-z][A-Za-z0-9_]*)(?:\s*\(\s*[A-Za-z_][\w, ]*\))?\s*=\s*(.+)", plain)
    output, expression = (match[1], match[2]) if match else ("y", plain)
    expression = re.sub(r"\b(sin|cos|tan|log|exp|sqrt|abs)\s+([A-Za-z0-9_.]+)", r"\1(\2)", expression)
    expression = normalize_scalar(expression)
    expression = re.sub(r"\)\s*(?=[A-Za-z_(])", ")*", expression)
    for _ in range(2):
        expression = re.sub(r"([A-Za-z0-9_.])\s+(?=[A-Za-z_(])", r"\1*", expression)
    expression = re.sub(r"\b([A-Za-z_]\w*)\s*\(",
                        lambda match: match[0] if match[1] in MATH_FUNCTIONS else f"{match[1]}*(", expression)
    tree = ast.parse(expression, mode="eval")
    functions = set()
    for node in ast.walk(tree):
        if not isinstance(node, _ALLOWED):
            raise ValueError("The equation uses notation that can't be converted to code.")
        if isinstance(node, ast.Call):
            if not isinstance(node.func, ast.Name) or node.func.id not in MATH_FUNCTIONS or node.keywords:
                raise ValueError("The equation uses an unsupported function.")
            functions.add(node.func)
        if isinstance(node, ast.Constant) and type(node.value) not in (int, float):
            raise ValueError("The equation contains a non-numeric constant.")
    inputs = [node.id for node in sorted((node for node in ast.walk(tree) if isinstance(node, ast.Name)
                                          and node not in functions and node.id not in ("pi", "e")),
                                         key=lambda node: node.col_offset)]
    inputs = list(dict.fromkeys(inputs))
    if any(name.startswith("_") for name in inputs):
        raise ValueError("The equation has unreadable variable names.")
    if output in inputs:
        output = f"{output}_next"
    return output, tree.body, inputs


def _emit(node, language):
    """Emit an expression AST in the target language syntax."""
    precedence = {ast.Add: 1, ast.Sub: 1, ast.Mult: 2, ast.Div: 2, ast.Pow: 3}

    def wrap(child, parent_op, right=False):
        text = emit(child)
        if isinstance(child, ast.BinOp):
            inner, outer = precedence[type(child.op)], precedence[type(parent_op)]
            if inner < outer or (inner == outer and (right or outer == 3)):
                return f"({text})"
        if isinstance(child, ast.UnaryOp) and isinstance(parent_op, ast.Pow):
            return f"({text})"
        return text

    def emit(node):
        if isinstance(node, ast.Constant):
            return repr(float(node.value)) if language == "C++" else repr(node.value)
        if isinstance(node, ast.Name):
            if node.id in ("pi", "e"):
                return {"Python": f"math.{node.id}", "PyTorch": f"math.{node.id}", "MATLAB":
                        "pi" if node.id == "pi" else "exp(1)", "C++": f"std::numbers::{node.id}"}.get(
                    language, f"Math.{node.id.upper() if node.id == 'pi' else 'E'}")
            return node.id
        if isinstance(node, ast.UnaryOp):
            return ("-" if isinstance(node.op, ast.USub) else "") + wrap(node.operand, ast.Mult())
        if isinstance(node, ast.Call):
            name, argument = node.func.id, ", ".join(emit(arg) for arg in node.args)
            if language in ("Python", "MATLAB") and name == "abs":
                return f"abs({argument})"
            prefix = {"Python": "math.", "PyTorch": "torch.", "C++": "std::", "MATLAB": ""}.get(language, "Math.")
            return f"{prefix}{name}({argument})"
        op = type(node.op)
        left, right = wrap(node.left, node.op), wrap(node.right, node.op, right=True)
        if op is ast.Pow:
            if isinstance(node.left, ast.Name) and node.left.id == "e":
                return emit(ast.Call(ast.Name("exp"), [node.right], []))
            if language == "C++":
                return f"std::pow({emit(node.left)}, {emit(node.right)})"
            return f"{left}{'.^' if language == 'MATLAB' else ' ** '}{right}"
        symbol = {ast.Add: "+", ast.Sub: "-", ast.Mult: "*", ast.Div: "/"}[op]
        if language == "MATLAB" and op in (ast.Mult, ast.Div):
            symbol = "." + symbol
        return f"{left} {symbol} {right}"

    return emit(node)


def function_stub(latex, language):
    """A function computing 'name = expression' in the selected language, or a commented stub."""
    extension = LANGUAGE_EXTENSIONS.get(language, ".py")
    comment = "#" if extension == ".py" else "//"
    documented = "\n".join(f"{comment} {line}" for line in equation_body(latex).splitlines())
    try:
        output, tree, inputs = split_equation(latex)
    except (ValueError, SyntaxError):
        if extension == ".py":
            return (f"{comment} Equation from paper:\n{documented}\n\n\ndef compute(*args):\n"
                    f"    # TODO: implement the equation above.\n    raise NotImplementedError\n")
        return f"{comment} Equation from paper:\n{documented}\n\n{comment} TODO: implement the equation above.\n"
    body = _emit(tree, language)
    if language in ("Python", "PyTorch"):
        uses_math = "math." in body
        header = ("import math\n" if uses_math else "") + ("import torch\n" if language == "PyTorch" else "")
        return (f"{header}{chr(10) * 2 if header else ''}def compute_{output}({', '.join(inputs)}):\n"
                f'    r"""Equation from paper: {" ".join(equation_body(latex).replace(chr(34), chr(39)).split()).rstrip(chr(92))}"""\n'
                f"    return {body}\n")
    if language == "C++":
        parameters = ", ".join(f"double {name}" for name in inputs)
        includes = "#include <cmath>\n" + ("#include <numbers>\n" if "std::numbers" in body else "")
        return (f"{includes}\n{documented}\ndouble compute_{output}({parameters}) {{\n"
                f"    return {body};\n}}\n")
    typed = language == "TypeScript"
    parameters = ", ".join(f"{name}{': number' if typed else ''}" for name in inputs)
    return (f"{documented}\nexport function compute_{output}({parameters}){': number' if typed else ''} {{\n"
            f"  return {body};\n}}\n")


def plot_spec(content):
    """Return (output, Python expression text, low, high) for a plot or equation clip."""
    number = r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)"
    low, high, expression, output = -5.0, 5.0, None, "y"
    for line in str(content).replace("−", "-").splitlines():
        line = line.strip().rstrip(";")
        if not line:
            continue
        if match := re.fullmatch(rf"({number})\s*(?:<=|≤|<)\s*x\s*(?:<=|≤|<)\s*({number})", line):
            low, high = float(match[1]), float(match[2])
        elif match := re.fullmatch(rf"x\s*=\s*(?:(?:np|numpy)\.)?linspace\(\s*({number})\s*,\s*({number}).*", line):
            low, high = float(match[1]), float(match[2])
        elif match := re.fullmatch(r"(?:plt\.)?plot\(\s*x\s*,\s*(.+)\)", line):
            expression = expression or match[1]
        elif expression is None:
            expression = line
    if expression is None:
        raise ValueError("The copied plot has no equation.")
    expression = re.sub(r"\b(?:np|numpy|math)\.", "", expression.replace(".^", "^").replace(".*", "*")
                        .replace("./", "/"))
    output, tree, inputs = split_equation(expression if "=" in expression else f"y = {expression}")
    if low >= high:
        low, high = -5.0, 5.0
    return output, tree, inputs, (low, high)


# Tables -------------------------------------------------------------------------------------------------

_NUMERIC = re.compile(r"[+\-−]?(?:\d{1,3}(?:,\d{3})+|\d+)?(?:\.\d+)?(?:[eE][+-]?\d+)?\s*%?")
_LATEX_ESCAPES = {"&": r"\&", "%": r"\%", "$": r"\$", "#": r"\#", "_": r"\_", "{": r"\{", "}": r"\}",
                  "~": r"\textasciitilde{}", "^": r"\textasciicircum{}", "\\": r"\textbackslash{}"}


def latex_escape(text):
    return "".join(_LATEX_ESCAPES.get(character, character) for character in str(text))


def is_number(cell):
    cell = cell.strip()
    return bool(cell) and any(character.isdigit() for character in cell) and bool(_NUMERIC.fullmatch(cell))


def split_header(rows):
    """Return (header or None, body). A first row without numbers above numeric rows is a header."""
    if len(rows) < 2:
        return None, rows
    first, rest = rows[0], rows[1:]
    if not any(is_number(cell) for cell in first) and (
            any(is_number(cell) for row in rest for cell in row) or all(cell.strip() for cell in first)):
        return first, rest
    return None, rows


def latex_table(text, caption="Table from paper.", label=None):
    """Booktabs tabular in a table float; numeric columns right-aligned, special characters escaped."""
    rows = table_rows(text)
    if not rows or len(rows[0]) < 1:
        raise ValueError("No table rows found. Separate cells with ' | ', one row per line.")
    header, body = split_header(rows)
    columns = len(rows[0])
    align = "".join("r" if body and all(is_number(row[index]) or not row[index].strip() for row in body)
                    and any(is_number(row[index]) for row in body) else "l" for index in range(columns))

    def line(cells, bold=False):
        cells = [latex_escape(cell.replace("−", "-")) for cell in cells]
        if bold:
            cells = [rf"\textbf{{{cell}}}" if cell else "" for cell in cells]
        return "    " + " & ".join(cells) + r" \\"

    lines = [r"\begin{table}[htbp]", r"  \centering", rf"  \begin{{tabular}}{{{align}}}", r"    \toprule"]
    if header:
        lines += [line(header, bold=True), r"    \midrule"]
    lines += [line(row) for row in body]
    lines += [r"    \bottomrule", r"  \end{tabular}", rf"  \caption{{{latex_escape(caption)}}}"]
    if label:
        lines.append(rf"  \label{{{label}}}")
    lines.append(r"\end{table}")
    return "\n".join(lines) + "\n"


def csv_text(text):
    output = io.StringIO()
    csv.writer(output, lineterminator="\n").writerows(table_rows(text))
    return output.getvalue()


def _matlab_name(text, index):
    name = re.sub(r"\W+", "_", text.strip()).strip("_")
    if not name or not name[0].isalpha():
        name = f"Var{index + 1}" if not name else f"x{name}"
    return name[:63]


def matlab_table(text):
    rows = table_rows(text)
    if not rows:
        raise ValueError("No table rows found.")
    header, body = split_header(rows)
    if header is None:
        header = [f"Var{index + 1}" for index in range(len(rows[0]))]
    names = []
    for index, cell in enumerate(header):
        name = _matlab_name(cell, index)
        while name in names:
            name += "_2"
        names.append(name)
    numeric = [bool(body) and all(is_number(row[index]) for row in body) for index in range(len(names))]
    quoted = "{" + ", ".join(f"'{name}'" for name in names) + "}"

    def number(cell):
        return cell.replace("−", "-").replace(",", "").replace("%", "").strip()

    if all(numeric):
        matrix = ";\n     ".join(" ".join(number(cell) for cell in row) for row in body)
        return (f"% Table from paper\nM = [{matrix}];\nT = array2table(M, 'VariableNames', {quoted});\n"
                "disp(T)\n")
    columns = []
    for index, is_numeric in enumerate(numeric):
        if is_numeric:
            columns.append("[" + "; ".join(number(row[index]) for row in body) + "]")
        else:
            columns.append("[" + "; ".join('"' + row[index].replace('"', '""') + '"' for row in body) + "]")
    return (f"% Table from paper\nT = table({', '.join(columns)}, ...\n    'VariableNames', {quoted});\n"
            "disp(T)\n")


# Rendering for each target app --------------------------------------------------------------------------

@dataclass(frozen=True)
class Rendering:
    """What Paste produces: a file (name + content) or a LaTeX fragment (folder) or a staged prompt."""
    description: str
    content: str
    filename: str = ""
    folder: str = ""
    packages: tuple = ()


def _latex_document(fragment, packages):
    uses = ",".join(("amsmath", *packages))
    return (f"\\documentclass{{article}}\n\\usepackage{{{uses}}}\n\\begin{{document}}\n\n"
            f"{fragment.strip()}\n\n\\end{{document}}\n")


def render_latex(clip, language="Python"):
    kind, content = clip.kind, clip.content
    if kind == "equation":
        return Rendering("an equation environment",
                         "\\begin{equation}\n  " + equation_body(content) + "\n\\end{equation}", folder="paperbridge")
    if kind == "table":
        return Rendering("a booktabs table", latex_table(content), folder="tables", packages=("booktabs",))
    if kind == "code":
        listing = LISTINGS_LANGUAGES.get(clip.source_app if clip.source_app == "MATLAB" else language, "Python")
        body = content.replace(r"\end{lstlisting}", r"\end {lstlisting}")
        return Rendering("a code listing", f"\\begin{{lstlisting}}[language={listing}]\n{body}\n\\end{{lstlisting}}",
                         folder="paperbridge", packages=("listings",))
    if kind == "todo":
        items = todo_items(content) or [(line, False) for line in content.splitlines() if line.strip()]
        lines = "\n".join(f"  \\item {'(done) ' if done else ''}{latex_escape(item)}" for item, done in items)
        return Rendering("an itemize list", f"\\begin{{itemize}}\n{lines}\n\\end{{itemize}}", folder="paperbridge")
    if kind == "diagram" or kind == "plot":
        body = content.replace(r"\end{verbatim}", r"\end {verbatim}")
        return Rendering("a verbatim block", f"\\begin{{verbatim}}\n{body}\n\\end{{verbatim}}", folder="paperbridge")
    paragraphs = "\n\n".join(latex_escape(" ".join(part.split())) for part in re.split(r"\n\s*\n", content)
                             if part.strip())
    return Rendering("a paragraph", paragraphs, folder="paperbridge")


def latex_draft_document(rendering):
    return _latex_document(rendering.content, rendering.packages)


def _python_plot(output, tree, inputs, domain):
    body = _emit(tree, "Python").replace("math.", "np.")
    variable = inputs[0] if len(inputs) == 1 else "x"
    extra = [name for name in inputs if name != variable]
    parameters = "".join(f"{name} = 1.0  # TODO: set {name}\n" for name in extra)
    return ("import matplotlib.pyplot as plt\nimport numpy as np\n\n" + parameters +
            f"{variable} = np.linspace({domain[0]:g}, {domain[1]:g}, 601)\n{output} = {body}\n\n"
            f"plt.plot({variable}, {output})\nplt.xlabel('{variable}')\nplt.ylabel('{output}')\n"
            "plt.grid(alpha=0.25)\nplt.show()\n")


def render_vscode(clip, language="Python"):
    kind, content = clip.kind, clip.content
    extension = LANGUAGE_EXTENSIONS.get(language, ".py")
    comment = "#" if extension == ".py" else "//"
    if kind == "equation":
        return Rendering(f"a {language} function", function_stub(content, language), f"equation{extension}")
    if kind == "code":
        return Rendering(f"a new {language} file", content.rstrip() + "\n", f"pasted_code{extension}")
    if kind == "table":
        return Rendering("a CSV file", csv_text(content), "table.csv")
    if kind == "todo":
        items = todo_items(content) or [(line, False) for line in content.splitlines() if line.strip()]
        lines = "\n".join(f"{comment} {'DONE' if done else 'TODO'}: {item}" for item, done in items)
        return Rendering("TODO comments", lines + "\n", f"todo{extension}")
    if kind == "plot":
        return Rendering("a matplotlib plotting script", _python_plot(*plot_spec(content)), "plot_from_paper.py")
    return Rendering("a Markdown note", (fenced_code(content) if kind == "diagram" else content) + "\n",
                     f"{kind}_from_paper.md")


def render_matlab(clip, language="MATLAB"):
    kind, content = clip.kind, clip.content
    if kind in ("equation", "plot"):
        output, tree, inputs, domain = (plot_spec(content) if kind == "plot"
                                        else (*split_equation(content), (-5.0, 5.0)))
        body = _emit(tree, "MATLAB")
        variable = inputs[0] if inputs else "x"
        extra = inputs[1:]
        lines = ["% Equation from paper: " + " ".join(equation_body(content).split()) if kind == "equation"
                 else "% Plot from paper"]
        lines += [f"{name} = 1;  % TODO: set {name}" for name in extra]
        lines += [f"{output} = @({variable}) {body};",
                  f"fplot({output}, [{domain[0]:g} {domain[1]:g}]);",
                  "grid on", f"xlabel('{variable}')", f"ylabel('{output}')"]
        name = "paper_plot.m" if kind == "plot" else "paper_equation.m"
        return Rendering("an anonymous function with fplot", "\n".join(lines) + "\n", name)
    if kind == "code":
        return Rendering("a MATLAB file", content.rstrip() + "\n", "paper_code.m")
    if kind == "table":
        return Rendering("a MATLAB table", matlab_table(content), "paper_table.m")
    if kind == "todo":
        items = todo_items(content) or [(line, False) for line in content.splitlines() if line.strip()]
        return Rendering("TODO comments", "\n".join(f"% {'DONE' if done else 'TODO'}: {item}"
                                                    for item, done in items) + "\n", "paper_todo.m")
    return Rendering("comments", "\n".join(f"% {line}" for line in content.splitlines()) + "\n", "paper_notes.m")


PROMPTS = {
    "equation": "Explain this equation step by step: define every symbol, say where it comes from, "
                "and check it for mistakes.",
    "code": "Review this code: explain what it does, point out bugs or edge cases, and suggest improvements.",
    "table": "Explain this table: summarize what it shows and point out patterns, outliers, or inconsistencies.",
    "plot": "Explain the function behind this plot: its shape, key points, and behavior over the range.",
    "diagram": "Explain this diagram: describe the components and relationships and flag anything unclear.",
    "todo": "Help me plan these tasks: order them, estimate effort, and point out dependencies.",
    "text": "Help me review and develop this text. Flag any assumptions or missing context.",
}


def render_chatgpt(clip, language=""):
    prompt = f"{PROMPTS[clip.kind]}\n\n{fenced_code(clip.content)}"
    image = clip.image((".png", ".jpg", ".jpeg"))
    if image is not None:
        prompt += f"\n\n(I can attach the rendered image: {image.name})"
    return Rendering("a review prompt", prompt, "brief.md")


RENDERERS = {"LaTeX": render_latex, "VS Code": render_vscode, "MATLAB": render_matlab, "ChatGPT": render_chatgpt}


def render(clip, app, language="Python"):
    if app not in RENDERERS:
        raise ValueError(f"{app} has no typed paste.")
    return RENDERERS[app](clip, language)


def describe(clip):
    source = f" from {clip.source_app}" if clip.source_app else ""
    return f"{clip.kind}{source}"
