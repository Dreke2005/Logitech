"""What the user is likely writing for each keypad app and key, and how to check a reading.

One table, keyed by (app, workflow), tells the model what to expect and in which form to
answer, and names the validator that accepts or rejects its reading. A rejected reading is
retried once with the reason, then read again from the photo (see paper_conversion).
"""

import ast
import re
import tempfile
from dataclasses import dataclass, replace
from pathlib import Path

from paper_math import normalize_scalar

APPS = {"vscode": "VS Code", "matlab": "MATLAB", "latex": "LaTeX", "chatgpt": "ChatGPT",
        "kicad": "KiCad", "git": "Git"}
# Keypad keys (paper_menus.json) that read handwriting, and the workflow each one starts.
KEY_WORKFLOWS = {"code": "code", "project": "project", "graph": "graph", "equation": "equation",
                 "table": "table", "brief": "brief", "circuit": "circuit", "repo": "repo",
                 "branch": "branch", "commit": "commit", "diagram": "diagram", "copy": "copy"}
COPY_KINDS = ("equation", "code", "table", "plot", "diagram", "todo", "text")
# The selected app is the prior for what a copied mark most likely is.
COPY_PRIOR = {"MATLAB": ("code", "equation", "plot"), "LaTeX": ("equation", "table"),
              "VS Code": ("code", "equation"), "ChatGPT": ("text", "todo", "code"),
              "KiCad": ("text", "equation"), "Git": ("diagram", "todo", "text")}


@dataclass(frozen=True)
class WritingContext:
    writing: str  # What the user is most likely writing with this key.
    output: str  # The form the reading must take.
    check: str  # Validator: equation, code, plot, table, branch, commit, diagram, copy, or text.
    kinds: tuple = ("text",)
    end_circle: bool = True  # Whether a small closed circle drawn last ends the recording.
    language: str = ""

    @property
    def prompt(self):
        return f"{self.writing} Expected output: {self.output}"


PROJECT = WritingContext(
    "A project sketch: class boxes with fields and methods, arrows between them, or a short prose idea.",
    "One line per class or idea with its members, and relations written as A -> B. Keep names exactly.",
    "text", ("diagram", "outline", "pseudocode"))
PLOT = WritingContext(
    "One function of x to plot, optionally with a written x range on its own line.",
    "y = <expression in x> using ^ for powers and * for products; a range goes on its own line.",
    "plot", ("equation",))

CONTEXTS = {
    ("VS Code", "code"): WritingContext(
        "Mathematics or {language} code: equations, assignments, loops, conditions, or function calls.",
        "The {language} code or equation exactly as written, one source line per written line, "
        "with indentation preserved.", "code", ("equation", "pseudocode", "code")),
    ("VS Code", "project"): PROJECT,
    ("VS Code", "graph"): PLOT,
    ("MATLAB", "code"): WritingContext(
        "MATLAB mathematics or code: matrices such as [1 2; 3 4], equations, or for/if blocks closed by end.",
        "MATLAB source exactly as written. Keep every coefficient, index, matrix row, operator, and line break.",
        "code", ("equation", "matrix", "pseudocode", "code"), language="MATLAB"),
    ("MATLAB", "graph"): PLOT,
    ("MATLAB", "project"): PROJECT,
    ("LaTeX", "equation"): WritingContext(
        "A mathematical equation or expression to typeset.",
        "LaTeX math without $ or \\[ delimiters, using ^{...} for exponents and \\frac{...}{...} for fractions. "
        "Preserve every coefficient, sign, index, and exponent.", "equation", ("equation", "matrix")),
    ("LaTeX", "report"): WritingContext(
        "A report outline: a title and an ordered list of section headings.",
        "The title on the first line, then one heading per line, in written order.", "text", ("outline",)),
    ("LaTeX", "table"): WritingContext(
        "A hand-drawn table: rows of cells separated by vertical lines, often with a header row.",
        "One row per line, cells separated by ' | ', the header row first if there is one. "
        "Every row has the same number of cells; write an empty cell as nothing between separators.",
        "table", ("table",)),
    ("ChatGPT", "brief"): WritingContext(
        "Free-form notes for a prompt: keywords, short sentences, arrows, and lists.",
        "The notes as short lines in reading order, arrows as ->.", "text", ("outline", "diagram", "text")),
    ("KiCad", "circuit"): WritingContext(
        "A circuit sketch with component references (R1, C2, L1, D1), values, and net labels.",
        "One component per line: reference, value, and its two net labels.", "text", ("diagram", "text")),
    ("Git", "repo"): WritingContext(
        "A software project idea for a new repository: its title, goals, components, and class names.",
        "The idea as short lines: title first, then goals and components.", "text",
        ("outline", "diagram", "pseudocode")),
    ("Git", "branch"): WritingContext(
        "A short git branch name or feature description of a few words, such as 'add login page'.",
        "Only the written words on one line. A circle drawn after the text ends the recording; leave it out.",
        "branch", ("text",)),
    ("Git", "commit"): WritingContext(
        "A git commit message, usually one short sentence.",
        "The message exactly as written. A circle drawn after the text ends the recording; leave it out.",
        "commit", ("text",)),
    ("Git", "diagram"): WritingContext(
        "A git branch diagram: horizontal lines are branches with their names written at the start, dots "
        "on a line are commits. A circle around a node means commit at that node, with the text next to it "
        "as the message. A circle around a branch name means push that branch. A line forking off a branch "
        "starts a new branch; a line joining back into a branch is a merge.",
        "One command per line, in time order from left to right, using only: branch <name> [from <parent>] | "
        'commit <branch> "<message>" | merge <source> into <target> | push <branch>.',
        "diagram", ("diagram",), end_circle=False),
    ("*", "copy"): WritingContext(
        "A marked item to copy into another app. In {app} it is most likely: {prior}.",
        "First line KIND: <equation|code|table|plot|diagram|todo|text>, then the content: an equation as "
        "LaTeX, code as code, a table as rows with cells separated by ' | ', a plot as y = <expression>, "
        "a to-do list as '- [ ] item' lines, a branch diagram as git commands.",
        "copy", ("equation", "code", "table", "text"), end_circle=False),
}
CONTEXTS[("*", "table")] = CONTEXTS[("LaTeX", "table")]
CONTEXTS[("*", "graph")] = PLOT
GENERIC = WritingContext("Handwritten notes.", "The writing transcribed literally.", "text", ("text",))


def canonical_workflow(app, workflow):
    """Map the workbench's internal workflow names to the context table's keys."""
    workflow = (workflow or "default").removeprefix("git_")
    if workflow == "new":
        workflow = "repo"
    if workflow == "default":
        return "circuit" if app == "KiCad" else "code"
    if workflow == "idea":
        return "brief"
    if workflow in ("matrix", "solve"):
        return "code"
    if workflow == "architecture":
        return "project"
    if workflow == "project" and app == "LaTeX":
        return "report"
    if workflow == "equation" and app != "LaTeX":
        return "code"
    return workflow


def writing_context(app, workflow, language=None):
    """The context for a key, with the selected language or app filled in."""
    key = canonical_workflow(app, workflow)
    context = CONTEXTS.get((app, key)) or CONTEXTS.get(("*", key)) or GENERIC
    language = context.language or language or ("MATLAB" if app == "MATLAB" else "Python")
    prior = COPY_PRIOR.get(app, ("equation", "text"))
    fill = {"{language}": language, "{app}": app or "the current app", "{prior}": " or ".join(prior)}

    def filled(text):
        for field, value in fill.items():
            text = text.replace(field, value)
        return text

    return replace(context, writing=filled(context.writing), output=filled(context.output), language=language)


def end_circle_enabled(workflow, app=None):
    """False where circles are content (branch diagrams, copy marks), so drawing one must not end recording."""
    return writing_context(app, workflow).end_circle


# Validators ------------------------------------------------------------------------------

PAIRS = {")": "(", "]": "[", "}": "{"}


def _balance(text):
    stack = []
    for index, char in enumerate(text):
        if char in "([{":
            stack.append(char)
        elif char in PAIRS:
            if not stack or stack.pop() != PAIRS[char]:
                return f"unmatched '{char}'"
    return f"unclosed '{stack[-1]}'" if stack else None


def _latex_to_plain(text):
    """A plain, Python-like form of simple LaTeX math, for a syntax check only."""
    text = re.sub(r"\\(?:left|right|displaystyle|,|;|!|quad|qquad)", " ", text)
    text = re.sub(r"\\(?:cdot|times|ast)", "*", text).replace("\\div", "/")
    text = re.sub(r"\\leq?(?![A-Za-z])", "<=", text)
    text = re.sub(r"\\geq?(?![A-Za-z])", ">=", text)
    text = re.sub(r"\\neq?(?![A-Za-z])", "!=", text)
    # Limits of sums, products, integrals, and limits are not part of the syntax check.
    text = re.sub(r"\\(sum|prod|int|oint|lim)\s*(?:_\s*(?:\{[^{}]*\}|\S))?\s*(?:\^\s*(?:\{[^{}]*\}|\S))?",
                  r" \1 ", text)
    for _ in range(8):
        new = re.sub(r"\\[dt]?frac\s*\{([^{}]*)\}\s*\{([^{}]*)\}", r"((\1)/(\2))", text)
        new = re.sub(r"\\sqrt\s*\[([^\[\]{}]*)\]\s*\{([^{}]*)\}", r"((\2)**(1/(\1)))", new)
        new = re.sub(r"\\sqrt\s*\{([^{}]*)\}", r"sqrt(\1)", new)
        new = re.sub(r"_\s*\{([^{}]*)\}", lambda m: "_" + re.sub(r"\W", "", m[1]), new)
        if new == text:
            break
        text = new
    text = re.sub(r"\\([A-Za-z]+)", r"\1", text)
    text = text.replace("{", "(").replace("}", ")").replace("^", "**")
    return text


def _plain_side(side):
    side = side.strip().rstrip(";,.")
    if not side:
        raise SyntaxError("empty side")
    side = normalize_scalar(side.replace("·", "*").replace("÷", "/"))
    side = re.sub(r"\)\s*\(", ")*(", side)
    side = re.sub(r"(?<=[A-Za-z0-9_)])\s+(?=[A-Za-z(])", "*", side)  # "a b" -> a*b
    ast.parse(side, mode="eval")


def check_equation(text):
    """An equation or expression that parses: sympy when installed, otherwise a Python-syntax check."""
    text = text.strip()
    text = re.sub(r"^\$+|\$+$|^\\\[|\\\]$|^\\\(|\\\)$", "", text).strip()
    text = re.sub(r"\\begin\{(?:equation|align|aligned|gather)\*?\}|\\end\{(?:equation|align|aligned|gather)\*?\}",
                  "", text).strip()
    if not text:
        return "empty reading"
    problem = _balance(text)
    if problem:
        return f"unbalanced brackets: {problem}"
    words = [word for word in re.findall(r"[A-Za-z]{4,}", re.sub(r"\\[A-Za-z]+", " ", text))
             if word.lower() not in {"sqrt", "sinh", "cosh", "tanh", "asin", "acos", "atan", "log10", "plot",
                                     "linspace", "frac", "sum", "prod", "lim", "exp"}]
    if len(words) >= 3:
        return "prose words instead of math"
    lines = [line for line in re.split(r"\\\\|\n", text) if line.strip()]
    for line in lines:
        line = line.replace("&", "")
        plain = _latex_to_plain(line).replace("≤", "<=").replace("≥", ">=").replace("−", "-")
        sides = re.split(r"<=|>=|!=|==|=|<|>|≈|\\approx", plain)
        try:
            for side in sides:
                _plain_side(side)
        except (SyntaxError, ValueError) as error:
            if _sympy_accepts(line):
                continue
            return f"'{line.strip()[:60]}' does not parse as math ({getattr(error, 'msg', error)})"
    return None


def _sympy_accepts(line):
    try:
        from sympy.parsing.latex import parse_latex
    except ImportError:
        return False
    try:
        for side in re.split(r"=", line):
            parse_latex(side)
        return True
    except Exception:  # noqa: BLE001 - sympy raises many parser types.
        return False


def check_code(text, language):
    if not text.strip():
        return "empty reading"
    if language in ("Python", "PyTorch"):
        try:
            ast.parse(text)
            return None
        except SyntaxError as error:
            if all(check_equation(line) is None for line in text.splitlines() if line.strip()):
                return None  # Handwritten math is a valid answer for the Code key.
            return f"Python syntax error near line {error.lineno}: {error.msg}"
    stripped = re.sub(r'"(?:[^"\\\n]|\\.)*"', '""', text)
    if language == "MATLAB":
        stripped = re.sub(r"%[^\n]*", "", stripped)
        problem = _balance(re.sub(r"(?<=[\w)\].'])'", "", re.sub(r"(?<![\w)\].'])'[^'\n]*'", "''", stripped)))
        if problem:
            return f"unbalanced brackets: {problem}"
        opened = len(re.findall(r"(?m)^\s*(?:for|parfor|while|if|switch|try)\b", stripped))
        closed = len(re.findall(r"(?m)(?:^|;)\s*end\b", stripped))
        if closed < opened:
            return f"{opened - closed} block(s) without a closing end"
        return None
    stripped = re.sub(r"//[^\n]*|/\*.*?\*/|'(?:[^'\\\n]|\\.)*'|`[^`]*`", "", stripped, flags=re.S)
    problem = _balance(stripped)
    return f"unbalanced brackets: {problem}" if problem else None


def check_table(text):
    rows = [line.strip().strip("|").strip() for line in text.strip().splitlines() if line.strip()]
    rows = [row for row in rows if not re.fullmatch(r"[-|: ]+", row)]  # Markdown separator rows.
    if not rows:
        return "empty table"
    counts = [len(row.split("|")) for row in rows]
    if counts[0] < 2:
        return "a table needs at least two columns separated by ' | '"
    for number, count in enumerate(counts, 1):
        if count != counts[0]:
            return f"row {number} has {count} cells but the first row has {counts[0]}"
    return None


def check_plot(text):
    from paper_transfer import graph

    with tempfile.TemporaryDirectory() as folder:
        try:
            graph(text, Path(folder))
        except (ValueError, ArithmeticError) as error:
            return str(error)
    return None


REF = r"[A-Za-z0-9][A-Za-z0-9._/-]*"
DIAGRAM = [re.compile(rf"branch ({REF})(?: from ({REF}))?"), re.compile(rf'commit ({REF}) "([^"]+)"'),
           re.compile(rf"merge ({REF}) into ({REF})"), re.compile(rf"push ({REF})")]


def _bad_ref(name):
    return (".." in name or name.endswith((".", "/", ".lock")) or "//" in name or "@{" in name
            or name.upper() == "HEAD")


def check_diagram(text):
    lines = [line.strip() for line in text.strip().splitlines() if line.strip()]
    if not lines:
        return "no git commands"
    for line in lines:
        match = next((pattern.fullmatch(line) for pattern in DIAGRAM if pattern.fullmatch(line)), None)
        if match is None:
            return f"'{line[:60]}' is not one of: branch, commit, merge, push"
        names = [match[1]] + ([match[2]] if match.re is not DIAGRAM[1] and match.lastindex == 2 else [])
        if any(name and _bad_ref(name) for name in names):
            return f"invalid branch name in '{line[:60]}'"
    return None


def check_copy(text, language=None):
    first, _, rest = text.strip().partition("\n")
    match = re.fullmatch(r"KIND:\s*(\w+)", first.strip(), re.IGNORECASE)
    if match is None or match[1].lower() not in COPY_KINDS:
        return f"the first line must be KIND: one of {', '.join(COPY_KINDS)}"
    kind, rest = match[1].lower(), rest.strip()
    if not rest:
        return "nothing after the KIND line"
    if kind == "todo":
        bad = [line for line in rest.splitlines() if line.strip() and not re.match(r"\s*- \[[ xX]\] \S", line)]
        return f"'{bad[0][:40]}' is not a '- [ ] item' line" if bad else None
    checks = {"equation": check_equation, "table": check_table, "plot": check_plot, "diagram": check_diagram,
              "code": lambda value: check_code(value, language if language in ("Python", "PyTorch") else "")}
    return checks[kind](rest) if kind in checks else None


def check_branch(text):
    from paper_git import GitError, branch_name

    lines = [line for line in text.strip().splitlines() if line.strip()]
    if len(lines) != 1:
        return "a branch name is a single line"
    if len(lines[0].split()) > 8:
        return "too many words for a branch name"
    try:
        name = branch_name(lines[0])
    except GitError as error:
        return str(error)
    return f"'{name}' is not a valid branch name" if _bad_ref(name) else None


def check_commit(text):
    lines = [line for line in text.strip().splitlines() if line.strip()]
    if not lines:
        return "empty commit message"
    if len(lines) > 4 or len(lines[0]) > 200:
        return "a commit message is one short line, optionally with a few more"
    return None


def check_text(text):
    return None if re.search(r"[A-Za-z0-9]", text) else "no readable text"


def validate_reading(app, workflow, text, language=None):
    """Return (ok, reason) for a model reading in the context of the selected key."""
    if not isinstance(text, str) or not text.strip():
        return False, "empty reading"
    context = writing_context(app, workflow, language)
    check = context.check
    if check == "code":
        reason = check_code(text, context.language)
    elif check == "copy":
        reason = check_copy(text, context.language)
    else:
        reason = {"equation": check_equation, "plot": check_plot, "table": check_table, "branch": check_branch,
                  "commit": check_commit, "diagram": check_diagram}.get(check, check_text)(text)
    return reason is None, reason or ""
