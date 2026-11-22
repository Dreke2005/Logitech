"""Translate common, unambiguous math without asking a model to invent inputs."""

import ast
import keyword
import re

FUNCTIONS = {"sqrt", "sin", "cos", "tan", "asin", "acos", "atan", "atan2",
             "log", "log10", "exp", "hypot", "floor", "ceil", "abs"}
CONSTANTS = {"pi", "e"}


def normalize_scalar(expression):
    superscripts = str.maketrans("⁰¹²³⁴⁵⁶⁷⁸⁹⁻⁺", "0123456789-+")
    expression = re.sub(r"[⁰¹²³⁴⁵⁶⁷⁸⁹⁻⁺]+",
                        lambda power: "^(" + power[0].translate(superscripts) + ")", expression)
    expression = expression.replace("^", "**").replace("−", "-").replace("×", "*")
    return re.sub(r"(?<![\w.])(\d+(?:\.\d*)?|\.\d+)([eE][+-]?\d+)?"
                  r"(?![eE][+-]?\d)\s*(?=[A-Za-z_(])", r"\1\2*", expression)


def python_equation(source):
    match = re.fullmatch(r"([a-zA-Z][A-Za-z0-9_]*)\s*=\s*([^\n]+)", source.strip())
    if not match:
        return None
    output, expression = match.groups()
    expression = normalize_scalar(expression)
    try:
        parsed = ast.parse(expression.strip(), mode="eval")
    except SyntaxError:
        return None
    operands = (ast.BinOp, ast.UnaryOp, ast.Add, ast.Sub, ast.Mult, ast.Div,
                ast.Pow, ast.Mod, ast.UAdd, ast.USub, ast.Load, ast.Expression)
    variables, imports, calls = [], set(), set()
    for node in ast.walk(parsed):
        if isinstance(node, ast.Call):
            if not isinstance(node.func, ast.Name) or node.func.id not in FUNCTIONS or node.keywords:
                return None
            calls.add(node.func)
            if node.func.id != "abs":
                imports.add(node.func.id)
        elif isinstance(node, ast.Name):
            if node in calls:
                continue
            if node.id in CONSTANTS:
                imports.add(node.id)
            elif not keyword.iskeyword(node.id):
                variables.append((node.col_offset, node.id))
        elif isinstance(node, ast.Constant):
            if type(node.value) not in (int, float):
                return None
        elif not isinstance(node, operands):
            return None
    inputs = list(dict.fromkeys(name for _, name in sorted(variables)))
    if output in inputs or set(inputs) & imports:
        return None
    header = f"from math import {', '.join(sorted(imports))}\n\n\n" if imports else ""
    return f"{header}def compute_{output}({', '.join(inputs)}):\n    return {ast.unparse(parsed.body)}"


def latex_equation(source):
    """Format a simple assignment from its transcription without changing the math."""
    if not re.fullmatch(r"[A-Za-z][A-Za-z0-9]*\s*=\s*[A-Za-z0-9. +−×*/^()⁰¹²³⁴⁵⁶⁷⁸⁹⁻⁺-]+",
                        source.strip()) or python_equation(source) is None:
        return None
    powers = str.maketrans("⁰¹²³⁴⁵⁶⁷⁸⁹⁻⁺", "0123456789-+")
    result = re.sub(r"[⁰¹²³⁴⁵⁶⁷⁸⁹⁻⁺]+", lambda match: "^{" + match[0].translate(powers) + "}",
                    source.strip())
    result = re.sub(r"(?:\*\*|\^)\s*(?:\(([A-Za-z0-9.+-]+)\)|([A-Za-z0-9.+-]+?)(?![A-Za-z0-9.]))",
                    lambda match: "^{" + (match[1] or match[2]) + "}", result)
    result = result.replace("−", "-").replace("×", r"\times ").replace("*", r"\cdot ")
    return "\\[\n" + result + "\n\\]"


MATRIX = re.compile(r"([A-Za-z][A-Za-z0-9_]*)\s*=\s*(\[[0-9eE .+\-,;\n]+\])\s*;?")
NUMBER = re.compile(r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?")
LINEAR = re.compile(r"([A-Z][A-Za-z0-9_]*)(?:\s*\*\s*|\s+)([a-z]\w*)\s*=\s*([A-Za-z]\w*)"
                    r"(?:\s*[\n;]\s*(?:find|solve for)\s+\2)?", re.ASCII | re.IGNORECASE)


def matlab_literal(source, workflow):
    if workflow == "matrix":
        matches = list(MATRIX.finditer(source))
        remainder = MATRIX.sub("", source)
        if matches and not remainder.strip() and all(_numeric_matrix(match[2]) for match in matches):
            return "\n".join(f"{match[1]} = {match[2]};" for match in matches)
    if workflow == "solve":
        expression = re.sub(r"\b([A-Z])([a-z])(\s*=)", r"\1*\2\3", source.strip())
        match = LINEAR.fullmatch(expression)
        if match and len(set(match.groups())) == 3:
            matrix, unknown, rhs = match.groups()
            return f"function {unknown} = paper_draft({matrix}, {rhs})\n    {unknown} = {matrix}\\{rhs};\nend"
    return None


def _numeric_matrix(value):
    rows = [re.split(r"[\s,]+", row.strip()) for row in re.split(r"[;\n]+", value[1:-1].strip())]
    return len({len(row) for row in rows}) == 1 and all(NUMBER.fullmatch(token) for row in rows for token in row)
