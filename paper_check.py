"""Check handwritten mathematics with sympy: each step of a derivation, and "= ?" results.

Only statements that claim an identity are checked. A line such as ``y = x^2 + 1``
defines y and is left alone; ``(x+1)^2 = x^2 + 2x + 1`` and the later links of a chain
``y = (x+1)^2 = x^2 + 2x + 2`` are compared, and the first step that does not follow
is named.
"""

import re

from paper_context import _latex_to_plain

RELATION = re.compile(r"<=|>=|!=|==|<|>|≈|≤|≥|≠")


def _parse(text):
    from sympy.parsing.sympy_parser import (convert_xor, implicit_multiplication, parse_expr,
                                            standard_transformations)

    text = text.strip().rstrip(";,.")
    if not text or len(text) > 300:
        raise ValueError("empty or too long")
    plain = _latex_to_plain(text).replace("−", "-").replace("·", "*").replace("×", "*").replace("÷", "/")
    transformations = (*standard_transformations, implicit_multiplication, convert_xor)
    return parse_expr(plain, transformations=transformations, evaluate=False)


def _same(left, right):
    from sympy import simplify

    difference = simplify(left - right)
    if difference == 0:
        return True
    symbols = sorted(difference.free_symbols, key=str)
    if not symbols:
        return False
    # simplify can miss identities; a few numeric points settle it.
    for values in ((0.37, 1.91, -2.3), (1.3, -0.7, 2.9), (-1.1, 0.53, 4.2)):
        point = {symbol: values[index % 3] + 0.11 * index for index, symbol in enumerate(symbols)}
        try:
            if abs(complex(difference.evalf(subs=point))) > 1e-7:
                return False
        except (TypeError, ValueError):
            return None
    return True


def check_derivation(text):
    """Notes about steps that do not follow, and results for lines ending in "= ?". Empty when all is fine."""
    try:
        import sympy
    except ImportError:
        return ()
    notes = []
    lines = [line.strip() for line in re.split(r"\\\\|\n", text.replace("&", "")) if line.strip()]
    for line in lines[:12]:
        line = re.sub(r"^\$+|\$+$|^\\\[|\\\]$", "", line).strip()
        if RELATION.search(line) or "=" not in line:
            continue
        parts = [part.strip() for part in line.split("=")]
        try:
            if parts[-1] in ("?", "") and len(parts) >= 2:
                value = sympy.simplify(_parse(parts[-2]).doit())
                notes.append(f"Result: {parts[-2]} = {sympy.sstr(value)}")
                parts = parts[:-1]
            expressions = [_parse(part) for part in parts]
        except Exception:  # noqa: BLE001 - sympy's parser raises many types; unreadable math is not checked.
            continue
        # The first link of "name = expression" is a definition, not a claim.
        start = 1 if len(expressions) > 1 and expressions[0].is_Symbol else 0
        for index in range(start, len(expressions) - 1):
            try:
                same = _same(expressions[index], expressions[index + 1])
            except Exception:  # noqa: BLE001
                same = None
            if same is False:
                notes.append(f"Check this step: {parts[index]} = {parts[index + 1]} does not hold "
                             f"(the left side is {sympy.sstr(sympy.expand(expressions[index]))}).")
                break
    return tuple(notes)
