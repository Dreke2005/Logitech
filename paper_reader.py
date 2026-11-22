"""Fast reading of clean ink pictures with the local models.

The reader is a small OCR model that returns plain text in well under a second: no JSON,
no notes. The key that was pressed chooses the recognition mode (formula, table, or text)
and the validator that checks the result. Only a reading that fails its check is read a
second time, by the general vision model, with the reason for the rejection.

``decipher`` is what the workbench uses for both the live path (ink registered while
writing) and the snapshot path (ink extracted from a photo). ``READERS`` holds the
variants that ``paper_bench.py`` compares.
"""

import base64
import json
import re
import time
from urllib.request import ProxyHandler, Request, build_opener

import cv2

from local_model import MODEL, OLLAMA_CHAT, SMALL_CONTEXT, _online, _start_portable, make_room
from paper_ink import normalize

OCR_MODELS = ("glm-ocr:q8_0", "glm-ocr:latest")  # The 8-bit build fits beside the vision model on a 4 GB GPU.
OCR_CONTEXT = 2048
GLYPH = int(__import__("os").environ.get("PAPER_GLYPH", "64"))  # Glyph height the reader is fed, in pixels.
INSTRUCTIONS = {
    "equation": "This is one handwritten equation or expression. Write it as plain text on one line, using ^ for "
                "powers, * only where a multiplication sign is written, and names like sin, cos, sqrt. "
                "Output only the equation.",
    "plot": "This is one handwritten function of x to plot. Write it as plain text on one line, using ^ for powers. "
            "Output only the equation.",
    "code": "This is handwritten source code or pseudocode. Write it exactly, one line per written line, keeping "
            "brackets, colons, operators, and indentation. Output only the code.",
    "table": "This is a handwritten table. Write one row per line with the cells separated by ' | '. "
             "Output only the rows.",
    "text": "This is handwritten text. Write it exactly as written, one line per written line. Output only the text.",
}
OCR_PROMPTS = {"equation": "Formula Recognition:", "plot": "Formula Recognition:", "table": "Table Recognition:"}
_installed = {"at": 0.0, "names": ()}


def installed_models():
    """Names of the local models, refreshed at most every 30 s."""
    if time.monotonic() - _installed["at"] > 30 or not _installed["names"]:
        try:
            opener = build_opener(ProxyHandler({}))
            if not _online(opener):
                _start_portable(opener)
            with opener.open(OLLAMA_CHAT.replace("/api/chat", "/api/tags"), timeout=3) as reply:
                _installed["names"] = tuple(model["name"] for model in json.load(reply).get("models", ()))
        except (OSError, ValueError, KeyError, TypeError):
            _installed["names"] = ()
        _installed["at"] = time.monotonic()
    return _installed["names"]


def ocr_model():
    """The installed OCR model, or None; then the general vision model reads everything."""
    names = installed_models()
    return next((name for name in OCR_MODELS if name in names), None)


def _ask(model, prompt, image, tokens=160, context=None, timeout=60):
    ok, data = cv2.imencode(".png", image)
    if not ok:
        raise ValueError("Could not encode the ink picture.")
    options = {"temperature": 0, "num_predict": tokens, "repeat_penalty": 1.1}
    if context:
        options["num_ctx"] = context
    body = json.dumps({"model": model, "stream": False, "think": False, "keep_alive": "30m", "options": options,
                       "messages": [{"role": "user", "content": prompt,
                                     "images": [base64.b64encode(data.tobytes()).decode("ascii")]}]}).encode()
    opener = build_opener(ProxyHandler({}))
    try:
        if not _online(opener):
            _start_portable(opener)
        make_room(model, context or SMALL_CONTEXT, opener)
        with opener.open(Request(OLLAMA_CHAT, body, {"Content-Type": "application/json"}), timeout=timeout) as reply:
            return json.load(reply)["message"]["content"]
    except OSError as error:
        raise RuntimeError(f"Local model {model} is unavailable. Check Ollama and the installed model.") from error
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("The local model returned an unreadable response.") from error


def warm_up():
    """Load the reader so the first reading is as fast as the rest. Errors are ignored."""
    model = ocr_model()
    if model is None:
        return
    try:
        opener = build_opener(ProxyHandler({}))
        body = json.dumps({"model": model, "keep_alive": "30m", "stream": False,
                           "messages": [{"role": "user", "content": "ok"}],
                           "options": {"num_predict": 1, "num_ctx": OCR_CONTEXT}}).encode()
        with opener.open(Request(OLLAMA_CHAT, body, {"Content-Type": "application/json"}), timeout=90) as reply:
            reply.read()
    except (OSError, ValueError):
        pass


def from_latex(text):
    """Plain one-line math from the LaTeX an OCR model returns."""
    text = text.replace("\\left", "").replace("\\right", "").replace("\\,", " ").replace("\\!", "").replace("\\;", " ")
    text = text.replace("\\quad", " ").replace("\\qquad", " ")
    text = re.sub(r"\^\s*\{([^{}]*)\}", lambda m: "^" + (m[1] if re.fullmatch(r"[\w.]+", m[1]) else f"({m[1]})"), text)
    text = re.sub(r"_\s*\{([^{}]*)\}", r"_\1", text)
    for _ in range(4):
        text = re.sub(r"\\[dt]?frac\s*\{([^{}]*)\}\s*\{([^{}]*)\}", r"(\1)/(\2)", text)
        text = re.sub(r"\\sqrt\s*\{([^{}]*)\}", r"sqrt(\1)", text)
        text = re.sub(r"\\(?:mathrm|operatorname|text|mathbf|mathit|mathcal)\s*\{([^{}]*)\}", r"\1", text)
    text = re.sub(r"\\(?:cdot|times|ast)\b", "*", text).replace("\\div", "/")
    text = re.sub(r"\\(?:leq|le)\b", "<=", text)
    text = re.sub(r"\\(?:geq|ge)\b", ">=", text)
    text = re.sub(r"\\(sin|cos|tan|log|ln|exp|pi|theta|alpha|beta|gamma|delta|omega|lambda|mu|sigma|phi|tau)\b", r"\1",
                  text)
    return text.replace("{", "(").replace("}", ")").replace("\\", "")


FUNCTION_NAMES = "sin|cos|tan|asin|acos|atan|sinh|cosh|tanh|log|ln|exp|sqrt|abs|min|max"


def tidy_math(line):
    """Plain math as a person would type it: 'f(x) = 2x^2 - 5x + 7', not 'f (x) = 2 x ^2-5 x+7'."""
    line = " ".join(line.split())
    line = re.sub(r"\s*\^\s*", "^", line)
    line = re.sub(rf"\b({FUNCTION_NAMES})\s+\(", r"\1(", line)
    line = re.sub(r"^([A-Za-z])\s+\(\s*([A-Za-z](?:\s*,\s*[A-Za-z])*)\s*\)\s*=", r"\1(\2) =", line)
    line = re.sub(r"(?<=\d)\s+(?=[A-Za-z(])", "", line)  # 2 x -> 2x
    line = re.sub(r"(?<=\d)\s+(?=\d)", "", line)  # 1 2 -> 12: digits written apart are one number
    line = " ".join(re.sub(r"\s*([=+\-])\s*", r" \1 ", line).split())
    line = re.sub(r"(^|[=(,]) ?- ", r"\1 -", line)  # a leading minus is a sign, not an operator
    return line.replace("( ", "(").replace(" )", ")").strip()


def tidy(text, kind="text"):
    """The reading without wrappers a model likes to add: fences, math delimiters, lead-ins."""
    text = text.strip()
    fence = re.fullmatch(r"\x60{3}[^\n]*\n(.*?)\n?\x60{3}", text, re.S)
    if fence:
        text = fence[1].strip()
    if kind in ("code", "text") and len(text.splitlines()) == 1 and text.startswith(("$", "\\[", "\\(")):
        kind = "equation"  # The reader marked a single line as math: give it the plain-math form.
    text = re.sub(r"^\$+\s*|\s*\$+$", "", text)
    text = re.sub(r"^\\[\[(]\s*|\s*\\[\])]$", "", text)
    text = re.sub(r"^(?:the (?:equation|text|code) (?:is|reads)|transcription|output)\s*:\s*", "", text, flags=re.I)
    if kind in ("equation", "plot"):
        lines = [tidy_math(from_latex(line)) for line in re.split(r"\\\\|\n", text)]
        text = "\n".join(line.rstrip(" .,;") for line in lines if line.strip())
    elif kind == "latex":
        text = "\n".join(line.strip().rstrip(" .,;") for line in text.splitlines() if line.strip())
    elif kind == "table":
        text = _table_rows(text)
    if kind in ("code", "equation", "plot"):
        # A capital O standing alone next to an operator is a zero: 'n > O' -> 'n > 0'.
        text = re.sub(r"(?<=[=<>+\-*/(,\[] )O(?![A-Za-z0-9_(.])|(?<=[=<>+\-*/(,\[])O(?![A-Za-z0-9_(.])", "0", text)
    return text.strip().strip("\x60").strip()


def _table_rows(text):
    """Rows with ' | ' between cells from Markdown, HTML, or LaTeX table output."""
    if "<tr" in text.lower():
        rows = re.findall(r"<tr[^>]*>(.*?)</tr>", text, flags=re.S | re.I)
        cells = [[re.sub(r"<[^>]+>", "", cell).strip() for cell in re.findall(r"<t[dh][^>]*>(.*?)</t[dh]>", row,
                                                                                flags=re.S | re.I)] for row in rows]
        return "\n".join(" | ".join(row) for row in cells if row)
    rows = []
    # A hand-drawn column line is sometimes read as a slash; with no bar anywhere, " / " separates cells.
    separator = r"\s*[|&]\s*" if re.search(r"[|&]", text) else r"\s+/\s+"
    for line in text.replace("\\\\", "\n").splitlines():
        line = line.strip().strip("|").strip()
        if not line or re.fullmatch(r"[-|: ]+", line) or line.startswith("\\"):
            continue
        cells = [cell.strip() for cell in re.split(separator, line)]
        rows.append(" | ".join(cells))
    return "\n".join(rows)


def _qwen_schema(image, kind, hints, feedback=None):
    from local_model import chat
    from paper_conversion import SYSTEM, _reading_schema

    reply = chat(MODEL, SYSTEM, f"{INSTRUCTIONS[kind]} Put the transcription in source.", _reading_schema((kind,)),
                 image=cv2.imencode(".jpg", image)[1].tobytes(), tokens=500)
    return str(reply.get("source", ""))


def _qwen_plain(image, kind, hints, feedback=None):
    prompt = INSTRUCTIONS[kind]
    if hints:
        prompt += f" Names that may appear: {', '.join(hints)}."
    if feedback:
        prompt += f" A previous reading was {feedback[0]!r}, which is not valid: {feedback[1]}. Read it again carefully."
    return _ask(MODEL, prompt, image, context=SMALL_CONTEXT)


def _ocr(image, kind, hints, feedback=None):
    model = ocr_model()
    if model is None:
        return _qwen_plain(image, kind, hints, feedback)
    return _ask(model, OCR_PROMPTS.get(kind, "Text Recognition:"), image, tokens=220, context=OCR_CONTEXT)


READERS = {"qwen-schema": _qwen_schema, "qwen-plain": _qwen_plain, "ocr": _ocr}
DEFAULT = "ocr"


def read_picture(image, text_height, kind="text", reader=None, hints=(), rotation=0, glyph=None, feedback=None,
                 latex=False):
    """The text in a clean ink picture. ``kind`` is equation, plot, code, table, or text.

    With ``latex`` an equation keeps the LaTeX the formula reader produced (fractions, roots,
    indices) instead of being flattened to plain text."""
    kind = kind if kind in INSTRUCTIONS else "text"
    prepared = normalize(image, text_height, glyph=glyph or GLYPH, rotation=rotation)
    reading = READERS[reader or DEFAULT](prepared, kind, tuple(hints), feedback)
    return tidy(reading, "latex" if latex and kind == "equation" and "\\" in reading else kind)


KIND_OF_CHECK = {"equation": "equation", "plot": "plot", "code": "code", "table": "table"}


LOOK_ALIKES = (("+", "t"), ("t", "+"))  # Only swaps that cannot turn one valid equation into another.


def repair_math(text, valid):
    """An invalid equation with one look-alike character swapped so that it becomes valid.

    A handwritten t and a + are the same two strokes. When a reading fails its check, each
    single swap is tried, and a repair is accepted only if exactly one swap makes
    the equation valid; otherwise the reading is left for the second look. Returns None then.
    """
    if not text or len(text) > 120:
        return None
    found = set()
    for wrong, right in LOOK_ALIKES:
        for index, char in enumerate(text):
            if char == wrong:
                candidate = tidy_math(text[:index] + right + text[index + 1:])
                if valid(candidate):
                    found.add(candidate)
                    if len(found) > 1:
                        return None
    return next(iter(found)) if len(found) == 1 else None


def decipher(image, text_height, target, workflow, language=None, hints=(), rotation=0):
    """Read a clean ink picture for a keypad key. Returns (text, notes, valid).

    1. The OCR model reads in the mode the key calls for.
    2. The key's validator checks the reading (an equation must parse, code must be
       balanced, table rows must match, ...).
    3. Only a rejected reading is read again, in the other recognition mode. The valid one
       wins; if neither is valid the first is returned with a note.
    """
    from paper_context import validate_reading, writing_context

    context = writing_context(target, workflow, language)
    kind = KIND_OF_CHECK.get(context.check, "text")
    latex = target == "LaTeX" and kind == "equation"  # LaTeX keeps the formula reader's own LaTeX.
    first = read_picture(image, text_height, kind, hints=hints, rotation=rotation, latex=latex)
    if context.check == "copy":
        first = _labelled(first)
    ok, reason = validate_reading(target, workflow, first, language)
    if ok:
        return first, (), True
    if kind in ("equation", "plot"):
        repaired = repair_math(first, lambda text: validate_reading(target, workflow, text, language)[0])
        if repaired:
            return repaired, (f"Read as {first!r}; a look-alike character was corrected.",), True
    # Rejected: read once more in the other recognition mode. The same small model does it, so
    # the second look costs a fraction of a second and no GPU memory is fought over.
    second = ""
    if kind in ("equation", "plot", "table") or not first:
        try:
            second = tidy(_ocr(normalize(image, text_height, glyph=GLYPH, rotation=rotation), "text", (), None), kind)
        except (RuntimeError, ValueError):
            second = ""
    if context.check == "copy" and second:
        second = _labelled(second)
    if second and validate_reading(target, workflow, second, language)[0]:
        return second, ("The first reading was rejected, so the writing was read a second time.",), True
    if not first and second:
        first = second
    return first, ((f"Check this reading: {reason}.",) if first else ()), False


def _labelled(text):
    """A copied item with its kind on the first line, as the copy key expects."""
    from paper_clipboard import classify

    if not text.strip() or re.match(r"\s*KIND\s*:", text, re.IGNORECASE):
        return text
    return f"KIND: {classify(text)}\n{text}"
