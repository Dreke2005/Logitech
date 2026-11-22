"""Reading and generation in the cloud, through the Claude Code CLI that is signed in on this computer.

``chat`` has the interface of ``local_model.chat`` and is what the workbench calls: with the
setting on "cloud" the request goes to Claude first, and to the local Ollama model when Claude is
not installed, offline, slow (every call has a timeout), or returns something unusable. ``MODE``
is the setting ("cloud" or "local", from settings.json or PAPER_GENERATION).

What leaves the computer is exactly what the caller passes in: the instruction for the pressed key,
the transcription (with the names a key already uses as hints), and the picture of the writing.
Nothing else is added here: no file paths, no project contents, no user or machine names. The CLI
runs with every tool, MCP server, settings file, and skill turned off, without saving a session,
and from a neutral folder (the public profile), because it reports its working directory, the
platform, and the Windows version to the model. It also attaches the e-mail address of the Claude
account it is signed in with; that goes to the service the account belongs to and cannot be
turned off without an API key.
"""

import base64
import contextlib
import json
import os
import re
import shutil
import subprocess
import threading
import time
from pathlib import Path

import local_model

MODE = "local" if os.environ.get("PAPER_GENERATION", "").lower() == "local" else "cloud"
FAST = os.environ.get("PAPER_CLOUD_MODEL", "claude-haiku-5-5")  # Reading and short code: about two seconds.
STRONG = os.environ.get("PAPER_CLOUD_MODEL_LARGE", "claude-sonnet-5-5")  # Projects and circuit drawings.
READER = os.environ.get("PAPER_CLOUD_READER", STRONG)  # Handwriting: as fast as FAST on small crops, and it misreads less.
RETRY_AFTER = 60  # Seconds the cloud is skipped after it failed, so being offline costs one timeout, not one per call.
_state = {"down_until": 0.0, "reason": None, "binary": None}
_trail = threading.local()


class CloudUnavailable(RuntimeError):
    """Claude could not be reached or did not answer in time; the caller falls back to the local model."""


def binary():
    """The Claude Code CLI, or None when it is not installed."""
    if _state["binary"] is None:
        found = shutil.which("claude")
        candidates = [Path(found)] if found else []
        candidates += [Path.home() / ".local/bin/claude.exe", Path.home() / ".local/bin/claude"]
        _state["binary"] = next((str(path) for path in candidates if path.is_file()), "")
    return _state["binary"] or None


def enabled():
    """True when the next request on this thread should try the cloud."""
    return (MODE == "cloud" and not getattr(_trail, "local", False) and time.monotonic() >= _state["down_until"]
            and binary() is not None)


@contextlib.contextmanager
def local_only():
    """Keep this thread's requests on the local models: the readings shown while the user still writes
    happen at every pause of the pen and must not each become a cloud request."""
    previous, _trail.local = getattr(_trail, "local", False), True
    try:
        yield
    finally:
        _trail.local = previous


def status():
    """One line for the status bar: which engine the next request uses, and why."""
    if MODE != "cloud":
        return "local models (setting)"
    if binary() is None:
        return "local models (the Claude CLI is not installed)"
    if time.monotonic() < _state["down_until"]:
        return f"local models for now ({_state['reason']})"
    return "cloud (Claude)"


def _neutral_folder():
    """A working directory whose path names neither the user nor a project: the CLI reports it to the model."""
    for name in ("PUBLIC", "ProgramData", "SystemRoot"):
        folder = os.environ.get(name)
        if folder and Path(folder).is_dir():
            return folder
    return os.path.abspath(os.sep)


def note(engine, seconds, what="generation"):
    """Record which engine produced a step of the result in progress (per worker thread)."""
    if not hasattr(_trail, "steps"):
        _trail.steps = []
    _trail.steps.append((what, engine, seconds))


def take_trail():
    """The steps recorded on this thread since the last call: [(what, engine, seconds)]."""
    steps, _trail.steps = getattr(_trail, "steps", []), []
    return steps


def describe(steps):
    """'read in the cloud 2.1 s, generated locally 8.0 s' for the status line and the notes."""
    parts = []
    combined = [seconds for kind, _, seconds in steps if kind == "orchestrated"]
    if combined:  # One request that read the capture and wrote the result (see paper_orchestrator).
        parts.append(f"read and generated in one cloud request (Claude) in {sum(combined):.1f} s")
    for what in ("reading", "generation"):
        chosen = [(engine, seconds) for kind, engine, seconds in steps if kind == what]
        if not chosen:
            continue
        engines = list(dict.fromkeys(engine for engine, _ in chosen))
        where = "in the cloud (Claude)" if engines == ["cloud"] else "by the local model" if engines == ["local"] \
            else "in the cloud and by the local model"
        parts.append(f"{'read' if what == 'reading' else 'generated'} {where} in "
                     f"{sum(seconds for _, seconds in chosen):.1f} s")
    return ", ".join(parts)


def _ask_claude(system, content, images=(), *, model=None, timeout=30):
    """Claude's plain-text answer. ``images`` are encoded pictures (PNG or JPEG bytes).

    Only ``system``, ``content``, and ``images`` are sent (see the module text). Raises
    CloudUnavailable when the CLI is missing, fails, or exceeds ``timeout`` seconds."""
    program = binary()
    if program is None:
        raise CloudUnavailable("The Claude CLI is not installed.")
    blocks = [{"type": "image", "source": {
        "type": "base64", "media_type": "image/png" if image[:4] == b"\x89PNG" else "image/jpeg",
        "data": base64.b64encode(image).decode("ascii")}} for image in images if image]
    blocks.append({"type": "text", "text": content})
    message = json.dumps({"type": "user", "message": {"role": "user", "content": blocks}}) + "\n"
    command = [program, "-p", "--model", model or FAST, "--effort", "low", "--tools", "", "--strict-mcp-config",
               "--disable-slash-commands", "--no-session-persistence", "--setting-sources", "",
               "--system-prompt", system, "--input-format", "stream-json", "--output-format", "stream-json",
               "--verbose"]
    try:
        done = subprocess.run(command, input=message, capture_output=True, text=True, encoding="utf-8",
                              errors="replace", timeout=timeout, cwd=_neutral_folder(),
                              creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
    except subprocess.TimeoutExpired:
        raise _unavailable(f"Claude did not answer within {timeout:g} s") from None
    except OSError as error:
        raise _unavailable(f"the Claude CLI could not be started: {error}") from error
    result = None
    for line in done.stdout.splitlines():
        if line.startswith("{") and '"result"' in line:
            try:
                event = json.loads(line)
            except ValueError:
                continue
            if isinstance(event, dict) and event.get("type") == "result":
                result = event
    if result is None or result.get("is_error") or not isinstance(result.get("result"), str):
        detail = (result or {}).get("result") or done.stderr.strip()[-200:] or f"exit code {done.returncode}"
        raise _unavailable(f"Claude returned an error: {str(detail)[:200]}")
    return result["result"].strip()


ask = _ask_claude  # The name callers use. Unit tests replace it, so no test reaches the cloud.


def _unavailable(reason):
    _state["down_until"], _state["reason"] = time.monotonic() + RETRY_AFTER, reason
    return CloudUnavailable(reason[0].upper() + reason[1:] + ".")


def reset():
    """Try the cloud again at once (after the setting changed, or in tests)."""
    _state["down_until"], _state["reason"] = 0.0, None


def strip_fence(text):
    fence = re.fullmatch(r"\s*\x60{3}[^\n]*\n(.*?)\n?\x60{3}\s*", text, re.S)
    return fence[1] if fence else text


def _object(text, schema):
    """The JSON object in Claude's answer, with the schema's required keys."""
    text = strip_fence(text).strip()
    if not text.startswith("{"):
        start, end = text.find("{"), text.rfind("}")
        text = text[start:end + 1] if 0 <= start < end else text
    try:
        parsed = json.loads(text)
    except ValueError as error:
        raise ValueError("Claude did not return a JSON object.") from error
    missing = [key for key in schema.get("required", ()) if key not in parsed] if isinstance(parsed, dict) else None
    if missing is None or missing:
        raise ValueError("Claude's answer is missing fields.")
    return parsed


def chat_cloud(system, content, schema, *, image=None, evidence=None, model=None, timeout=30):
    """One structured reply from Claude, as a dict that follows ``schema``."""
    request = (f"{content}\n\nAnswer with one JSON object and nothing else, following this JSON schema:\n"
               f"{json.dumps(schema, separators=(',', ':'))}")
    return _object(ask(system, request, [image, evidence], model=model, timeout=timeout), schema)


def chat(model, system, content, schema, *, image=None, evidence=None, tokens=1800, timeout=100, what=None,
         large=False):
    """``local_model.chat`` with the cloud in front of it.

    With the setting on cloud, Claude answers; the local model ``model`` answers when the cloud is
    off, unavailable, too slow, or unreadable. ``large`` picks the stronger cloud model (projects,
    circuits). Which one answered is recorded for the status line (see ``take_trail``)."""
    what = what or ("reading" if image is not None else "generation")
    if enabled():
        start = time.perf_counter()
        try:
            reply = chat_cloud(system, content, schema, image=image, evidence=evidence,
                               model=STRONG if large else READER if image is not None else FAST,
                               timeout=min(timeout, 90 if large else 40))
            note("cloud", time.perf_counter() - start, what)
            return reply
        except (CloudUnavailable, ValueError):
            pass
    start = time.perf_counter()
    reply = local_model.chat(model, system, content, schema, image=image, evidence=evidence, tokens=tokens,
                             timeout=timeout)
    note("local", time.perf_counter() - start, what)
    return reply
