"""Structured text and image inference through the local Ollama service."""

import base64
import json
import os
import subprocess
import time
from pathlib import Path
from threading import Lock
from urllib.error import HTTPError
from urllib.request import ProxyHandler, Request, build_opener

MODEL = "qwen3-vl:2b-instruct"
OLLAMA_CHAT = "http://127.0.0.1:11434/api/chat"
OLLAMA_VERSION = "http://127.0.0.1:11434/api/version"
START_LOCK = Lock()


def _online(opener):
    try:
        with opener.open(OLLAMA_VERSION, timeout=0.8) as response:
            return bool(json.load(response).get("version"))
    except (OSError, ValueError, AttributeError):
        return False


def _start_portable(opener):
    """Start the verified per-user portable installation when the API is down."""
    installation = os.environ.get("LOCALAPPDATA")
    if not installation:
        return
    tools = Path(installation) / "LogiInsightTools"
    binary = tools / "ollama.exe"
    if not binary.is_file() or not (tools / "models").is_dir():
        return
    with START_LOCK:
        if _online(opener):
            return
        environment = {**os.environ, "OLLAMA_HOST": "127.0.0.1:11434",
                       "OLLAMA_MODELS": str(tools / "models"), "OLLAMA_NO_CLOUD": "1"}
        worker = subprocess.Popen([str(binary), "serve"], env=environment, cwd=tools,
                                  stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                                  creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
        for _ in range(24):
            if _online(opener) or worker.poll() is not None:
                return
            time.sleep(0.25)


class ModelOutputLimit(ValueError):
    """The model stopped at its output budget before completing the result."""


def chat(model, system, content, schema, *, image=None, evidence=None, tokens=1800, timeout=100):
    """One structured reply from the local model.

    A small model sometimes repeats itself until it hits the output budget, which leaves
    the JSON unfinished. That reply is retried once with a larger budget, a stronger
    repetition penalty, and a request to be brief; only a second cut-off is an error.
    """
    try:
        return _chat(model, system, content, schema, image, evidence, tokens, timeout, 1.1)
    except ModelOutputLimit:
        brief = f"{content} Answer briefly and do not repeat yourself."
        return _chat(model, system, brief, schema, image, evidence, 2 * tokens, timeout, 1.3)


SMALL_CONTEXT, LARGE_CONTEXT = 2560, 4096


def context_window(system, content, tokens, image=None, evidence=None):
    """The small window whenever the request fits. With it the reader and this model share a 4 GB
    GPU without pushing each other into system memory; only long generations (projects, circuits
    with a picture) need the large one, at the price of a reload."""
    prompt = (len(system) + len(content)) // 3 + 1150 * ((image is not None) + (evidence is not None))
    return SMALL_CONTEXT if prompt + tokens + 60 <= SMALL_CONTEXT else LARGE_CONTEXT


def make_room(model, context, opener=None):
    """Unload other models that cannot share the GPU with this request.

    Two small-context models fit together on 4 GB. A large context does not fit beside
    another model: left alone, both spill into system memory and a reply takes minutes.
    So a large-context request unloads the others first, and any request unloads a
    model that is still loaded with a large context.
    """
    opener = opener or build_opener(ProxyHandler({}))
    try:
        with opener.open(OLLAMA_CHAT.replace("/api/chat", "/api/ps"), timeout=3) as response:
            loaded = json.load(response).get("models", ())
        for other in loaded:
            name = other.get("name")
            if name and name != model and (context > SMALL_CONTEXT or other.get("context_length", 0) > SMALL_CONTEXT):
                body = json.dumps({"model": name, "keep_alive": 0}).encode()
                request = Request(OLLAMA_CHAT.replace("/api/chat", "/api/generate"), body,
                                  {"Content-Type": "application/json"})
                with opener.open(request, timeout=20) as response:
                    response.read()
    except (OSError, ValueError, AttributeError):
        pass


def warm_up(model=MODEL):
    """Load the model into memory so the first reading does not wait for it. Errors are ignored."""
    try:
        opener = build_opener(ProxyHandler({}))
        if not _online(opener):
            _start_portable(opener)
        body = json.dumps({"model": model, "keep_alive": "30m", "stream": False,
                           "messages": [{"role": "user", "content": "ok"}],
                           "options": {"num_predict": 1, "num_ctx": SMALL_CONTEXT}}).encode()
        with opener.open(Request(OLLAMA_CHAT, body, {"Content-Type": "application/json"}), timeout=60) as response:
            response.read()
    except (OSError, ValueError):
        pass


def _chat(model, system, content, schema, image, evidence, tokens, timeout, repeat_penalty):
    if model.lower().endswith(("-cloud", ":cloud")):
        raise ValueError("Use a model installed locally in Ollama.")
    prompt = {"role": "user", "content": content}
    if image is not None:
        prompt["images"] = [base64.b64encode(image).decode("ascii")]
        if evidence is not None:
            prompt["images"].append(base64.b64encode(evidence).decode("ascii"))
    window = context_window(system, content, tokens, image, evidence)
    body = json.dumps({
        "model": model, "stream": False, "format": schema, "think": False, "keep_alive": "30m",
        "messages": [{"role": "system", "content": system}, prompt],
        "options": {"temperature": 0, "num_predict": tokens, "repeat_penalty": repeat_penalty,
                    "num_ctx": window},
    }).encode()
    request = Request(OLLAMA_CHAT, body, {"Content-Type": "application/json"})
    opener = build_opener(ProxyHandler({}))
    try:
        if not _online(opener):
            _start_portable(opener)
        # Reading a picture needs extra GPU memory for a moment, so it gets the GPU to itself.
        make_room(model, LARGE_CONTEXT if image is not None else window, opener)
        try:
            with opener.open(request, timeout=timeout) as response:
                result = json.load(response)
        except HTTPError as error:
            if error.code < 500:
                raise
            # A server error right after a model swap on a small GPU: free the GPU and try once more.
            make_room(model, LARGE_CONTEXT, opener)
            time.sleep(0.5)
            with opener.open(request, timeout=timeout) as response:
                result = json.load(response)
        try:
            parsed = json.loads(result["message"]["content"])
        except json.JSONDecodeError:
            if result.get("done_reason") == "length":
                raise ModelOutputLimit("The model's reply was cut off. Select just the handwritten equation.") from None
            raise
    except OSError as error:
        raise RuntimeError(f"Local model {model} is unavailable. Check Ollama and the installed model.") from error
    except (KeyError, TypeError, json.JSONDecodeError) as error:
        raise ValueError("The local model returned an unreadable response.") from error
    if not isinstance(parsed, dict):
        raise ValueError("The local model did not return an object.")
    return parsed
