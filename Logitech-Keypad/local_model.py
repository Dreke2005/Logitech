"""Structured text and image inference through the local Ollama service."""

import base64
import json
import os
import subprocess
import time
from pathlib import Path
from threading import Lock
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
    if model.lower().endswith(("-cloud", ":cloud")):
        raise ValueError("Use a model installed locally in Ollama.")
    prompt = {"role": "user", "content": content}
    if image is not None:
        prompt["images"] = [base64.b64encode(image).decode("ascii")]
        if evidence is not None:
            prompt["images"].append(base64.b64encode(evidence).decode("ascii"))
    body = json.dumps({
        "model": model, "stream": False, "format": schema, "think": False,
        "messages": [{"role": "system", "content": system}, prompt],
        "options": {"temperature": 0, "num_predict": tokens},
    }).encode()
    request = Request(OLLAMA_CHAT, body, {"Content-Type": "application/json"})
    opener = build_opener(ProxyHandler({}))
    try:
        if not _online(opener):
            _start_portable(opener)
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
