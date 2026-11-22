"""Coordinate camera ownership between the RGB preview and PaperWorkbench."""

import json
import os
import time
from pathlib import Path
from threading import Event, Lock, Thread
from uuid import uuid4

MODES = ("keyboard", "touchpad", "eye", "lip", "notes")
ROOT = Path(os.environ.get("LOCALAPPDATA", Path.home())) / "LogiInsightPaper" / "workspace"
MAX_AGE = 5


def _read(name):
    try:
        with (ROOT / name).open("rb") as source:
            data = source.read(2049)
        if len(data) > 2048:
            return None
        value = json.loads(data)
        if isinstance(value, dict):
            return value
    except (OSError, ValueError, RecursionError):
        pass
    return None


def _write(name, value):
    ROOT.mkdir(parents=True, exist_ok=True)
    temporary = ROOT / f"{uuid4().hex}.tmp"
    try:
        temporary.write_text(json.dumps(value), encoding="utf-8")
        # Virus scanning or a reader may hold a Windows handle momentarily.
        for attempt in range(4):
            try:
                temporary.replace(ROOT / name)
                break
            except PermissionError:
                if attempt == 3:
                    raise
                time.sleep(0.015)
    finally:
        temporary.unlink(missing_ok=True)


def workspace_active():
    state = _read("state.json")
    if (
        state
        and isinstance(state.get("session"), str)
        and state.get("mode") in MODES
        and type(state.get("notes_enabled")) is bool
        and type(state.get("updated")) in (int, float)
        and 0 <= time.time() - state["updated"] < MAX_AGE
    ):
        return state
    return None


def request_notes(state):
    """Ask the active workspace to release RGB and enter PaperWorkbench."""
    _write("request.json", {"session": state["session"], "id": uuid4().hex})


def notes_command(paper_session):
    """Read a command addressed to this live PaperWorkbench session, if any."""
    state = workspace_active()
    command = _read("command.json")
    if (
        state
        and command
        and command.get("session") == state["session"]
        and isinstance(command.get("id"), str)
        and command.get("paper_session") in (None, paper_session)
        and command.get("action") in ("start", "stop")
        and (command["action"] == "start") == state["notes_enabled"]
    ):
        return command
    return None


def acknowledge_notes(command, camera_active):
    """Publish readiness only after completing the corresponding camera operation."""
    _write("ack.json", {"session": command["session"], "id": command["id"], "camera_active": bool(camera_active)})


class WorkspaceIpc:
    """Publish mode ownership and serialize access by a single Windows process."""

    def __init__(self):
        self.session = uuid4().hex
        self.mode = MODES[0]
        self.notes_enabled = False
        self._last_request = None
        self._lock = Lock()
        self._stop = Event()

    def __enter__(self):
        import msvcrt

        ROOT.mkdir(parents=True, exist_ok=True)
        self._owner = (ROOT / "owner.lock").open("a+b")
        try:
            self._owner.seek(0)
            msvcrt.locking(self._owner.fileno(), msvcrt.LK_NBLCK, 1)
        except OSError as error:
            self._owner.close()
            raise RuntimeError("A LogiInsight workspace is already running.") from error
        try:
            self._publish()
            self._worker = Thread(target=self._heartbeat, daemon=True)
            self._worker.start()
        except BaseException:
            self._owner.close()
            raise
        return self

    def __exit__(self, *_):
        self._stop.set()
        self._worker.join()
        try:
            for name in ("state.json", "command.json", "request.json", "ack.json"):
                (ROOT / name).unlink(missing_ok=True)
        finally:
            self._owner.close()

    def _publish(self):
        _write("state.json", {"session": self.session, "mode": self.mode,
                              "notes_enabled": self.notes_enabled, "updated": time.time()})

    def _heartbeat(self):
        while not self._stop.wait(0.65):
            with self._lock:
                try:
                    self._publish()
                except OSError:
                    pass

    def set_mode(self, mode, *, notes_enabled=False):
        with self._lock:
            self.mode = mode
            self.notes_enabled = notes_enabled
            self._publish()

    def consume_notes_request(self):
        request = _read("request.json")
        if request and request.get("session") == self.session and isinstance(request.get("id"), str):
            identity = request["id"]
            if identity != self._last_request:
                self._last_request = identity
                return True
        return False

    def command(self, action, paper_session=None, *, device=None):
        command = {"session": self.session, "id": uuid4().hex, "action": action, "paper_session": paper_session}
        if device is not None:
            command["device"] = device
        _write("command.json", command)
        return command

    def acknowledged(self, command, camera_active):
        match = _read("ack.json")
        return bool(match and match.get("session") == self.session and match.get("id") == command["id"]
                    and match.get("camera_active") is camera_active)
