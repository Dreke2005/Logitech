"""Local command and display exchange with the Logi Actions plugin."""

import json
import os
import time
from pathlib import Path
from threading import Event, Lock, Thread
from uuid import uuid4

ROOT = Path(os.environ.get("LOCALAPPDATA", Path.home())) / "LogiInsightPaper"
STATE = ROOT / "state.json"
COMMANDS = ROOT / "commands"
MENUS = json.loads(Path(__file__).with_name("paper_menus.json").read_text(encoding="utf-8"))


def _write_json(path, value):
    temporary = path.with_name(f"{uuid4().hex}.tmp")
    try:
        temporary.write_text(json.dumps(value), encoding="utf-8")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def current_state():
    try:
        value = json.loads(STATE.read_text(encoding="utf-8"))
        if (0 <= time.time() * 1000 - value["updated"] < 6000
                and isinstance(value["session"], str) and isinstance(value["revision"], int)):
            return value
    except (OSError, ValueError, KeyError, TypeError):
        pass
    return None


def send_slot(state, slot):
    COMMANDS.mkdir(parents=True, exist_ok=True)
    name = f"{time.time_ns()}-{uuid4().hex}.json"
    _write_json(COMMANDS / name, {"session": state["session"], "revision": state["revision"], "slot": slot})


def send_route(state, route):
    COMMANDS.mkdir(parents=True, exist_ok=True)
    _write_json(COMMANDS / f"{time.time_ns()}-{uuid4().hex}.json",
                {"session": state["session"], "menu": "home", "action": route})


class KeypadSession:
    def __init__(self):
        self.id = uuid4().hex
        self.revision = 0
        self._state = None
        self._state_lock = Lock()
        self._stop = Event()
        self._worker = None
        COMMANDS.mkdir(parents=True, exist_ok=True)

    def publish(self, labels, menu="home", camera_active=False, active=None, status=None, active_menu=None):
        """``active`` is the key the workbench is working on and ``status`` its state
        (recording, busy, done, error); the keypad draws it as a coloured bar on that key."""
        with self._state_lock:
            self._state = {"session": self.id, "revision": self.revision, "labels": labels,
                           "menu": menu, "cameraActive": camera_active, "active": active, "status": status,
                           "activeMenu": active_menu}
            self._refresh()

    def _refresh(self):
        if self._state is not None:
            _write_json(STATE, {**self._state, "updated": int(time.time() * 1000)})

    def start_heartbeat(self):
        self._worker = Thread(target=self._heartbeat, daemon=True)
        self._worker.start()

    def _heartbeat(self):
        while not self._stop.wait(1.5):
            with self._state_lock:
                try:
                    self._refresh()
                except OSError:
                    pass

    def presses(self):
        for path in sorted(COMMANDS.glob("*.json"))[:32]:
            try:
                message = json.loads(path.read_text(encoding="utf-8")) if path.stat().st_size < 512 else {}
                menu, action = message.get("menu"), message.get("action")
                if isinstance(menu, str) and any(action == button[0] for button in MENUS.get(menu, ())):
                    if message.get("session") == self.id:
                        yield menu, action
                    continue
                slot = message.get("slot")
                if (message.get("session") == self.id and message.get("revision") == self.revision
                        and type(slot) is int and 0 <= slot < 9):
                    yield slot
            except (OSError, ValueError, AttributeError):
                pass
            finally:
                try:
                    path.unlink(missing_ok=True)
                except OSError:
                    pass

    def close(self):
        self._stop.set()
        if self._worker is not None:
            self._worker.join(timeout=1)
        latest = current_state()
        if latest and latest["session"] == self.id:
            STATE.unlink(missing_ok=True)
