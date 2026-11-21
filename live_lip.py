"""Asynchronous visual speech recognition for the RGB preview."""

import hmac
import json
import math
import secrets
import socket
import subprocess
import time
from pathlib import Path
from queue import Empty, SimpleQueue
from threading import Event, Thread

import cv2

from silent_auto_avsr import FRAME_HEADER, MAX_EDGE, MAX_LIVE_FRAMES, MIN_FRAMES, TARGET_FPS
from silent_transcription import backend_paths


class LiveLipReader:
    """Keep a local model loaded and send recorded phrases as raw frames in memory."""

    def __init__(self, repo, python, checkpoint, device="cpu"):
        self.phase = "loading"
        self.status = "Loading the visual speech model..."
        self.frame_count = 0
        self._frames = []
        self._first_time = None
        self._commands = SimpleQueue()
        self._events = SimpleQueue()
        self._closed = Event()
        self._connection = None
        self._process = None
        self._thread = Thread(
            target=self._run, args=(Path(repo), Path(python), Path(checkpoint), device),
            daemon=True, name="visual-speech-worker",
        )
        self._thread.start()

    def start(self):
        if self.phase != "ready":
            return False
        self.phase = "recording"
        self.status = "Mouth a phrase. Press Space to finish."
        self.frame_count = 0
        self._first_time = None
        return True

    def add(self, frame_bgr, timestamp):
        if self.phase != "recording" or not math.isfinite(timestamp):
            return
        if self._first_time is None:
            self._first_time = timestamp
        index = math.floor((timestamp - self._first_time) * TARGET_FPS + 0.5)
        if index < self.frame_count:
            return
        if index >= MAX_LIVE_FRAMES:
            self.stop()
            return

        height, width = frame_bgr.shape[:2]
        if self._frames:
            new_height, new_width = self._frames[0].shape[:2]
            # Repeat the preceding image if the camera delivered fewer than 25 FPS.
            self._frames.extend([self._frames[-1]] * (index - self.frame_count))
        else:
            scale = min(1, MAX_EDGE / max(height, width))
            new_width, new_height = max(1, round(width * scale)), max(1, round(height * scale))
        if (width, height) == (new_width, new_height):
            frame = frame_bgr.copy()
        else:
            frame = cv2.resize(frame_bgr, (new_width, new_height), interpolation=cv2.INTER_AREA)
        self._frames.append(frame)
        self.frame_count = len(self._frames)
        if self.frame_count == MAX_LIVE_FRAMES:
            self.stop()

    def stop(self):
        if self.phase != "recording":
            return False
        frames, self._frames = self._frames, []
        if len(frames) < MIN_FRAMES:
            self.phase = "ready"
            self.status = "Phrase too short. Record at least half a second."
            return False
        self.phase = "transcribing"
        self.status = "Transcribing the phrase..."
        self._commands.put(frames)
        return True

    def poll(self):
        """Return the latest draft when one is ready. Call on the preview thread."""
        if self.phase == "closed":
            return None
        try:
            event, text = self._events.get_nowait()
        except Empty:
            return None
        if event == "error":
            self.phase = "error"
            self.status = f"Visual speech error: {text}"
        else:
            self.phase = "ready"
            self.status = text if event == "retry" else "Press Space to start a phrase."
        return text if event == "transcript" else None

    def close(self):
        if self._closed.is_set():
            return
        self.phase = "closed"
        self.status = "Visual speech closed."
        self._frames.clear()
        self._closed.set()
        try:
            self._commands.get_nowait()
        except Empty:
            pass
        self._commands.put(None)
        if self._connection is not None:
            try:
                self._connection.shutdown(socket.SHUT_RDWR)
            except OSError:
                pass
        self._thread.join(timeout=1)

    def _run(self, repo, python, checkpoint, device):
        try:
            repo, python, checkpoint = backend_paths(repo, python, checkpoint)
            worker = Path(__file__).with_name("silent_auto_avsr.py").resolve()
            token = secrets.token_bytes(32)
            with socket.socket() as listener:
                listener.bind(("127.0.0.1", 0))
                listener.listen(1)
                listener.settimeout(0.2)
                port = listener.getsockname()[1]
                if self._closed.is_set():
                    return
                self._process = subprocess.Popen(
                    [str(python), "-I", str(worker), "--serve", str(repo), str(checkpoint), device,
                     str(port), token.hex()], cwd=repo,
                    stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                    creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
                )
                connection = self._accept(listener, token)
            if connection is not None:
                self._exchange_frames(connection)
        except (OSError, EOFError, ValueError, RuntimeError) as error:
            if not self._closed.is_set():
                self._events.put(("error", str(error) or type(error).__name__))
        finally:
            if self._process is not None:
                if self._process.poll() is None:
                    self._process.terminate()
                self._process.wait()

    def _accept(self, listener, token):
        deadline = time.monotonic() + 30
        while not self._closed.is_set():
            try:
                connection, _ = listener.accept()
            except socket.timeout:
                if self._process.poll() is not None:
                    raise RuntimeError(f"Auto-AVSR exited during startup ({self._process.returncode}).") from None
                if time.monotonic() > deadline:
                    raise RuntimeError("Auto-AVSR could not start its local connection.") from None
                continue
            connection.settimeout(1)
            try:
                supplied = bytearray()
                while len(supplied) < len(token):
                    part = connection.recv(len(token) - len(supplied))
                    if not part:
                        break
                    supplied.extend(part)
                if hmac.compare_digest(supplied, token):
                    return connection
            except OSError:
                pass
            connection.close()
        return None

    def _exchange_frames(self, connection):
        self._connection = connection
        with connection, connection.makefile("rb") as stream:
            connection.settimeout(120)
            if self._closed.is_set() or self._read_event(stream) != "ready":
                return
            while not self._closed.is_set():
                frames = self._commands.get()
                if frames is None or self._closed.is_set():
                    return
                height, width = frames[0].shape[:2]
                connection.sendall(FRAME_HEADER.pack(width, height, len(frames)))
                for frame in frames:
                    connection.sendall(memoryview(frame).cast("B"))
                del frame, frames
                if self._read_event(stream) == "error":
                    return

    def _read_event(self, stream):
        line = stream.readline(8193)
        if not line or len(line) > 8192:
            raise EOFError("Auto-AVSR stopped responding.")
        reply = json.loads(line)
        if not isinstance(reply, dict):
            raise ValueError("Invalid response from Auto-AVSR.")
        event, text = reply.get("event"), reply.get("text")
        if event not in {"ready", "error", "retry", "transcript"} or not isinstance(text, str):
            raise ValueError("Invalid response from Auto-AVSR.")
        self._events.put((event, text))
        return event
