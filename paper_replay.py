"""Record a writing session's frames and hand landmarks, and replay them through the tracker.

    uv run paper_replay.py <debug record folder>            # replay, print the selection, save replay.jpg
    uv run paper_replay.py <folder> --read --app "VS Code" --workflow default

The workbench keeps a ``Recorder`` on each recording. Its debug record (see
paper_workbench.debug_record) then holds ``replay/``: the reference frame, every frame as a
JPEG, and the hand landmarks with their times. Replaying feeds exactly what the camera and the
hand tracker delivered into a new WritingSession, so a change to the tracker can be checked
on real writing without anyone writing again.
"""

import argparse
import json
import queue
import time
from pathlib import Path
from threading import Thread

import cv2
import numpy as np

LIMIT = 600  # Frames kept per recording (about 40 s at the camera's rate).


def _plain(value):
    """Hand samples as JSON: numpy numbers and arrays become plain lists and numbers."""
    if isinstance(value, dict):
        return {key: _plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


class Recorder:
    """Frames and hand samples of one recording, compressed off the UI thread."""

    def __init__(self, baseline, area=None, sheet=None):
        self.baseline = baseline.copy()
        self.area, self.sheet = area, None if sheet is None else np.asarray(sheet, float).tolist()
        self.frames, self.samples = [], []
        self._queue = queue.SimpleQueue()
        self._worker = Thread(target=self._encode, daemon=True)
        self._worker.start()

    def add(self, frame, hand_sample, now):
        if len(self.samples) < LIMIT:
            self.samples.append({"time": now, "hands": _plain(hand_sample) if hand_sample else None})
            self._queue.put(frame)

    def _encode(self):
        while True:
            frame = self._queue.get()
            if frame is None:
                return
            self.frames.append(cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 90])[1].tobytes())

    def save(self, folder):
        """Write replay/ into a debug record. Waits for the frames still being compressed."""
        self._queue.put(None)
        self._worker.join(timeout=10)
        folder = Path(folder) / "replay"
        folder.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(folder / "reference.png"), self.baseline)
        for index, data in enumerate(self.frames):
            (folder / f"{index:04d}.jpg").write_bytes(data)
        count = len(self.frames)
        (folder / "samples.json").write_text(json.dumps({"area": self.area, "sheet": self.sheet,
                                                         "samples": self.samples[:count]}), encoding="utf-8")
        return folder


def replay(folder, session_options=None):
    """Feed a recorded replay/ folder into a new WritingSession. Returns (session, last frame, seconds spent)."""
    from paper_writing import WritingSession

    folder = Path(folder)
    if (folder / "replay").is_dir():
        folder = folder / "replay"
    meta = json.loads((folder / "samples.json").read_text(encoding="utf-8"))
    baseline = cv2.imread(str(folder / "reference.png"))
    area = tuple(meta["area"]) if meta.get("area") else None
    session = WritingSession(baseline, area, sheet=meta.get("sheet"), **(session_options or {}))
    frame, spent = None, 0.0
    for index, sample in enumerate(meta["samples"]):
        frame = cv2.imread(str(folder / f"{index:04d}.jpg"))
        if frame is None:
            break
        start = time.perf_counter()
        session.observe(frame, sample["hands"], sample["time"])
        spent += time.perf_counter() - start
    return session, frame, spent


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("record", type=Path, help="A debug record folder (or its replay/ folder)")
    parser.add_argument("--read", action="store_true", help="Also read the selection with the local models")
    parser.add_argument("--app", default="VS Code")
    parser.add_argument("--workflow", default="default")
    parser.add_argument("--out", type=Path, help="Where to save the overlay (default: the record folder)")
    args = parser.parse_args()
    session, frame, spent = replay(args.record)
    start = time.perf_counter()
    selection = session.select(frame, prefer_recent=True)
    selecting = time.perf_counter() - start
    tips, down = session.summary()
    print(f"{len(json.loads((args.record / 'replay' / 'samples.json').read_text())['samples'])} frames, "
          f"tracking {1000 * spent:.0f} ms, selection {1000 * selecting:.0f} ms; {tips} tip samples, {down} pen-down, "
          f"tip offset {session.tip_offset.tolist()}")
    overlay = session.draw(frame, 1e9)
    if selection is None:
        print("No selection.")
    else:
        x0, y0, x1, y1 = selection.page.bounds
        cv2.rectangle(overlay, (x0, y0), (x1 - 1, y1 - 1), (45, 150, 25), 4)
        print(f"bounds {selection.page.bounds}, {len(selection.strokes)} strokes, text height "
              f"{selection.text_height:.0f}, rotation {selection.rotation}")
        if selection.ink is not None:
            cv2.imwrite(str((args.out or args.record) / "replay_ink.png"), selection.ink)
        if args.read:
            from paper_conversion import read_recording

            start = time.perf_counter()
            reading = read_recording(selection, args.app, "Python", args.workflow, "qwen3-vl:2b-instruct")
            print(f"read {reading.source!r} from {reading.path} in {time.perf_counter() - start:.2f} s")
    cv2.imwrite(str((args.out or args.record) / "replay.jpg"), overlay)


if __name__ == "__main__":
    main()
