"""Check that a camera can drive PaperBridge, using the same capture path as the workbench.

    uv run paper_camera_check.py            # the camera the workbench would pick
    uv run paper_camera_check.py --camera 0 --seconds 15 --save check.png

It runs the workbench's capture worker (frames plus hand landmarks) and a WritingSession
for a few seconds and reports what the workbench depends on: resolution, frame rate,
whether a hand and a pen grip are seen, whether the paper is found, how many pen-tip
samples were placed, and how many of them counted as pen-down. Hold a pen over a sheet
of paper and write a few letters while it runs.
"""

import argparse
import multiprocessing as mp
import queue
import time
from pathlib import Path

import cv2

from fingertip_tracker import DEFAULT_MODEL
from paper_pen import pen_grip
from paper_writing import WritingSession, paper_bounds
from rgb_preview import _capture


def choose_camera(names, requested=None):
    """(index, is LogiInsight) of the camera to use: the LogiInsight RGB stream, else the first webcam."""
    if requested is not None:
        return (requested, "lvnext" in names[requested].lower()) if 0 <= requested < len(names) else (None, False)
    skip = ("cpd", "hello", " ir", "infrared", "virtual", "obs")
    for wanted in (True, False):
        for index, name in enumerate(names):
            lower = name.lower()
            if ("lvnext" in lower) == wanted and not any(part in lower for part in skip):
                return index, wanted
    return None, False


def main():
    from pygrabber.dshow_graph import FilterGraph

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--camera", type=int, help="DirectShow camera index")
    parser.add_argument("--seconds", type=float, default=12)
    parser.add_argument("--save", type=Path, help="Save the last frame with the tracking overlay")
    args = parser.parse_args()
    names = FilterGraph().get_input_devices()
    device, logitech = choose_camera(names, args.camera)
    if device is None:
        print(f"No usable camera. Found: {names}")
        return 1
    print(f"Camera {device}: {names[device]}" + ("" if logitech else "  (not the LogiInsight camera)"))
    ctx = mp.get_context("spawn")
    output, stop = ctx.Queue(1), ctx.Event()
    hand_model = DEFAULT_MODEL if DEFAULT_MODEL.is_file() else None
    process = ctx.Process(target=_capture, args=(device, "DSHOW", 1920, 1080, output, stop, hand_model), daemon=True)
    process.start()
    session, frames, with_hand, with_grip, first, frame = None, 0, 0, 0, None, None
    started = time.monotonic()
    try:
        while time.monotonic() - started < args.seconds + (first is None) * 20:
            try:
                kind, sample = output.get(timeout=0.5)
            except queue.Empty:
                if not process.is_alive():
                    print("The capture worker stopped.")
                    return 1
                continue
            if kind == "error":
                print(f"Camera error: {sample}")
                return 1
            frame, hands, _ = sample
            now = time.monotonic()
            if first is None:
                first, started = now, now
                area = paper_bounds(frame)
                session = WritingSession(frame, area)
                print(f"Resolution {frame.shape[1]}x{frame.shape[0]}; paper area {area or 'not found (whole frame is used)'}")
                print(f"Recording for {args.seconds:g} s: write a few letters now...")
            frames += 1
            found = hands["hands"] if hands else ()
            with_hand += bool(found)
            with_grip += any(pen_grip(hand) is not None for hand in found)
            session.observe(frame, hands, now)
    finally:
        stop.set()
        process.join(timeout=1)
        if process.is_alive():
            process.terminate()
    if not frames:
        print("No frames arrived.")
        return 1
    elapsed = max(time.monotonic() - first, 1e-3)
    tips = [sample for samples in session.tips.values() for sample in samples]
    down = sum(1 for sample in tips if sample.down)
    selection = session.select(frame, prefer_recent=True)
    print(f"{frames} frames, {frames / elapsed:.1f} per second")
    print(f"Hand seen in {with_hand} frames, pen grip in {with_grip}")
    print(f"Pen-tip samples {len(tips)}, counted as pen-down {down}")
    if selection is None:
        print("No writing region yet: no new ink was seen near the pen.")
    else:
        x0, y0, x1, y1 = selection.page.bounds
        print(f"Writing region {x1 - x0}x{y1 - y0} px at ({x0},{y0}); {len(selection.strokes)} strokes; "
              f"letter height about {selection.text_height:.0f} px; rotation {selection.rotation}")
    if args.save:
        overlay = session.draw(frame, time.monotonic())
        if selection is not None:
            cv2.rectangle(overlay, (x0, y0), (x1 - 1, y1 - 1), (45, 150, 25), 3)
        cv2.imwrite(str(args.save), overlay)
        print(f"Saved {args.save}")
    return 0


if __name__ == "__main__":
    mp.freeze_support()
    raise SystemExit(main())
