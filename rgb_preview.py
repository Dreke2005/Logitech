"""RGB preview with fingertip tracking. No SSH connection or credentials required."""

import argparse
import json
import multiprocessing as mp
import queue
import sys
import time
from pathlib import Path

import cv2


def _capture(device, backend, width, height, output, stop, model_path=None, mirror=False,
             eye_model_path=None):
    """Isolate native camera calls so a blocked driver can be stopped."""
    from camera_utils import open_camera

    cap = None
    tracker = None
    try:
        if eye_model_path is not None:
            from eye_pointer import EyeLandmarker

            tracker = EyeLandmarker(eye_model_path)
        elif model_path is not None:
            from fingertip_tracker import FingertipTracker

            tracker = FingertipTracker(model_path)
        cap, _ = open_camera(
            device, backend=getattr(cv2, "CAP_" + backend),
            width=width, height=height, fourcc="MJPG", fps=30,
        )
        while not stop.is_set():
            ok, frame = cap.read()
            if not ok or frame is None:
                stop.wait(0.05)
                continue
            if mirror:
                frame = cv2.flip(frame, 1)
            tracking = tracker.process(frame) if tracker is not None else None
            try:
                output.put_nowait(("frame", (frame, tracking)))
            except queue.Full:
                pass  # Bound buffering instead of building a video backlog.
    except Exception as exc:
        try:
            output.put(("error", str(exc)), timeout=0.2)
        except queue.Full:
            pass
    finally:
        if cap is not None:
            cap.release()
        if tracker is not None:
            tracker.close()


def _desktop_banner(frame, status):
    top, width = frame.shape[:2]
    frame = cv2.copyMakeBorder(frame, 0, 60, 0, 0, cv2.BORDER_CONSTANT, value=(25, 30, 41))
    font = cv2.FONT_HERSHEY_SIMPLEX
    for text, baseline in ((status, top + 24), ("Ctrl+Alt+F8   Pause / resume", top + 47)):
        text_width = max(1, cv2.getTextSize(text, font, 1, 1)[0][0])
        scale = min(0.57, max(1, width - 24) / text_width)
        cv2.putText(frame, text, (12, baseline), font, scale, (243, 243, 248), 1, cv2.LINE_AA)
    return frame


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--list", action="store_true", help="List DirectShow cameras")
    parser.add_argument("--device", type=int, help="Camera index; default: LVNext RGB camera")
    parser.add_argument("--backend", choices=("DSHOW", "MSMF"), default="DSHOW")
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--timeout", type=float, default=10, help="Seconds without a frame before stopping")
    parser.add_argument("--frames", type=int, default=0, help="Stop after N frames; 0 means unlimited")
    parser.add_argument("--no-display", action="store_true", help="Test capture without opening a window")
    parser.add_argument("--no-hands", action="store_true", help="Plain RGB video without hand inference")
    parser.add_argument("--no-keyboard", action="store_true", help="Show fingertip tracking without the keyboard")
    parser.add_argument("--mode", choices=("keyboard", "touchpad", "eye"), default="keyboard",
                        help="Choose the typing sandbox or an opt-in Windows desktop input mode")
    parser.add_argument("--mirror", action="store_true", help="Mirror video before tracking; coordinates follow the display")
    parser.add_argument("--model", type=Path, help="Local hand_landmarker.task; default downloads Google's model once")
    parser.add_argument("--eye-model", type=Path, help="Local face_landmarker.task for --mode eye")
    parser.add_argument("--landmarks-jsonl", type=Path, help="Optional JSONL output for keyboard integration")
    args = parser.parse_args()
    if min(args.width, args.height, args.timeout) <= 0 or args.frames < 0:
        parser.error("Dimensions and timeout must be positive; frames must be nonnegative.")
    if args.mode != "keyboard" and args.no_display:
        parser.error("Desktop input modes require the preview window.")
    if args.mode == "touchpad" and args.no_hands:
        parser.error("Touchpad mode requires hand tracking.")
    if args.mode == "eye" and args.landmarks_jsonl:
        parser.error("--landmarks-jsonl records hand landmarks. Use keyboard or touchpad mode.")

    from pygrabber.dshow_graph import FilterGraph

    cameras = FilterGraph().get_input_devices()
    for index, name in enumerate(cameras):
        print(f"  {index}: {name}", flush=True)
    if args.list:
        return 0
    if args.device is None:
        # DirectShow indexes belong to its named enumeration. MSMF ordering
        # need not match, so require an explicit index for that backend.
        if args.backend == "MSMF":
            parser.error("For MSMF, supply --device INDEX; indexes may differ from DirectShow.")
        args.device = next((i for i, name in enumerate(cameras)
                            if "lvnext" in name.lower()
                            and "cpd" not in name.lower()
                            and "hello" not in name.lower()), None)
        if args.device is None:
            print("No LVNext RGB camera detected. Reconnect it, or use --device 0 for the laptop camera.")
            return 2
    if args.device < 0:
        parser.error("Camera index must be nonnegative.")

    model_path = None
    eye_model_path = None
    if args.mode == "eye":
        from eye_pointer import ensure_eye_model

        eye_model_path = args.eye_model.resolve() if args.eye_model else ensure_eye_model()
        if not eye_model_path.is_file():
            parser.error(f"Eye model not found: {eye_model_path}")
    elif not args.no_hands:
        from fingertip_tracker import ensure_model

        model_path = args.model.resolve() if args.model else ensure_model()
        if not model_path.is_file():
            parser.error(f"Hand model not found: {model_path}")
    if args.no_hands and args.landmarks_jsonl:
        parser.error("--landmarks-jsonl requires hand tracking; remove --no-hands.")
    print(f"Opening camera {args.device} via {args.backend}; press q or Escape to quit.", flush=True)
    ctx = mp.get_context("spawn")
    output, stop = ctx.Queue(maxsize=1), ctx.Event()
    worker = ctx.Process(target=_capture, args=(args.device, args.backend, args.width,
                                               args.height, output, stop, model_path, args.mirror,
                                               eye_model_path), daemon=True)
    keyboard = None
    desktop = controller = pause_controller = None
    desktop_enabled = shortcut_held = False
    status = "Paused. Press Ctrl+Alt+F8 to enable desktop control."
    if args.mode != "keyboard":
        from desktop_input import WindowsInput

        desktop = WindowsInput()
        if args.mode == "touchpad":
            from gesture_touchpad import TouchpadController

            controller = TouchpadController(desktop)
            pause_controller = controller.reset
        else:
            from eye_pointer import EyePointer

            controller = EyePointer(desktop)
            pause_controller = controller.pause
    elif not (args.no_hands or args.no_keyboard or args.no_display):
        from virtual_keyboard import VirtualKeyboard

        keyboard = VirtualKeyboard()
    preview_name = "LogiInsight RGB"
    if keyboard is not None:
        preview_name += " + virtual keyboard"
    elif desktop is not None:
        preview_name += f" + {args.mode} control"
    elif not args.no_hands:
        preview_name += " + fingertips"
    title = preview_name + " - q / Escape to quit"
    last_frame = time.monotonic()
    count = 0
    max_detected = 0
    log_file = None
    reset_for_gap = False
    try:
        if args.landmarks_jsonl:
            log_file = args.landmarks_jsonl.open("w", encoding="utf-8")
        worker.start()
        while True:
            if desktop is not None:
                toggle, shortcut_held = desktop.poll_pause_shortcut()
                if toggle:
                    desktop_enabled = not desktop_enabled
                    pause_controller()
                    desktop.release_buttons()
                    status = "Enabled." if desktop_enabled else "Paused."
            try:
                kind, value = output.get(timeout=0.05)
            except queue.Empty:
                if not worker.is_alive():
                    raise RuntimeError("Camera capture process stopped without delivering a frame.") from None
                if desktop_enabled and not reset_for_gap and time.monotonic() - last_frame > 0.25:
                    pause_controller()
                    desktop.release_buttons()
                    reset_for_gap = True
            else:
                if kind == "error":
                    raise RuntimeError(value)
                frame, tracking = value
                last_frame = time.monotonic()
                reset_for_gap = False
                count += 1
                if count == 1:
                    print(f"Receiving RGB video: {frame.shape[1]}x{frame.shape[0]}", flush=True)
                if tracking is not None and args.mode != "eye":
                    from fingertip_tracker import draw_hands

                    max_detected = max(max_detected, len(tracking["hands"]))
                    if log_file:
                        log_file.write(json.dumps(tracking) + "\n")
                    if not args.no_display:
                        if keyboard is not None:
                            keyboard.update(tracking)
                            keyboard.draw_keys(frame)
                        draw_hands(frame, tracking, show_status=keyboard is None,
                                   show_labels=keyboard is None)
                        if keyboard is not None:
                            frame = keyboard.add_text_panel(frame)
                if desktop is not None:
                    if desktop_enabled and not shortcut_held:
                        status = controller.update(tracking)
                    frame = _desktop_banner(frame, status)
                if not args.no_display:
                    cv2.imshow(title, frame)
                if args.frames and count >= args.frames:
                    break
            if not args.no_display:
                key = cv2.waitKey(1) & 0xFF
                if key in (27, ord("q")):
                    break
                if keyboard is not None and key == ord("r"):
                    keyboard.reset_calibration()
                if args.mode == "eye" and desktop_enabled and not shortcut_held:
                    if key == ord("c"):
                        status = controller.begin_calibration()
                    elif key == ord(" "):
                        status = controller.capture_target()
                if count and cv2.getWindowProperty(title, cv2.WND_PROP_VISIBLE) < 1:
                    break
            if time.monotonic() - last_frame > args.timeout:
                raise RuntimeError(f"No video frame for {args.timeout:g}s. Close other camera apps, "
                                   "reconnect USB, or try --backend MSMF --device INDEX.")
    finally:
        if desktop is not None:
            try:
                pause_controller()
                desktop.release_buttons()
            except OSError as exc:
                print(f"Could not release desktop input: {exc}", file=sys.stderr)
        stop.set()
        if worker.pid is not None:
            worker.join(timeout=1)
            if worker.is_alive():
                worker.terminate()
                worker.join(timeout=2)
        output.close()
        if log_file:
            log_file.close()
        if not args.no_display:
            cv2.destroyAllWindows()
    summary = f"Stopped after {count} frames."
    if args.mode != "eye":
        summary += f" Maximum hands detected: {max_detected}."
    print(summary)
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("Stopped.")
        sys.exit(130)
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        sys.exit(1)
