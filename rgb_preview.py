"""RGB preview with fingertip tracking, gaze control, and visual speech recognition."""

import argparse
import json
import multiprocessing as mp
import queue
import subprocess
import sys
import time
from contextlib import ExitStack
from pathlib import Path
from threading import Thread

import cv2
import numpy as np

CALIBRATION_WINDOW = "Gaze calibration"


class CameraOwnershipError(RuntimeError):
    """A camera had an owner whose release could not be verified."""


def _capture(device, backend, width, height, output, stop, model_path=None, mirror=False,
             eye_model_path=None, exposure_lock=None):
    """Isolate native camera calls so a blocked driver can be stopped."""
    from camera_utils import open_camera

    try:
        with ExitStack() as resources:
            hands = eyes = None
            if model_path is not None:
                from fingertip_tracker import FingertipTracker

                hands = FingertipTracker(model_path)
                resources.callback(hands.close)
            if eye_model_path is not None:
                from eye_pointer import EyeLandmarker

                eyes = EyeLandmarker(eye_model_path)
                resources.callback(eyes.close)
            cap, _ = open_camera(
                device, backend=getattr(cv2, "CAP_" + backend),
                width=width, height=height, fourcc="MJPG", fps=30,
            )
            resources.callback(cap.release)
            exposure = None
            if exposure_lock is not None:
                # Paper mode: keep the paper bright but not clipped, and steady while a recording runs.
                from paper_exposure import PaperExposure, PaperFocus

                exposure = PaperExposure(cap)
                resources.callback(exposure.release)
                focus = PaperFocus(cap)  # The autofocus hunts between hand and paper; lock it on the paper.
                resources.callback(focus.release)
            # Stop when the program that started this worker is gone, even if it was killed: a
            # worker left behind keeps the camera, and the next program then gets black frames.
            import multiprocessing

            parent = multiprocessing.parent_process()
            while not stop.is_set() and (parent is None or parent.is_alive()):
                ok, frame = cap.read()
                if not ok or frame is None:
                    stop.wait(0.05)
                    continue
                if mirror:
                    frame = cv2.flip(frame, 1)
                if exposure is not None:
                    exposure.update(frame, locked=exposure_lock.is_set())
                    focus.update(frame, locked=exposure_lock.is_set())
                hand_sample = hands.process(frame) if hands is not None else None
                eye_sample = eyes.process(frame) if eyes is not None else None
                try:
                    output.put_nowait(("frame", (frame, hand_sample, eye_sample)))
                except queue.Full:
                    pass  # Bound buffering instead of building a video backlog.
    except Exception as exc:
        try:
            output.put(("error", str(exc)), timeout=0.2)
        except queue.Full:
            pass
    finally:
        # Do not wait for a frame nobody will read: without this the worker hangs at exit when
        # its parent is gone, and keeps the process (and on some drivers the camera) alive.
        output.cancel_join_thread()


def _desktop_banner(frame, status, gesture_status, navigation_status=None, *, workspace=False, pause_control=True):
    top, width = frame.shape[:2]
    lines = [status]
    if gesture_status:
        lines.append(gesture_status)
    if navigation_status:
        lines.append(navigation_status)
    if pause_control:
        lines.append("Ctrl+Alt+F8   Pause / resume")
    if workspace:
        lines.append("Ctrl+Alt+F9   Switch mode")
    frame = cv2.copyMakeBorder(frame, 0, 14 + 23 * len(lines), 0, 0, cv2.BORDER_CONSTANT,
                               value=(25, 30, 41))
    font = cv2.FONT_HERSHEY_SIMPLEX
    for index, text in enumerate(lines):
        text_width = max(1, cv2.getTextSize(text, font, 1, 1)[0][0])
        scale = min(0.57, max(1, width - 24) / text_width)
        cv2.putText(frame, text, (12, top + 24 + 23 * index), font, scale, (243, 243, 248), 1, cv2.LINE_AA)
    return frame


def _lip_banner(frame, status, transcript, *, workspace=False):
    height, width = frame.shape[:2]
    font = cv2.FONT_HERSHEY_SIMPLEX
    margin = min(12, width // 5)
    available = max(1, width - 2 * margin)
    lines = []
    line = ""
    for word in transcript.split():
        joined = f"{line} {word}" if line else word
        if line and cv2.getTextSize(joined, font, 0.62, 1)[0][0] > available:
            lines.append(line)
            line = word
        else:
            line = joined
    if line:
        lines.append(line)
    hints = [status, "Space   Start / finish phrase"]
    if workspace:
        hints.append("Ctrl+Alt+F9   Switch mode")
    frame = cv2.copyMakeBorder(frame, 0, 13 + 25 * (len(hints) + len(lines)), 0, 0,
                               cv2.BORDER_CONSTANT, value=(25, 30, 41))
    rows = [*hints, *lines]
    for index, label in enumerate(rows):
        max_scale = 0.62 if index >= len(hints) else 0.49
        text_width = max(1, cv2.getTextSize(label, font, 1, 1)[0][0])
        scale = min(max_scale, available / text_width)
        color = (240, 243, 248) if index >= len(hints) else (171, 207, 201)
        cv2.putText(frame, label, (margin, height + 23 + 25 * index), font, scale,
                    color, 1, cv2.LINE_AA)
    return frame


def _lip_model_paths(root):
    if root is not None:
        return root / "upstream", root / ".venv/Scripts/python.exe", root / "checkpoint.pth"
    parent = Path(__file__).resolve().parent.parent
    return (parent / "auto_avsr", parent / ".venv-auto-avsr/Scripts/python.exe",
            parent / "vsr_trlrs2lrs3vox2avsp_base.pth")


def _correct_lip_draft(output, draft, model, topic):
    from lip_correction import polish

    try:
        output.put((draft, polish(draft, model, topic), None))
    except (ValueError, RuntimeError) as error:
        output.put((draft, draft, str(error)))


def _pause_controls(gestures, gaze):
    if gestures is not None:
        gestures.reset()
    if gaze is not None:
        gaze.pause()


def _show_calibration(screen_size, target, progress, status):
    width, height = screen_size
    image = np.full((height, width, 3), (23, 26, 33), dtype=np.uint8)
    center = (round(target[0] * (width - 1)), round(target[1] * (height - 1)))
    radius = max(18, round(height * 0.025))
    cv2.circle(image, center, radius, (83, 96, 108), 2, cv2.LINE_AA)
    cv2.circle(image, center, max(5, radius // 3), (248, 248, 250), -1, cv2.LINE_AA)
    cv2.ellipse(image, center, (radius, radius), -90, 0, round(360 * progress),
                (118, 225, 165), 3, cv2.LINE_AA)
    font = cv2.FONT_HERSHEY_SIMPLEX
    for text, bottom in ((status, 58), ("Follow the dot with your eyes   |   Esc to cancel", 24)):
        scale = min(0.8, width * 0.9 / max(1, cv2.getTextSize(text, font, 1, 1)[0][0]))
        text_width = cv2.getTextSize(text, font, scale, 1)[0][0]
        cv2.putText(image, text, ((width - text_width) // 2, height - bottom), font, scale,
                    (220, 224, 232), 1, cv2.LINE_AA)
    cv2.imshow(CALIBRATION_WINDOW, image)


class _Workspace:
    def __init__(self, ipc, device):
        from desktop_input import WindowsInput

        self.ipc = ipc
        self.device = device
        self.desktop = WindowsInput()
        self.keyboard = self.gaze = None
        # The standalone workbench can launch our workspace on the same press.
        self.desktop.poll_mode_shortcut()

    def poll(self, mode):
        from workspace_modes import MODES

        switch, held = self.desktop.poll_mode_shortcut()
        if self.ipc.consume_notes_request() and mode != "notes":
            return "notes", held
        return (MODES[(MODES.index(mode) + 1) % len(MODES)] if switch else None), held

    def stop_notes(self, next_mode):
        from paper_keypad import current_state

        self.ipc.set_mode(next_mode)
        state = current_state()
        if not state:
            return
        # Even a published cameraActive=false can be stale while Tk is opening a worker.
        # A command and matching ACK synchronize with the Tk event loop.
        command = self.ipc.command("stop", state["session"])
        deadline = time.monotonic() + 10
        while time.monotonic() < deadline:
            if self.ipc.acknowledged(command, False):
                return
            time.sleep(0.04)
        raise CameraOwnershipError("PaperWorkbench did not confirm camera release. Restart it and the workspace.")

    def run_notes(self):
        from paper_keypad import current_state

        self.ipc.set_mode("notes", notes_enabled=True)
        state = current_state()
        process = None
        if state is None:
            path = Path(__file__).resolve().with_name("paper_workbench.py")
            process = subprocess.Popen([sys.executable, str(path), "--camera", str(self.device)],
                                       cwd=path.parent)
        self.ipc.command("start", state["session"] if state else None, device=self.device)
        print("Notes mode. Press Ctrl+Alt+F9 to return to the virtual keyboard.", flush=True)
        deadline = time.monotonic() + 10
        while True:
            if (next_mode := self.poll("notes")[0]) is not None:
                return next_mode
            if current_state():
                deadline = time.monotonic() + 7
            elif (process is not None and process.poll() is not None) or time.monotonic() > deadline:
                raise RuntimeError("PaperWorkbench closed or did not start within 10 seconds.")
            time.sleep(0.04)


def _workspace_mode_error(workspace, mode, error):
    title = "LogiInsight workspace"
    print(f"{mode.capitalize()} mode: {error}", file=sys.stderr, flush=True)
    frame = np.full((115, 780, 3), (25, 30, 41), dtype=np.uint8)
    font = cv2.FONT_HERSHEY_SIMPLEX
    for index, line in enumerate((f"{mode.capitalize()} mode unavailable", str(error),
                                  "Ctrl+Alt+F9   Switch mode   |   Q   Quit")):
        scale = min(0.62, 744 / max(1, cv2.getTextSize(line, font, 1, 1)[0][0]))
        cv2.putText(frame, line, (18, 25 + index * 33), font, scale, (240, 243, 248), 1, cv2.LINE_AA)
    try:
        cv2.imshow(title, frame)
        while True:
            if (next_mode := workspace.poll(mode)[0]) is not None:
                return next_mode
            if (cv2.waitKey(40) & 0xFF in (27, ord("q"))
                    or cv2.getWindowProperty(title, cv2.WND_PROP_VISIBLE) < 1):
                return None
    finally:
        cv2.destroyAllWindows()


def _run_workspace(args):
    from workspace_modes import WorkspaceIpc

    with WorkspaceIpc() as ipc:
        workspace = _Workspace(ipc, args.device)
        mode = "keyboard"
        workspace.stop_notes(mode)
        while mode:
            args.mode = mode
            try:
                if mode == "notes":
                    next_mode = workspace.run_notes()
                else:
                    ipc.set_mode(mode)
                    next_mode = _run_preview(args, workspace=workspace)
            except CameraOwnershipError:
                raise
            except (OSError, RuntimeError, ValueError) as error:
                next_mode = _workspace_mode_error(workspace, mode, error)
            if mode == "notes" and next_mode:
                workspace.stop_notes(next_mode)
            mode = next_mode
    return 0


def _workspace_startup_error(error):
    if sys.stderr is not None:
        print(f"Workspace: {error}", file=sys.stderr, flush=True)
    if sys.stderr is None and sys.platform == "win32":
        import ctypes

        ctypes.windll.user32.MessageBoxW(None, str(error), "LogiInsight workspace", 0x10)
    return 2


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
    parser.add_argument("--mode", choices=("keyboard", "desktop", "touchpad", "eye", "lip", "workspace"),
                        default="keyboard", help="Use one tool or switch between tools in the workspace")
    parser.add_argument("--mirror", action="store_true", help="Mirror video before tracking; coordinates follow the display")
    parser.add_argument("--model", type=Path, help="Local hand_landmarker.task; default downloads Google's model once")
    parser.add_argument("--eye-model", type=Path, help="Local face_landmarker.task for eye or desktop mode")
    parser.add_argument("--landmarks-jsonl", type=Path, help="Optional JSONL output for keyboard integration")
    parser.add_argument("--record-video", type=Path, help="Record a video-only MP4 with Space in plain RGB mode")
    parser.add_argument("--vsr-root", type=Path,
                        help="Alternate lip model folder with upstream, .venv, and checkpoint.pth")
    parser.add_argument("--lip-device", choices=("cpu", "cuda"), default="cpu", help="Lip model compute device")
    parser.add_argument("--lip-correct", metavar="MODEL", help="Suggest conservative edits using a local Ollama model")
    parser.add_argument("--lip-topic", default="", help="Optional subject for lip-reading correction")
    args = parser.parse_args()
    if min(args.width, args.height, args.timeout) <= 0 or args.frames < 0:
        parser.error("Dimensions and timeout must be positive; frames must be nonnegative.")
    if args.mode != "keyboard" and args.no_display:
        parser.error("This mode requires the preview window.")
    workspace_mode = args.mode == "workspace"
    if workspace_mode and (args.backend != "DSHOW" or args.no_hands or args.no_keyboard
                           or args.frames or args.record_video or args.landmarks_jsonl):
        parser.error("Workspace mode needs the DirectShow camera and all input tools enabled.")
    hand_tracking = not args.no_hands and args.mode in ("keyboard", "desktop", "touchpad")
    if args.mode in ("desktop", "touchpad") and args.no_hands:
        parser.error("Desktop and touchpad modes require hand tracking.")
    if args.landmarks_jsonl and not hand_tracking:
        parser.error("--landmarks-jsonl requires a mode with hand tracking enabled.")
    clip = None
    if args.record_video:
        from video_clip import VideoClip

        if args.mode != "keyboard" or args.no_display or not args.no_hands:
            parser.error("--record-video requires --no-hands and a visible RGB preview.")
        path = args.record_video.resolve()
        if path.suffix.lower() != ".mp4" or path.exists() or not path.parent.is_dir():
            parser.error("--record-video needs a new .mp4 file in an existing directory.")
        clip = VideoClip(path)

    from pygrabber.dshow_graph import FilterGraph

    cameras = FilterGraph().get_input_devices()
    for index, name in enumerate(cameras):
        print(f"  {index}: {name}", flush=True)
    if args.list:
        return 0
    if workspace_mode:
        from camera_utils import choose_rgb_camera

        args.device, _ = choose_rgb_camera(cameras, args.device)
        if args.device is None:
            return _workspace_startup_error("No usable RGB camera was found for the workspace.")
        try:
            return _run_workspace(args)
        except (OSError, RuntimeError, ValueError) as error:
            return _workspace_startup_error(error)
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

    return _run_preview(args, clip=clip)


def _run_preview(args, *, clip=None, workspace=None):
    """Run one preview using the same feature controllers for standalone and workspace modes."""
    hand_gestures = args.mode in ("desktop", "touchpad")
    eye_control = args.mode in ("desktop", "eye")
    lip_mode = args.mode == "lip"
    hand_tracking = not args.no_hands and args.mode in ("keyboard", "desktop", "touchpad")
    model_path = None
    eye_model_path = None
    if eye_control:
        from eye_pointer import ensure_eye_model

        eye_model_path = args.eye_model.resolve() if args.eye_model else ensure_eye_model()
        if not eye_model_path.is_file():
            raise ValueError(f"Eye model not found: {eye_model_path}")
    if hand_tracking:
        from fingertip_tracker import ensure_model

        model_path = args.model.resolve() if args.model else ensure_model()
        if not model_path.is_file():
            raise ValueError(f"Hand model not found: {model_path}")
    if lip_mode:
        lip_paths = _lip_model_paths(args.vsr_root)
        repo, python, checkpoint = lip_paths
        if not repo.is_dir():
            raise ValueError(f"Auto-AVSR checkout is missing at {repo}. See silent_transcription_setup.md.")
        for path in (python, checkpoint):
            if not path.is_file():
                raise ValueError(f"Lip model file is missing at {path}. See silent_transcription_setup.md.")
    print(f"Opening camera {args.device} via {args.backend}; press q or Escape to quit.", flush=True)
    ctx = mp.get_context("spawn")
    output, stop = ctx.Queue(maxsize=1), ctx.Event()
    worker = ctx.Process(target=_capture, args=(args.device, args.backend, args.width,
                                               args.height, output, stop, model_path, args.mirror,
                                               eye_model_path), daemon=True)
    keyboard = None
    desktop = gestures = gaze = None
    desktop_enabled = shortcut_held = False
    status = "Paused. Press Ctrl+Alt+F8 to enable desktop control."
    gesture_status = None
    calibration_screen = None
    if hand_gestures or eye_control:
        from desktop_input import WindowsInput

        desktop = workspace.desktop if workspace is not None else WindowsInput()
        if hand_gestures:
            from gesture_touchpad import TouchpadController

            gestures = TouchpadController(desktop)
        if eye_control:
            from eye_pointer import EyePointer

            gaze = workspace.gaze if workspace is not None else None
            if gaze is None:
                gaze = EyePointer(desktop)
                if workspace is not None:
                    workspace.gaze = gaze
    elif hand_tracking and not (args.no_keyboard or args.no_display):
        from virtual_keyboard import VirtualKeyboard

        keyboard = workspace.keyboard if workspace is not None else None
        if keyboard is None:
            keyboard = VirtualKeyboard()
            if workspace is not None:
                workspace.keyboard = keyboard
    preview_name = "LogiInsight RGB"
    if keyboard is not None:
        preview_name += " + virtual keyboard"
    elif desktop is not None:
        preview_name += f" + {args.mode} control"
    elif lip_mode:
        preview_name += " + lip reading"
    elif hand_tracking:
        preview_name += " + fingertips"
    title = preview_name + " - q / Escape to quit"
    last_frame = time.monotonic()
    count = 0
    max_detected = 0
    log_file = None
    lip_reader = last_lip_frame = None
    transcript = corrected = correction_status = ""
    correction = None
    show_raw = False
    reset_for_gap = False
    next_mode = None
    try:
        if args.landmarks_jsonl:
            log_file = args.landmarks_jsonl.open("w", encoding="utf-8")
        if lip_mode:
            from live_lip import LiveLipReader

            lip_reader = LiveLipReader(*lip_paths, device=args.lip_device)
        worker.start()
        while True:
            mode_shortcut_held = False
            if workspace is not None:
                next_mode, mode_shortcut_held = workspace.poll(args.mode)
                if next_mode is not None:
                    break
            new_transcript = lip_reader.poll() if lip_reader is not None else None
            if new_transcript:
                transcript = new_transcript
                if args.lip_correct:
                    correction = queue.SimpleQueue()
                    correction_status = "Checking draft with local model..."
                    Thread(target=_correct_lip_draft, args=(correction, transcript, args.lip_correct,
                                                           args.lip_topic), daemon=True).start()
            redraw_lips = bool(new_transcript)
            if correction is not None:
                try:
                    raw, corrected, error = correction.get_nowait()
                except queue.Empty:
                    pass
                else:
                    correction = None
                    correction_status = error or ("Suggested correction. R toggles original." if corrected != raw
                                                  else "Original draft. No grounded correction found.")
                    redraw_lips = True
            caption = transcript if show_raw or not corrected else corrected
            if desktop is not None:
                toggle, shortcut_held = desktop.poll_pause_shortcut()
                if toggle:
                    desktop_enabled = not desktop_enabled
                    _pause_controls(gestures, gaze)
                    status = "Enabled." if desktop_enabled else "Paused."
                    gesture_status = None
            try:
                kind, value = output.get(timeout=0.05)
            except queue.Empty:
                if not worker.is_alive():
                    raise RuntimeError("Camera capture process stopped without delivering a frame.") from None
                if redraw_lips and last_lip_frame is not None:
                    cv2.imshow(title, _lip_banner(last_lip_frame, correction_status or lip_reader.status, caption,
                                                 workspace=workspace is not None))
                if desktop_enabled and not reset_for_gap and time.monotonic() - last_frame > 0.25:
                    if gestures is not None:
                        gestures.tracking_lost()
                    if gaze is not None and not gaze.calibrating:
                        gaze.pause()
                    reset_for_gap = True
                    gesture_status = None
            else:
                if kind == "error":
                    raise RuntimeError(value)
                frame, tracking, eye_sample = value
                last_frame = time.monotonic()
                count += 1
                if count == 1:
                    print(f"Receiving RGB video: {frame.shape[1]}x{frame.shape[0]}", flush=True)
                if lip_reader is not None:
                    lip_reader.add(frame, last_frame)
                    last_lip_frame = frame
                    frame = _lip_banner(frame, correction_status or lip_reader.status, caption,
                                        workspace=workspace is not None)
                if clip is not None:
                    clip.add_frame(frame, last_frame)
                    frame = frame.copy()
                    cv2.putText(frame, clip.status, (10, 27), cv2.FONT_HERSHEY_SIMPLEX,
                                0.5, (245, 230, 90), 1, cv2.LINE_AA)
                reset_for_gap = False
                if tracking is not None:
                    from fingertip_tracker import draw_hands

                    max_detected = max(max_detected, len(tracking["hands"]))
                    if log_file:
                        log_file.write(json.dumps(tracking) + "\n")
                    if not args.no_display:
                        if keyboard is not None and not mode_shortcut_held:
                            keyboard.update(tracking)
                        if keyboard is not None:
                            keyboard.draw_keys(frame)
                        draw_hands(frame, tracking, show_status=keyboard is None,
                                   show_labels=keyboard is None)
                        if keyboard is not None:
                            frame = keyboard.add_text_panel(frame)
                            if workspace is not None:
                                frame = _desktop_banner(frame, "Virtual keyboard", None,
                                                        workspace=True, pause_control=False)
                if desktop is not None:
                    navigation_status = None
                    if desktop_enabled and not (shortcut_held or mode_shortcut_held):
                        if gaze is not None:
                            status = gaze.update(eye_sample)
                        if gestures is not None and (gaze is None or not gaze.calibrating):
                            hand_status = gestures.update(tracking)
                            navigation_status = desktop.navigation_feedback()
                            if gaze is None:
                                status = hand_status
                            else:
                                gesture_status = hand_status
                    frame = _desktop_banner(frame, status, gesture_status, navigation_status,
                                            workspace=workspace is not None)
                if not args.no_display and (gaze is None or not gaze.calibrating):
                    cv2.imshow(title, frame)
                if args.frames and count >= args.frames:
                    break
            if not args.no_display:
                target = gaze.target if gaze is not None else None
                if target is not None:
                    if calibration_screen is None:
                        calibration_screen = desktop.screen_size()
                        cv2.namedWindow(CALIBRATION_WINDOW, cv2.WINDOW_NORMAL)
                        cv2.moveWindow(CALIBRATION_WINDOW, 0, 0)
                        cv2.setWindowProperty(CALIBRATION_WINDOW, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
                    _show_calibration(calibration_screen, target, gaze.target_progress, status)
                elif calibration_screen is not None:
                    cv2.destroyWindow(CALIBRATION_WINDOW)
                    calibration_screen = None
                key = cv2.waitKey(1) & 0xFF
                if key == 27 and target is not None:
                    gaze.pause()
                    status = "Eye calibration cancelled. Press C to try again."
                    continue
                if key in (27, ord("q")):
                    break
                if clip is not None and key == ord(" "):
                    clip.toggle()
                if lip_reader is not None and key == ord(" "):
                    if lip_reader.phase == "recording":
                        lip_reader.stop()
                    elif lip_reader.phase == "ready" and lip_reader.start():
                        transcript = corrected = correction_status = ""
                        correction = None
                        show_raw = False
                if lip_reader is not None and key == ord("r") and corrected and corrected != transcript:
                    show_raw = not show_raw
                    correction_status = ("Original draft. R shows suggested correction." if show_raw
                                         else "Suggested correction. R toggles original.")
                if keyboard is not None and key == ord("r"):
                    keyboard.reset_calibration()
                if gaze is not None and desktop_enabled and not (shortcut_held or mode_shortcut_held):
                    if key == ord("c"):
                        if gestures is not None:
                            gestures.reset()
                            gesture_status = None
                        status = gaze.begin_calibration()
                    elif key == ord(" "):
                        status = gaze.capture_target()
                if count and cv2.getWindowProperty(title, cv2.WND_PROP_VISIBLE) < 1:
                    break
            if time.monotonic() - last_frame > args.timeout:
                raise RuntimeError(f"No video frame for {args.timeout:g}s. Close other camera apps, "
                                   "reconnect USB, or try --backend MSMF --device INDEX.")
    finally:
        camera_unreleased = False
        if clip is not None:
            clip.finish()
        if lip_reader is not None:
            lip_reader.close()
        if keyboard is not None and workspace is not None:
            keyboard.reset_calibration()
        _pause_controls(gestures, gaze)
        stop.set()
        if worker.pid is not None:
            worker.join(timeout=1)
            if worker.is_alive():
                worker.terminate()
                worker.join(timeout=2)
            camera_unreleased = workspace is not None and worker.is_alive()
        output.close()
        if log_file:
            log_file.close()
        if not args.no_display:
            cv2.destroyAllWindows()
        if camera_unreleased:
            raise CameraOwnershipError("The RGB capture process did not release the camera.")
    summary = f"Stopped after {count} frames."
    if hand_tracking:
        summary += f" Maximum hands detected: {max_detected}."
    print(summary)
    return next_mode if workspace is not None else 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("Stopped.")
        sys.exit(130)
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        sys.exit(1)
