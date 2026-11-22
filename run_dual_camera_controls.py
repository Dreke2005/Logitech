"""Run camera 0 hand controls with either keyboard typing or paper reading.

Start with ``uv run python run_dual_camera_controls.py`` from this folder.
Camera 0 runs ``rgb_preview.py --mode touchpad``; camera 1 runs
``keyboard_typing_detector.py`` by default. Use ``--paper`` to run Paper
Workbench on camera 1 instead. Press Q in either window to stop both.
"""

# Author: Victor Lancieux
# Date: 2026-11-22 (integration launcher created)
# Purpose: Run left-hand pointer/drag and right-hand navigation on camera 0,
# with keyboard typing on camera 1.

import argparse
import ctypes
import subprocess
import sys
import time
from pathlib import Path
from typing import List
from ctypes import wintypes


PROJECT_DIR = Path(__file__).resolve().parent
DESKTOP_SCRIPT = PROJECT_DIR / "rgb_preview.py"
TYPING_SCRIPT = PROJECT_DIR / "keyboard_typing_detector.py"
PAPER_SCRIPT = PROJECT_DIR / "paper_workbench.py"
COPIED_PROJECT_DIR = PROJECT_DIR / "Logitech-Keypad"


def _start(command: List[str], cwd: Path, label: str) -> subprocess.Popen:
    print(f"Starting {label}...", flush=True)
    return subprocess.Popen(command, cwd=str(cwd))


def _python_for(project_dir: Path) -> str:
    """Prefer the copied project's uv environment for its own dependencies."""
    candidates = (
        project_dir / ".venv" / "Scripts" / "python.exe",
        project_dir / ".venv" / "bin" / "python",
    )
    return str(next((path for path in candidates if path.is_file()), Path(sys.executable)))


class _ModeSwitchHotkey:
    """Detect a new Ctrl+Alt+F9 press on Windows without requiring window focus."""

    def __init__(self):
        self.user32 = ctypes.WinDLL("user32", use_last_error=True)
        self.was_held = False

    def poll(self) -> bool:
        held = all(self.user32.GetAsyncKeyState(key) & 0x8000
                   for key in (0x11, 0x12, 0x78))
        pressed = held and not self.was_held
        self.was_held = held
        return pressed


def _close_window_and_wait(process: subprocess.Popen, timeout: float = 5.0) -> None:
    """Ask the selected app to close so it can release its camera cleanly."""
    if process.poll() is not None:
        return
    if sys.platform == "win32":
        user32 = ctypes.WinDLL("user32", use_last_error=True)
        enum_windows = user32.EnumWindows
        enum_windows.argtypes = (ctypes.WINFUNCTYPE(wintypes.BOOL, wintypes.HWND, wintypes.LPARAM),
                                 wintypes.LPARAM)
        get_window_pid = user32.GetWindowThreadProcessId
        get_window_pid.argtypes = (wintypes.HWND, ctypes.POINTER(wintypes.DWORD))
        post_message = user32.PostMessageW
        post_message.argtypes = (wintypes.HWND, wintypes.UINT, wintypes.WPARAM, wintypes.LPARAM)

        def close_matching_window(hwnd, _param):
            pid = wintypes.DWORD()
            get_window_pid(hwnd, ctypes.byref(pid))
            if pid.value == process.pid and user32.IsWindowVisible(hwnd):
                post_message(hwnd, 0x0010, 0, 0)  # WM_CLOSE
            return True

        callback = ctypes.WINFUNCTYPE(wintypes.BOOL, wintypes.HWND, wintypes.LPARAM)(
            close_matching_window
        )
        enum_windows(callback, 0)
    try:
        process.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        if sys.platform == "win32":
            subprocess.run(["taskkill", "/PID", str(process.pid), "/T", "/F"],
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=False)
        else:
            process.terminate()
        process.wait(timeout=3)


def run(confidence: float = 0.55, finger_mode: str = "all",
        text_only: bool = False, keyboard_output: bool = True,
        paper: bool = False, keypad: bool = False) -> int:
    secondary_script = PAPER_SCRIPT if paper else TYPING_SCRIPT
    missing = [str(path) for path in (DESKTOP_SCRIPT, secondary_script) if not path.is_file()]
    if missing:
        raise FileNotFoundError("Required detector file(s) not found:\n  " + "\n  ".join(missing))
    if keypad and not paper:
        raise ValueError("--keypad can only be used with --paper.")

    desktop_python = _python_for(COPIED_PROJECT_DIR)
    desktop_command = [
        desktop_python, str(DESKTOP_SCRIPT), "--mode", "touchpad", "--device", "0",
        "--mirror", "--right-hand-only", "--hand-pointer", "--wink-clicks",
    ]
    if paper:
        secondary_command = [sys.executable, str(PAPER_SCRIPT), "--camera", "1", "--route", "latex"]
        if keypad:
            secondary_command.append("--keypad")
        secondary_label = "paper formulas on camera 1"
    else:
        secondary_command = [
            sys.executable, str(TYPING_SCRIPT), "--camera", "1", "--confidence", str(confidence),
            "--finger", finger_mode,
        ]
        if text_only:
            secondary_command.append("--text-only")
        if not keyboard_output:
            secondary_command.append("--no-system-output")
        secondary_label = "keyboard typing on camera 1"

    processes: List[subprocess.Popen] = []
    mode_hotkey = _ModeSwitchHotkey() if sys.platform == "win32" else None
    use_paper = paper
    try:
        processes.append(_start(desktop_command, DESKTOP_SCRIPT.parent,
                                "right-hand touchpad controls on camera 0"))
        # Let camera 0 initialize before opening camera 1.
        time.sleep(0.8)
        if processes[0].poll() is None:
            processes.append(_start(secondary_command, secondary_script.parent, secondary_label))
        else:
            return int(processes[0].returncode or 1)

        if mode_hotkey is not None:
            print("Press Ctrl+Alt+F9 to switch camera 1 between keyboard and paper formulas.",
                  flush=True)
        print("Both camera controls are running. Close either app to stop both.", flush=True)
        while all(process.poll() is None for process in processes):
            if mode_hotkey is not None and mode_hotkey.poll():
                secondary = processes[1]
                if secondary.poll() is None:
                    _close_window_and_wait(secondary)
                use_paper = not use_paper
                if use_paper:
                    command = [sys.executable, str(PAPER_SCRIPT), "--camera", "1",
                               "--route", "latex"]
                    if keypad:
                        command.append("--keypad")
                    label = "paper formulas on camera 1"
                else:
                    command = [sys.executable, str(TYPING_SCRIPT), "--camera", "1",
                               "--confidence", str(confidence), "--finger", finger_mode]
                    if text_only:
                        command.append("--text-only")
                    if not keyboard_output:
                        command.append("--no-system-output")
                    label = "keyboard typing on camera 1"
                processes[1] = _start(command, (PAPER_SCRIPT if use_paper else TYPING_SCRIPT).parent,
                                      label)
            time.sleep(0.2)
        return next((int(process.returncode or 0) for process in processes
                     if process.poll() is not None), 0)
    except KeyboardInterrupt:
        print("Stopping both detectors...", flush=True)
        return 130
    finally:
        for process in processes:
            if process.poll() is None:
                process.terminate()
        for process in processes:
            if process.poll() is None:
                try:
                    process.wait(timeout=3)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--confidence", type=float, default=0.55,
                        help="minimum MediaPipe detection confidence")
    parser.add_argument("--finger", choices=("all", "indexes-only"), default="all",
                        help="fingers used by the keyboard detector")
    parser.add_argument("--text-only", action="store_true",
                        help="limit keyboard output to text-entry keys")
    parser.add_argument("--no-keyboard-output", action="store_true",
                        help="preview detected keyboard keys without typing into other apps")
    parser.add_argument("--paper", action="store_true",
                        help="run Paper Workbench on camera 1 instead of keyboard typing")
    parser.add_argument("--keypad", action="store_true",
                        help="enable the optional MX Keypad bridge (requires --paper)")
    args = parser.parse_args()
    if args.keypad and not args.paper:
        parser.error("--keypad requires --paper.")
    raise SystemExit(run(args.confidence, args.finger, args.text_only,
                         not args.no_keyboard_output, args.paper, args.keypad))


if __name__ == "__main__":
    main()
