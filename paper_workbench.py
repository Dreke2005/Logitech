"""Capture handwriting with the LogiInsight RGB camera and review model drafts."""

import argparse
from collections import deque
import json
import multiprocessing as mp
import os
import queue
import re
import shutil
import subprocess
import sys
import time
import tkinter as tk
import webbrowser
from dataclasses import replace
from functools import partial
from pathlib import Path
from threading import Event, Thread
from tkinter import filedialog, ttk

import cv2
import numpy as np

from fingertip_tracker import DEFAULT_MODEL
from local_model import MODEL, warm_up
from paper_camera_ai import CameraAI
from paper_camera_check import choose_camera
from paper_check import check_derivation
from paper_context import end_circle_enabled, project_hints, validate_reading, writing_context
from paper_conversion import (
    DIRECT_TARGETS,
    LANGUAGES,
    Draft,
    convert_page,
    convert_transcription,
    make_plot,
    read_again,
    read_page_source,
    read_recording,
    ink_usable,
    read_static,
    read_still,
    trace_usable,
)
import paper_browser
import paper_clipboard
from paper_clipboard import write_new_file
import paper_cloud
import paper_conversion
import paper_orchestrator
import paper_forge
import paper_git
import paper_ltspice
import paper_plan
import paper_project
import paper_reader
from paper_handoff import (ask_vscode, matlab_running, open_code, open_matlab, package_latex, save_generated,
                           save_project)
from paper_keypad import APPS, MENUS, ROOT, KeypadSession, current_state, send_route, send_slot
import paper_system
from paper_math import display_latex, latex_equation
from paper_selection import select_paper_region
from paper_transfer import fenced_code, graph, import_equation, import_figure, import_snippet, next_numbered
from paper_pen import find_pen, rotate_image
from paper_replay import Recorder
from paper_sheet import SheetCalibration
from paper_snapshot import find_paper, snapshot_region
from paper_writing import WritingSelection, WritingSession, paper_bounds
from rgb_preview import _capture



# What ChatGPT is asked to do with a drawing (the Image key). One line: Enter sends it.
IMAGE_REQUEST = ("Generate a finished, polished image from this hand-drawn sketch. Keep its composition, proportions, "
                 "and any labels, and clean up the lines.")


class ReadingCancelled(RuntimeError):
    """The user cancelled the reading check."""


def _signature(selection):
    """Identifies the pen strokes of a selection, to tell whether a live reading is still current."""
    strokes = getattr(selection, "strokes", None) or ()
    ink = getattr(selection, "ink", None)
    amount = 0 if ink is None else int((ink[:, :, 0] < 170).sum()) // 25  # Coarse, so camera noise is ignored.
    return len(strokes) + amount, sum(len(stroke) for stroke in strokes) + amount


def _read_tracked(bench, selected, target, language, workflow, model):
    """read_recording that remembers whether the pen trace or the photo fallback produced the text.

    A reading made while the user was still writing is reused when the strokes have not
    changed since, so finishing usually costs no further model call."""
    live = getattr(bench, "live", None)
    # With the cloud setting the finished writing is read by Claude; the local preview is not reused.
    if live is not None and _signature(selected)[0] and not paper_cloud.enabled():
        key = (_signature(selected), target, workflow)
        thread = live.get("thread")
        if live.get("pending") == key and thread is not None and thread.is_alive():
            thread.join(timeout=40)
        if live.get("key") == key and live.get("reading") is not None:
            reading = live["reading"]
            bench.reading_path = getattr(reading, "path", None)
            return reading
    repo, saved = getattr(bench, "git_repo", None), getattr(bench, "saved", None)
    folder = {"Git": repo, "GitHub": repo, "GitLab": repo, "LaTeX": getattr(bench, "report_folder", None)}.get(
        target, saved if saved and saved.is_dir() else repo)
    reading = read_recording(selected, target, language, workflow, model, hints=project_hints(target, folder))
    bench.reading_path = getattr(reading, "path", None)
    return reading

class PaperWorkbench:
    def __init__(self, window, args):
        self.window = window
        self.args = args
        self.keypad = KeypadSession()
        self.keypad.start_heartbeat()
        self.target = None
        self.menu = "start"  # The keypad's own keys plus the webcam key that starts PaperBridge.
        self.languages = False
        self.language = "Python"
        self.workflow = "default"
        self.icons = {}
        self.frame = self.page = self.draft = self.saved = self.camera = None
        self.figure = self.report_folder = self.git_repo = None
        self.writing = WritingSession()
        self.writing_workflow = None
        self.writing_area = None
        self.paper_clipboard = None
        self.page_pinned = False
        self.image_geometry = self.drag_start = self.roi = None
        self.results = queue.SimpleQueue()
        self.busy = False
        self.pending_action = None
        self.pending_capture = None
        self.dialog_open = False
        self.preview_live = False
        self.camera_failed = False
        self.pasted = False
        self.preview_update = 0
        self.last_app = "vscode"  # The app page used last; version control acts on its project.
        self.writing_started = 0.0  # When the current recording began; a second press soon after is a snapshot.
        self.finish_wait = None  # Deadline while waiting for the hand to uncover the writing after a key press.
        self.snapshot_reason = None  # Which region a snapshot read: pointed, single, paper, or frame.
        self.selection_overlay = None
        self.sheet = self.sheet_bounds = None  # The outline (and bounds) of the sheet the recordings use.
        self.recorder = None  # Frames of the current recording for the debug record (paper_replay.Recorder).
        self.confirm_window = self.confirm_state = None  # The open "Is this right?" check (confirm_reading).
        self.exposure_lock = None  # Set while recording: the capture worker keeps the exposure as it is.
        self.recent_frames = deque(maxlen=20)  # (time, frame) of the last frames, for a sharp snapshot.
        self.writer_rotation = 0  # Clockwise turn that makes the page upright for the writer (saved in settings).
        self.camera_ai = CameraAI()  # The camera's own face detection (SDK): tells where the writer sits.
        self.sheet_calibration = SheetCalibration()  # The sheet's outline, found and followed in the background.
        self.confirm_enabled = getattr(self, "confirm_enabled", True)
        self.debug_folder = None  # The debug record of the reading in progress (see debug_record).
        self.first_frame_at = None  # When the running camera delivered its first frame.
        # Reading while writing: the latest background reading and the strokes it belongs to.
        self.live = {"key": None, "reading": None, "pending": None, "thread": None, "tips": 0, "at": 0.0}
        self.live_results = queue.SimpleQueue()
        self.front_app = None  # The app in front on the computer; the keypad follows it.
        self.front_checked = 0.0
        self.quiet = False
        self.repo_pin = None  # A repository picked with Browse, kept while the same app is used.
        self.project_of = {}  # Per app page: the file or folder the workbench opened last.
        self.active_key = None  # (menu, key) of the action being worked on, shown on the keypad.
        self.key_result = None  # ("done" | "error", time) of the last finished action.
        self.key_state = None
        self.toast_window = None
        self.status = tk.StringVar(value="Start the RGB camera or open a photo of your handwriting.")
        self.status.trace_add("write", lambda *_: self.announce())
        self.load_settings()
        self.destination = tk.StringVar(value="Choose an app")
        window.title("LogiInsight Paper")
        width = min(1090, window.winfo_screenwidth() - 30)
        height = min(700, window.winfo_screenheight() - 110)
        window.geometry(f"{width}x{height}")
        window.minsize(800, 480)
        window.protocol("WM_DELETE_WINDOW", self.close)
        window.bind("<F5>", lambda _event: self.capture())

        header = ttk.Frame(window, padding=(16, 12))
        header.pack(fill="x")
        ttk.Button(header, text="Toggle camera", command=self.toggle_camera).pack(side="left", padx=4)
        ttk.Button(header, text="Open photo", command=self.open_photo).pack(side="left", padx=4)
        ttk.Button(header, text="Capture page", command=self.capture).pack(side="left", padx=4)
        ttk.Button(header, text="Resume live", command=self.resume_live).pack(side="left", padx=4)
        self.editor_ai = ttk.Button(header, text="Open VS Code Chat", command=self.vscode_ai)
        self.plot_preview = ttk.Button(header, text="Open plot preview", command=self.open_plot)
        # Where reading and generation run: Claude in the cloud, or the local models (kept in settings.json).
        self.cloud = tk.BooleanVar(value=paper_cloud.MODE == "cloud")
        ttk.Checkbutton(header, text="Cloud", variable=self.cloud, command=self.set_generation).pack(side="right")
        ttk.Label(header, textvariable=self.destination).pack(side="right", padx=10)
        body = ttk.PanedWindow(window, orient="horizontal")
        body.pack(fill="both", expand=True, padx=16)
        left, right = ttk.Frame(body, padding=8), ttk.Frame(body, padding=8)
        body.add(left, weight=1)
        body.add(right, weight=2)
        self.preview = ttk.Label(left, text="Aim the camera at one well-lit section of paper.",
                                 anchor="center", justify="center", width=45)
        self.preview.pack(fill="both", expand=True)
        self.preview.bind("<ButtonPress-1>", self.begin_selection)
        self.preview.bind("<ButtonRelease-1>", self.end_selection)
        self.buttons = ttk.Frame(left)
        self.buttons.pack(fill="x", pady=(12, 2))
        ttk.Label(right, text="Read from paper").pack(anchor="w")
        self.source = tk.Text(right, height=6, wrap="word", font=("Consolas", 11))
        self.source.pack(fill="x", pady=(5, 12))
        ttk.Label(right, text="Generated draft").pack(anchor="w")
        self.result = tk.Text(right, height=15, wrap="none", undo=True, font=("Consolas", 11))
        self.result.pack(fill="both", expand=True, pady=(5, 12))
        self.notes = ttk.Label(right, text="", wraplength=530, justify="left")
        self.notes.pack(fill="x")
        ttk.Label(window, textvariable=self.status, padding=(18, 12), wraplength=1020).pack(fill="x")
        icon_root = Path(__file__).with_name("keypad-plugin") / "icons"
        for path in icon_root.rglob("*.png"):
            image = tk.PhotoImage(file=str(path))
            factor = max(1, round(image.width() / 44))
            self.icons[path.stem] = image.subsample(factor, factor)
        self.render_buttons()
        self.tick()

    # Hands-off use: remembered folders, key status on the keypad, and pop-up results --------------

    def load_settings(self):
        """Folders chosen once are remembered, so the keypad never has to ask again."""
        try:
            saved = json.loads((ROOT / "settings.json").read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return
        for name in ("git_repo", "report_folder"):
            folder = saved.get(name) if isinstance(saved, dict) else None
            if isinstance(folder, str) and Path(folder).is_dir():
                setattr(self, name, Path(folder))
        if isinstance(saved, dict) and saved.get("language") in LANGUAGES:
            self.language = saved["language"]
        if isinstance(saved, dict) and saved.get("rotation") in (0, 90, 180, 270):
            self.writer_rotation = saved["rotation"]
        if isinstance(saved, dict) and saved.get("generation") in ("cloud", "local"):
            paper_cloud.MODE = saved["generation"]  # Where reading and generation run (see paper_cloud).
        colour = saved.get("pen_colour") if isinstance(saved, dict) else None
        if isinstance(colour, list) and len(colour) == 3:
            WritingSession.pen_colour = tuple(float(v) for v in colour)

    def set_generation(self):
        paper_cloud.MODE = "cloud" if self.cloud.get() else "local"
        paper_cloud.reset()
        self.save_settings()
        self.status.set(f"Reading and generation: {paper_cloud.status()}.")

    def save_settings(self):
        try:
            ROOT.mkdir(parents=True, exist_ok=True)
            (ROOT / "settings.json").write_text(json.dumps({
                "git_repo": str(self.git_repo) if self.git_repo else None,
                "report_folder": str(self.report_folder) if self.report_folder else None,
                "language": self.language, "rotation": self.writer_rotation, "generation": paper_cloud.MODE,
                "pen_colour": list(WritingSession.pen_colour) if WritingSession.pen_colour else None}),
                encoding="utf-8")
        except OSError:
            pass

    def key_status(self):
        """(key, status) for the keypad: recording, busy, then done or error for a few seconds."""
        if self.active_key is None:
            return None, None, None
        menu, key = self.active_key
        if self.writing_workflow is not None and not self.busy:
            return key, "recording", menu
        if self.busy:
            return key, "busy", menu
        if self.key_result is not None and time.monotonic() - self.key_result[1] < 5:
            return key, self.key_result[0], menu
        return None, None, None

    def announce(self):
        """Show the status line as a small pop-up when the workbench window is not in front."""
        text = self.status.get()
        if self.quiet:
            return
        try:
            hidden = self.window.state() == "iconic" or self.window.focus_displayof() is None
        except tk.TclError:
            return
        if not hidden or not text:
            return
        if self.toast_window is not None:
            try:
                self.toast_window.destroy()
            except tk.TclError:
                pass
        toast = self.toast_window = tk.Toplevel(self.window)
        toast.overrideredirect(True)
        toast.attributes("-topmost", True)
        toast.configure(background="#181c24")
        title = "PaperBridge" if self.target is None else f"PaperBridge · {self.target}"
        tk.Label(toast, text=title, foreground="#86e3c4", background="#181c24", anchor="w",
                 font=("Segoe UI", 9, "bold")).pack(fill="x", padx=14, pady=(10, 0))
        tk.Label(toast, text=text, foreground="#f0f0f0", background="#181c24", wraplength=360, justify="left",
                 anchor="w", font=("Segoe UI", 10)).pack(fill="x", padx=14, pady=(2, 12))
        toast.update_idletasks()
        width, height = max(300, toast.winfo_reqwidth()), toast.winfo_reqheight()
        x = toast.winfo_screenwidth() - width - 24
        y = toast.winfo_screenheight() - height - 72
        toast.geometry(f"{width}x{height}+{x}+{y}")
        toast.after(5500, lambda: toast.winfo_exists() and toast.destroy())

    def actions(self):
        menu = self.current_menu()
        return [(button[1], partial(self.dispatch, menu, button[0])) for button in MENUS[menu]]

    def current_menu(self):
        return "vscode_languages" if self.languages else self.menu

    def dispatch(self, menu, action):
        if self.confirm_state is not None:
            # The reading is waiting for a check: the action key confirms it, Camera or Home cancels it.
            if action in ("code", "graph", "equation", "table"):
                self._answer("ok")
            elif action in ("camera", "home", "back"):
                self._answer("cancel")
            return
        if self.dialog_open:
            return
        if self.busy:
            if menu == self.menu and {"code": "default", "graph": "graph", "project": "project",
                                      "equation": "equation", "table": "table", "flow": "flow", "image": "image",
                                      "issue": "forge_issue", "request": "forge_request", "gantt": "gantt",
                                      "diagram": "diagram"}.get(action) == self.workflow:
                return
            self.pending_action = menu, action
            return
        if menu == "start":
            if action == "webcam":
                self.start_from_webcam()
            else:
                paper_system.run_stock(action)
            return
        if menu == "home":
            if action == "start":
                # Leave PaperBridge: the camera stops and the keypad shows its own keys again.
                self.stop_camera()
                self.select("start")
                self.status.set("Webcam off. Press the webcam key to start again.")
            elif self.menu != action or self.languages:
                self.select(action)
            return
        if menu in ("git", "git_remote") and self.menu not in ("git", "git_remote") and action in (
                "commit", "push", "pull", "web", "publish", "browse"):
            # Pressed on the keypad's version-control page while working in an app: act on that
            # app's project without leaving its page.
            self.active_key, self.key_result = (menu, action), None
            self.git_dispatch(action)
            return
        if action == "home":
            self.select("home")
            return
        target_menu = "vscode" if menu == "vscode_languages" else menu
        if self.menu != target_menu:
            self.select(target_menu)
        if action == "git":
            # The Version control key of an app page: git for that app's project. With the camera on,
            # recording starts right away (see arm_git_auto).
            self.select("git")
            repo = self.active_repo()
            if repo is not None:
                self.status.set(f"Version control for {repo.name}. Commit, push, pull, branch, or Host.")
                self.arm_git_auto(first=True)
            return
        if menu == "vscode_languages":
            self.writing_workflow = None
            if action in LANGUAGES:
                self.language = action
                self.save_settings()
                self.clear_draft()
            self.languages = False
            self._show_menu()
            return
        if action not in ("camera", "home", "languages", "photo", "save_photo", "browse", "open", "save", "remote",
                          "back"):
            self.active_key, self.key_result = (target_menu, action), None
        if target_menu in ("git", "git_remote"):
            self.git_dispatch(action)
            return
        if action == "gantt":
            self.writing_action("gantt")
            if self.writing_workflow == "gantt":
                self.status.set("Write the tasks top to bottom with durations (2w, 3d) and mark milestones. Press "
                                "Gantt again or lift your hand when done.")
            return
        if action == "image":
            self.writing_action("image")
            if self.writing_workflow == "image":
                self.status.set("Draw the picture. Press Image again or lift your hand when done; press twice "
                                "quickly for a drawing that is already on the page.")
            return
        if action == "flow":
            self.writing_action("flow")
            if self.writing_workflow == "flow":
                self.status.set("Draw boxes and arrows. Press Flow again or lift your hand when done.")
            return
        handlers = {
            "camera": self.toggle_camera, "open": self.open_draft, "save": self.save,
            "photo": self.open_photo, "copy": self.copy_draft, "paste": self.paste,
            "chat": self.chatgpt, "save_photo": self.save_photo,
            "browse": self.browse_report, "overleaf": self.open_overleaf, "simulate": self.simulate,
        }
        workflows = {"code": "default", "equation": "equation", "project": "project",
                     "brief": "idea", "circuit": "default", "graph": "graph"}
        if action == "languages":
            self.languages = True
            self.writing_workflow = None
            self._show_menu()
        elif action in ("code", "graph", "project", "equation", "table"):
            self.writing_action(workflows.get(action, action))
        elif action in handlers:
            if action in ("photo", "copy", "paste"):
                self.writing_workflow = None
            handlers[action]()
        elif action in workflows:
            self.convert(workflows[action])

    def render_buttons(self):
        for child in self.buttons.winfo_children():
            child.destroy()
        self.actions_drawn = self.actions()
        self.keypad.revision += 1
        for index, (name, action) in enumerate(self.actions_drawn):
            key, _, *icons = MENUS[self.current_menu()][index]
            icon = self.icons.get("camera_stop" if key == "camera" and self.camera else
                                  icons[0] if icons else key)
            button = ttk.Button(self.buttons, command=action)
            if icon is not None:
                button.configure(image=icon)
                button.bind("<Enter>", lambda _event, label=name: self.describe(label))
                button.bind("<FocusIn>", lambda _event, label=name: self.describe(label))
            else:
                button.configure(text=name)
            button.grid(row=index // 3, column=index % 3, padx=2, pady=2, sticky="ew")
        self.publish()

    def describe(self, label):
        if not self.busy and self.writing_workflow is None:
            self.status.set(label)

    def publish(self):
        try:
            self.key_state = self.key_status()
            self.keypad.publish([name for name, _ in self.actions_drawn] + [""] * (9 - len(self.actions_drawn)),
                                self.current_menu(), self.camera is not None, *self.key_state)  # key, status, menu
        except OSError as error:
            self.status.set(f"Could not update keypad labels: {error}")

    def follow_front_app(self, now):
        """Show the page of the application the user brings to the front on the computer."""
        if now - self.front_checked < 0.8:
            return
        self.front_checked = now
        try:
            app = paper_project.foreground_app()
        except OSError:
            return
        if app == self.front_app:
            return
        self.front_app = app
        if (app in APPS and app != self.menu and self.menu != "start" and not self.busy
                and self.writing_workflow in (None, "git_auto")
                and not self.dialog_open and not (self.menu in ("git", "git_remote") and self.last_app == app)):
            self.quiet = True  # No pop-up for a page change the user did not ask for by key.
            try:
                self.select(app, keep=True)
            finally:
                self.quiet = False

    def select(self, menu, keep=False):
        """Show a menu. ``keep`` leaves the current draft in place (used when following the front app)."""
        if self.busy:
            return
        if menu not in ("start", "home", "git", "git_remote"):
            if menu != self.last_app:
                self.repo_pin = None
            self.last_app = menu
        self.menu, self.languages = menu, False
        self.writing_workflow = None
        self.target = "Git" if menu in ("git", "git_remote") else APPS.get(menu)
        if not keep:
            self.clear_draft()
        self._show_menu()
        self.status.set("Press the webcam key to start." if menu == "start" else "Choose an app."
                        if self.target is None else "Use the camera icon to start RGB, then choose a conversion.")

    def start_from_webcam(self):
        """The webcam key on the start page: camera on, pen tracking running, and the apps shown."""
        self.start_camera()
        self.select("home")
        if self.camera:
            self.status.set("Webcam on, tracking the pen. Choose an app.")

    def arm_git_auto(self, first=False):
        """Record on the version-control page without a key press.

        A drawn branch tree runs as git operations and a written sentence becomes a commit
        (see paper_conversion.read_recording for how the two are told apart). After each run
        the recording starts again, so the next drawing on the page is picked up too. The
        keys still work: pressing Commit, Branch, or Diagram replaces this recording."""
        if (self.menu != "git" or self.camera is None or self.busy or self.dialog_open
                or self.writing_workflow is not None or self.git_repo is None or self.page_pinned
                or self.roi is not None):
            return
        previous = self.status.get()
        self.quiet = not first  # Re-arming must not hide the result of the last run behind a new pop-up.
        try:
            # Start from the page as it is now: writing done on an app page is not a git request.
            self.writing = WritingSession()
            self.writing_action("git_auto")
            self.active_key, self.key_result = ("git", "diagram"), None
            if first and self.writing_workflow == "git_auto":
                self.status.set(f"Recording on {self.git_repo.name}: draw branches and circled commits, or write a "
                                "commit message, then lift your hand. Keys work as usual.")
            elif not first:
                self.status.set(previous)
        finally:
            self.quiet = False

    def _show_menu(self):
        title = "Choose an app" if self.target is None else self.target
        if self.target == "VS Code":
            title += f" / {self.language}"
            self.editor_ai.pack(side="left", padx=4)
        else:
            self.editor_ai.pack_forget()
        self.destination.set(title)
        self.render_buttons()

    def toggle_camera(self):
        if self.camera:
            self.stop_camera()
            self.status.set("Camera stopped.")
        else:
            self.start_camera()

    def start_camera(self):
        if self.camera:
            self.preview_live = True
            return
        from pygrabber.dshow_graph import FilterGraph

        try:
            names = FilterGraph().get_input_devices()
            # The LogiInsight RGB stream when connected, otherwise the first ordinary webcam. Both go
            # through the same capture worker, hand tracking, and writing session.
            device, logitech = choose_camera(names, self.args.camera)
            if device is None:
                self.camera_failed = True
                self.status.set("No camera found. Connect the LogiInsight camera or a webcam.")
                return
            ctx = mp.get_context("spawn")
            output, stop = ctx.Queue(1), ctx.Event()
            # Held while a recording runs, so the exposure does not change under the writing.
            self.exposure_lock = ctx.Event()
            hand_model = DEFAULT_MODEL if DEFAULT_MODEL.is_file() else None
            process = ctx.Process(target=_capture, args=(device, "DSHOW", 1920, 1080, output, stop, hand_model),
                                  kwargs={"exposure_lock": self.exposure_lock}, daemon=True)
            process.start()
            self.camera_ai.start()  # Runs only with the SDK credentials set; otherwise it stays idle.
            self.sheet_calibration.start()
            self.writing = WritingSession()
            self.writing_area = None
            self.page_pinned = False
            self.camera = process, output, stop, time.monotonic()
            # Load both models while the camera starts, so the first reading is as fast as the rest.
            Thread(target=lambda: (paper_reader.warm_up(), warm_up(self.args.model)), daemon=True).start()
            self.first_frame_at = None
            self.preview_live = True
            self.preview_update = 0
            self.render_buttons()
            self.status.set(f"Opening {names[device]}..." if logitech else
                            f"LogiInsight camera not connected: using {names[device]}. Aim it at the paper.")
        except (OSError, RuntimeError) as error:
            self.camera_failed = True
            self.status.set(f"Camera error: {error}")

    def stop_camera(self):
        self.sheet_calibration.stop()
        if self.camera:
            process, output, stop, _ = self.camera
            stop.set()
            process.join(timeout=0.3)
            if process.is_alive():
                process.terminate()
                process.join(timeout=1)
            output.close()
            self.camera = None
            self.frame = None
            self.preview_live = False
            self.pending_capture = None
            self.writing_workflow = None
            self.render_buttons()

    def resume_live(self):
        self.roi = None
        self.page_pinned = False
        self.start_camera()

    def show_image(self, frame):
        height, width = frame.shape[:2]
        columns = max(250, min(540, self.preview.winfo_width()))
        scale = min(columns / width, 450 / height)
        small = cv2.resize(frame, (max(1, round(width * scale)), max(1, round(height * scale))))
        self.image_geometry = width, height, small.shape[1], small.shape[0]
        height, width = small.shape[:2]
        rgb = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
        image = tk.PhotoImage(data=f"P6 {width} {height} 255\n".encode() + rgb.tobytes(), format="PPM")
        self.preview.configure(image=image, text="")
        self.preview.image = image

    def begin_selection(self, event):
        if (self.preview_live or self.page is not None) and not self.busy:
            self.drag_start = event.x, event.y

    def end_selection(self, event):
        if self.drag_start is None:
            return
        start_x, start_y = self.drag_start
        self.drag_start = None
        if self.busy or not self.image_geometry:
            return
        width, height, view_width, view_height = self.image_geometry
        left = (self.preview.winfo_width() - view_width) / 2
        top = (self.preview.winfo_height() - view_height) / 2
        x0, x1 = sorted(round((x - left) * width / view_width) for x in (start_x, event.x))
        y0, y1 = sorted(round((y - top) * height / view_height) for y in (start_y, event.y))
        roi = max(0, x0), max(0, y0), min(width, x1), min(height, y1)
        if self.preview_live:
            if roi[2] - roi[0] > 30 and roi[3] - roi[1] > 30:
                self.writing_area = roi
                if self.writing_workflow and self.frame is not None:
                    self.writing = WritingSession(self.frame, roi)
                    self.status.set("Writing area set. Write now, then move your writing hand away.")
                else:
                    self.status.set("Writing area set. Press Code or Plot before writing.")
            return
        self.roi = roi if roi[2] - roi[0] > 8 and roi[3] - roi[1] > 8 else None
        self.clear_draft()
        self.selected_page()
        self.status.set("Manual selection ready." if self.roi else "Automatic selection restored.")

    def capture(self):
        if self.busy:
            return
        if self.frame is None:
            self.start_camera()
            return
        self.set_page(self.frame.copy())

    def open_photo(self):
        if self.busy:
            return
        path = self._dialog(filedialog.askopenfilename, filetypes=[("Photos", "*.png *.jpg *.jpeg *.bmp *.webp")])
        if path:
            self.load_image(path)

    def _dialog(self, prompt, **options):
        self.dialog_open = True
        try:
            return prompt(parent=self.window, **options)
        finally:
            self.dialog_open = False

    def load_image(self, path):
        import numpy as np

        try:
            frame = cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_COLOR)
            if frame is None:
                raise ValueError("The selected file is not a readable image.")
            self.set_page(frame)
        except (OSError, ValueError) as error:
            self.status.set(str(error))

    def set_page(self, image):
        self.writing_workflow = None
        self.page = image
        self.page_pinned = True
        self.camera_failed = False
        self.roi = None
        self.preview_live = False
        self.clear_draft()
        self.selected_page()
        self.status.set("Page ready. Drag to adjust the selection, or choose a conversion.")

    def selected_page(self, *, narrow=False, marked_only=False, pointed=False):
        """The region to read from the current page. ``pointed`` (a snapshot from the live camera):
        the writing the pen tip points at, else the only block of writing, else the whole sheet."""
        now = time.monotonic()
        rotation = 0
        if pointed:
            # Upright for the writer: from the pen in view, else from where the camera's own AI sees the
            # writer's face, else as the last recording or pen showed.
            seen = self.writing.pen_rotation(now)
            if seen is None:
                seen = self.camera_ai.writer_rotation((self.page.shape[1], self.page.shape[0]))
            if seen is not None and seen != self.writer_rotation:
                self.writer_rotation = seen
                self.save_settings()
            rotation = self.writer_rotation
        if pointed and self.roi is None and not marked_only:
            tip = self.writing.pointer(now)
            if tip is None and self.writing.pen_rotation(now) is not None:
                tip = self.writing.pen.point  # The nib, even off the detected sheet.
            # The pen points at the writing just in front of its point; that block is read.
            pen = self.writing.pen if self.writing.pen_rotation(now) is not None else None
            ahead = np.subtract(pen.point, pen.back) if pen is not None else None
            bounds, reason = snapshot_region(self.page, tip, self.writing.hidden_mask(self.page.shape), ahead)
            self.snapshot_reason = reason
            writing = WritingSelection(select_paper_region(self.page, bounds), rotation=rotation)
            self.show_selection(writing.page, tip)
            return writing
        writing = self.writing.select(self.page, roi=self.roi, narrow=narrow, marked_only=marked_only)
        if rotation:
            writing = replace(writing, rotation=rotation)
        self.show_selection(writing.page)
        return writing

    def show_selection(self, selection, tip=None):
        overlay = self.page.copy()
        if selection.bounds:
            x0, y0, x1, y1 = selection.bounds
            cv2.rectangle(overlay, (x0, y0), (x1 - 1, y1 - 1), (45, 150, 25), 4)
        if tip is not None:
            cv2.drawMarker(overlay, tip, (230, 40, 230), cv2.MARKER_CROSS, 36, 3)
        self.selection_overlay = overlay
        self.show_image(overlay)

    def debug_record(self, kind, selected, reason=None):
        """Keep what a reading is based on, to check a demo afterwards: ROOT/debug/<time>-<kind>/ with the
        frame (ink in blue, pen tip, chosen region in green), the crop, the ink, and the reading (see
        debug_result). Only the newest records are kept."""
        if selected is None or self.page is None:
            return
        try:
            now = time.monotonic()
            folder = ROOT / "debug" / f"{time.strftime('%Y%m%d-%H%M%S')}-{kind}"
            folder.mkdir(parents=True, exist_ok=True)
            overlay = self.writing.draw(self.page, now)
            if selected.page.bounds:
                x0, y0, x1, y1 = selected.page.bounds
                cv2.rectangle(overlay, (x0, y0), (x1 - 1, y1 - 1), (45, 150, 25), 4)
            tip = self.writing.pointer(now, within=3.0)
            if tip is not None:
                cv2.drawMarker(overlay, tip, (230, 40, 230), cv2.MARKER_CROSS, 36, 3)
            cv2.imwrite(str(folder / "frame.jpg"), overlay, [cv2.IMWRITE_JPEG_QUALITY, 85])
            cv2.imwrite(str(folder / "raw.jpg"), self.page, [cv2.IMWRITE_JPEG_QUALITY, 92])
            if self.writing.reference is not None:
                cv2.imwrite(str(folder / "reference.jpg"), self.writing.reference, [cv2.IMWRITE_JPEG_QUALITY, 92])
            cv2.imwrite(str(folder / "crop.png"), selected.page.image)
            for name in ("ink", "trace"):
                image = getattr(selected, name, None)
                if image is not None:
                    cv2.imwrite(str(folder / f"{name}.png"), image)
            tips, down = self.writing.summary()
            info = {"kind": kind, "reason": reason, "bounds": selected.page.bounds, "app": self.target,
                    "language": self.language, "tip_samples": tips, "pen_down": down, "pointer": tip,
                    "tip_offset": [float(v) for v in self.writing.tip_offset],
                    "strokes": len(getattr(selected, "strokes", ()) or ()),
                    "text_height": float(getattr(selected, "text_height", 0.0) or 0.0),
                    "rotation": getattr(selected, "rotation", 0), "frame": list(self.page.shape[:2]),
                    "camera_ai": self.camera_ai.status(), "face": self.camera_ai.face_box(within=5.0),
                    "sheet": {"confidence": self.sheet_calibration.status()[1], "state": self.sheet_calibration.status()[2],
                              "outline": None if self.sheet_calibration.status()[0] is None
                              else self.sheet_calibration.status()[0].round().tolist(),
                              "seconds": round(self.sheet_calibration.cost, 3),
                              "error": self.sheet_calibration.error}}
            (folder / "selection.json").write_text(json.dumps(info, indent=1, default=str), encoding="utf-8")
            self.debug_folder, self.debug_started = folder, time.perf_counter()
            recorder, self.recorder = self.recorder, None
            if recorder is not None and kind.startswith("recording"):
                Thread(target=recorder.save, args=(folder,), daemon=True).start()
            records = sorted((ROOT / "debug").iterdir())
            for old in records[:-10]:  # Replays take up to 100 MB each.
                shutil.rmtree(old, ignore_errors=True)
        except (OSError, cv2.error, ValueError, TypeError):
            self.debug_folder = None

    # Checking the reading with the user -------------------------------------------------------------

    CONFIRMED = ("default", "equation", "graph", "table")  # Keys whose reading is shown for a check first.

    def confirm_reading(self, source, workflow, again=None, summary=None):
        """Ask whether the reading is right before anything is generated. Called from the worker thread.

        A small window on top of everything shows the reading, editable. Correct (Enter, or the same
        key again) uses it, with any edits; Read again calls ``again`` for a second reading and asks
        again; Cancel stops. Returns the text to use, or None when cancelled. Corrections are kept with
        the debug record and in corrections.jsonl, to measure and improve reading on real writing.
        ``summary`` is a function returning one line about what will be produced, shown under the reading."""
        if workflow not in self.CONFIRMED or not source or not self.confirm_enabled:
            return source
        first = source
        while True:
            answer = {}
            done = Event()
            line = summary() if summary is not None else None
            self.window.after(0, self._show_confirm, source, workflow, answer, done, *((line,) if line else ()))
            if not done.wait(300):
                self.window.after(0, self._close_confirm)
                return None
            if answer.get("choice") == "again" and again is not None:
                self.window.after(0, self.status.set, "Reading again...")
                try:
                    source = again(source) or source
                except (OSError, RuntimeError, ValueError):
                    pass
                continue
            if answer.get("choice") != "ok":
                return None
            text = answer.get("text", source).strip()
            if text != first:
                self._log_correction(first, text, workflow)
            return text

    def _show_confirm(self, source, workflow, answer, done, summary=None):
        self._close_confirm()
        what = {"default": "Code", "equation": "Equation", "graph": "Plot", "table": "Table"}.get(workflow, "Reading")
        dialog = self.confirm_window = tk.Toplevel(self.window)
        dialog.title(f"PaperBridge · Is this {what.lower()} right?")
        dialog.attributes("-topmost", True)
        dialog.configure(background="#181c24")
        dialog.protocol("WM_DELETE_WINDOW", lambda: self._answer("cancel"))
        tk.Label(dialog, text=f"{what} read from your writing. Correct it here if needed.",
                 foreground="#86e3c4", background="#181c24", font=("Segoe UI", 10, "bold"),
                 anchor="w").pack(fill="x", padx=14, pady=(12, 4))
        lines = source.count("\n") + 1
        text = tk.Text(dialog, height=min(14, max(2, lines + 1)), width=60, font=("Consolas", 14), wrap="none",
                       background="#f7f7f7", undo=True)
        text.insert("1.0", source)
        text.pack(fill="both", expand=True, padx=14)
        preview = self._math_preview(source) if workflow in ("equation", "graph") and lines == 1 else None
        if preview is not None:
            label = tk.Label(dialog, image=preview, background="#ffffff")
            label.image = preview
            label.pack(padx=14, pady=(6, 0))
        if summary:  # What the orchestrator will produce from this reading; an edit produces it anew.
            tk.Label(dialog, text=f"Will make: {summary}", foreground="#d8dee9", background="#181c24",
                     font=("Segoe UI", 9), anchor="w", wraplength=560, justify="left").pack(fill="x", padx=14,
                                                                                           pady=(6, 0))
        buttons = tk.Frame(dialog, background="#181c24")
        buttons.pack(fill="x", padx=14, pady=10)
        for label, choice in (("Correct  ✓", "ok"), ("Read again", "again"), ("Cancel", "cancel")):
            ttk.Button(buttons, text=label, command=partial(self._answer, choice)).pack(side="left", padx=(0, 8))
        # Enter confirms a one-line reading; code keeps Enter for new lines and confirms with Ctrl+Enter.
        if lines == 1 and workflow != "default":
            text.bind("<Return>", lambda _event: (self._answer("ok"), "break")[1])
        text.bind("<Control-Return>", lambda _event: (self._answer("ok"), "break")[1])
        dialog.bind("<Escape>", lambda _event: self._answer("cancel"))
        self.confirm_state = answer, done, text, workflow
        dialog.update_idletasks()
        width, height = dialog.winfo_reqwidth(), dialog.winfo_reqheight()
        dialog.geometry(f"+{dialog.winfo_screenwidth() - width - 24}+{dialog.winfo_screenheight() - height - 90}")
        dialog.lift()
        dialog.focus_force()
        text.focus_set()
        text.tag_add("sel", "1.0", "end-1c")
        self.status.set(f"Check the {what.lower()} that was read: Correct, edit it, or Read again.")

    def _answer(self, choice):
        if self.confirm_state is None:
            return
        answer, done, text, _ = self.confirm_state
        answer["choice"], answer["text"] = choice, text.get("1.0", "end-1c")
        self.confirm_state = None
        self._close_confirm()
        done.set()

    def _close_confirm(self):
        if self.confirm_window is not None:
            try:
                self.confirm_window.destroy()
            except tk.TclError:
                pass
            self.confirm_window = None

    def _math_preview(self, source):
        """The equation typeset, so a misread sign or exponent stands out. None when it cannot be drawn."""
        try:
            import io

            import matplotlib
            matplotlib.use("Agg")
            from matplotlib import pyplot

            latex = display_latex(source)
            latex = latex.replace("\\[", "").replace("\\]", "").strip().strip("$")
            latex = re.sub(r"\bsum_\(([^()]*)\)", r"\\sum_{\1}", latex)  # 'sum_(i=2)^4' as the reader writes it
            figure = pyplot.figure(figsize=(0.01, 0.01))
            figure.text(0, 0, f"${latex}$", fontsize=22)
            buffer = io.BytesIO()
            figure.savefig(buffer, format="png", dpi=110, bbox_inches="tight", pad_inches=0.12, facecolor="white")
            pyplot.close(figure)
            return tk.PhotoImage(data=buffer.getvalue())
        except Exception:  # A preview is optional; any rendering failure just leaves it out.
            return None

    def _log_correction(self, read, corrected, workflow):
        entry = {"time": time.strftime("%Y-%m-%d %H:%M:%S"), "app": self.target, "workflow": workflow,
                 "read": read, "corrected": corrected,
                 "record": str(self.debug_folder) if self.debug_folder else None}
        try:
            with (ROOT / "corrections.jsonl").open("a", encoding="utf-8") as log:
                log.write(json.dumps(entry) + "\n")
            if self.debug_folder is not None:
                (self.debug_folder / "corrected.json").write_text(json.dumps(entry, indent=1), encoding="utf-8")
        except OSError:
            pass

    def debug_result(self, result):
        """Add the reading and the result to the debug record of the reading that just finished."""
        folder, self.debug_folder = self.debug_folder, None
        if folder is None:
            return
        try:
            if isinstance(result, Exception):
                info = {"error": str(result)}
            else:
                draft, figure, workflow = result
                info = {"workflow": workflow, "read_from": getattr(self, "reading_path", None),
                        "source": draft.source, "content": draft.content[:4000], "notes": list(draft.notes),
                        "files": [name for name, _ in draft.files or ()], "figure": str(figure) if figure else None}
            info["seconds"] = round(time.perf_counter() - self.debug_started, 2)
            (folder / "result.json").write_text(json.dumps(info, indent=1, default=str), encoding="utf-8")
        except (OSError, TypeError, ValueError):
            pass

    def clear_draft(self):
        self.draft = self.saved = self.figure = None
        self.pasted = False
        self.plot_preview.pack_forget()
        self._source_text("")
        self.result.configure(state="normal")
        self.result.delete("1.0", "end")
        self.notes.configure(text="")

    def _source_text(self, text):
        self.source.delete("1.0", "end")
        self.source.insert("1.0", text)

    def writing_action(self, workflow):
        if (not self.camera and not self.page_pinned and self.roi is None and self.page is None
                and not (workflow == "graph" and self.pasted)):
            self.start_camera()  # One key press: the action key starts the camera when it is off.
        if not self.camera or self.page_pinned or self.roi is not None or (workflow == "graph" and self.pasted):
            self.convert(workflow)
            return
        if self.writing_workflow == workflow:
            if time.monotonic() - self.writing_started < 1.5:
                # Pressed twice quickly: take a snapshot and work from what is already on the page.
                self.writing_workflow = None
                self.preview_live = True
                self.status.set("Reading what is already on the page...")
                self.convert(workflow)
            elif self.frame is not None and self.writing.hand_over_page(time.monotonic()):
                # The hand hides the last letters: finish as soon as it moves away (see finish_after_wait).
                self.finish_wait = time.monotonic() + 1.5
                self.status.set("Lift your hand for a moment so the last letters are visible...")
            elif self.frame is not None:
                self.finish_writing(self.writing.select(self.frame, prefer_recent=True))
            return
        self.writing_workflow = workflow
        self.finish_wait = None
        self.live.update(key=None, reading=None, pending=None, tips=0)
        self.writing_started = time.monotonic()
        self.preview_live = True
        self.pending_capture = None
        purpose = {"default": "code", "graph": "a plot", "project": "a project", "equation": "an equation",
                   "git_repo": "a repository idea", "git_branch": "a branch name",
                   "git_commit": "a commit message", "diagram": "a branch diagram",
                   "git_auto": "a branch diagram or a commit message",
                   "forge_issue": "an issue", "forge_request": "a request title", "flow": "a flowchart",
                   "image": "a drawing",
                   "gantt": "a project plan",
                   "table": "a table"}.get(workflow, "writing")
        if not self.camera_settled(time.monotonic()):
            self.writing = WritingSession()  # The recording starts in tick once the picture has settled.
            self.recorder = None
            self.status.set(f"Camera starting. Wait for \"Camera ready\" before writing for {purpose}.")
        else:
            self.begin_writing_frame()
            area_note = "" if self.writing.area else " Drag around the paper if the pen is hard to see."
            finish = ("draw a small circle, lift your hand, or press the same key to finish"
                      if self.writing.end_circle_enabled else "lift your hand or press the same key to finish")
            self.status.set(f"Recording for {purpose}. Write now, then {finish}." + area_note)

    def begin_writing_frame(self):
        sheet = None
        area = self.writing_area
        calibrated = self.sheet_calibration.sheet() if area is None else None
        if calibrated is not None:
            # The sheet as calibrated on a clear view and followed since: no search at the key press, when
            # a hand and the pen cover part of it.
            area, sheet = calibrated
        elif area is None:
            # The sheet of paper: writing counts only on it, never on the desk or things around it. With the
            # pen in view it is the sheet under the nib, even on a white desk or next to other sheets.
            nib = None
            if self.writing.pen_rotation(time.monotonic()) is not None:
                nib = self.writing.pen.point
            elif WritingSession.pen_colour is not None:
                # Not followed yet: look for the pen of the learned colour in this frame.
                found = find_pen(self.frame, None, (), colour=WritingSession.pen_colour)
                nib = found.point if found is not None else None
            paper = find_paper(self.frame, pen=nib)
            height, width = self.frame.shape[:2]
            plausible = paper.found and (paper.bounds[2] - paper.bounds[0]) * (paper.bounds[3] - paper.bounds[1]) \
                <= 0.5 * width * height
            if plausible:
                area, sheet = paper.bounds, paper.corners
            elif self.sheet is not None and self.sheet_bounds is not None:
                area, sheet = self.sheet_bounds, self.sheet  # The sheet of the last recording, still on the desk.
            else:
                area = paper_bounds(self.frame)
        # Branch diagrams and copy marks contain circles, so a circle must not end those recordings.
        previous = self.writing
        self.writing = WritingSession(self.frame, area,
                                      end_circle=end_circle_enabled(self.writing_workflow, self.target),
                                      settle=1.6 if self.writing_workflow == "git_auto" else 0.9, sheet=sheet)
        self.writing.pen, self.writing.pen_seen = previous.pen, previous.pen_seen  # Keep following the same pen.
        if sheet is not None:
            self.sheet, self.sheet_bounds = sheet, area
        # Every frame and hand sample of this recording, so it can be replayed offline (paper_replay.py).
        self.recorder = Recorder(self.frame, area, sheet)

    def _points_at_writing(self):
        """True when the pen's point rests at writing that is already on the page."""
        if self.frame is None:
            return False
        now = time.monotonic()
        pen = self.writing.pen if self.writing.pen_rotation(now) is not None else None
        tip = self.writing.pointer(now) or (pen.point if pen is not None else None)
        if tip is None:
            return False
        ahead = np.subtract(pen.point, pen.back) if pen is not None else None
        try:
            _, reason = snapshot_region(self.frame, tip, self.writing.hidden_mask(self.frame.shape), ahead)
        except cv2.error:
            return False
        return reason == "pointed"

    @staticmethod
    def _wrote_little(selected):
        """True when a recording holds next to no new ink: specks at the pen's point, not writing."""
        ink = getattr(selected, "ink", None)
        dark = 0 if ink is None else int(np.count_nonzero(ink[:, :, 0] < 170))
        x0, y0, x1, y1 = selected.page.bounds or (0, 0, 0, 0)
        return dark < 150 or max(x1 - x0, y1 - y0) < 110

    def finish_writing(self, selected):
        if (self.writing_workflow in self.CONFIRMED and (selected is None or self._wrote_little(selected))
                and self._points_at_writing()):
            # Key, point at an equation, key again: nothing was written, so this is the same request as
            # a double tap. Read the writing the pen points at instead of a few specks at its point.
            workflow, self.writing_workflow = self.writing_workflow, None
            self.finish_wait = None
            self.preview_live = True
            self.status.set("Nothing new was written: reading the writing the pen points at...")
            self.convert(workflow)
            return
        if selected is None:
            # The key was pressed but neither strokes nor new ink were found. The usual cause is the
            # hand still covering the writing, so give it a moment to move away, then read a photo
            # of the area the hand worked in (see finish_after_wait).
            if self.finish_wait is None:
                self.finish_wait = time.monotonic() + 2.5
                self.status.set("Move your hand off the writing for a moment...")
            return
        self.finish_wait = None
        workflow, self.writing_workflow = self.writing_workflow, None
        if self.writing.tip_count() >= 5:
            self.writer_rotation = selected.rotation  # Remembered for snapshots, where no writing is followed.
            self.save_settings()  # Also keeps the pen's learned colour for the next session.
        self.page = self.frame.copy()
        self.preview_live = False
        self.show_selection(selected.page)
        self.debug_record(f"recording-{workflow}", selected, reason=self.writing.finish_reason)
        self.generate_selection(workflow, selected, recorded=True)
        if selected.end_marked:
            self.status.set(f"End circle detected. Reading the writing session: {paper_cloud.status()}...")

    def read_while_writing(self, now):
        """During a recording, read the strokes so far whenever the pen pauses.

        The reading shows up as a pop-up while you write, and when you finish with the same
        strokes it is used as is, so the result follows almost at once."""
        live = self.live
        try:
            key, reading = self.live_results.get_nowait()
        except queue.Empty:
            pass
        else:
            live["key"], live["reading"] = key, reading
            if reading.source and self.writing_workflow and not self.busy:
                self.status.set(f"Reading so far: {' '.join(reading.source.split())[:110]}")
        thread = live["thread"]
        if self.writing_workflow == "image":
            return  # A drawing for ChatGPT is not read at all.
        if (self.busy or self.writing.reference is None or (thread is not None and thread.is_alive())
                or now - live["at"] < 0.8):
            return
        tips = self.writing.tip_count()
        if tips == live["tips"] or tips < 8 or now - self.writing.last_tip_time() < 0.5:
            return  # Nothing new, too little, or the pen is still moving.
        live["tips"], live["at"] = tips, now
        selected = self.writing._select_recording(self.frame, None, follow_sheet=False)  # Fast preview.
        if selected is None or not (ink_usable(selected) or trace_usable(selected.strokes)):
            return
        key = (_signature(selected), self.target, self.writing_workflow)
        if key == live["key"]:
            return
        live["pending"] = key
        arguments = (selected, self.target, self.language, self.writing_workflow, self.args.model)
        folder = self.git_repo if self.target == "Git" else self.report_folder if self.target == "LaTeX" else (
            self.saved if self.saved and self.saved.is_dir() else self.git_repo)
        hints = project_hints(self.target, folder)

        def work():
            try:
                # The preview while writing stays local: it runs at every pause of the pen.
                self.live_results.put((key, read_recording(*arguments, hints=hints, cloud=False)))
            except (OSError, RuntimeError, ValueError, cv2.error):
                pass  # The final reading reports problems; a failed preview is just skipped.

        live["thread"] = Thread(target=work, daemon=True)
        live["thread"].start()

    def finish_after_wait(self, now):
        """Second attempt after the key press: the recording if it can be read now, else a photo of
        the area where the hand wrote."""
        if now < self.finish_wait and self.writing.hand_over_page(now):
            return
        self.finish_wait = None
        workflow = self.writing_workflow
        selected = self.writing.select(self.frame, prefer_recent=True)
        if selected is not None:
            self.finish_writing(selected)
            return
        tips, down = self.writing.summary()
        bounds = self.writing.motion_bounds((self.frame.shape[1], self.frame.shape[0]))
        self.writing_workflow = None
        self.page, self.roi, self.preview_live = self.frame.copy(), bounds, False
        chosen = self.capture_selection(workflow)
        self.roi = None  # The region belongs to this one reading; the next key press records again.
        if chosen is None:
            return
        self.generate_selection(workflow, chosen)
        where = "the area where you wrote" if bounds else "the whole page"
        self.status.set(f"No pen strokes were registered ({tips} pen-tip samples, {down} on ink), so a photo of "
                        f"{where} is being read...")

    def convert(self, workflow):
        if self.busy or self.target is None:
            return
        pasted_plot = workflow == "graph" and self.pasted
        if not pasted_plot and self.camera is None and self.page is None:
            self.start_camera()
        if not pasted_plot and self.preview_live and self.frame is None:
            self.pending_capture = self.menu, workflow
            self.status.set("Camera starting. Conversion will begin when the preview is ready.")
            return
        if pasted_plot:
            selected = self.result.get("1.0", "end-1c").strip()
        else:
            selected = self.capture_selection(workflow)
            if selected is None:
                return
        self.writing_workflow = None
        self.generate_selection(workflow, selected)

    def generate_selection(self, workflow, selected, *, recorded=False):
        self.clear_draft()
        self.busy = True
        self.workflow = workflow
        if isinstance(selected, str):
            self.status.set("Rendering the plot...")
        elif workflow == "image":
            self.status.set("Taking the drawing for ChatGPT...")
        elif recorded:
            self.status.set(f"Reading the writing session: {paper_cloud.status()}...")
        else:
            self.status.set({"pointed": "Reading the writing at the pen tip...",
                             "single": "Reading the writing on the page...",
                             "paper": "Several pieces of writing and no pen pointing: reading the whole sheet..."}
                            .get(getattr(self, "snapshot_reason", None),
                                 f"Reading the page and generating a draft: {paper_cloud.status()}..."))
            self.snapshot_reason = None
        Thread(target=self._generate, args=(selected, self.target, self.language, workflow, recorded),
               daemon=True).start()

    def sharpest_frame(self, within=1.2):
        """The sharpest frame of the last ``within`` seconds (variance of the Laplacian), else the latest."""
        now = time.monotonic()
        best, chosen = None, self.frame
        for when, frame in self.recent_frames:
            if now - when > within or frame.shape != self.frame.shape:
                continue
            small = cv2.cvtColor(cv2.resize(frame, None, fx=0.25, fy=0.25, interpolation=cv2.INTER_AREA),
                                 cv2.COLOR_BGR2GRAY)
            sharpness = float(cv2.Laplacian(small, cv2.CV_32F).var())
            if best is None or sharpness > best:
                best, chosen = sharpness, frame
        return chosen

    def capture_selection(self, workflow):
        if self.camera_failed:
            self.status.set("Restart the camera and wait for new RGB frames, or open a photo.")
            return None
        live = self.frame is not None and (self.preview_live or (self.roi is None and not self.page_pinned))
        if live:
            # The sharpest of the last moments: pressing a key shakes the desk and the camera with it.
            self.page = self.sharpest_frame().copy()
            self.roi = None
            self.preview_live = False
        if self.page is None:
            self.status.set("Capture the paper or open a photo first.")
            return
        if float(self.page[::8, ::8].mean()) < 12:
            # Never send a black picture to the reader: it answers with an explanation instead of a reading.
            self.status.set("The camera picture is black. Open the lens cover or add light, then press the key again.")
            self.key_result = ("error", time.monotonic())
            return None
        selected = self.selected_page(narrow=workflow in ("graph", "default", "equation"),
                                      marked_only=workflow == "copy", pointed=live)
        self.debug_record(f"snapshot-{workflow}", selected,
                          reason=getattr(self, "snapshot_reason", None) if live else "still")
        if selected.page.reason == "multiple circles":
            self.status.set("Several circles found. Drag around the item you want to use.")
            return
        if workflow == "copy" and selected.page.kind == "full":
            self.status.set("Draw a dashed oval around one equation, or drag a selection in the preview.")
            return None
        return selected

    def _generate(self, selected, target, language, workflow, recorded=False):
        self.reading_path = None
        self.engines = self.plot_script = self.plan_actions = None
        paper_cloud.take_trail()  # Which engine reads and generates is recorded per worker thread.
        page = evidence = None
        if getattr(selected, "page", None) is not None:
            # A snapshot is read upright for the writer (see selected_page); a recording turns its own ink.
            turn = getattr(selected, "rotation", 0) if not recorded else 0
            orchestrated = paper_cloud.enabled() and paper_orchestrator.key_for(target, workflow, language)
            if not recorded and workflow in self.CONFIRMED and not orchestrated:
                # (The orchestrator reads a turned picture as it is, so no reading is spent on finding the turn.)
                turn = self._upright_turn(selected.page.image, turn, target, language, workflow)
            page = rotate_image(selected.page.image, turn)
            evidence = rotate_image(selected.evidence, turn) if selected.evidence is not None else None
        try:
            figure = None
            planned = PaperWorkbench._orchestrated(self, selected, target, language, workflow, recorded, page)
            if planned is not None:
                draft, figure = planned
            elif workflow == "image":
                # Nothing is read: the picture of the drawing itself goes to ChatGPT (see finish_image).
                self.sketch = rotate_image(selected.page.image, getattr(selected, "rotation", 0))
                draft = Draft("", IMAGE_REQUEST, (), ".md")
            elif workflow == "copy":
                # The reading starts with "KIND: <kind>"; a copied plot is rendered here, off the UI thread.
                source, notes = _read_tracked(self, selected, target, language, "copy", self.args.model)
                kind, content = paper_clipboard.parse_reading(source)
                if kind == "plot" and content:
                    try:
                        figure, domain = graph(content, ROOT / "figures")
                    except ValueError:
                        try:
                            figure, domain = graph(paper_clipboard.latex_to_plain(content), ROOT / "figures")
                        except ValueError as error:
                            notes = (*notes, f"Could not render the plot: {error}")
                    if figure is not None:
                        notes = (*notes, f"Plotted over x = {domain[0]:g} to {domain[1]:g}.")
                draft = Draft(source, content, notes, ".txt")
            elif workflow in ("table", "diagram", "git_auto", "flow", "gantt", "forge_issue", "forge_request"):
                if recorded or workflow in ("table", "diagram", "git_auto"):
                    source, notes = _read_tracked(self, selected, target, language, workflow, self.args.model)
                else:
                    source, notes = read_still(page, target, language, workflow, self.args.model,
                                               evidence=evidence)
                source = self._checked(source, workflow, selected, target, language)
                draft = Draft(source, source, notes,
                              {"table": ".tex", "flow": ".md", "gantt": ".md"}.get(workflow, ".txt"))
            elif workflow == "graph":
                if isinstance(selected, str):
                    source, notes = selected, ()
                elif recorded:
                    source, notes = _read_tracked(self, selected, target, language, workflow, self.args.model)
                else:
                    request = "Preserve the exact equation y = f(x), the written domain, or plot call. Use ^ for powers."
                    source, notes = read_page_source(page, self.args.model, request,
                                                     evidence=evidence, key=(target, language, "graph"))
                source = self._checked(source, workflow, selected, target, language)
                # A plotting program in the app's language, saved and run: its figure is the plot.
                draft, figure, self.plot_script = make_plot(source, target, language, self.args.model,
                                                            ROOT / "drafts" / self.menu)
                draft = replace(draft, notes=(*notes, *draft.notes))
            elif workflow in paper_git.CONTEXTS:
                if recorded:
                    source, notes = _read_tracked(self, selected, "Git", language, workflow, self.args.model)
                else:
                    request = paper_git.CONTEXTS[workflow][0] + " Transcribe the writing literally."
                    source, notes = read_page_source(page, self.args.model, request,
                                                     evidence=evidence, key=("Git", language, workflow))
                draft = (convert_transcription(source, notes, "VS Code", language, self.args.model, "project")
                         if workflow == "git_repo" else Draft(source, source, notes, ".txt"))
            elif recorded or (workflow in ("default", "equation") and target in ("VS Code", "MATLAB", "LaTeX")):
                if recorded:
                    source, notes = _read_tracked(self, selected, target, language, workflow, self.args.model)
                else:
                    # A snapshot: read first, so the reading can be checked before code is generated from it.
                    request = writing_context(target, workflow, language).prompt + " Transcribe the writing literally."
                    source, notes = read_page_source(page, self.args.model, request,
                                                     evidence=evidence, key=(target, language, workflow))
                source = self._checked(source, workflow, selected, target, language)
                draft = convert_transcription(source, notes, target, language, self.args.model, workflow)
            else:
                # A circuit is run once in LTspice before it opens; an error goes back for one corrected reading.
                draft = convert_page(page, target, language, self.args.model, workflow, evidence=evidence,
                                     check=self._circuit_check if target in ("LTspice", "KiCad") else None)
            if workflow in ("default", "equation") and draft.source and target in ("VS Code", "MATLAB", "LaTeX"):
                checks = check_derivation(draft.source)
                if checks:
                    draft = replace(draft, notes=(*draft.notes, *checks))
            if (workflow == "default" and target in ("VS Code", "MATLAB") and draft.content and not draft.files
                    and draft.source):
                # The handwriting goes in as a comment, so the editor's inline AI sees what was meant.
                mark = "%" if target == "MATLAB" else "#" if draft.extension == ".py" else "//"
                header = "\n".join(f"{mark} {line}" for line in ["From paper:", *draft.source.splitlines()][:14])
                draft = replace(draft, content=f"{header}\n\n{draft.content}")
            self.engines = paper_cloud.describe(paper_cloud.take_trail())
            generated = [part for part in self.engines.split(", ") if part.startswith("generated")]
            if generated:
                draft = replace(draft, notes=(*draft.notes, generated[0][0].upper() + generated[0][1:] + "."))
            self.results.put((draft, figure, workflow))
        except (OSError, RuntimeError, ValueError, cv2.error) as error:
            self.results.put(error)

    @staticmethod
    def _circuit_check(content):
        """Run a generated schematic once in LTspice's batch mode (about a second): (ok, lines), or None
        when LTspice is not installed. A circuit that cannot be simulated is then reported with the result,
        and corrected once, instead of failing after Simulate."""
        import tempfile

        try:
            with tempfile.TemporaryDirectory(prefix="paper-circuit-") as folder:
                path = Path(folder) / "paper_circuit.asc"
                path.write_text(content, encoding="utf-8", newline="\n")
                return paper_ltspice.simulate(path, timeout=30)
        except OSError:
            return None

    @staticmethod
    def _upright_turn(image, preferred, target, language, workflow):
        """Which way up a snapshot is, from the writing itself.

        The writer's side (remembered from recordings and the pen) is only a guess for writing that is
        already on the page: the sheet may have been turned since. Lines run along the longer side of
        the crop, so only two turns are possible; the expected one is read first, and the opposite one
        only when that reading fails the key's check. The reading is cached, so it costs nothing more."""
        height, width = image.shape[:2]
        flat = (0, 180) if width >= height else (90, 270)
        first = preferred if preferred in flat else flat[0] if preferred in (0, 90) else flat[1]
        for turn in (first, (first + 180) % 360):
            reading = read_static(rotate_image(image, turn), target, language, workflow)
            if reading is not None and validate_reading(target, workflow, reading.source, language)[0]:
                return turn
        return first

    def _orchestrated(self, selected, target, language, workflow, recorded, page):
        """One cloud request for the key press: read the capture and write the result (paper_orchestrator).

        Returns (draft, figure), or None when this key is not orchestrated or the cloud is off or failed,
        so the step-by-step pipeline below runs (it is also the offline path). The reading is shown for
        the usual check, with a line about what will be made. A reading the user corrects is converted
        again from the corrected text: exactly and at once where a direct conversion exists, otherwise by
        a text-only request. The answer is checked before it is used: Python must compile, a plot program
        must run, LaTeX must parse; a failed check goes back once for a fix."""
        key = paper_orchestrator.key_for(target, workflow, language)
        if key is None or isinstance(selected, str) or not paper_cloud.enabled():
            return None
        try:
            pictures = (paper_conversion._selection_pictures(selected) if recorded
                        else [paper_conversion._picture_bytes(page)])
        except (AttributeError, ValueError, cv2.error):
            return None
        repo, saved = getattr(self, "git_repo", None), getattr(self, "saved", None)
        folder = getattr(self, "report_folder", None) if target == "LaTeX" else (
            saved if saved and saved.is_dir() else repo)
        hints = project_hints(target, folder)
        plan = paper_orchestrator.plan(key, pictures, hints)
        if plan is None or not plan.transcription or plan.kind == "unreadable":
            return None
        box = {"plan": plan}

        def again(previous):
            second = paper_orchestrator.plan(key, pictures, hints, previous=previous)
            if second is None or not second.transcription:
                return None
            box["plan"] = second
            return second.transcription

        checked = self.confirm_reading(plan.transcription, workflow, again, summary=lambda: box["plan"].summary)
        if checked is None:
            raise ReadingCancelled("Cancelled: nothing was generated. Press the key to write again.")
        plan = box["plan"]
        self.reading_path = "cloud"
        model = self.args.model
        if checked.strip() != plan.transcription.strip() or not plan.content or "stage" in plan.actions:
            # The user corrected the reading (or nothing usable was written for it): the result is made anew
            # from the checked text. Direct conversions are exact and instant; the rest is a text-only request.
            if workflow == "graph":
                draft, figure, self.plot_script = make_plot(checked, target, language, model,
                                                            ROOT / "drafts" / self.menu)
                return draft, figure
            return convert_transcription(checked, plan.notes, target, language, model, workflow), None
        self.plan_actions = plan.actions
        notes, content = plan.notes, plan.content
        if workflow == "graph":
            draft, figure, self.plot_script = make_plot(
                plan.transcription, target, language, model, ROOT / "drafts" / self.menu,
                program=content if target == "MATLAB" or "run_python" in plan.actions else None)
            return replace(draft, notes=(*notes, *draft.notes)), figure
        problem = None
        if target == "LaTeX":
            content = paper_orchestrator.tidy_latex(content)
            ok, reason = validate_reading(target, workflow, content, language)
            problem = None if ok else reason
            content = "\\[\n" + content + "\n\\]"
        elif key.filename.endswith(".py"):
            ok, error = paper_conversion.compiles(content)
            problem = None if ok else f"Python syntax error near line {error.lineno}: {error.msg}"
        elif target == "MATLAB":
            from paper_context import check_code

            problem = check_code(content, "MATLAB")
        if problem:
            fixed = paper_orchestrator.plan(key, hints=hints, text=plan.transcription, problem=(plan.content, problem))
            if fixed is None or not fixed.content:
                return convert_transcription(plan.transcription, notes, target, language, model, workflow), None
            content = fixed.content
            if target == "LaTeX":
                content = "\\[\n" + paper_orchestrator.tidy_latex(content) + "\n\\]"
            notes = (*notes, f"The first result failed a check ({problem}) and was corrected.")
            if key.filename.endswith(".py"):
                content, repaired = paper_conversion.repair_python(content)
                notes = (*notes, *repaired)
        return Draft(plan.transcription, content.rstrip("\n") if target == "LaTeX" else content, notes,
                     Path(key.filename).suffix), None

    def _checked(self, source, workflow, selected, target, language):
        """The reading after the user checked it (see confirm_reading). Raises ReadingCancelled on Cancel."""
        again = None
        if not isinstance(selected, str):
            def again(previous):
                return read_again(selected, target, language, workflow, self.args.model, previous)
        checked = self.confirm_reading(source, workflow, again)
        if checked is None:
            raise ReadingCancelled("Cancelled: nothing was generated. Press the key to write again.")
        return checked

    def save(self):
        if not self.draft or not self.result.get("1.0", "end-1c").strip():
            self.status.set("Choose Read page before saving.")
            return None
        if self.draft.files:
            destination = self._dialog(filedialog.askdirectory, title="Save a new project folder inside")
            if not destination:
                return None
            try:
                language = self.language if self.target == "VS Code" else self.target
                self.saved = save_project(self.draft.files, language, destination)
            except (OSError, ValueError) as error:
                self.status.set(str(error))
                return None
            if self.target == "LaTeX":
                self.report_folder = self.saved
            self.status.set(f"Saved {len(self.draft.files)} files to {self.saved}")
            return self.saved
        suffix = self.draft.extension
        path = self._dialog(filedialog.asksaveasfilename, defaultextension=suffix, initialfile="paper_draft" + suffix,
                            filetypes=[(f"{self.target} draft", "*" + suffix)])
        if not path:
            return None
        path = Path(path)
        if path.suffix.lower() != suffix:
            self.status.set(f"Use the {suffix} extension for this target.")
            return None
        try:
            with path.open("x", encoding="utf-8", newline="\n") as output:
                output.write(self.result.get("1.0", "end-1c").rstrip() + "\n")
        except OSError as error:
            self.status.set(str(error))
            return None
        self.saved = path
        self.status.set(f"Saved {path}")
        return path

    def open_draft(self):
        path = self.saved or self.save()
        if path is None:
            return
        self.open_path(path)

    def open_path(self, path):
        if getattr(self, "project_of", None) is not None:
            self.project_of[self.menu] = path
        try:
            if self.target == "KiCad":
                program_files = Path(os.environ.get("ProgramFiles", "C:/Program Files"))
                editors = sorted(program_files.glob("KiCad/*/bin/eeschema.exe"), reverse=True)
                editor = shutil.which("eeschema.exe") or (editors[0] if editors else None)
                if not editor:
                    raise OSError("KiCad was not found. Install KiCad or open the saved .sch on another PC.")
                subprocess.Popen([str(editor), str(path)])
            elif self.target == "LTspice":
                paper_ltspice.open_ltspice(path)
            elif self.target == "MATLAB":
                path = open_matlab(path)
            elif self.target == "LaTeX" and path.is_dir() and (path / "main.tex").is_file():
                open_code(path / "main.tex")
            else:
                open_code(path)
            self.status.set(f"Opened {path}")
            return True
        except OSError as error:
            self.status.set(f"Saved {path}. Could not open it: {error}")
            return False

    def open_generated(self):
        try:
            language = self.language if self.target == "VS Code" else self.target
            self.saved = save_generated(self.draft, language, ROOT / "drafts" / self.menu)
        except (OSError, ValueError) as error:
            self.status.set(f"Could not save the generated draft: {error}")
            return
        if self.target == "LaTeX" and self.saved.is_dir():
            self.report_folder = self.saved
        self.open_path(self.saved)

    def open_circuit(self):
        """Save the generated schematic under drafts and open it in LTspice or KiCad. What the reader
        assumed (a default value, a ground it added) is named in the status line, as there is no check
        window for drawings."""
        try:
            self.saved = write_new_file(ROOT / "drafts" / self.menu, "paper_circuit" + self.draft.extension,
                                        self.result.get("1.0", "end-1c").rstrip() + "\n")
        except OSError as error:
            self.status.set(f"Could not save the schematic: {error}")
            return
        routine = ("Netlist", "*", "Check", "LTspice", "Every part", "Net ", "Read in", "The first reading", "Open in")
        assumed = [note for note in self.draft.notes if note and not note.startswith(routine)][:2]
        tail = f" Assumed: {' '.join(assumed)}"[:260] if assumed else ""
        if self.open_path(self.saved):
            self.status.set((f"Opened {self.saved.name} in LTspice. Press Simulate to run it."
                             if self.target == "LTspice" else f"Opened {self.saved.name} in KiCad.") + tail)
        elif tail:
            self.status.set(self.status.get() + tail)

    def simulate(self):
        """Run the current schematic or netlist in LTspice and report what the log says."""
        if self.busy:
            return
        path = self.saved if self.saved and self.saved.suffix.lower() in (".asc", ".cir", ".net") else None
        if path is None:
            if not self.draft or not self.result.get("1.0", "end-1c").strip():
                self.status.set("Draw a circuit and press Schematic first, or paste a netlist.")
                return
            try:
                path = write_new_file(ROOT / "drafts" / "ltspice", "paper_circuit" + self.draft.extension,
                                      self.result.get("1.0", "end-1c").rstrip() + "\n")
            except OSError as error:
                self.status.set(f"Could not save the schematic: {error}")
                return
        self.busy = True
        self.workflow = "ltspice_sim"
        self.status.set(f"Simulating {path.name} in LTspice...")
        Thread(target=self._simulate, args=(path, self.draft), daemon=True).start()

    def _simulate(self, path, draft):
        try:
            ok, lines = paper_ltspice.simulate(path)
            if ok:
                paper_ltspice.open_ltspice(path, run=True)
        except OSError as error:
            self.results.put(error)
            return
        self.simulated = path, ok, lines
        self.results.put((draft, None, "ltspice_sim"))

    def copy(self, text):
        self.paper_clipboard = None
        self.window.clipboard_clear()
        self.window.clipboard_append(text)

    def open_plot(self):
        if self.figure:
            try:
                os.startfile(self.figure.with_suffix(".png"))
            except OSError as error:
                self.status.set(str(error))

    def copy_clip(self, clip):
        """Put a typed clip on the Windows clipboard and remember its kind for Paste in any app."""
        self.copy(clip.text)
        self.paper_clipboard = clip
        try:
            paper_clipboard.save(clip)
        except OSError as error:
            self.status.set(f"Copied {clip.kind}, but the typed clipboard was not saved: {error}")
            return False
        return True

    def draft_clip(self, text, selection=False):
        """Describe the current draft, or the text selected in it, as a typed clip."""
        source = self.draft.source if self.draft else ""
        if self.figure is not None:
            return paper_clipboard.Clip("plot", source or str(self.figure), self.target or "",
                                        files=[self.figure, self.figure.with_suffix(".png")])
        kind = None
        if not selection and self.draft:
            kind = {"equation": "equation", "graph": "equation", "table": "table", "idea": "text"}.get(self.workflow)
            if self.workflow == "default" and self.target in ("VS Code", "MATLAB"):
                kind = "code"
            if (kind == "table" or self.workflow == "graph") and source:
                text = source  # The reading, not the table markup or the plotting program made from it.
            elif self.workflow == "copy" and source:
                kind, text = paper_clipboard.parse_reading(source)
        if kind is None:
            kind, text = paper_clipboard.parse_reading(text)
        return paper_clipboard.Clip(kind, text, self.target or "")

    def finish_copy(self, draft, figure):
        kind, content = paper_clipboard.parse_reading(draft.source)
        if not content:
            self.status.set("Nothing readable inside the oval. Write a little larger or drag around it.")
            return
        files = [figure, figure.with_suffix(".png")] if figure else []
        clip = paper_clipboard.Clip(kind, content, self.target or "", files=files)
        if figure:
            self.plot_preview.pack(side="left", padx=4)
        if self.copy_clip(clip):
            self.status.set(f"Copied {kind} from paper. Open LaTeX, VS Code, MATLAB, ChatGPT, or Git and press "
                            "Paste.")

    def copy_draft(self, capture=True):
        if self.figure is not None:
            if self.copy_clip(self.draft_clip("")):
                self.status.set("Copied plot. Paste it in LaTeX (figure), VS Code or MATLAB (plot script), "
                                "or ChatGPT.")
            return True
        if capture and self.preview_live:
            self.convert("copy")
            return False
        if (capture and self.frame is not None and not self.page_pinned and not self.result.tag_ranges("sel")
                and select_paper_region(self.frame, allow_dashed=True).kind == "circle"):
            self.convert("copy")
            return False
        selection = bool(self.result.tag_ranges("sel"))
        draft = self.result.get("sel.first", "sel.last") if selection else (
            self.result.get("1.0", "end-1c").strip())
        if not draft:
            if capture:
                self.convert("copy")
            return False
        clip = self.draft_clip(draft, selection)
        if self.copy_clip(clip):
            self.status.set(f"Copied {clip.kind}.")
        return True

    def paste(self):
        try:
            text = self.window.clipboard_get().strip()
        except tk.TclError:
            text = ""
        clip = paper_clipboard.load()
        if self.target == "Git" or (clip is not None and clip.matches(text)
                                    and self.target in paper_clipboard.RENDERERS):
            if self.target == "Git" and (clip is None or not clip.matches(text)):
                clip = paper_clipboard.Clip(*paper_clipboard.parse_reading(text), "Windows") if text else None
            if clip is None:
                self.status.set("Copy a commit message or a todo list first.")
                return
            try:
                self.paste_clip(clip)
            except (OSError, RuntimeError, ValueError, SyntaxError, paper_git.GitError) as error:
                self.status.set(f"Could not paste the copied {clip.kind}: {error}")
            return
        if self.target == "LaTeX":
            try:
                literal = latex_equation(text)
                if literal is not None:
                    self.paste_equation(literal)
                    return
                if self.paste_figure(text):
                    return
            except (OSError, RuntimeError, ValueError) as error:
                self.status.set(str(error))
                return
        if not text:
            self.status.set("Copy text or a figure from the source app first.")
            return
        extension = LANGUAGES[self.language] if self.target == "VS Code" else DIRECT_TARGETS[self.target]
        if self.target == "LTspice":
            extension = ".cir"  # Pasted text is a SPICE netlist; drawn circuits are .asc schematics.
        if self.target == "ChatGPT":
            text = "Help me review and develop this material. Flag any assumptions or missing context.\n\n" + (
                fenced_code(text))
        self.show_draft(Draft("", text, (), extension))
        self.pasted = True
        self.status.set("Pasted into the workbench. Review it before opening or copying.")

    def paste_clip(self, clip):
        """Paste a typed clip the way the target app expects it (see the copy/paste table in the guide)."""
        copied = paper_clipboard.describe(clip)
        if self.target == "Git":
            self.paste_git(clip)
            return
        if self.target == "LaTeX" and clip.kind in ("plot", "diagram") and clip.image() is not None:
            self.paste_figure_with_source(clip, copied)
            return
        rendering = paper_clipboard.render(clip, self.target, self.language)
        message = f"Pasted {copied} as {rendering.description}"
        if self.target == "ChatGPT":
            self.show_draft(Draft(clip.content, rendering.content, (), ".md"))
            self.pasted = True
            self.status.set(f"{message}. Press ChatGPT to copy it and open the chat.")
            return
        if self.target == "LaTeX":
            self.place_latex(rendering, message, source=clip.content)
            return
        folder = ROOT / "drafts" / self.menu
        project = self.saved if self.target == "VS Code" and self.saved and self.saved.is_dir() else None
        if self.target == "VS Code" and clip.kind == "code" and project is None and self.git_repo:
            project = self.git_repo if self.git_repo.is_dir() else None
        path = write_new_file(project or folder, rendering.filename, rendering.content, unique_folder=project is None)
        self.show_draft(Draft(clip.content, rendering.content, (str(path),), path.suffix))
        self.saved = path
        opened = self.open_path(path)
        self.status.set(f"{message}: {path.name}" + (" (opened)." if opened else f". {self.status.get()}"))

    def paste_figure_with_source(self, clip, copied):
        """Paste a copied plot into the LaTeX project as what it was made from plus the figure.

        A plot generated from a handwritten equation carries that equation, and one made from
        code carries the code. Both go into the text in one fragment: the equation (or a code
        listing) followed by the figure, whose file is copied into ``figures/``. A figure
        without such a source is inserted on its own.
        """
        project = self.active_report()
        if project is None:
            return
        figure = import_figure(clip.image(), project)
        content, packages, parts = clip.content.strip(), ["graphicx"], ["the figure"]
        is_path = "\n" not in content and Path(content.strip('"')).suffix.lower() in (".pdf", ".png", ".jpg", ".jpeg")
        text = ""
        if content and not is_path and clip.kind == "plot":
            from paper_context import check_code, check_equation

            if check_equation(content) is None:
                text = "\\begin{equation}\n  " + paper_clipboard.equation_body(content) + "\n\\end{equation}"
                parts.insert(0, "the equation")
            elif clip.source_app in ("VS Code", "MATLAB") and check_code(
                    content, "MATLAB" if clip.source_app == "MATLAB" else "Python") is None:
                language = "Matlab" if clip.source_app == "MATLAB" else "Python"
                body = content.replace("\\end{lstlisting}", "\\end {lstlisting}")
                text = f"\\begin{{lstlisting}}[language={language}]\n{body}\n\\end{{lstlisting}}"
                packages.append("listings")
                parts.insert(0, "the code")
        fragment = f"{text}\n\n{figure}" if text else figure
        rendering = paper_clipboard.Rendering(" and ".join(parts), fragment, folder="paperbridge",
                                              packages=tuple(packages))
        self.place_latex(rendering, f"Pasted {copied} as {rendering.description}", source=clip.content, ask=False)

    def place_latex(self, rendering, message, *, source="", ask=True, name=None):
        """Insert a LaTeX fragment into the selected project, or save a standalone draft if there is none."""
        project = self.report_folder if self.report_folder and self.report_folder.is_dir() else None
        if project is None and ask:
            project = self.choose_report()
        if project is None:
            document = paper_clipboard.latex_draft_document(rendering)
            path = write_new_file(ROOT / "drafts" / "latex", "paper_draft.tex", document)
            self.show_draft(Draft(source, document, (str(path),), ".tex"))
            self.saved = path
            self.open_path(path)
            self.status.set(f"{message} in a standalone draft {path.name}. Use Browse to choose a project.")
            return path
        if rendering.folder == "tables" and name is None:
            name = next_numbered(project / "tables", "table")
        snippet, reference, entry = import_snippet(rendering.content, project, rendering.folder, name,
                                                   rendering.packages)
        self.show_draft(Draft(source, rendering.content, (reference,), ".tex"))
        self.saved = snippet
        if entry:
            self.open_path(entry)
            self.status.set(f"{message} in {snippet.relative_to(project).as_posix()}, included from {entry.name}.")
        else:
            self.copy(reference)
            self.open_path(snippet)
            self.status.set(f"{message} in {snippet.name}. The \\input line is on the clipboard.")
        return snippet

    def finish_table(self, draft):
        rows = paper_clipboard.table_rows(draft.source)
        if not rows:
            self.status.set("No table found. Separate the columns with vertical lines and write one row per line.")
            return
        project = self.report_folder if self.report_folder and self.report_folder.is_dir() else None
        name = next_numbered(project / "tables", "table") if project else None
        label = f"tab:{name.removesuffix('.tex')}" if name else None
        rendering = paper_clipboard.Rendering("a booktabs table", paper_clipboard.latex_table(draft.source, label=label),
                                              folder="tables", packages=("booktabs",))
        try:
            self.place_latex(rendering, f"Converted a {len(rows)}-row table", source=draft.source, ask=False,
                             name=name)
        except (OSError, ValueError) as error:
            self.status.set(f"Could not save the table: {error}")

    def paste_git(self, clip):
        repo = self.active_repo()
        if repo is None:
            return
        if clip.kind == "todo":
            items = [item for item, done in paper_clipboard.todo_items(clip.content) if not done]
            path = repo / "TODO.md"
            existing = path.read_text(encoding="utf-8") if path.is_file() else "# TODO\n"
            new = [item for item in items if f"- [ ] {item}" not in existing]
            if not new:
                self.status.set("Every copied todo item is already in TODO.md.")
                return
            with path.open("w", encoding="utf-8", newline="\n") as output:
                output.write(existing.rstrip("\n") + "\n" + "".join(f"- [ ] {item}\n" for item in new))
            body = "\n".join(f"- {item}" for item in new)
            short, _ = paper_git.commit_file(repo, "TODO.md", f"Add {len(new)} TODO items from paper", body)
            self.status.set(f"Pasted todo list: added {len(new)} items to TODO.md and committed {short}.")
        elif clip.kind == "diagram" and paper_git.parse_diagram(clip.content)[0]:
            self.run_git_diagram(clip.content)
        elif clip.kind in ("text", "equation", "code") and len(clip.content.splitlines()) <= 12:
            short, message = paper_git.commit(repo, clip.content)
            self.status.set(f"Pasted {clip.kind} as commit message: committed {short} {message.splitlines()[0]}")
        else:
            self.status.set(f"Git paste takes text (a commit message) or a todo list, not a {clip.kind}.")

    def browse_report(self):
        folder = self.choose_report()
        if folder is not None:
            self.open_path(folder)
            self.status.set(f"Selected LaTeX project {folder}. Paste and Table now save into it.")

    def choose_report(self):
        options = {"initialdir": str(self.report_folder)} if self.report_folder else {}
        choice = self._dialog(filedialog.askdirectory, title="Choose the LaTeX project folder", **options)
        if not choice:
            return None
        folder = Path(choice)
        if not any(folder.glob("**/*.tex")):
            self.status.set("Choose a folder containing a .tex project.")
            return None
        self.report_folder = folder
        self.save_settings()
        self.status.set(f"Selected LaTeX project {folder}")
        return folder

    def active_report(self):
        return self.report_folder if self.report_folder and self.report_folder.is_dir() else self.choose_report()

    def paste_equation(self, content):
        project = self.active_report()
        if project is None:
            return
        snippet, reference, entry = import_equation(content, project)
        self.show_draft(Draft("", content, (reference,), ".tex"))
        self.saved = snippet
        self.copy(reference)
        if entry:
            self.open_path(entry)
            self.status.set(f"Equation inserted into {entry.name} in {project}.")
        else:
            self.open_path(snippet)
            self.status.set(f"Equation saved in {project}. The include command is on the clipboard.")

    def overleaf_fragment(self):
        """The LaTeX to hand to Overleaf: the current LaTeX draft, else the copied item typeset as Paste
        into LaTeX would. None when there is neither (then the key offers the ZIP upload)."""
        text = self.result.get("1.0", "end-1c").strip()
        if self.target == "LaTeX" and self.draft is not None and not self.draft.files and text:
            if self.workflow == "equation" and "\\begin{" not in text:
                return paper_clipboard.render(paper_clipboard.Clip("equation", text, "LaTeX"), "LaTeX").content
            if "\\begin{document}" in text and "\\end{document}" in text:
                # A standalone draft: only what is inside the document goes into the user's own document.
                text = text.split("\\begin{document}", 1)[1].rsplit("\\end{document}", 1)[0].strip()
            return text
        try:
            copied = self.window.clipboard_get().strip()
        except tk.TclError:
            copied = ""
        clip = paper_clipboard.load()
        if clip is not None and clip.matches(copied):
            try:
                if clip.kind == "plot":  # A figure file cannot be pasted as text; its equation can.
                    clip = paper_clipboard.Clip("equation", clip.content, clip.source_app)
                return paper_clipboard.render(clip, "LaTeX", self.language).content
            except (ValueError, SyntaxError):
                return None
        return None

    def open_overleaf(self):
        """Overleaf key: paste the current equation (or the copied item, as LaTeX) at the cursor of the
        Overleaf project that is open in the browser. Overleaf has no local interface, so the LaTeX goes
        on the clipboard and Ctrl+V is pressed in the Overleaf tab, and only there. Without an Overleaf
        project in front of a browser window, the project list opens and the LaTeX waits on the
        clipboard. With nothing drafted or copied, the key offers the ZIP upload of the local project."""
        fragment = self.overleaf_fragment()
        if not fragment:
            self.upload_overleaf()
            return
        try:
            paper_browser.copy_text(fragment)
        except OSError as error:
            self.status.set(f"Could not put the LaTeX on the clipboard: {error}")
            return
        self.paper_clipboard = None
        window = paper_browser.find_window(paper_browser.is_overleaf_editor)
        if window is None:
            webbrowser.open(paper_browser.OVERLEAF_PROJECTS)
            self.status.set("Overleaf opened. Open your project, click where the LaTeX should go, and press "
                            "Overleaf again (or Ctrl+V): it is on the clipboard.")
            return
        self.status.set("Pasting into the Overleaf project...")

        def work():
            pasted = paper_browser.paste_into(paper_browser.is_overleaf_editor, window)
            message = ("Pasted the LaTeX at the cursor in your Overleaf project." if pasted else
                       "Overleaf did not come to the front, so nothing was typed. The LaTeX is on the clipboard: "
                       "click in the Overleaf editor and press Ctrl+V.")
            self.window.after(0, self.status.set, message)

        Thread(target=work, daemon=True).start()

    def upload_overleaf(self):
        """Package the selected local LaTeX project as a ZIP for Overleaf's upload."""
        project = self.active_report()
        if project is None:
            return
        archive = self._dialog(filedialog.asksaveasfilename, title="Save the Overleaf upload",
                               defaultextension=".zip", initialfile=f"{project.name}.zip",
                               filetypes=[("Overleaf project", "*.zip")])
        if not archive:
            return
        try:
            bundle = package_latex(project, Path(archive))
            webbrowser.open("https://www.overleaf.com/project")
            self.status.set(f"Archive saved to {bundle}. In Overleaf, choose New project → Existing project (.zip).")
        except (OSError, ValueError) as error:
            self.status.set(str(error))

    def paste_figure(self, text):
        from PIL import Image, ImageGrab

        path = Path(text.strip('"'))
        if path.suffix.lower() in (".pdf", ".png", ".jpg", ".jpeg") and len(text) < 1024 and "\n" not in text:
            source = path
        else:
            image = ImageGrab.grabclipboard()
            if isinstance(image, Image.Image):
                import tempfile

                with tempfile.TemporaryDirectory() as folder:
                    source = Path(folder) / "clipboard.png"
                    image.save(source)
                    self.stage_figure(source)
                return True
            if isinstance(image, list):
                source = next((Path(name) for name in image if Path(name).suffix.lower() in
                               (".pdf", ".png", ".jpg", ".jpeg")), None)
            else:
                source = None
            if source is None:
                if text:
                    return False
                choice = self._dialog(filedialog.askopenfilename, title="Select an exported figure",
                                      filetypes=[("Figures", "*.pdf *.png *.jpg *.jpeg")])
                if not choice:
                    return True
                source = Path(choice)
        self.stage_figure(source)
        return True

    def stage_figure(self, source):
        folder = self.active_report()
        if folder is None:
            return
        snippet = import_figure(source, folder)
        self.show_draft(Draft("", snippet, (r"Load \usepackage{graphicx} in your LaTeX preamble.",), ".tex"))
        self.copy(snippet)
        self.status.set(f"Figure imported into {folder / 'figures'}. Paste the copied snippet into your report.")

    def chatgpt(self):
        copied = self.copy_draft(capture=False)
        # ChatGPT is the AI of this page: a brief that fits in a link is handed over with the link, so the
        # chat opens with it already asked. It stays on the clipboard for a longer one, or as a fallback.
        from urllib.parse import quote

        brief = self.result.get("1.0", "end-1c").strip() if copied else ""
        query = quote(brief, safe="")
        sent = bool(brief) and len(query) <= 6000
        if webbrowser.open(f"https://chatgpt.com/?q={query}" if sent else "https://chatgpt.com/") and copied:
            self.status.set("ChatGPT opened with the brief asked. It is also on your clipboard, in case the chat "
                            "opened empty." if sent else
                            "The brief is on your clipboard. Paste it in ChatGPT when ready.")

    def save_photo(self):
        if self.preview_live and self.frame is not None:
            self.set_page(self.frame.copy())
        if self.page is None:
            self.status.set("Capture or open a photo first.")
            return
        path = self._dialog(filedialog.asksaveasfilename, defaultextension=".png",
                            initialfile="paper.png", filetypes=[("PNG photo", "*.png")])
        if not path:
            return
        try:
            ok, data = cv2.imencode(".png", self.page)
            if not ok or Path(path).suffix.lower() != ".png":
                raise ValueError("Save the page with a .png filename.")
            with Path(path).open("xb") as output:
                output.write(data.tobytes())
            self.status.set(f"Saved {path}. Attach it in ChatGPT if useful.")
        except (OSError, ValueError, cv2.error) as error:
            self.status.set(str(error))

    def vscode_ai(self):
        source = self.source.get("1.0", "end-1c").strip()
        if not self.draft or not source:
            self.status.set("Read the page and check the transcription before opening VS Code AI.")
            return
        try:
            ask_vscode(source, self.draft.notes, self.language, self.workflow, ROOT / "handoffs", self.saved)
        except OSError as error:
            self.status.set(str(error))
            return
        self.status.set("Opened VS Code Chat with the reviewed transcription.")

    def finish_plot(self, draft, figure):
        """Plot key: the program that draws the plot is open in the app, with the figure it made."""
        script = getattr(self, "plot_script", None)
        if figure:
            self.plot_preview.pack(side="left", padx=4)
        if script is None:
            self.status.set("Plot ready. Use Copy to transfer it to LaTeX." if figure else "Review the draft.")
            return
        self.saved = script
        if self.target == "MATLAB":
            # A MATLAB that is started for this runs the script; one that is already open only shows it.
            running = matlab_running()
            self.project_of[self.menu] = script
            try:
                open_matlab(script, run=True)
                where = (f"{script.name} is open in MATLAB: press F5 there to run it" if running else
                         f"MATLAB is starting and runs {script.name}")
            except OSError as error:
                where = f"Saved {script}. Could not open it: {error}"
            self.status.set(f"Plot script ready. {where}. Copy takes the figure, for example into LaTeX.")
            return
        opened = self.open_path(script)
        if not figure:
            self.status.set(f"{script.name} is open in VS Code, but it did not produce a figure: "
                            f"{draft.notes[-1] if draft.notes else 'see the notes'}")
            self.key_result = ("error", time.monotonic())
            return
        try:
            open_code(figure.with_suffix(".png"))  # The figure the program made, next to its source.
        except OSError:
            pass
        self.status.set(f"Plot ready: {script.name} ran and drew {figure.with_suffix('.png').name}; both are open "
                        "in VS Code. Copy takes the figure, for example into LaTeX." if opened else
                        f"Plot ready, saved as {script}. Copy takes the figure, for example into LaTeX.")

    def finish_equation(self, draft):
        """LaTeX Equation key with a project selected: the equation goes into the document, like a table."""
        rendering = paper_clipboard.render(paper_clipboard.Clip("equation", draft.content, "LaTeX"), "LaTeX")
        try:
            self.place_latex(rendering, "Typeset the equation", source=draft.source, ask=False)
            if draft.notes:  # Keep the derivation check and the reading notes in view.
                self.notes.configure(text="\n".join((*draft.notes, self.notes.cget("text"))))
        except (OSError, ValueError) as error:
            self.status.set(f"Could not add the equation to the project: {error}")
            self.open_generated()

    def finish_plan(self, draft):
        """Schedule the written plan, render the Gantt chart, and open it. Copy then carries the figure."""
        items, problems = paper_plan.parse_plan(draft.source)
        if len(items) < 2:
            self.status.set("No plan read. Write one task per line with a duration, such as 'Prototype 3w'.")
            self.key_result = ("error", time.monotonic())
            return
        problems += paper_plan.schedule(items)
        fence = "`" * 3
        document = (f"# Project plan\n\n{paper_plan.summary(items)}\n\n{fence}mermaid\n"
                    f"{paper_plan.mermaid(items)}\n{fence}\n")
        try:
            path = write_new_file(ROOT / "drafts" / "plan", "project_plan.md", document)
            chart = paper_plan.render(items, path.with_name("gantt.png"))
        except (OSError, ValueError) as error:
            self.status.set(f"Could not save the plan: {error}")
            return
        self.show_draft(Draft(draft.source, document, (*draft.notes, *problems), ".md"))
        self.saved, self.figure = path, chart.with_suffix(".pdf")
        self.plot_preview.pack(side="left", padx=4)
        try:
            os.startfile(chart)
            open_code(path)
        except OSError:
            pass
        self.status.set(f"Gantt chart ready: {paper_plan.summary(items)}. Copy takes the chart, for example into "
                        "LaTeX as a figure; the Mermaid source is open in VS Code.")

    def finish_image(self, draft):
        """Image key: hand the drawing to ChatGPT with a request to generate a finished image from it.

        The picture goes on the Windows clipboard as an image and is saved as a PNG. ChatGPT opens, and
        once its window is in front the picture is pasted, the request is typed, and Enter sends it.
        Keys are pressed only while ChatGPT is the window in front; otherwise the picture stays on the
        clipboard and the status line says to paste it."""
        picture = getattr(self, "sketch", None)
        if picture is None or picture.size == 0:
            self.status.set("No drawing was captured. Draw it, then press Image again.")
            self.key_result = ("error", time.monotonic())
            return
        height, width = picture.shape[:2]
        if min(height, width) < 512:  # A small crop is enlarged, so the drawing is clear to ChatGPT.
            factor = min(4.0, 512 / max(1, min(height, width)))
            picture = cv2.resize(picture, None, fx=factor, fy=factor, interpolation=cv2.INTER_CUBIC)
        try:
            folder = ROOT / "drafts" / "chatgpt" / f"sketch-{time.strftime('%Y%m%d-%H%M%S')}"
            folder.mkdir(parents=True, exist_ok=True)
            ok, data = cv2.imencode(".png", picture)
            if not ok:
                raise OSError("the picture could not be encoded")
            (folder / "sketch.png").write_bytes(data.tobytes())
            self.saved = folder / "sketch.png"
            paper_browser.copy_image(picture)
        except (OSError, cv2.error) as error:
            self.status.set(f"Could not stage the drawing for ChatGPT: {error}")
            self.key_result = ("error", time.monotonic())
            return
        self.paper_clipboard = None
        webbrowser.open(paper_browser.CHATGPT)
        self.status.set("ChatGPT is opening with your drawing...")
        request, saved = draft.content, self.saved

        def work():
            time.sleep(1.5)  # The new tab takes the place of whatever the browser showed before.
            sent = paper_browser.paste_into(paper_browser.is_chatgpt, wait=12, settle=2.0)
            if sent:
                time.sleep(1.0)
                _, process, title = paper_browser.foreground()
                if paper_browser.is_chatgpt(process, title):
                    paper_system.type_text(request)
                    time.sleep(2.5)  # The picture uploads; Enter before that is ignored by ChatGPT.
                    _, process, title = paper_browser.foreground()
                    if paper_browser.is_chatgpt(process, title):
                        paper_system.press("enter")
            message = ("ChatGPT has your drawing and the request to generate an image from it. If it has not "
                       "started, press Enter there." if sent else
                       "ChatGPT did not come to the front, so nothing was typed. The drawing is on the clipboard "
                       f"(and saved as {saved}): click in ChatGPT's message box, press Ctrl+V, and ask for an "
                       "image; the request is in the draft panel.")
            self.window.after(0, self.status.set, message)

        Thread(target=work, daemon=True).start()

    def finish_flow(self, draft):
        """Save the drawn flowchart as a Mermaid block, open it in VS Code, and copy it as a typed diagram."""
        if not draft.source:
            self.status.set("No flowchart read. Draw the boxes a little larger with clear arrows.")
            return
        fence = "`" * 3
        document = f"# Flowchart from paper\n\n{fence}mermaid\n{draft.source.strip()}\n{fence}\n"
        try:
            path = write_new_file(ROOT / "drafts" / "chatgpt", "flowchart.md", document)
        except OSError as error:
            self.status.set(f"Could not save the flowchart: {error}")
            return
        self.show_draft(Draft(draft.source, document, draft.notes, ".md"))
        self.saved = path
        self.copy_clip(paper_clipboard.Clip("diagram", document, self.target or ""))
        try:
            open_code(path)
            self.status.set(f"Flowchart saved as Mermaid in {path.name}, opened in VS Code and copied. Paste it "
                            "into ChatGPT or a Markdown file.")
        except OSError as error:
            self.status.set(f"Flowchart saved as {path} and copied. Could not open VS Code: {error}")

    def finish_forge(self, draft, workflow):
        repo = self.git_repo
        if repo is None:
            return
        try:
            kind = paper_forge.kind_of(repo)
        except (OSError, paper_git.GitError) as error:
            self.status.set(str(error))
            return
        if not draft.source:
            self.status.set("The writing was unclear. Write it again a little larger.")
            return
        try:
            if workflow == "forge_issue":
                webbrowser.open(paper_forge.issue_url(repo, kind, draft.source))
                self.status.set(f"Issue \"{paper_forge.split_text(draft.source)[0]}\" is filled in on "
                                f"{paper_forge.HOSTS[kind]}. Submit it in the browser.")
                return
            url = paper_forge.request_url(repo, kind, draft.source)
        except (OSError, paper_git.GitError) as error:
            self.status.set(str(error))
            self.key_result = ("error", time.monotonic())
            return
        self.busy = True
        self.workflow = "git_push"
        self.status.set(f"Pushing {repo.name} before opening the request...")
        Thread(target=self._forge_request, args=(repo, kind, url, draft.source), daemon=True).start()

    def _forge_request(self, repo, kind, url, text):
        name = "pull request" if kind == "github" else "merge request"
        try:
            paper_git.push(repo)
            webbrowser.open(url)
            message = (f"Pushed {paper_git.current_branch(repo)} and filled in the {name} "
                       f"\"{paper_forge.split_text(text)[0]}\". Submit it in the browser.")
            self.results.put((Draft(text, text, (message,), ".txt"), None, "git_push"))
        except (OSError, paper_git.GitError) as error:
            self.results.put(paper_git.GitError(f"The {name} was not opened: {error}"))

    def git_dispatch(self, action):
        """Version control for the project being worked on (see active_repo)."""
        if action == "camera":
            self.toggle_camera()
        elif action == "paste":
            self.paste()
        elif action == "remote":
            self.select("git_remote")
        elif action == "back":
            # From the host keys back to version control; from version control back to the app.
            self.select("git" if self.menu == "git_remote" else self.last_app)
        elif action in ("browse", "open"):
            repo = self.choose_repo() if action == "browse" else self.active_repo()
            if repo is not None:
                try:
                    open_code(repo)
                    self.status.set(f"Version control now uses {repo.name}; opened it in VS Code.")
                except OSError as error:
                    self.status.set(str(error))
        elif action == "commit":
            repo = self.active_repo()
            if repo is None:
                return
            # The message is typed: a console opens in the repository, shows the changes, and asks for it.
            python = Path(sys.executable).with_name("python.exe")
            try:
                subprocess.Popen([str(python if python.is_file() else sys.executable),
                                  str(Path(__file__).with_name("paper_commit.py")), str(repo)], cwd=repo,
                                 creationflags=getattr(subprocess, "CREATE_NEW_CONSOLE", 0))
                self.status.set(f"Commit prompt opened for {repo.name}. Type the message in the terminal.")
                self.key_result = ("done", time.monotonic())
            except OSError as error:
                self.status.set(f"Could not open the commit prompt: {error}")
                self.key_result = ("error", time.monotonic())
        elif action in ("push", "pull"):
            repo = self.active_repo()
            if repo is not None:
                self.busy = True
                self.workflow = "git_push"
                self.status.set(f"{'Pushing' if action == 'push' else 'Pulling'} {repo.name}...")
                Thread(target=self._git_push if action == "push" else self._git_pull, args=(repo,),
                       daemon=True).start()
        elif action in ("issue", "request", "web", "publish"):
            self.forge_action(action)
        elif action in ("repo", "branch", "diagram"):
            if action != "repo":
                repo = self.active_repo()
                if repo is None:
                    return
            self.writing_action(paper_git.WORKFLOWS[action])
            if action == "diagram" and self.writing_workflow == "diagram":
                self.status.set("Draw branches as horizontal lines, circle commits and write their message beside "
                                "them, circle a branch name to push. Press Diagram again when done.")
            elif action == "branch" and self.writing_workflow == "git_branch":
                self.status.set(f"Write the branch name for {repo.name}, then draw a small circle to finish.")

    def forge_action(self, action):
        """GitHub or GitLab, whichever the project's remote is: issue, request, publish, and the web page."""
        repo = self.active_repo()
        if repo is None:
            return
        try:
            if action == "publish":
                if paper_git.run(repo, "remote").split():
                    self.status.set(f"{repo.name} already has a remote. Use Push.")
                    return
                try:
                    copied = self.window.clipboard_get().strip()
                except tk.TclError:
                    copied = ""
                parsed = paper_forge.parse_remote(copied) if len(copied) < 300 else None
                kind = next((kind for kind in paper_forge.HOSTS if parsed and kind in parsed[0]), None)
                if kind is not None:
                    paper_forge.connect(repo, kind, copied)
                    self.busy = True
                    self.workflow = "git_push"
                    self.status.set(f"Connected {repo.name} to {parsed[1]}. Pushing...")
                    Thread(target=self._git_push, args=(repo,), daemon=True).start()
                else:
                    webbrowser.open(paper_forge.new_repository_url(repo, "github"))
                    self.status.set(f"Create the repository for {repo.name} in the browser (GitHub is open; use "
                                    "GitLab if you prefer), copy its URL, then press Publish again.")
                return
            kind = paper_forge.kind_of(repo)
            if action == "web":
                webbrowser.open(paper_forge.web_url(repo, kind))
                self.status.set(f"Opened {repo.name} on {paper_forge.HOSTS[kind]}.")
                return
        except (OSError, paper_git.GitError) as error:
            self.status.set(str(error))
            self.key_result = ("error", time.monotonic())
            return
        self.writing_action("forge_issue" if action == "issue" else "forge_request")
        if self.writing_workflow:
            what = "issue" if action == "issue" else "pull request" if kind == "github" else "merge request"
            self.status.set(f"Write the {what} title for {repo.name} on {paper_forge.HOSTS[kind]}, then draw a "
                            "small circle to finish.")

    def _git_pull(self, repo):
        try:
            branch = paper_git.pull(repo)
            self.results.put((Draft("", "", (f"Pulled {repo.name} ({branch}) up to date.",), ".txt"), None,
                              "git_push"))
        except (OSError, paper_git.GitError) as error:
            self.results.put(paper_git.GitError(f"Pull not done: {error}"))

    def run_git_diagram(self, text):
        """Show the plan from a drawn branch diagram, then run it in the background. The key press is the
        confirmation; nothing is forced, deleted, or rewritten."""
        repo = self.git_repo
        steps, problems, original = paper_git.plan_diagram(repo, text)
        plan = paper_git.describe_plan(steps, problems)
        self.show_draft(Draft(text, plan, tuple(problems), ".txt"))
        self.busy = True
        self.workflow = "git_diagram_run"
        self.status.set(f"Running {len(steps)} steps on {repo.name}: "
                        + "; ".join(step.describe() for step in steps[:4]) + ("; ..." if len(steps) > 4 else ""))
        Thread(target=self._git_diagram, args=(repo, text, steps, original, plan), daemon=True).start()

    def _git_diagram(self, repo, text, steps, original, plan):
        try:
            report, ok = paper_git.execute_plan(repo, steps, original)
        except (OSError, paper_git.GitError) as error:
            report, ok = [f"Stopped: {error}"], False
        summary = f"Git diagram {'done' if ok else 'stopped'} in {repo.name}."
        self.results.put((Draft(text, f"{plan}\n\nResult:\n" + "\n".join(report), (summary, *report), ".txt"),
                          None, "git_diagram_run"))

    def _git_push(self, repo):
        try:
            pushed = paper_git.push(repo)
            self.results.put((Draft("", "", (f"Pushed {repo.name} to {pushed}.",), ".txt"), None, "git_push"))
        except (OSError, paper_git.GitError) as error:
            self.results.put(paper_git.GitError(f"Push not done: {error}"))

    def choose_repo(self):
        options = {"initialdir": str(self.git_repo)} if self.git_repo else {}
        choice = self._dialog(filedialog.askdirectory, title="Choose a git repository", **options)
        if not choice:
            return None
        repo = paper_git.repository_root(choice)
        if repo is None:
            self.status.set(f"{choice} is not inside a git repository.")
            return None
        self.git_repo = self.repo_pin = repo
        if self.last_app == "latex" and any(repo.glob("**/*.tex")):
            self.report_folder = repo
        self.save_settings()
        self.status.set(f"Selected repository {repo}")
        return repo

    def active_repo(self):
        """The repository of the project being worked on: the one picked with Browse, else the
        project of the app page used last (for VS Code, its most recently used window), else
        the last repository used. Only when none of these exists does it ask."""
        repo = self.repo_pin if self.repo_pin and self.repo_pin.is_dir() else None
        if repo is None:
            app = self.menu if self.menu not in ("start", "home", "git", "git_remote") else self.last_app
            try:
                folder = paper_project.current_folder(app, self.project_of.get(app), self.report_folder)
                repo = paper_git.repository_root(folder) if folder else None
            except OSError:
                repo = None
        if repo is None and self.git_repo and self.git_repo.is_dir():
            repo = self.git_repo
        if repo is None:
            return self.choose_repo()
        if repo != self.git_repo:
            self.git_repo = repo
            self.save_settings()
        return repo

    def finish_git(self, draft, workflow):
        try:
            if workflow in ("git_push", "git_diagram_run"):
                self.status.set(" ".join(draft.notes))
            elif not draft.source:
                self.status.set("The writing was unclear. Write it again a little larger.")
            elif workflow != "git_repo" and self.git_repo is None:
                self.status.set("Select a repository with the repository key first.")
            elif workflow == "git_repo":
                if not draft.files:
                    self.status.set("The idea did not give a project scaffold. Name the main parts and try again.")
                    return
                folder = save_generated(draft, self.language, ROOT / "repos")
                paper_git.init_repository(folder, "Initial scaffold from a handwritten idea")
                self.git_repo = folder
                self.save_settings()
                open_code(folder)
                self.status.set(f"Created repository {folder} with an initial commit and opened it in VS Code.")
            elif workflow == "git_branch":
                name = paper_git.create_branch(self.git_repo, draft.source)
                self.status.set(f"Created and switched to branch {name} in {self.git_repo.name}.")
            elif workflow == "git_commit":
                short, message = paper_git.commit(self.git_repo, draft.source)
                self.status.set(f"Committed {short}: {message.splitlines()[0]}")
            elif workflow in ("diagram", "git_auto"):
                self.run_git_diagram(draft.source)
        except (OSError, ValueError, paper_git.GitError) as error:
            self.status.set(f"Git: {error}")

    def tick(self):
        for slot in self.keypad.presses():
            if isinstance(slot, tuple):
                self.dispatch(*slot)
            elif not self.dialog_open and slot < len(self.actions_drawn):
                self.actions_drawn[slot][1]()
        now = time.monotonic()
        self.follow_front_app(now)
        if self.camera:
            process, output, _, last = self.camera
            try:
                kind, sample = output.get_nowait()
            except queue.Empty:
                timeout = 20 if self.frame is None else 10
                if not process.is_alive() or now - last > timeout:
                    self.camera_failed = True
                    self.stop_camera()
                    self.status.set("RGB frames stopped. Reconnect the camera and try again.")
            else:
                if kind == "error":
                    self.camera_failed = True
                    self.stop_camera()
                    self.status.set(f"Camera error: {sample}")
                else:
                    self.frame, hands, _ = sample
                    self.recent_frames.append((now, self.frame))
                    self.sheet_calibration.feed(self.frame, hands, now)
                    if self.first_frame_at is None:
                        self.first_frame_at = now
                    if self.writing_workflow and self.writing.baseline is None and self.camera_settled(now):
                        # The reference frame must show the page as it is: a camera's first frames
                        # are black or still adjusting exposure.
                        self.begin_writing_frame()
                        self.status.set("Camera ready. Write now, then draw a small circle, lift your hand, or "
                                        "press the same key to finish.")
                    if not self.writing_workflow and not self.busy and self.writing.reference is not None:
                        # No recording is running (finished, cancelled, or the page was changed): stop
                        # comparing the page with that recording's reference frame, or a new sheet put down
                        # later would count as new ink all over. The pen is still followed.
                        previous, self.writing = self.writing, WritingSession()
                        self.writing.pen, self.writing.pen_seen = previous.pen, previous.pen_seen
                    recording = bool(self.writing_workflow) and self.writing.baseline is not None
                    if self.exposure_lock is not None and recording != self.exposure_lock.is_set():
                        (self.exposure_lock.set if recording else self.exposure_lock.clear)()
                    self.writing.observe(self.frame, hands, now)
                    if self.recorder is not None and self.writing_workflow:
                        self.recorder.add(self.frame, hands, now)
                    self.camera_failed = False
                    self.camera = (*self.camera[:3], now)
                    if self.preview_live and now - self.preview_update > 0.12:
                        preview = self.writing.draw(self.frame, now)
                        area = self.writing_area or (self.writing.area if self.writing_workflow else None)
                        outline, sure, _ = self.sheet_calibration.status()
                        if outline is not None:
                            # The calibrated sheet, followed as it moves: bold when it is held firmly.
                            cv2.polylines(preview, [np.rint(outline).astype(np.int32).reshape(-1, 1, 2)], True,
                                          (45, 150, 25), 4 if sure >= 0.5 else 1, cv2.LINE_AA)
                        elif self.sheet is not None and self.writing.baseline is not None:
                            cv2.polylines(preview, [np.rint(self.sheet).astype(np.int32).reshape(-1, 1, 2)], True,
                                          (45, 150, 25), 2, cv2.LINE_AA)
                            area = None if not self.writing_workflow else area
                        if area:
                            x0, y0, x1, y1 = area
                            cv2.rectangle(preview, (x0, y0), (x1 - 1, y1 - 1), (45, 150, 25), 3)
                        self.show_image(preview)
                        if self.preview_update == 0:
                            print(f"RGB ready: {self.frame.shape[1]}x{self.frame.shape[0]}", flush=True)
                            self.status.set("Camera starting: wait a moment before you write."
                                            if self.writing_workflow else
                                            "Press Code, Plot, or Project before writing, or capture a still.")
                            if self.pending_capture is not None:
                                self.window.after(450, self.convert_after_start)
                        self.preview_update = now
                    if self.writing_workflow:
                        self.read_while_writing(now)
                    if self.writing_workflow and self.finish_wait is not None and not self.busy:
                        self.finish_after_wait(now)
                    if self.writing_workflow and not self.busy:
                        # Without a detected paper edge, new ink alone still ends the recording once a
                        # pen tip was tracked: the crop is tied to those tip samples, not to the page.
                        completed = self.writing.ready(
                            self.frame, now, require_pen=self.writing.area is None and not self.writing.tips)
                        if completed is not None and self.writing_workflow == "git_auto" and not ink_usable(
                                completed):
                            # Recording by itself: a hand passing over the page without writing is not a request.
                            completed = None
                        if completed is not None:
                            self.finish_writing(completed)
                    elif self.menu == "git" and not self.busy and self.camera_settled(now):
                        self.arm_git_auto()
        try:
            result = self.results.get_nowait()
        except queue.Empty:
            pass
        else:
            self.busy = False
            self.key_result = ("error" if isinstance(result, Exception) else "done", now)
            self.debug_result(result)
            if isinstance(result, Exception):
                self.status.set(str(result))
            else:
                draft, figure, workflow = result
                self.show_draft(draft)
                self.figure = figure
                if getattr(self, "plan_actions", None) and "copy" in self.plan_actions and draft.content:
                    self.copy(draft.content)  # An action the orchestrator may choose (paper_orchestrator.ACTIONS).
                if workflow in paper_git.CONTEXTS or workflow in ("git_push", "diagram", "git_auto", "git_diagram_run"):
                    self.finish_git(draft, workflow)
                elif workflow == "copy":
                    self.finish_copy(draft, figure)
                elif workflow == "table":
                    self.finish_table(draft)
                elif workflow == "image":
                    self.finish_image(draft)
                elif workflow == "flow":
                    self.finish_flow(draft)
                elif workflow == "gantt":
                    self.finish_plan(draft)
                elif workflow in ("forge_issue", "forge_request"):
                    self.finish_forge(draft, workflow)
                elif workflow == "ltspice_sim":
                    path, ok, lines = self.simulated
                    self.saved = path
                    self.notes.configure(text="\n".join(lines))
                    self.status.set(f"Simulation finished: {'; '.join(lines[:4])}. {path.name} is open in LTspice "
                                    "with its waveforms." if ok else
                                    "The simulation failed. " + " | ".join(lines[:3]))
                    if not ok:
                        self.key_result = ("error", now)
                elif draft.content and self.target in ("LTspice", "KiCad") and workflow == "default":
                    self.open_circuit()
                elif (getattr(self, "plan_actions", None) and "open_in_app" not in self.plan_actions
                      and workflow in ("default", "equation", "graph")):
                    # The orchestrator chose not to open the result: it stays in the workbench for review.
                    if figure:
                        self.plot_preview.pack(side="left", padx=4)
                    self.status.set("The result is staged in the workbench for review; Open shows it in the app.")
                elif workflow == "graph":
                    self.finish_plot(draft, figure)
                elif figure:
                    self.plot_preview.pack(side="left", padx=4)
                    self.status.set("Plot ready. Use Copy to transfer it to LaTeX.")
                elif (draft.content and self.target == "LaTeX" and workflow == "equation" and self.report_folder
                      and self.report_folder.is_dir()):
                    self.finish_equation(draft)
                elif draft.content and (self.target in ("VS Code", "MATLAB") and workflow in ("default", "project")
                                        or self.target == "LaTeX" and workflow in ("equation", "project")):
                    self.open_generated()
                else:
                    self.status.set("Review the draft before saving." if draft.content
                                    else "The page was unclear. Try a closer, well-lit image.")
            if not isinstance(result, Exception) and getattr(self, "reading_path", None):
                label = {"ink": "read from the ink you wrote",
                         "ink+photo": "read from the ink you wrote, checked against the photo",
                         "trace": "read from the pen trace",
                         "trace+photo": "read from the pen trace, checked against the photo", "visual": "read from the photo (trace fallback)"}
                if self.reading_path != "cloud":  # The cloud reading is named with its time just below.
                    self.status.set(f"{self.status.get()} [{label.get(self.reading_path, self.reading_path)}]")
                self.reading_path = None
            if not isinstance(result, Exception) and getattr(self, "engines", None):
                self.status.set(f"{self.status.get()} [{self.engines}]")  # Cloud or local, and how long it took.
                self.engines = None
            if not isinstance(result, Exception) and not draft.content and not draft.source and workflow not in (
                    "git_push", "git_diagram_run", "ltspice_sim"):
                self.key_result = ("error", now)  # Nothing was read: show it on the key.
            if self.pending_action is not None:
                self.window.after_idle(partial(self.dispatch, *self.pending_action))
                self.pending_action = None
            elif self.menu == "git":
                self.window.after(1200, self.arm_git_auto)  # Ready for the next drawing on the page.
        if self.key_status() != self.key_state:
            self.publish()
        self.window.after(65, self.tick)

    def camera_settled(self, now):
        """True once the camera has delivered a usable picture: about a second and a half after its
        first frame, and not black. Gives up waiting after six seconds so a dark room still works."""
        if self.frame is None or self.first_frame_at is None:
            return False
        age = now - self.first_frame_at
        if age < 1.5:
            return False
        return age > 6 or float(self.frame[::8, ::8].mean()) >= 30

    def convert_after_start(self):
        if (self.pending_capture is not None and self.camera is not None
                and not self.camera_settled(time.monotonic())):
            self.window.after(300, self.convert_after_start)  # Wait for a real picture, not the black first frames.
            return
        pending, self.pending_capture = self.pending_capture, None
        if pending and pending[0] == self.menu and self.preview_live:
            self.convert(pending[1])

    def close(self):
        self.save_settings()
        self.camera_ai.stop()
        self.stop_camera()
        self.keypad.close()
        self.window.destroy()

    def show_draft(self, draft):
        self.clear_draft()
        self.draft = draft
        self._source_text(draft.source)
        self.result.insert("1.0", draft.content)
        if draft.files:
            self.result.configure(state="disabled")
        self.notes.configure(text="\n".join(draft.notes))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--camera", type=int, help="DirectShow camera index")
    parser.add_argument("--model", default=MODEL, help="Installed Ollama vision model")
    parser.add_argument("--image", type=Path, help="Open an existing photo")
    parser.add_argument("--route", choices=list(APPS), help=argparse.SUPPRESS)
    parser.add_argument("--webcam", action="store_true", help="Start the camera and show the apps (the webcam key)")
    parser.add_argument("--slot", type=int, choices=range(9), help=argparse.SUPPRESS)
    parser.add_argument("--background", action="store_true", help="Start minimized; results show as pop-ups")
    args = parser.parse_args()
    active = current_state()
    if active:
        if args.slot is not None:
            send_slot(active, args.slot)
        elif args.webcam:
            send_route(active, "webcam", menu="start")
        elif args.route:
            send_route(active, args.route)
        else:
            print("LogiInsight Paper is already open.")
        return
    window = tk.Tk()
    app = PaperWorkbench(window, args)
    if args.background:
        window.iconify()
    if args.webcam:
        app.start_from_webcam()
    elif args.route:
        app.select(args.route)
    if args.image:
        window.update_idletasks()
        app.load_image(args.image)
    if args.slot is not None and args.slot < len(app.actions_drawn):
        app.actions_drawn[args.slot][1]()
    window.mainloop()


if __name__ == "__main__":
    mp.freeze_support()
    main()
