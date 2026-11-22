"""Capture handwriting with the LogiInsight RGB camera and review model drafts."""

import argparse
import json
import multiprocessing as mp
import os
import queue
import shutil
import subprocess
import sys
import time
import tkinter as tk
import webbrowser
from dataclasses import replace
from functools import partial
from pathlib import Path
from threading import Thread
from tkinter import filedialog, ttk

import cv2

from fingertip_tracker import DEFAULT_MODEL
from local_model import MODEL
from paper_camera_check import choose_camera
from paper_check import check_derivation
from paper_context import end_circle_enabled, project_hints
from paper_conversion import (
    DIRECT_TARGETS,
    LANGUAGES,
    Draft,
    convert_page,
    convert_transcription,
    read_page_source,
    read_recording,
    read_still,
)
import paper_clipboard
from paper_clipboard import write_new_file
import paper_forge
import paper_git
import paper_ltspice
import paper_plan
import paper_project
from paper_handoff import ask_vscode, open_code, open_matlab, package_latex, save_generated, save_project
from paper_keypad import MENUS, ROOT, KeypadSession, current_state, send_route, send_slot
from paper_math import latex_equation
from paper_selection import select_paper_region
from paper_transfer import fenced_code, graph, import_equation, import_figure, import_snippet, next_numbered
from paper_writing import WritingSession, paper_bounds
from rgb_preview import _capture



def _read_tracked(bench, selected, target, language, workflow, model):
    """read_recording that remembers whether the pen trace or the photo fallback produced the text."""
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
        self.menu = "home"
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

    def save_settings(self):
        try:
            ROOT.mkdir(parents=True, exist_ok=True)
            (ROOT / "settings.json").write_text(json.dumps({
                "git_repo": str(self.git_repo) if self.git_repo else None,
                "report_folder": str(self.report_folder) if self.report_folder else None,
                "language": self.language}), encoding="utf-8")
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
        if self.dialog_open:
            return
        if self.busy:
            if menu == self.menu and {"code": "default", "graph": "graph", "project": "project",
                                      "equation": "equation", "table": "table", "flow": "flow",
                                      "issue": "forge_issue", "request": "forge_request", "gantt": "gantt",
                                      "diagram": "diagram"}.get(action) == self.workflow:
                return
            self.pending_action = menu, action
            return
        if menu == "home":
            if self.menu != action or self.languages:
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

    def select(self, menu):
        if self.busy:
            return
        if menu not in ("home", "git", "git_remote"):
            if menu != self.last_app:
                self.repo_pin = None
            self.last_app = menu
        self.menu, self.languages = menu, False
        self.writing_workflow = None
        self.target = "Git" if menu in ("git", "git_remote") else dict(MENUS["home"]).get(menu)
        self.clear_draft()
        self._show_menu()
        self.status.set("Choose an app." if self.target is None else
                        "Use the camera icon to start RGB, then choose a conversion.")

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
            hand_model = DEFAULT_MODEL if DEFAULT_MODEL.is_file() else None
            process = ctx.Process(target=_capture, args=(device, "DSHOW", 1920, 1080, output, stop, hand_model),
                                  daemon=True)
            process.start()
            self.writing = WritingSession()
            self.writing_area = None
            self.page_pinned = False
            self.camera = process, output, stop, time.monotonic()
            self.preview_live = True
            self.preview_update = 0
            self.render_buttons()
            self.status.set(f"Opening {names[device]}..." if logitech else
                            f"LogiInsight camera not connected: using {names[device]}. Aim it at the paper.")
        except (OSError, RuntimeError) as error:
            self.camera_failed = True
            self.status.set(f"Camera error: {error}")

    def stop_camera(self):
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

    def selected_page(self, *, narrow=False, marked_only=False):
        writing = self.writing.select(self.page, roi=self.roi, narrow=narrow, marked_only=marked_only)
        self.show_selection(writing.page)
        return writing

    def show_selection(self, selection):
        overlay = self.page.copy()
        if selection.bounds:
            x0, y0, x1, y1 = selection.bounds
            cv2.rectangle(overlay, (x0, y0), (x1 - 1, y1 - 1), (45, 150, 25), 4)
        self.show_image(overlay)

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
            if self.frame is not None:
                self.finish_writing(self.writing.select(self.frame, prefer_recent=True))
            return
        self.writing_workflow = workflow
        self.preview_live = True
        self.pending_capture = None
        purpose = {"default": "code", "graph": "a plot", "project": "a project", "equation": "an equation",
                   "git_repo": "a repository idea", "git_branch": "a branch name",
                   "git_commit": "a commit message", "diagram": "a branch diagram",
                   "forge_issue": "an issue", "forge_request": "a request title", "flow": "a flowchart",
                   "gantt": "a project plan",
                   "table": "a table"}.get(workflow, "writing")
        if self.frame is None:
            self.writing = WritingSession()
            self.status.set(f"Camera starting. Wait for the live image before writing for {purpose}.")
        else:
            self.begin_writing_frame()
            area_note = "" if self.writing.area else " Drag around the paper if the pen is hard to see."
            finish = ("draw a small circle, lift your hand, or press the same key to finish"
                      if self.writing.end_circle_enabled else "lift your hand or press the same key to finish")
            self.status.set(f"Recording for {purpose}. Write now, then {finish}." + area_note)

    def begin_writing_frame(self):
        area = self.writing_area or paper_bounds(self.frame)
        # Branch diagrams and copy marks contain circles, so a circle must not end those recordings.
        self.writing = WritingSession(self.frame, area,
                                      end_circle=end_circle_enabled(self.writing_workflow, self.target))

    def finish_writing(self, selected):
        if selected is None:
            self.status.set("No writing region detected yet. Keep writing or capture a still and select the text.")
            return
        workflow, self.writing_workflow = self.writing_workflow, None
        self.page = self.frame.copy()
        self.preview_live = False
        self.show_selection(selected.page)
        self.generate_selection(workflow, selected, recorded=True)
        if selected.end_marked:
            self.status.set("End circle detected. Reading the writing session locally...")

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
        elif recorded:
            self.status.set("Reading the writing session locally...")
        else:
            self.status.set("Reading the page and generating a draft locally...")
        Thread(target=self._generate, args=(selected, self.target, self.language, workflow, recorded),
               daemon=True).start()

    def capture_selection(self, workflow):
        if self.camera_failed:
            self.status.set("Restart the camera and wait for new RGB frames, or open a photo.")
            return None
        if self.frame is not None and (self.preview_live or (self.roi is None and not self.page_pinned)):
            self.page = self.frame.copy()
            self.roi = None
            self.preview_live = False
        if self.page is None:
            self.status.set("Capture the paper or open a photo first.")
            return
        selected = self.selected_page(narrow=workflow in ("graph", "default", "equation"),
                                      marked_only=workflow == "copy")
        if selected.page.reason == "multiple circles":
            self.status.set("Several circles found. Drag around the item you want to use.")
            return
        if workflow == "copy" and selected.page.kind == "full":
            self.status.set("Draw a dashed oval around one equation, or drag a selection in the preview.")
            return None
        return selected

    def _generate(self, selected, target, language, workflow, recorded=False):
        self.reading_path = None
        try:
            figure = None
            if workflow == "copy":
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
            elif workflow in ("table", "diagram", "flow", "gantt", "forge_issue", "forge_request"):
                if recorded or workflow in ("table", "diagram"):
                    source, notes = _read_tracked(self, selected, target, language, workflow, self.args.model)
                else:
                    source, notes = read_still(selected.page.image, target, language, workflow, self.args.model,
                                               evidence=selected.evidence)
                draft = Draft(source, source, notes,
                              {"table": ".tex", "flow": ".md", "gantt": ".md"}.get(workflow, ".txt"))
            elif workflow == "graph":
                if isinstance(selected, str):
                    source, notes = selected, ()
                elif recorded:
                    source, notes = _read_tracked(self, selected, target, language, workflow, self.args.model)
                else:
                    request = "Preserve the exact equation y = f(x), the written domain, or plot call. Use ^ for powers."
                    source, notes = read_page_source(selected.page.image, self.args.model,
                                                    request, evidence=selected.evidence)
                figure, domain = graph(source, ROOT / "figures")
                notes = (*notes, f"Plotted over x = {domain[0]:g} to {domain[1]:g}.")
                extension = LANGUAGES[language] if target == "VS Code" else DIRECT_TARGETS[target]
                draft = Draft(source, source, notes, extension)
            elif workflow in paper_git.CONTEXTS:
                if recorded:
                    source, notes = _read_tracked(self, selected, "Git", language, workflow, self.args.model)
                else:
                    request = paper_git.CONTEXTS[workflow][0] + " Transcribe the writing literally."
                    source, notes = read_page_source(selected.page.image, self.args.model, request,
                                                     evidence=selected.evidence)
                draft = (convert_transcription(source, notes, "VS Code", language, self.args.model, "project")
                         if workflow == "git_repo" else Draft(source, source, notes, ".txt"))
            elif recorded:
                source, notes = _read_tracked(self, selected, target, language, workflow, self.args.model)
                draft = convert_transcription(source, notes, target, language, self.args.model, workflow)
            else:
                draft = convert_page(selected.page.image, target, language, self.args.model, workflow,
                                     evidence=selected.evidence)
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
            self.results.put((draft, figure, workflow))
        except (OSError, RuntimeError, ValueError, cv2.error) as error:
            self.results.put(error)

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
        """Save the generated schematic under drafts and open it in LTspice."""
        try:
            self.saved = write_new_file(ROOT / "drafts" / "ltspice", "paper_circuit" + self.draft.extension,
                                        self.result.get("1.0", "end-1c").rstrip() + "\n")
        except OSError as error:
            self.status.set(f"Could not save the schematic: {error}")
            return
        if self.open_path(self.saved):
            self.status.set(f"Opened {self.saved.name} in LTspice. Press Simulate to run it.")

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
            if kind == "table" and source:
                text = source
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

    def open_overleaf(self):
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
        if webbrowser.open("https://chatgpt.com/") and copied:
            self.status.set("The brief is on your clipboard. Paste it in ChatGPT when ready.")

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
        elif action in ("remote", "back"):
            self.select("git_remote" if action == "remote" else "git")
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
        self.save_settings()
        self.status.set(f"Selected repository {repo}")
        return repo

    def active_repo(self):
        """The repository of the project being worked on: the one picked with Browse, else the
        project of the app page used last (for VS Code, its most recently used window), else
        the last repository used. Only when none of these exists does it ask."""
        repo = self.repo_pin if self.repo_pin and self.repo_pin.is_dir() else None
        if repo is None:
            app = self.menu if self.menu not in ("home", "git", "git_remote") else self.last_app
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
            elif workflow == "diagram":
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
                    if self.writing_workflow and self.writing.baseline is None:
                        self.begin_writing_frame()
                    self.writing.observe(self.frame, hands, now)
                    self.camera_failed = False
                    self.camera = (*self.camera[:3], now)
                    if self.preview_live and now - self.preview_update > 0.12:
                        preview = self.writing.draw(self.frame, now)
                        area = self.writing_area or (self.writing.area if self.writing_workflow else None)
                        if area:
                            x0, y0, x1, y1 = area
                            cv2.rectangle(preview, (x0, y0), (x1 - 1, y1 - 1), (45, 150, 25), 3)
                        self.show_image(preview)
                        if self.preview_update == 0:
                            print(f"RGB ready: {self.frame.shape[1]}x{self.frame.shape[0]}", flush=True)
                            self.status.set("Write now, then draw a small circle, lift your hand, or press the "
                                            "same key to finish." if self.writing_workflow else
                                            "Press Code, Plot, or Project before writing, or capture a still.")
                            if self.pending_capture is not None:
                                self.window.after(450, self.convert_after_start)
                        self.preview_update = now
                    if self.writing_workflow and not self.busy:
                        completed = self.writing.ready(self.frame, now, require_pen=self.writing.area is None)
                        if completed is not None:
                            self.finish_writing(completed)
        try:
            result = self.results.get_nowait()
        except queue.Empty:
            pass
        else:
            self.busy = False
            self.key_result = ("error" if isinstance(result, Exception) else "done", now)
            if isinstance(result, Exception):
                self.status.set(str(result))
            else:
                draft, figure, workflow = result
                self.show_draft(draft)
                self.figure = figure
                if workflow in paper_git.CONTEXTS or workflow in ("git_push", "diagram", "git_diagram_run"):
                    self.finish_git(draft, workflow)
                elif workflow == "copy":
                    self.finish_copy(draft, figure)
                elif workflow == "table":
                    self.finish_table(draft)
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
                    self.status.set(f"Simulation finished; {path.name} is open in LTspice with its waveforms."
                                    if ok else "LTspice reported a problem: " + " | ".join(lines[:3]))
                elif draft.content and self.target == "LTspice":
                    self.open_circuit()
                elif figure:
                    self.plot_preview.pack(side="left", padx=4)
                    self.status.set("Plot ready. Use Copy to transfer it to LaTeX.")
                elif draft.content and (self.target in ("VS Code", "MATLAB") and workflow in ("default", "project")
                                        or self.target == "LaTeX" and workflow in ("equation", "project")):
                    self.open_generated()
                else:
                    self.status.set("Review the draft before saving." if draft.content
                                    else "The page was unclear. Try a closer, well-lit image.")
            if not isinstance(result, Exception) and getattr(self, "reading_path", None):
                label = {"trace": "read from the pen trace",
                         "trace+photo": "read from the pen trace, checked against the photo", "visual": "read from the photo (trace fallback)"}
                self.status.set(f"{self.status.get()} [{label.get(self.reading_path, self.reading_path)}]")
                self.reading_path = None
            if not isinstance(result, Exception) and not draft.content and not draft.source and workflow not in (
                    "git_push", "git_diagram_run", "ltspice_sim"):
                self.key_result = ("error", now)  # Nothing was read: show it on the key.
            if self.pending_action is not None:
                self.window.after_idle(partial(self.dispatch, *self.pending_action))
                self.pending_action = None
        if self.key_status() != self.key_state:
            self.publish()
        self.window.after(65, self.tick)

    def convert_after_start(self):
        pending, self.pending_capture = self.pending_capture, None
        if pending and pending[0] == self.menu and self.preview_live:
            self.convert(pending[1])

    def close(self):
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
    parser.add_argument("--route", choices=[key for key, _ in MENUS["home"]], help=argparse.SUPPRESS)
    parser.add_argument("--slot", type=int, choices=range(9), help=argparse.SUPPRESS)
    parser.add_argument("--background", action="store_true", help="Start minimized; results show as pop-ups")
    args = parser.parse_args()
    active = current_state()
    if active:
        if args.slot is not None:
            send_slot(active, args.slot)
        elif args.route:
            send_route(active, args.route)
        else:
            print("LogiInsight Paper is already open.")
        return
    window = tk.Tk()
    app = PaperWorkbench(window, args)
    if args.background:
        window.iconify()
    if args.route:
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
