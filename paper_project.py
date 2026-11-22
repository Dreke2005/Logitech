"""Find the project the user is working in, so version control needs no folder picking.

For VS Code this is the folder of the most recently used VS Code window: window titles
give the order (top-most first) and VS Code's own state file gives the folder paths.
Other apps use the file the workbench last opened or saved for them.
"""

import ctypes
import json
import os
from pathlib import Path
from urllib.parse import unquote, urlparse


def _window_titles(suffix):
    """Titles of visible top-level windows ending in ``suffix``, front-most first."""
    if os.name != "nt":
        return []
    user32 = ctypes.windll.user32
    titles = []

    @ctypes.WINFUNCTYPE(ctypes.c_bool, ctypes.c_void_p, ctypes.c_void_p)
    def visit(handle, _):
        if user32.IsWindowVisible(handle):
            length = user32.GetWindowTextLengthW(handle)
            if length:
                buffer = ctypes.create_unicode_buffer(length + 1)
                user32.GetWindowTextW(handle, buffer, length + 1)
                if buffer.value.endswith(suffix):
                    titles.append(buffer.value)
        return True

    user32.EnumWindows(visit, 0)
    return titles


def _folder(uri):
    if not isinstance(uri, str) or not uri.startswith("file://"):
        return None
    path = unquote(urlparse(uri).path)
    if len(path) > 2 and path[0] == "/" and path[2] == ":":
        path = path[1:]
    return Path(path)


def vscode_folders(storage=None):
    """Folders VS Code has open or had open last, the last active one first."""
    storage = storage or Path(os.environ.get("APPDATA", Path.home())) / "Code/User/globalStorage/storage.json"
    try:
        state = json.loads(Path(storage).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return []
    windows = state.get("windowsState", {}) if isinstance(state, dict) else {}
    entries = [windows.get("lastActiveWindow") or {}, *(windows.get("openedWindows") or [])]
    uris = [entry.get("folder") for entry in entries if isinstance(entry, dict)]
    backups = state.get("backupWorkspaces", {}) if isinstance(state, dict) else {}
    uris += [entry.get("folderUri") for entry in backups.get("folders", []) if isinstance(entry, dict)]
    folders = []
    for folder in map(_folder, uris):
        if folder is not None and folder not in folders:
            folders.append(folder)
    return folders


def vscode_project(titles=None, folders=None):
    """The folder of the VS Code window used most recently, or None."""
    folders = vscode_folders() if folders is None else folders
    titles = _window_titles("Visual Studio Code") if titles is None else titles
    for title in titles:
        parts = [part.strip() for part in title.split(" - ")]
        # "file - folder - Visual Studio Code": the folder name is the part before the app name.
        for name in reversed(parts[:-1]):
            match = next((folder for folder in folders if folder.name == name), None)
            if match is not None and match.is_dir():
                return match
    if titles:
        return None  # A window is open on a folder VS Code has not recorded yet.
    return next((folder for folder in folders if folder.is_dir()), None)


def current_folder(app, saved=None, report_folder=None):
    """The folder the user is working in for a keypad app ('vscode', 'matlab', 'latex', ...)."""
    saved = Path(saved) if saved else None
    own = (saved if saved.is_dir() else saved.parent) if saved is not None and saved.exists() else None
    if app == "latex" and report_folder and Path(report_folder).is_dir():
        return Path(report_folder)
    if app == "vscode":
        return vscode_project() or own
    return own or vscode_project()
