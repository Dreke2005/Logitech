"""Open reviewed paper drafts in local desktop tools."""

import os
import shutil
import subprocess
import tempfile
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

from paper_conversion import architecture_files


def _vscode():
    command = shutil.which("code")
    user_programs = Path(os.environ.get("LOCALAPPDATA", Path.home())) / "Programs"
    system_programs = Path(os.environ.get("ProgramFiles", "C:/Program Files"))
    roots = [user_programs / "Microsoft VS Code", system_programs / "Microsoft VS Code"]
    if command:
        roots.insert(0, Path(command).resolve().parent.parent)
    for root in roots:
        editor = root / "Code.exe"
        if editor.is_file():
            return root, editor
    raise OSError("VS Code was not found.")


def open_code(path):
    _, editor = _vscode()
    subprocess.Popen([str(editor), "--reuse-window", str(path)])


def _matlab_running():
    try:
        listing = subprocess.run(["tasklist", "/FI", "IMAGENAME eq MATLAB.exe", "/FO", "CSV", "/NH"],
                                 capture_output=True, text=True, timeout=5,
                                 creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0)).stdout
    except (OSError, subprocess.SubprocessError):
        return False
    return "matlab.exe" in listing.lower()


def _matlab_association():
    if os.name != "nt":
        return False
    import winreg

    try:
        return winreg.QueryValue(winreg.HKEY_CLASSES_ROOT, ".m").lower().startswith("matlab")
    except OSError:
        return False


def matlab_entry(path):
    path = Path(path)
    if path.is_dir():
        path = path / "main.m" if (path / "main.m").is_file() else next(iter(sorted(path.glob("**/*.m"))), path)
    if not path.is_file() or path.suffix.lower() != ".m":
        raise OSError("The saved project does not contain a MATLAB file.")
    return path


def open_matlab(path):
    """Open a .m file (or a project's entry file) in the MATLAB editor, reusing a running MATLAB."""
    path = matlab_entry(path)
    if _matlab_running() and _matlab_association():
        os.startfile(path)  # MATLAB's shell verb opens the file in the running instance via DDE.
        return path
    installed = sorted(Path(os.environ.get("ProgramFiles", "C:/Program Files")).glob("MATLAB/*/bin/matlab.exe"),
                       reverse=True)
    matlab = shutil.which("matlab.exe") or (installed[0] if installed else None)
    if not matlab:
        raise OSError("MATLAB was not found. Open the saved .m in your editor.")
    quoted = path.as_posix().replace("'", "''")
    folder = path.parent.as_posix().replace("'", "''")
    subprocess.Popen([str(matlab), "-sd", folder, "-r", f"cd('{folder}'); edit('{quoted}')"])
    return path


def ask_vscode(source, notes, language, workflow, handoffs, project=None):
    root, editor = _vscode()
    paths = [root / "resources/app/out/cli.js", *root.glob("*/resources/app/out/cli.js")]
    cli = next((path for path in paths if path.is_file()), None)
    if cli is None:
        raise OSError("The VS Code chat CLI was not found in this installation.")
    handoffs.mkdir(parents=True, exist_ok=True)
    folder = Path(tempfile.mkdtemp(prefix="paper-", dir=handoffs))
    context = folder / "paper-context.md"
    context.write_text(f"# Reviewed paper transcription\n\n{source}\n\n"
                       f"## Uncertainties\n\n" + "\n".join(f"- {note}" for note in notes),
                       encoding="utf-8")
    task = ("Suggest a file-by-file implementation of the class diagram" if workflow == "architecture"
            else "Suggest code for the handwritten equation or algorithm")
    prompt = (f"{task} in {language}. Use the attached transcription as reference data. "
              "Preserve symbols and relationships, list ambiguities, and show proposed changes for review.")
    environment = {**os.environ, "ELECTRON_RUN_AS_NODE": "1"}
    subprocess.Popen([str(editor), str(cli), "chat", "--reuse-window", "--mode", "ask",
                      "--add-file", str(context), prompt], env=environment,
                     cwd=project if project and project.is_dir() else None,
                     creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))


def save_project(files, language, destination):
    checked = architecture_files([{"path": name, "content": text} for name, text in files], language)
    folder = Path(tempfile.mkdtemp(prefix="paper-project-", dir=destination))
    try:
        for relative, content in checked:
            path = folder / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("x", encoding="utf-8", newline="\n") as output:
                output.write(content)
    except OSError:
        shutil.rmtree(folder)
        raise
    return folder


def save_generated(draft, language, destination):
    """Persist an editable draft in its own folder before launching the editor."""
    destination.mkdir(parents=True, exist_ok=True)
    files = draft.files or (("paper_draft" + draft.extension, draft.content),)
    folder = save_project(files, language, destination)
    return folder if draft.files else folder / files[0][0]


def package_latex(project, archive):
    """Package common LaTeX source, reference, and figure files for Overleaf."""
    suffixes = {".tex", ".bib", ".sty", ".cls", ".bst", ".pdf", ".svg", ".eps", ".png",
                ".jpg", ".jpeg", ".tikz", ".csv", ".dat", ".txt", ".otf", ".ttf", ".bbx", ".cbx"}
    project = Path(project).resolve()
    paths = []
    for path in project.rglob("*"):
        relative = path.relative_to(project)
        if (path.suffix.lower() in suffixes and not any(part.startswith(".") for part in relative.parts)
                and path.is_file() and path.resolve().is_relative_to(project)
                and not path.is_symlink()):
            paths.append(path)
    if not any(path.suffix.lower() == ".tex" for path in paths):
        raise ValueError("Choose a project with at least one .tex source file.")
    if len(paths) > 180 or sum(path.stat().st_size for path in paths) > 50 * 1024**2:
        raise ValueError("Choose a smaller LaTeX project for export to Overleaf.")
    created = False
    try:
        with ZipFile(archive, mode="x", compression=ZIP_DEFLATED) as output:
            created = True
            for path in sorted(paths):
                output.write(path, path.relative_to(project).as_posix())
    except OSError:
        if created:
            Path(archive).unlink(missing_ok=True)
        raise
    return Path(archive)
