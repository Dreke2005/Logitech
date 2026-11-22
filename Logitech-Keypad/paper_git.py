"""Small, safe git operations for the PaperBridge Git keypad page."""

import os
import re
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path

WORKFLOWS = {"repo": "git_repo", "branch": "git_branch", "commit": "git_commit", "diagram": "diagram"}
CONTEXTS = {
    "git_repo": ("A software project idea for a new repository. Read its title, goals, components, class "
                 "names, and relationships as written.", ("outline", "diagram", "pseudocode")),
    "git_branch": ("A short git branch name or feature description. Transcribe only the written words; "
                   "a circle or loop drawn after the text marks the end and is not part of the name.",
                   ("text",)),
    "git_commit": ("A git commit message. Transcribe the written sentence literally. A circle or loop drawn "
                   "after the text marks the end of the message and is not part of it.", ("text",)),
}
_END_MARKS = re.compile(r"(?:^|\s+)(?:[oO°○◯⭕●◦]|\(\s*\))\s*$")


class GitError(RuntimeError):
    """A git command failed or the request is unsafe."""


def _git():
    command = shutil.which("git")
    if command is None:
        raise GitError("Git was not found. Install Git for Windows and reopen the workbench.")
    return command


def run(repo, *args, timeout=60):
    if args and args[0] == "push" and any(
            arg in ("-f", "--force", "--force-with-lease", "--mirror", "--delete", "--prune")
            or arg.startswith(("+", ":", "--force")) for arg in args[1:]):
        raise GitError("PaperBridge never force-pushes or deletes remote refs.")
    environment = {**os.environ, "GIT_TERMINAL_PROMPT": "0"}
    try:
        result = subprocess.run([_git(), "-C", str(repo), *args], capture_output=True, text=True,
                                encoding="utf-8", errors="replace", timeout=timeout, env=environment,
                                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
    except subprocess.TimeoutExpired as error:
        raise GitError(f"git {args[0]} timed out.") from error
    if result.returncode != 0:
        detail = (result.stderr or result.stdout).strip().splitlines()
        raise GitError(f"git {args[0]} failed: {detail[-1] if detail else result.returncode}")
    return result.stdout.strip()


def repository_root(path):
    """Return the top-level folder of the repository that contains path, or None."""
    path = Path(path)
    if not path.is_dir():
        return None
    try:
        return Path(run(path, "rev-parse", "--show-toplevel"))
    except GitError:
        return None


def branch_name(text):
    """Turn handwriting into a valid branch name, e.g. 'Add login page' -> 'add-login-page'."""
    text = _END_MARKS.sub("", " ".join(str(text).split()))
    parts = []
    for part in text.lower().split("/"):
        part = re.sub(r"[^a-z0-9._-]+", "-", part)
        part = re.sub(r"\.{2,}", ".", part)
        part = re.sub(r"-{2,}", "-", part).strip("-.")
        part = part.removesuffix(".lock").strip("-.")
        if part:
            parts.append(part[:60].strip("-."))
    name = "/".join(parts[:3])[:100].strip("/-.")
    if not name or name == "head" or name.startswith("-"):
        raise GitError("Write a short branch name, such as 'add login page'.")
    return name


def commit_message(text):
    """Clean a handwritten commit message and drop the end-of-message circle."""
    lines = [" ".join(line.split()) for line in str(text).strip().splitlines()]
    lines = [line for line in lines if line]
    while lines:
        cleaned = _END_MARKS.sub("", lines[-1]).rstrip()
        if cleaned:
            lines[-1] = cleaned
            break
        lines.pop()
    message = "\n".join(lines).strip()
    if not message:
        raise GitError("The commit message was empty. Write it again and circle the end.")
    return message[:2000]


def _ensure_branch_free(repo, name):
    run(repo, "check-ref-format", "--branch", name)
    try:
        run(repo, "rev-parse", "--verify", "--quiet", f"refs/heads/{name}")
    except GitError:
        return
    raise GitError(f"Branch {name} already exists.")


def create_branch(repo, text):
    name = branch_name(text)
    _ensure_branch_free(repo, name)
    run(repo, "switch", "-c", name)
    return name


def init_repository(folder, message="Initial commit from paper"):
    """Create a repository with an initial commit of every file in folder."""
    folder = Path(folder)
    if (folder / ".git").exists():
        raise GitError(f"{folder} is already a git repository.")
    run(folder, "init", "--initial-branch=main")
    run(folder, "add", "--all")
    run(folder, "commit", "-m", commit_message(message))
    return folder


def commit(repo, text):
    """Commit staged changes, or every change when nothing is staged. Returns (short hash, message)."""
    message = commit_message(text)
    if not run(repo, "status", "--porcelain"):
        raise GitError("Nothing to commit. The working tree is clean.")
    staged = run(repo, "diff", "--cached", "--name-only")
    if not staged:
        run(repo, "add", "--all")
    run(repo, "commit", "-m", message)
    return run(repo, "rev-parse", "--short", "HEAD"), message


def push(repo, timeout=120, branch=None):
    """Push a branch (the current one by default) to its upstream, or set one on the first push.

    Never forces."""
    remotes = run(repo, "remote").split()
    if not remotes:
        raise GitError("This repository has no remote. Add one with 'git remote add origin URL' first.")
    if branch is None:
        try:
            branch = run(repo, "symbolic-ref", "--quiet", "--short", "HEAD")
        except GitError as error:
            raise GitError("Check out a branch before pushing.") from error
    try:
        upstream = run(repo, "rev-parse", "--abbrev-ref", "--symbolic-full-name", f"{branch}@{{upstream}}")
    except GitError:
        remote = "origin" if "origin" in remotes else remotes[0]
        run(repo, "push", "--set-upstream", remote, f"refs/heads/{branch}:refs/heads/{branch}", timeout=timeout)
        return f"{remote}/{branch}"
    remote, _, target = upstream.partition("/")
    run(repo, "push", remote, f"refs/heads/{branch}:refs/heads/{target or branch}", timeout=timeout)
    return branch


def commit_file(repo, relative, message, body=""):
    """Commit only one file (added if new), leaving every other change untouched."""
    message = commit_message(message)
    run(repo, "add", "--", relative)
    run(repo, "commit", "--only", "-m", message, *(("-m", body.strip()) if body.strip() else ()), "--", relative)
    return run(repo, "rev-parse", "--short", "HEAD"), message


# Git diagrams drawn on paper ------------------------------------------------------------------------------
#
# The reader turns a drawing into one command per line:
#     branch <name> [from <parent>]      a fork from a horizontal branch line
#     commit <branch> "<message>"        a circled node with its text
#     merge <source> into <target>       a join
#     push <branch>                      a circled branch label
#     tag <name> [on <branch>]           (optional) a flag at a node

_TOKEN = r"(?:\"[^\"]*\"|'[^']*'|[^\s\"']+)"
_DIAGRAM_PATTERNS = {
    "branch": re.compile(rf"branch\s+(?P<name>{_TOKEN})(?:\s+(?:from|off|of|on)\s+(?P<parent>{_TOKEN}))?", re.I),
    "commit": re.compile(rf"commit(?:\s+(?:on|to|in)\b)?\s+(?P<branch>{_TOKEN})(?:\s*[:\-–]?\s*(?P<message>.*))?",
                         re.I),
    "merge": re.compile(rf"merge\s+(?P<source>{_TOKEN})(?:\s+(?:into|to|->|→|=>)\s+(?P<target>{_TOKEN}))?", re.I),
    "push": re.compile(rf"push(?:\s+(?P<branch>{_TOKEN}))?(?:\s+(?:to\s+)?origin)?", re.I),
    "tag": re.compile(rf"tag\s+(?P<name>{_TOKEN})(?:\s+(?:on|at|to)\s+(?P<branch>{_TOKEN}))?", re.I),
}
_QUOTES = str.maketrans({"“": '"', "”": '"', "„": '"', "‘": "'", "’": "'", "`": "'"})
DEFAULT_MESSAGE = "Commit from paper diagram"


@dataclass(frozen=True)
class Step:
    action: str
    branch: str
    other: str = ""
    note: str = ""

    def describe(self):
        return {
            "branch": f"create branch {self.branch} from {self.other}",
            "commit": f'commit on {self.branch}: "{self.other}"',
            "merge": f"merge {self.other} into {self.branch} (--no-ff)",
            "push": f"push {self.branch}",
            "tag": f"tag {self.branch} at {self.other}",
        }[self.action] + (f" ({self.note})" if self.note else "")


def _unquote(token):
    token = (token or "").strip().strip(".,;:")
    if len(token) >= 2 and token[0] == token[-1] and token[0] in "\"'":
        token = token[1:-1]
    return token.strip()


def parse_diagram(text):
    """Parse the diagram DSL. Returns (commands, ignored lines). Tolerates bullets, numbering, case, quotes."""
    commands, ignored = [], []
    for raw in str(text).translate(_QUOTES).splitlines():
        line = raw.strip()
        line = re.sub(r"^(?:[-*•>]+|\d+[.)]|\x60+)\s*", "", line).strip().rstrip("\x60").strip()
        if not line or line.startswith(("#", "//")) or re.fullmatch(r"['\x60]{3,}\w*", line):
            continue
        line = re.sub(r"^([A-Za-z]+)\s*:\s*", r"\1 ", line)
        word = line.split(maxsplit=1)[0].lower().rstrip(":")
        match = _DIAGRAM_PATTERNS[word].fullmatch(line) if word in _DIAGRAM_PATTERNS else None
        if match is None:
            ignored.append(raw.strip())
            continue
        values = {key: value for key, value in match.groupdict().items()}
        if word == "commit":
            branch, message = _unquote(values["branch"]), _unquote(values.get("message") or "")
            if values["branch"][:1] in "\"'" and not message:
                branch, message = "", branch  # commit "message" on the current branch
            commands.append(("commit", branch, message))
        elif word == "branch":
            commands.append(("branch", _unquote(values["name"]), _unquote(values.get("parent"))))
        elif word == "merge":
            commands.append(("merge", _unquote(values["source"]), _unquote(values.get("target"))))
        elif word == "push":
            commands.append(("push", _unquote(values.get("branch")), ""))
        else:
            commands.append(("tag", _unquote(values["name"]), _unquote(values.get("branch"))))
    return commands, ignored


def local_branches(repo):
    return set(run(repo, "for-each-ref", "--format=%(refname:short)", "refs/heads").split())


def current_branch(repo):
    try:
        return run(repo, "symbolic-ref", "--quiet", "--short", "HEAD")
    except GitError as error:
        raise GitError("Check out a branch before running a diagram.") from error


def plan_diagram(repo, text):
    """Validate the drawing against the repository. Returns (steps, problems, original branch)."""
    commands, ignored = parse_diagram(text)
    problems = [f"Ignored line: {line}" for line in ignored]
    if not commands:
        raise GitError("No branch, commit, merge, or push found in the drawing.")
    original = current_branch(repo)
    existing = local_branches(repo)
    known, steps, tags = set(existing), [], set()
    head = original

    def name_of(text, fallback=None):
        if not text:
            return fallback
        try:
            name = branch_name(text)
        except GitError:
            problems.append(f"Unusable branch name: {text}")
            return None
        if name.lower() in {branch.lower() for branch in existing} and name not in existing:
            name = next(branch for branch in existing if branch.lower() == name.lower())
        return name

    def ensure(name, parent=None, why="created for a later step"):
        if name in known:
            return True
        parent = parent or original
        if parent not in known:
            return False
        steps.append(Step("branch", name, parent, why))
        known.add(name)
        return True

    for action, first, second in commands:
        if action == "branch":
            name = name_of(first)
            parent = name_of(second, original)
            if name is None or parent is None:
                continue
            if name in known:
                if name not in existing or second:
                    problems.append(f"Branch {name} already exists; kept it as is.")
                head = name
                continue
            if parent not in known and not ensure(parent, why="parent drawn without a fork"):
                problems.append(f"Parent {parent} of {name} is unknown; skipped.")
                continue
            steps.append(Step("branch", name, parent))
            known.add(name)
            head = name
        elif action == "commit":
            name = name_of(first, head)
            if name is None:
                continue
            if not ensure(name):
                problems.append(f"Cannot create branch {name}; skipped its commit.")
                continue
            steps.append(Step("commit", name, commit_message(second or DEFAULT_MESSAGE)))
            head = name
        elif action == "merge":
            source, target = name_of(first), name_of(second, original)
            if source is None or target is None:
                continue
            if source not in known or target not in known or source == target:
                problems.append(f"Cannot merge {first} into {second or target}: "
                                + ("same branch." if source == target else "unknown branch."))
                continue
            steps.append(Step("merge", target, source))
            head = target
        elif action == "push":
            name = name_of(first, head)
            if name is None:
                continue
            if name not in known:
                problems.append(f"Cannot push unknown branch {name}; skipped.")
                continue
            steps.append(Step("push", name))
        else:
            target = name_of(second, head)
            tag = re.sub(r"[^A-Za-z0-9._/-]+", "-", first).strip("-./")
            if not tag or target not in known or tag in tags:
                problems.append(f"Cannot tag {first}; skipped.")
                continue
            tags.add(tag)
            steps.append(Step("tag", tag, target))
    if not steps:
        raise GitError("Nothing to run. " + " ".join(problems[:3]))
    return steps, problems, original


def describe_plan(steps, problems=()):
    lines = [f"{index}. {step.describe()}" for index, step in enumerate(steps, 1)]
    return "\n".join([*lines, *(f"! {problem}" for problem in problems)])


def execute_plan(repo, steps, original, timeout=120):
    """Run the planned steps in order, then return to the original branch.

    Never forces, deletes branches, or rewrites history. Stops at the first failure; a conflicting
    merge is aborted. Returns (report lines, ok)."""
    report, ok = [], True

    def switch(name):
        if current_branch(repo) != name:
            run(repo, "switch", name)

    try:
        for step in steps:
            try:
                if step.action == "branch":
                    _ensure_branch_free(repo, step.branch)
                    run(repo, "switch", "-c", step.branch, step.other)
                    report.append(f"Created branch {step.branch} from {step.other}.")
                elif step.action == "commit":
                    switch(step.branch)
                    if run(repo, "status", "--porcelain"):
                        if not run(repo, "diff", "--cached", "--name-only"):
                            run(repo, "add", "--all")
                        run(repo, "commit", "-m", step.other)
                        kind = "with your current changes"
                    else:
                        run(repo, "commit", "--allow-empty", "-m", step.other)
                        kind = "as an empty commit (no changes to commit)"
                    short = run(repo, "rev-parse", "--short", "HEAD")
                    report.append(f'Committed {short} on {step.branch} {kind}: "{step.other.splitlines()[0]}".')
                elif step.action == "merge":
                    switch(step.branch)
                    try:
                        run(repo, "merge", "--no-ff", "--no-edit", "-m",
                            f"Merge {step.other} into {step.branch}", step.other)
                    except GitError as error:
                        try:
                            run(repo, "rev-parse", "--verify", "--quiet", "MERGE_HEAD")
                        except GitError:
                            raise error from None
                        run(repo, "merge", "--abort")
                        raise GitError("it conflicted, so the merge was aborted and nothing changed") from None
                    report.append(f"Merged {step.other} into {step.branch}.")
                elif step.action == "push":
                    if not run(repo, "remote").split():
                        report.append(f"Skipped push of {step.branch}: the repository has no remote.")
                        continue
                    pushed = push(repo, timeout, branch=step.branch)
                    report.append(f"Pushed {step.branch} to {pushed}.")
                elif step.action == "tag":
                    run(repo, "tag", step.branch, f"refs/heads/{step.other}")
                    report.append(f"Tagged {step.other} as {step.branch}.")
            except GitError as error:
                report.append(f"Stopped at '{step.describe()}': {error}")
                ok = False
                break
    finally:
        try:
            if current_branch(repo) != original:
                run(repo, "switch", original)
            report.append(f"Back on {original}.")
        except GitError as error:
            report.append(f"Could not return to {original}: {error}")
            ok = False
    return report, ok


def run_diagram(repo, text):
    steps, problems, original = plan_diagram(repo, text)
    report, ok = execute_plan(repo, steps, original)
    return steps, problems, report, ok
