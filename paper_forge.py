"""GitHub and GitLab actions for a local repository, driven from handwriting.

Issues and pull/merge requests open in the browser with the written title and text
already filled in, so the signed-in browser session does the authentication and the
user confirms with one click. Nothing is created on the host without that click.
"""

import re
from urllib.parse import quote, urlencode

import paper_git

HOSTS = {"github": "GitHub", "gitlab": "GitLab"}
NEW_REPOSITORY = {"github": "https://github.com/new", "gitlab": "https://gitlab.com/projects/new"}
REMOTE = re.compile(r"(?:https?://(?:[^@/]+@)?|ssh://(?:[^@/]+@)?|[^@/\s]+@)(?P<host>[^/:]+)(?::\d+)?[:/](?P<path>.+?)"
                    r"(?:\.git)?/?$")


def parse_remote(url):
    """(host, 'owner/name') of an https or ssh remote URL, or None."""
    match = REMOTE.match(url.strip())
    if match is None or "/" not in match["path"]:
        return None
    return match["host"].lower(), match["path"].strip("/")


def remote(repo, kind):
    """The repository's remote on the given host kind, as (host, path). Raises GitError with what to do."""
    names = paper_git.run(repo, "remote").split()
    if not names:
        raise paper_git.GitError(f"This repository has no remote yet. Press Publish to create it on {HOSTS[kind]}.")
    found = []
    for name in sorted(names, key=lambda value: value != "origin"):
        parsed = parse_remote(paper_git.run(repo, "remote", "get-url", name))
        if parsed is not None:
            found.append(parsed)
            if kind in parsed[0]:
                return parsed
    host = found[0][0] if found else "an unknown host"
    raise paper_git.GitError(f"This repository lives on {host}, not on {HOSTS[kind]}. Open the matching page.")


def kind_of(repo):
    """'github' or 'gitlab' for the repository's remote; GitError when there is none or it is another host."""
    names = paper_git.run(repo, "remote").split()
    if not names:
        raise paper_git.GitError("This repository has no remote yet. Press Publish first.")
    hosts = []
    for name in sorted(names, key=lambda value: value != "origin"):
        parsed = parse_remote(paper_git.run(repo, "remote", "get-url", name))
        if parsed is not None:
            hosts.append(parsed[0])
            for kind in HOSTS:
                if kind in parsed[0]:
                    return kind
    raise paper_git.GitError(f"The remote {hosts[0] if hosts else ''} is neither GitHub nor GitLab.".replace("  ", " "))


def web_url(repo, kind):
    host, path = remote(repo, kind)
    return f"https://{host}/{path}"


def split_text(text):
    """First written line as the title, the rest as the description."""
    lines = [line.strip() for line in text.strip().splitlines() if line.strip()]
    if not lines:
        raise paper_git.GitError("Write a title first.")
    return lines[0][:200], "\n".join(lines[1:])


def issue_url(repo, kind, text):
    title, body = split_text(text)
    base = web_url(repo, kind)
    if kind == "github":
        return f"{base}/issues/new?" + urlencode({"title": title, "body": body}, quote_via=quote)
    return f"{base}/-/issues/new?" + urlencode({"issue[title]": title, "issue[description]": body}, quote_via=quote)


def default_branch(repo):
    try:
        return paper_git.run(repo, "symbolic-ref", "--quiet", "--short", "refs/remotes/origin/HEAD").split("/", 1)[1]
    except (paper_git.GitError, IndexError):
        branches = paper_git.local_branches(repo)
        return next((name for name in ("main", "master") if name in branches), None)


def request_url(repo, kind, text):
    """URL that opens a pull request (GitHub) or merge request (GitLab) for the current branch."""
    title, body = split_text(text)
    branch = paper_git.current_branch(repo)
    if not branch:
        raise paper_git.GitError("Check out a branch first.")
    target = default_branch(repo)
    if branch == target:
        raise paper_git.GitError(f"You are on {branch}. Create a branch for the change first (Git page, Branch key).")
    base = web_url(repo, kind)
    if kind == "github":
        compare = f"{quote(target, safe='')}...{quote(branch, safe='')}" if target else quote(branch, safe="")
        return f"{base}/compare/{compare}?" + urlencode({"expand": 1, "title": title, "body": body}, quote_via=quote)
    fields = {"merge_request[source_branch]": branch, "merge_request[title]": title,
              "merge_request[description]": body}
    if target:
        fields["merge_request[target_branch]"] = target
    return f"{base}/-/merge_requests/new?" + urlencode(fields, quote_via=quote)


def new_repository_url(repo, kind):
    name = re.sub(r"[^A-Za-z0-9._-]+", "-", repo.name).strip("-") or "paper-project"
    if kind == "github":
        return f"{NEW_REPOSITORY[kind]}?" + urlencode({"name": name})
    return f"{NEW_REPOSITORY[kind]}#blank_project"


def connect(repo, kind, url):
    """Add a copied repository URL as origin. Returns the remote path, or raises GitError."""
    parsed = parse_remote(url)
    if parsed is None or kind not in parsed[0] or len(url) > 300 or any(char.isspace() for char in url.strip()):
        raise paper_git.GitError(f"Copy the new repository's {HOSTS[kind]} URL, then press Publish again.")
    if paper_git.run(repo, "remote").split():
        raise paper_git.GitError("This repository already has a remote. Use Push.")
    paper_git.run(repo, "remote", "add", "origin", url.strip())
    return parsed[1]
