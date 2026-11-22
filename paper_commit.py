"""Terminal prompt for a commit: shows what changed, asks for the message, commits.

Started by the keypad's Commit key in its own console window, in the repository of the
project the user is working in.
"""

import argparse
import subprocess
import sys
from pathlib import Path

import paper_git


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("repo", type=Path)
    repo = parser.parse_args().repo
    try:
        branch = paper_git.current_branch(repo) or "(detached)"
        # Porcelain lines are "XY path": X is the staged state, Y the unstaged one.
        status = subprocess.run([paper_git._git(), "-C", str(repo), "status", "--porcelain"], capture_output=True,
                                text=True, encoding="utf-8", errors="replace", check=False).stdout
        entries = [line for line in status.splitlines() if len(line) > 3]
        print(f"\n  Repository  {repo}\n  Branch      {branch}\n")
        if not entries:
            print("  Nothing to commit: the working tree is clean.")
            input("\n  Press Enter to close. ")
            return 0
        staged = [line[3:] for line in entries if line[0] not in " ?"]
        unstaged = [line[3:] for line in entries if line[1] != " " or line[0] == "?"]
        for title, files in (("Staged", staged), ("Not staged", unstaged)):
            if files:
                print(f"  {title} ({len(files)})")
                print("\n".join("    " + name for name in files[:30]) + ("\n    ..." if len(files) > 30 else ""))
        if unstaged:
            default = "n" if staged else "y"
            answer = input(f"\n  Stage all {len(unstaged)} unstaged file{'s' if len(unstaged) != 1 else ''} too? "
                           f"[{'Y/n' if default == 'y' else 'y/N'}] ").strip().lower() or default
            if answer in ("y", "yes", "j", "ja"):
                paper_git.run(repo, "add", "--all")
                print("  Staged everything.")
            elif not staged:
                print("  Nothing is staged, so there is nothing to commit.")
                input("\n  Press Enter to close. ")
                return 0
        message = input("\n  Commit message (empty cancels): ").strip()
        if not message:
            print("  Cancelled; nothing was committed.")
            return 0
        short, message = paper_git.commit(repo, message)
        print(f"\n  Committed {short}  {message.splitlines()[0]}")
        if paper_git.run(repo, "remote").split() and input("  Push now? [y/N] ").strip().lower() in ("y", "yes", "j"):
            print(f"  Pushed to {paper_git.push(repo)}.")
    except (OSError, paper_git.GitError) as error:
        print(f"\n  Git: {error}")
        input("\n  Press Enter to close. ")
        return 1
    except (EOFError, KeyboardInterrupt):
        return 1
    input("\n  Press Enter to close. ")
    return 0


if __name__ == "__main__":
    sys.exit(main())
