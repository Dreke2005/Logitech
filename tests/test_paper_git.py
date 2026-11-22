import os
import shutil
import stat
import subprocess
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import paper_git
import paper_handoff
from paper_conversion import Draft


def _remove(path):
    def writable(function, name, _error):
        os.chmod(name, stat.S_IWRITE)
        function(name)

    shutil.rmtree(path, onexc=writable)


@unittest.skipUnless(shutil.which("git"), "git is not installed")
class GitTestCase(unittest.TestCase):
    def setUp(self):
        self.folder = Path(tempfile.mkdtemp(prefix="paper-git-test-"))
        self.addCleanup(_remove, self.folder)
        config = self.folder / "gitconfig"
        config.write_text("", encoding="utf-8")
        environment = {"GIT_CONFIG_GLOBAL": str(config), "GIT_CONFIG_NOSYSTEM": "1",
                       "GIT_AUTHOR_NAME": "Paper Test", "GIT_AUTHOR_EMAIL": "paper@example.invalid",
                       "GIT_COMMITTER_NAME": "Paper Test", "GIT_COMMITTER_EMAIL": "paper@example.invalid"}
        patcher = patch.dict(os.environ, environment)
        patcher.start()
        self.addCleanup(patcher.stop)

    def make_repo(self, name="repo"):
        repo = self.folder / name
        repo.mkdir()
        (repo / "main.py").write_text("print('paper')\n", encoding="utf-8")
        return paper_git.init_repository(repo, "Initial scaffold")

    def log(self, repo, *args):
        return paper_git.run(repo, "log", "--format=%s", *args).splitlines()


class NameTests(unittest.TestCase):
    def test_branch_names_become_valid_refs(self):
        cases = {
            "Add login page": "add-login-page",
            "  Feature/Paper Keys!! ": "feature/paper-keys",
            "fix..double dots.lock": "fix.double-dots",
            "-leading dash": "leading-dash",
            "Refactor io": "refactor-io",
            "add plots O": "add-plots",
            "wip ○": "wip",
        }
        for text, expected in cases.items():
            with self.subTest(text=text):
                self.assertEqual(paper_git.branch_name(text), expected)

    def test_unusable_branch_names_are_refused(self):
        for text in ("", "  ", "@{}", "...", "HEAD", "O"):
            with self.subTest(text=text), self.assertRaises(paper_git.GitError):
                paper_git.branch_name(text)

    def test_commit_message_drops_end_circle(self):
        self.assertEqual(paper_git.commit_message("Fix the  plot axis O"), "Fix the plot axis")
        self.assertEqual(paper_git.commit_message("Add paper keys\n\nwith icons\n○"), "Add paper keys\nwith icons")
        self.assertEqual(paper_git.commit_message("Set default to 0"), "Set default to 0")
        self.assertEqual(paper_git.commit_message("Handle io"), "Handle io")
        with self.assertRaises(paper_git.GitError):
            paper_git.commit_message(" o ")

    def test_force_push_arguments_are_refused(self):
        for args in (("push", "--force"), ("push", "-f"), ("push", "origin", "+main"),
                     ("push", "--force-with-lease"), ("push", "origin", ":main")):
            with self.subTest(args=args), patch("subprocess.run") as run, \
                    self.assertRaises(paper_git.GitError):
                paper_git.run(".", *args)
            run.assert_not_called()


class RepositoryTests(GitTestCase):
    def test_init_creates_initial_commit_on_main(self):
        repo = self.make_repo()
        self.assertEqual(self.log(repo), ["Initial scaffold"])
        self.assertEqual(paper_git.run(repo, "branch", "--show-current"), "main")
        self.assertEqual(paper_git.repository_root(repo / ".git" / ".."), repo.resolve())
        self.assertIsNone(paper_git.repository_root(self.folder))
        with self.assertRaises(paper_git.GitError):
            paper_git.init_repository(repo)

    def test_create_branch_switches_and_refuses_duplicates(self):
        repo = self.make_repo()
        self.assertEqual(paper_git.create_branch(repo, "Add paper keys"), "add-paper-keys")
        self.assertEqual(paper_git.run(repo, "branch", "--show-current"), "add-paper-keys")
        paper_git.run(repo, "switch", "main")
        with self.assertRaisesRegex(paper_git.GitError, "already exists"):
            paper_git.create_branch(repo, "add paper keys")

    def test_commit_stages_all_changes_or_only_staged_ones(self):
        repo = self.make_repo()
        with self.assertRaisesRegex(paper_git.GitError, "Nothing to commit"):
            paper_git.commit(repo, "Nothing here")
        (repo / "main.py").write_text("print('changed')\n", encoding="utf-8")
        (repo / "notes.txt").write_text("new\n", encoding="utf-8")
        short, message = paper_git.commit(repo, "Update the greeting O")
        self.assertEqual(message, "Update the greeting")
        self.assertEqual(self.log(repo)[0], "Update the greeting")
        self.assertTrue(short)
        self.assertEqual(paper_git.run(repo, "status", "--porcelain"), "")

        (repo / "main.py").write_text("print('staged')\n", encoding="utf-8")
        (repo / "notes.txt").write_text("unstaged\n", encoding="utf-8")
        paper_git.run(repo, "add", "main.py")
        paper_git.commit(repo, "Only the staged file")
        self.assertIn("notes.txt", paper_git.run(repo, "status", "--porcelain"))

    def test_push_requires_a_remote_and_sets_upstream(self):
        repo = self.make_repo()
        with self.assertRaisesRegex(paper_git.GitError, "no remote"):
            paper_git.push(repo)
        remote = self.folder / "remote.git"
        subprocess.run(["git", "init", "--bare", str(remote)], check=True, capture_output=True)
        paper_git.run(repo, "remote", "add", "origin", str(remote))
        self.assertEqual(paper_git.push(repo), "origin/main")
        (repo / "main.py").write_text("print('again')\n", encoding="utf-8")
        paper_git.commit(repo, "Second commit")
        self.assertEqual(paper_git.push(repo), "main")
        self.assertEqual(paper_git.run(remote, "log", "--format=%s", "main").splitlines()[0], "Second commit")


class WorkbenchGitTests(GitTestCase):
    def workbench(self, repo=None):
        statuses = []
        return SimpleNamespace(git_repo=repo, language="Python", status=SimpleNamespace(set=statuses.append),
                               save_settings=lambda: None,
                               statuses=statuses)

    def test_repo_from_idea_scaffolds_commits_and_opens_vscode(self):
        import paper_workbench

        draft = Draft("Paper keypad idea", "preview", (), ".md", (("app/main.py", "def run():\n    return 1\n"),))
        bench = self.workbench()
        with patch.object(paper_workbench, "ROOT", self.folder), \
                patch.object(paper_workbench, "open_code") as opened:
            paper_workbench.PaperWorkbench.finish_git(bench, draft, "git_repo")
        repo = bench.git_repo
        self.assertTrue(repo.is_relative_to(self.folder / "repos"))
        self.assertTrue((repo / "app" / "main.py").is_file())
        self.assertEqual(self.log(repo), ["Initial scaffold from a handwritten idea"])
        opened.assert_called_once_with(repo)
        self.assertIn("Created repository", bench.statuses[-1])

    def test_branch_and_commit_from_handwriting(self):
        import paper_workbench

        repo = self.make_repo()
        bench = self.workbench(repo)
        paper_workbench.PaperWorkbench.finish_git(bench, Draft("New Plot Page", "", (), ".txt"), "git_branch")
        self.assertEqual(paper_git.run(repo, "branch", "--show-current"), "new-plot-page")
        (repo / "plot.py").write_text("x = 1\n", encoding="utf-8")
        paper_workbench.PaperWorkbench.finish_git(bench, Draft("Add plot page O", "", (), ".txt"), "git_commit")
        self.assertEqual(self.log(repo)[0], "Add plot page")
        self.assertIn("Committed", bench.statuses[-1])
        paper_workbench.PaperWorkbench.finish_git(bench, Draft("Again", "", (), ".txt"), "git_commit")
        self.assertIn("Nothing to commit", bench.statuses[-1])

    def test_diagram_reading_shows_plan_then_runs_in_background(self):
        import queue
        from functools import partial

        import paper_workbench

        repo = self.make_repo()
        bench = self.workbench(repo)
        shown = []
        bench.results = queue.SimpleQueue()
        bench.show_draft = shown.append
        bench._git_diagram = partial(paper_workbench.PaperWorkbench._git_diagram, bench)
        bench.run_git_diagram = partial(paper_workbench.PaperWorkbench.run_git_diagram, bench)
        reading = "branch docs from main\ncommit docs \"Write docs\"\nmerge docs into main\npush main"
        paper_workbench.PaperWorkbench.finish_git(bench, Draft(reading, reading, (), ".txt"), "diagram")
        self.assertTrue(bench.busy)
        self.assertIn("1. create branch docs from main", shown[0].content)
        self.assertIn("Running 4 steps", bench.statuses[-1])
        draft, figure, workflow = bench.results.get(timeout=60)
        self.assertEqual(workflow, "git_diagram_run")
        paper_workbench.PaperWorkbench.finish_git(bench, draft, workflow)
        self.assertIn("Git diagram done", bench.statuses[-1])
        self.assertIn("empty commit", bench.statuses[-1])
        self.assertIn("no remote", bench.statuses[-1])
        self.assertEqual(self.log(repo, "main")[0], "Merge docs into main")

    def test_paste_todo_commits_todo_file_and_text_becomes_commit_message(self):
        from functools import partial

        import paper_clipboard
        import paper_workbench

        repo = self.make_repo()
        bench = self.workbench(repo)
        bench.target = "Git"
        bench.active_repo = lambda: repo
        bench.run_git_diagram = lambda text: bench.statuses.append("diagram")
        paste = partial(paper_workbench.PaperWorkbench.paste_git, bench)
        (repo / "scratch.txt").write_text("unrelated\n", encoding="utf-8")
        paste(paper_clipboard.Clip("todo", "- [ ] write tests\n- [x] done already\n- [ ] add icons"))
        self.assertIn("added 2 items to TODO.md", bench.statuses[-1])
        self.assertEqual(self.log(repo)[0], "Add 2 TODO items from paper")
        self.assertIn("- add icons", paper_git.run(repo, "log", "-1", "--format=%b"))
        self.assertIn("?? scratch.txt", paper_git.run(repo, "status", "--porcelain"))
        paste(paper_clipboard.Clip("todo", "- [ ] write tests"))
        self.assertIn("already in TODO.md", bench.statuses[-1])
        paste(paper_clipboard.Clip("text", "Add scratch notes"))
        self.assertEqual(self.log(repo)[0], "Add scratch notes")
        paste(paper_clipboard.Clip("diagram", "commit main 'x'"))
        self.assertEqual(bench.statuses[-1], "diagram")
        paste(paper_clipboard.Clip("table", "a | b\n1 | 2"))
        self.assertIn("not a table", bench.statuses[-1])


class DiagramParseTests(unittest.TestCase):
    def test_parse_tolerates_noise_quotes_and_case(self):
        text = ("Here is the diagram:\n1. Branch Feature/Login from main\n- commit feature/login “Add login form”\n"
                "COMMIT on feature/login: fix typo\ncommit \"standalone\"\nMerge feature/login INTO main\n"
                "push main to origin\n```\ntag v1.0 on main\nsomething else\n")
        commands, ignored = paper_git.parse_diagram(text)
        self.assertEqual(commands, [
            ("branch", "Feature/Login", "main"), ("commit", "feature/login", "Add login form"),
            ("commit", "feature/login", "fix typo"), ("commit", "", "standalone"),
            ("merge", "feature/login", "main"), ("push", "main", ""), ("tag", "v1.0", "main")])
        self.assertEqual(ignored, ["Here is the diagram:", "something else"])


class DiagramTests(GitTestCase):
    def remote(self, repo):
        remote = self.folder / "remote.git"
        subprocess.run(["git", "init", "--bare", str(remote)], check=True, capture_output=True)
        paper_git.run(repo, "remote", "add", "origin", str(remote))
        return remote

    def test_plan_and_run_branches_commits_merge_push_and_tag(self):
        repo = self.make_repo()
        remote = self.remote(repo)
        (repo / "feature.py").write_text("x = 1\n", encoding="utf-8")
        text = ("branch feature from main\ncommit feature \"Add feature file\"\ncommit feature Second step\n"
                "merge feature into main\npush main\npush feature\ntag v1 on main\nnot a command")
        steps, problems, original = paper_git.plan_diagram(repo, text)
        self.assertEqual(original, "main")
        self.assertEqual([step.action for step in steps], ["branch", "commit", "commit", "merge", "push", "push", "tag"])
        self.assertEqual(problems, ["Ignored line: not a command"])
        plan = paper_git.describe_plan(steps, problems)
        self.assertIn("1. create branch feature from main", plan)
        self.assertIn("! Ignored line", plan)

        report, ok = paper_git.execute_plan(repo, steps, original)
        self.assertTrue(ok, report)
        self.assertEqual(paper_git.current_branch(repo), "main")
        self.assertEqual(report[-1], "Back on main.")
        self.assertTrue(any("with your current changes" in line for line in report))
        self.assertTrue(any("empty commit" in line for line in report))
        self.assertEqual(self.log(repo, "feature")[:2], ["Second step", "Add feature file"])
        self.assertEqual(self.log(repo, "main")[0], "Merge feature into main")
        self.assertEqual(len(paper_git.run(repo, "rev-list", "--parents", "-n", "1", "main").split()), 3)
        self.assertTrue((repo / "feature.py").is_file())
        self.assertEqual(paper_git.run(remote, "log", "--format=%s", "-n", "1", "main"), "Merge feature into main")
        self.assertEqual(paper_git.run(remote, "log", "--format=%s", "-n", "1", "feature"), "Second step")
        self.assertEqual(paper_git.run(repo, "tag", "--list"), "v1")

    def test_missing_parent_is_created_and_push_without_remote_is_skipped(self):
        repo = self.make_repo()
        paper_git.run(repo, "switch", "-c", "work")
        steps, problems, original = paper_git.plan_diagram(
            repo, "branch fix from develop\ncommit fix 'Fix it'\npush fix\nmerge ghost into main")
        self.assertEqual(original, "work")
        self.assertEqual(steps[0], paper_git.Step("branch", "develop", "work", "parent drawn without a fork"))
        self.assertIn("unknown branch", problems[0])
        report, ok = paper_git.execute_plan(repo, steps, original)
        self.assertTrue(ok, report)
        self.assertIn("Skipped push of fix: the repository has no remote.", report)
        self.assertEqual(paper_git.current_branch(repo), "work")
        self.assertEqual(paper_git.local_branches(repo), {"main", "work", "develop", "fix"})

    def test_conflicting_merge_is_aborted_and_the_original_branch_restored(self):
        repo = self.make_repo()
        steps, _, original = paper_git.plan_diagram(repo, "branch other from main")
        paper_git.execute_plan(repo, steps, original)
        for branch, text in (("other", "print('other')\n"), ("main", "print('main')\n")):
            paper_git.run(repo, "switch", branch)
            (repo / "main.py").write_text(text, encoding="utf-8")
            paper_git.commit(repo, f"Change on {branch}")
        head = paper_git.run(repo, "rev-parse", "main")
        steps, _, original = paper_git.plan_diagram(repo, "merge other into main\npush main")
        report, ok = paper_git.execute_plan(repo, steps, original)
        self.assertFalse(ok)
        self.assertTrue(any("conflicted" in line for line in report), report)
        self.assertEqual(paper_git.run(repo, "rev-parse", "main"), head)
        self.assertEqual(paper_git.run(repo, "status", "--porcelain"), "")
        self.assertEqual(paper_git.current_branch(repo), "main")

    def test_existing_branches_are_reused_and_nothing_is_forced(self):
        repo = self.make_repo()
        with self.assertRaisesRegex(paper_git.GitError, "No branch"):
            paper_git.plan_diagram(repo, "hello\nworld")
        steps, problems, _ = paper_git.plan_diagram(repo, "branch main\ncommit main 'Docs'")
        self.assertEqual([step.action for step in steps], ["commit"])
        self.assertEqual(problems, [])


class OpenTargetTests(unittest.TestCase):
    def test_matlab_reuses_a_running_instance(self):
        with tempfile.TemporaryDirectory() as location:
            project = Path(location)
            (project / "main.m").write_text("disp(1)\n", encoding="utf-8")
            with patch.object(paper_handoff, "_matlab_running", return_value=True), \
                    patch.object(paper_handoff, "_matlab_association", return_value=True), \
                    patch.object(paper_handoff.os, "startfile", create=True) as started, \
                    patch.object(paper_handoff.subprocess, "Popen") as launched:
                self.assertEqual(paper_handoff.open_matlab(project), project / "main.m")
            started.assert_called_once_with(project / "main.m")
            launched.assert_not_called()

    def test_matlab_starts_in_the_project_folder(self):
        with tempfile.TemporaryDirectory() as location:
            project = Path(location)
            (project / "solver.m").write_text("disp(1)\n", encoding="utf-8")
            with patch.object(paper_handoff, "_matlab_running", return_value=False), \
                    patch.object(paper_handoff.shutil, "which", return_value="matlab.exe"), \
                    patch.object(paper_handoff.subprocess, "Popen") as launched:
                paper_handoff.open_matlab(project)
            command = launched.call_args.args[0]
            self.assertEqual(command[:3], ["matlab.exe", "-sd", project.as_posix()])
            self.assertIn("edit('", command[-1])
            self.assertIn("solver.m", command[-1])
            with self.assertRaises(OSError):
                paper_handoff.open_matlab(project / "missing")

    def test_generated_code_and_projects_open_automatically(self):
        import paper_workbench

        bench = SimpleNamespace(target="VS Code", language="Python", menu="vscode", report_folder=None,
                                draft=Draft("x", "print(1)\n", (), ".py"), saved=None,
                                status=SimpleNamespace(set=lambda _text: None))
        bench.open_path = lambda path: paper_workbench.PaperWorkbench.open_path(bench, path)
        with tempfile.TemporaryDirectory() as location, \
                patch.object(paper_workbench, "ROOT", Path(location)), \
                patch.object(paper_workbench, "open_code") as opened:
            paper_workbench.PaperWorkbench.open_generated(bench)
            self.assertEqual(opened.call_args.args[0].suffix, ".py")
            bench.target, bench.menu = "LaTeX", "latex"
            bench.draft = Draft("t", "x", (), ".tex", (("main.tex", "\\documentclass{article}\n"),))
            paper_workbench.PaperWorkbench.open_generated(bench)
            self.assertEqual(opened.call_args.args[0].name, "main.tex")


if __name__ == "__main__":
    unittest.main()
