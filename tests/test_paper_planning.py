import json
import subprocess
import tempfile
import unittest
from datetime import date
from pathlib import Path

import paper_forge
import paper_git
import paper_plan
import paper_project
from paper_context import validate_reading


def git(repo, *args):
    subprocess.run(["git", "-C", str(repo), *args], check=True, capture_output=True)


class PlanTests(unittest.TestCase):
    PLAN = ("task: Literature review, 2w\ntask: Prototype, 3w\nmilestone: Demo\n"
            "task: Report, 10d, after Prototype\nmilestone: Hand-in")

    def test_tasks_follow_each_other_and_after_overrides(self):
        items, problems = paper_plan.parse_plan(self.PLAN)
        self.assertEqual((len(items), problems), (5, []))
        self.assertEqual(paper_plan.schedule(items, date(2026, 1, 5)), [])
        review, prototype, demo, report, hand_in = items
        self.assertEqual((review.start, review.end), (date(2026, 1, 5), date(2026, 1, 19)))
        self.assertEqual(prototype.end, date(2026, 2, 9))
        self.assertTrue(demo.milestone and demo.start == prototype.end)
        self.assertEqual((report.start, report.end), (prototype.end, date(2026, 2, 19)))
        self.assertEqual(hand_in.start, report.end)
        chart = paper_plan.mermaid(items)
        self.assertIn("Prototype :t1, 2026-01-19, 21d", chart)
        self.assertIn("Demo :milestone, m2, 2026-02-09, 0d", chart)

    def test_unknown_lines_and_references_are_reported(self):
        items, problems = paper_plan.parse_plan("task: A, 1w\nsomething else\ntask: B, 2d, after Nowhere")
        self.assertEqual(len(items), 2)
        self.assertIn("Not a task or milestone", problems[0])
        self.assertIn("Nowhere", paper_plan.schedule(items, date(2026, 1, 5))[0])
        self.assertEqual(validate_reading("Plan", "gantt", self.PLAN), (True, ""))
        self.assertFalse(validate_reading("Plan", "gantt", "buy milk")[0])

    def test_chart_is_rendered_as_png_and_pdf(self):
        items, _ = paper_plan.parse_plan(self.PLAN)
        paper_plan.schedule(items, date(2026, 1, 5))
        with tempfile.TemporaryDirectory() as folder:
            chart = paper_plan.render(items, Path(folder) / "gantt.png")
            self.assertGreater(chart.stat().st_size, 2000)
            self.assertTrue(chart.with_suffix(".pdf").read_bytes().startswith(b"%PDF-"))


class CurrentProjectTests(unittest.TestCase):
    def test_the_front_vscode_window_decides_the_project(self):
        with tempfile.TemporaryDirectory() as folder:
            first, second = Path(folder) / "alpha", Path(folder) / "beta"
            first.mkdir()
            second.mkdir()
            storage = Path(folder) / "storage.json"
            storage.write_text(json.dumps({"windowsState": {
                "lastActiveWindow": {"folder": first.as_uri()},
                "openedWindows": [{"folder": second.as_uri()}]}}), encoding="utf-8")
            folders = paper_project.vscode_folders(storage)
            self.assertEqual(folders, [first, second])
            titles = ["main.py - beta - Visual Studio Code", "notes.md - alpha - Visual Studio Code"]
            self.assertEqual(paper_project.vscode_project(titles, folders), second)
            self.assertEqual(paper_project.vscode_project([], folders), first)
            self.assertIsNone(paper_project.vscode_project(["x - gamma - Visual Studio Code"], folders))
            report = Path(folder) / "report"
            report.mkdir()
            self.assertEqual(paper_project.current_folder("latex", None, report), report)
            saved = first / "main.m"
            saved.write_text("x = 1;\n", encoding="utf-8")
            self.assertEqual(paper_project.current_folder("matlab", saved), first)


class RemoteTests(unittest.TestCase):
    def test_host_kind_and_pull(self):
        with tempfile.TemporaryDirectory() as folder:
            origin, repo = Path(folder) / "origin.git", Path(folder) / "work"
            git(folder, "init", "--bare", "-b", "main", str(origin))
            git(folder, "clone", str(origin), str(repo))
            with self.assertRaisesRegex(paper_git.GitError, "neither GitHub nor GitLab"):
                paper_forge.kind_of(repo)
            git(repo, "remote", "set-url", "origin", "git@gitlab.example.org:grp/r.git")
            self.assertEqual(paper_forge.kind_of(repo), "gitlab")
            git(repo, "remote", "set-url", "origin", "https://github.com/o/r.git")
            self.assertEqual(paper_forge.kind_of(repo), "github")
            git(repo, "remote", "remove", "origin")
            with self.assertRaisesRegex(paper_git.GitError, "no remote"):
                paper_git.pull(repo)


if __name__ == "__main__":
    unittest.main()
