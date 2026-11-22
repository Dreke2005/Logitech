import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

import paper_forge
import paper_git
from paper_check import check_derivation
from paper_context import project_hints, validate_reading, writing_context
from paper_conversion import _dangling_nets, convert_page, read_recording
from paper_selection import PaperSelection
from paper_writing import WritingSelection


def git(repo, *args):
    subprocess.run(["git", "-C", str(repo), *args], check=True, capture_output=True)


class DerivationCheckTests(unittest.TestCase):
    def test_a_wrong_step_is_named_and_definitions_are_left_alone(self):
        self.assertEqual(check_derivation("y = x^2 + 1"), ())
        self.assertEqual(check_derivation("(x+1)^2 = x^2 + 2x + 1"), ())
        notes = check_derivation("y = (x+1)^{2} = x^{2} + 2x + 2")
        self.assertEqual(len(notes), 1)
        self.assertIn("does not hold", notes[0])

    def test_question_mark_gives_the_result(self):
        self.assertEqual(check_derivation(r"\frac{1}{2} + \frac{1}{3} = ?"), (r"Result: \frac{1}{2} + \frac{1}{3} = 5/6",))

    def test_text_that_is_not_math_is_not_checked(self):
        self.assertEqual(check_derivation("for i in range(3):\n    total = total + i"), ())


class ProjectHintTests(unittest.TestCase):
    def test_names_come_from_the_project_and_its_branches(self):
        with tempfile.TemporaryDirectory() as folder:
            repo = Path(folder)
            (repo / "model.py").write_text("class Trainer:\n    pass\n\ndef gradient_step(w):\n    return w\n",
                                           encoding="utf-8")
            (repo / ".venv").mkdir()
            (repo / ".venv" / "junk.py").write_text("def hidden_helper():\n    pass\n", encoding="utf-8")
            git(repo, "init", "-b", "main")
            git(repo, "-c", "user.name=t", "-c", "user.email=t@example.com", "commit", "--allow-empty", "-m", "x")
            git(repo, "branch", "feature/login")
            self.assertEqual(project_hints("VS Code", repo), ("Trainer", "gradient_step"))
            self.assertEqual(project_hints("Git", repo)[:2], ("feature/login", "main"))
        self.assertEqual(project_hints("VS Code", None), ())

    def test_hints_reach_the_reader_prompt(self):
        strokes = tuple(np.asarray([(300 + 20 * i, 200), (310 + 20 * i, 230), (300 + 20 * i, 215)]) for i in range(6))
        selection = WritingSelection(PaperSelection(np.full((60, 160, 3), 225, np.uint8), None, "full"),
                                     strokes=strokes, text_height=30.0)
        reply = {"source": "gradient_step(w)", "notes": [], "kind": "code", "clear": True}
        with patch("paper_conversion.chat", return_value=reply) as model:
            read_recording(selection, "VS Code", "Python", "default", "local", hints=("Trainer", "gradient_step"))
        self.assertIn("Trainer, gradient_step", model.call_args.args[2])


class FlowchartTests(unittest.TestCase):
    def test_mermaid_reading_is_validated(self):
        good = "flowchart TD\nA[Start] --> B{Valid?}\nB -->|yes| C[Save]"
        self.assertEqual(validate_reading("ChatGPT", "flow", good), (True, ""))
        self.assertFalse(validate_reading("ChatGPT", "flow", "Start then save")[0])
        self.assertFalse(validate_reading("ChatGPT", "flow", "flowchart TD\nA[Start --> B")[0])
        self.assertFalse(writing_context("ChatGPT", "flow").end_circle)


class CircuitTests(unittest.TestCase):
    def test_a_net_on_one_pin_is_read_again(self):
        broken = {"source": "", "notes": [], "directive": "", "components": [
            {"reference": "V1", "value": "5", "nets": ["in", "GND"]},
            {"reference": "R1", "value": "1k", "nets": ["n1", "out"]},
            {"reference": "C1", "value": "100n", "nets": ["out", "GND"]}]}
        fixed = {**broken, "components": [broken["components"][0],
                                         {"reference": "R1", "value": "1k", "nets": ["in", "out"]},
                                         broken["components"][2]]}
        self.assertEqual(_dangling_nets(broken["components"]), ["in", "n1"])
        with patch("paper_conversion.chat", side_effect=[broken, fixed]) as model:
            draft = convert_page(np.zeros((40, 60, 3), np.uint8), "LTspice", None, "local")
        self.assertEqual(model.call_count, 2)
        self.assertIn("in, n1", model.call_args.args[2])
        self.assertTrue(any("R1 in out 1k" in note for note in draft.notes))


class ForgeTests(unittest.TestCase):
    def setUp(self):
        self.folder = tempfile.TemporaryDirectory()
        self.addCleanup(self.folder.cleanup)
        self.repo = Path(self.folder.name) / "paper-demo"
        self.repo.mkdir()
        git(self.repo, "init", "-b", "main")
        git(self.repo, "-c", "user.name=t", "-c", "user.email=t@example.com", "commit", "--allow-empty", "-m", "x")

    def test_remote_urls_are_parsed(self):
        self.assertEqual(paper_forge.parse_remote("https://github.com/o/r.git"), ("github.com", "o/r"))
        self.assertEqual(paper_forge.parse_remote("git@gitlab.example.org:grp/sub/r.git"),
                         ("gitlab.example.org", "grp/sub/r"))
        self.assertIsNone(paper_forge.parse_remote("not a url"))

    def test_issue_and_request_links_carry_the_writing(self):
        git(self.repo, "remote", "add", "origin", "git@github.com:o/r.git")
        url = paper_forge.issue_url(self.repo, "github", "Crop too large\nOn a light desk")
        self.assertEqual(url, "https://github.com/o/r/issues/new?title=Crop%20too%20large&body=On%20a%20light%20desk")
        with self.assertRaisesRegex(paper_git.GitError, "Create a branch"):
            paper_forge.request_url(self.repo, "github", "Title")
        git(self.repo, "switch", "-c", "feature/login")
        self.assertEqual(paper_forge.request_url(self.repo, "github", "Add login"),
                         "https://github.com/o/r/compare/main...feature%2Flogin?expand=1&title=Add%20login&body=")
        with self.assertRaisesRegex(paper_git.GitError, "not on GitLab"):
            paper_forge.issue_url(self.repo, "gitlab", "x")

    def test_gitlab_merge_request_and_publish(self):
        with self.assertRaisesRegex(paper_git.GitError, "Publish"):
            paper_forge.web_url(self.repo, "gitlab")
        self.assertEqual(paper_forge.new_repository_url(self.repo, "github"), "https://github.com/new?name=paper-demo")
        with self.assertRaisesRegex(paper_git.GitError, "Copy the new repository"):
            paper_forge.connect(self.repo, "gitlab", "https://github.com/o/r.git")
        self.assertEqual(paper_forge.connect(self.repo, "gitlab", "https://gitlab.com/grp/r.git"), "grp/r")
        git(self.repo, "switch", "-c", "fix")
        url = paper_forge.request_url(self.repo, "gitlab", "Fix crop\nDetails")
        self.assertTrue(url.startswith("https://gitlab.com/grp/r/-/merge_requests/new?"))
        self.assertIn("merge_request%5Bsource_branch%5D=fix", url)
        self.assertIn("merge_request%5Btarget_branch%5D=main", url)


if __name__ == "__main__":
    unittest.main()
