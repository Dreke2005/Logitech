import queue
import time
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np

from paper_conversion import Draft
from paper_selection import PaperSelection
from paper_workbench import PaperWorkbench
from paper_writing import WritingSelection


class WorkbenchWorkspaceTests(unittest.TestCase):
    def test_camera_action_defers_to_the_workspace_when_another_mode_is_active(self):
        resume = Mock()
        bench = SimpleNamespace(menu="vscode", status=Mock(), workspace_workflow=None)
        state = {"session": "workspace", "mode": "keyboard", "notes_enabled": False}
        with patch("paper_workbench.workspace_active", return_value=state), patch(
                "paper_workbench.request_notes") as request:
            self.assertTrue(PaperWorkbench.request_notes_camera(bench, resume))
        request.assert_called_once_with(state)
        self.assertEqual(bench.workspace_workflow, ("vscode", resume))

    def test_sharper_desk_and_different_paper_are_ignored(self):
        frame = np.full((160, 224, 3), 212, np.uint8)
        changed_desk = frame.copy()
        changed_desk[:, :36, :] = np.tile([0, 255], 18)[None, :, None]
        changed_page = frame.copy()
        changed_page[30:120, 60:180, :] = np.tile([0, 0, 0, 0, 255, 255, 255, 255], 15)[None, :, None]
        now = time.monotonic()
        bench = SimpleNamespace(frame=frame, writing_area=(60, 30, 180, 120),
                                writing=SimpleNamespace(area=None), sheet_bounds=None,
                                recent_frames=[(now, changed_desk), (now, changed_page)])
        self.assertIs(PaperWorkbench.sharpest_frame(bench), frame)

    def test_snapshot_ignores_a_pen_point_outside_the_selected_area(self):
        frame = np.full((160, 224, 3), 212, np.uint8)
        pen = SimpleNamespace(point=(190, 50), back=(181, 50))
        writing = SimpleNamespace(pointer=Mock(return_value=pen.point), pen=pen, area=None,
                                  pen_rotation=Mock(return_value=0), hidden_mask=Mock(return_value=None))
        bench = SimpleNamespace(page=frame, roi=None, writer_rotation=0, writing=writing,
                                writing_area=(45, 30, 168, 125), show_selection=Mock())
        with patch("paper_workbench.snapshot_region", return_value=((0, 0, 224, 160), "frame")) as snapshot:
            selected = PaperWorkbench.selected_page(bench, pointed=True)
        self.assertEqual(selected.page.bounds, bench.writing_area)
        self.assertIsNone(snapshot.call_args.args[1])
        self.assertIsNone(snapshot.call_args.args[3])

    def test_live_circuit_drawing_goes_through_structured_reader(self):
        frame = np.full((40, 60, 3), 220, np.uint8)
        selection = WritingSelection(PaperSelection(frame, (0, 0, 60, 40), "manual"),
                                     evidence=frame.copy(), rotation=90)
        bench = SimpleNamespace(args=SimpleNamespace(model="model"), results=queue.SimpleQueue(),
                                _circuit_check=Mock())
        draft = Draft("R1 1k IN OUT", "schematic", (), ".asc")
        with patch("paper_workbench.PaperWorkbench._orchestrated", return_value=None), patch(
                "paper_workbench.convert_page", return_value=draft) as convert:
            PaperWorkbench._generate(bench, selection, "LTspice", "Python", "circuit", recorded=True)
        self.assertEqual(bench.results.get_nowait(), (draft, None, "circuit"))
        self.assertEqual(convert.call_args.args[0].shape, (60, 40, 3))
        self.assertEqual(convert.call_args.kwargs["evidence"].shape, (60, 40, 3))
        self.assertIs(convert.call_args.kwargs["check"], bench._circuit_check)
