"""Verify workspace transitions without opening a camera or using real hotkeys."""

import tempfile
import unittest
from contextlib import ExitStack
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import hermetic  # noqa: F401

import rgb_preview
import workspace_modes
from desktop_input import WindowsInput


class WorkspaceIpcTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        workspace = patch.object(workspace_modes, "ROOT", Path(temporary.name))
        workspace.start()
        self.addCleanup(workspace.stop)
        self.ipc = workspace_modes.WorkspaceIpc()
        self.ipc.set_mode("keyboard")

    def test_paper_commands_require_current_session_and_matching_acknowledgment(self):
        self.ipc.set_mode("notes", notes_enabled=True)
        start = self.ipc.command("start", "workbench", device=2)
        self.assertIsNone(workspace_modes.notes_command("old"))
        self.assertEqual(workspace_modes.notes_command("workbench"), start)
        workspace_modes.acknowledge_notes(start, True)
        self.assertTrue(self.ipc.acknowledged(start, True))

        self.ipc.set_mode("keyboard")
        self.assertIsNone(workspace_modes.notes_command("workbench"))
        stop = self.ipc.command("stop", "workbench")
        self.assertEqual(workspace_modes.notes_command("workbench"), stop)
        self.assertFalse(self.ipc.acknowledged(stop, False))
        workspace_modes.acknowledge_notes(stop, False)
        self.assertTrue(self.ipc.acknowledged(stop, False))

    def test_notes_requests_are_consumed_once_and_malformed_state_is_ignored(self):
        active = workspace_modes.workspace_active()
        self.assertFalse(self.ipc.consume_notes_request())
        workspace_modes.request_notes(active)
        self.assertTrue(self.ipc.consume_notes_request())
        self.assertFalse(self.ipc.consume_notes_request())
        (workspace_modes.ROOT / "state.json").write_text("[]", encoding="utf-8")
        self.assertIsNone(workspace_modes.workspace_active())
        self.assertIsNone(workspace_modes.notes_command("workbench"))

    def test_workbench_acknowledges_the_start_and_releases_camera_before_stop_ack(self):
        from paper_workbench import PaperWorkbench

        bench = SimpleNamespace(keypad=SimpleNamespace(id="paper"), args=SimpleNamespace(camera=None),
                                camera=None, menu="start", window=Mock(), workspace_command=None,
                                workspace_workflow=None, workspace_hotkey=None, workspace_process=None)

        def start_camera():
            bench.camera = object()

        bench.start_from_webcam = Mock(side_effect=start_camera)
        self.ipc.set_mode("notes", notes_enabled=True)
        start = self.ipc.command("start", bench.keypad.id, device=2)
        PaperWorkbench.follow_workspace(bench)
        self.assertEqual(bench.args.camera, 2)
        bench.start_from_webcam.assert_called_once_with()
        self.assertTrue(self.ipc.acknowledged(start, True))

        self.ipc.set_mode("keyboard")
        stop = self.ipc.command("stop", bench.keypad.id)
        bench.stop_camera = Mock(return_value=False)
        PaperWorkbench.follow_workspace(bench)
        self.assertFalse(self.ipc.acknowledged(stop, False))

        def stop_camera():
            bench.camera = None
            return True

        bench.stop_camera.side_effect = stop_camera
        PaperWorkbench.follow_workspace(bench)
        self.assertTrue(self.ipc.acknowledged(stop, False))


class WorkspaceTransitionsTests(unittest.TestCase):
    def test_launch_chord_is_consumed_and_each_press_advances_one_mode(self):
        api = Mock()
        keys = {0x11, 0x12, 0x78}
        api.GetAsyncKeyState.side_effect = lambda code: 0x8000 if code in keys else 0
        desktop = WindowsInput(api)
        ipc = Mock()
        ipc.consume_notes_request.return_value = False
        with patch("desktop_input.WindowsInput", return_value=desktop):
            workspace = rgb_preview._Workspace(ipc, 0)
        self.assertEqual(workspace.poll("keyboard"), (None, True))
        keys.clear()
        self.assertEqual(workspace.poll("keyboard"), (None, False))
        for current, following in zip(workspace_modes.MODES, (*workspace_modes.MODES[1:], "keyboard")):
            keys.update((0x11, 0x12, 0x78))
            self.assertEqual(workspace.poll(current), (following, True))
            self.assertEqual(workspace.poll(current), (None, True))
            keys.clear()
            self.assertEqual(workspace.poll(current), (None, False))
        ipc.consume_notes_request.return_value = True
        self.assertEqual(workspace.poll("eye"), ("notes", False))

    def test_rgb_waits_for_workbench_ack_even_with_camera_flag_off(self):
        ipc = Mock()
        ipc.acknowledged.side_effect = (False, True)
        with ExitStack() as stack:
            stack.enter_context(patch("desktop_input.WindowsInput"))
            stack.enter_context(patch("paper_keypad.current_state",
                                      return_value={"session": "paper", "cameraActive": False}))
            sleep = stack.enter_context(patch("rgb_preview.time.sleep"))
            workspace = rgb_preview._Workspace(ipc, 0)
            workspace.stop_notes("keyboard")
        ipc.set_mode.assert_called_once_with("keyboard")
        ipc.command.assert_called_once_with("stop", "paper")
        self.assertEqual(ipc.acknowledged.call_count, 2)
        sleep.assert_called_once()
