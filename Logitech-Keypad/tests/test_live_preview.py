"""Exercise the live RGB lip-reading entry point without capturing a webcam."""

import queue
import sys
import unittest
from contextlib import ExitStack
from pathlib import Path
from unittest.mock import patch

import cv2
import numpy as np

import rgb_preview


class LiveLipPreviewTests(unittest.TestCase):
    def test_space_submits_raw_rgb_and_result_appears_even_without_a_new_frame(self):
        image = np.zeros((32, 200, 3), dtype=np.uint8)
        frame = ("frame", (image, None, None))
        here = Path(__file__).resolve()
        with ExitStack() as stack:
            enter = stack.enter_context
            enter(patch.object(sys, "argv", ["rgb_preview.py", "--mode", "lip"]))
            enter(patch.object(rgb_preview, "_lip_model_paths", return_value=(here.parent, here, here)))
            cameras = enter(patch("pygrabber.dshow_graph.FilterGraph")).return_value
            cameras.get_input_devices.return_value = ["LVNext Camera"]
            desktop = enter(patch("desktop_input.WindowsInput"))
            ctx = enter(patch.object(rgb_preview.mp, "get_context")).return_value
            ctx.Queue.return_value.get.side_effect = [frame, frame, queue.Empty]
            reader = enter(patch("live_lip.LiveLipReader")).return_value
            reader.phase = "ready"

            def start():
                reader.phase = "recording"
                return True

            reader.start.side_effect = start
            reader.status = "Visual speech status"
            reader.poll.side_effect = [None, None, "Hello world"]
            display = enter(patch.object(cv2, "imshow"))
            put_text = enter(patch.object(cv2, "putText", wraps=cv2.putText))
            enter(patch.object(cv2, "waitKey", side_effect=[ord(" "), ord(" "), ord("q")]))
            enter(patch.object(cv2, "getWindowProperty", return_value=1))
            enter(patch.object(cv2, "destroyAllWindows"))
            self.assertEqual(rgb_preview.main(), 0)

        desktop.assert_not_called()
        reader.start.assert_called_once_with()
        reader.stop.assert_called_once_with()
        reader.close.assert_called_once_with()
        self.assertIs(reader.add.call_args.args[0], image)
        self.assertIn("Hello world", [call.args[1] for call in put_text.call_args_list])
        self.assertGreater(display.call_args.args[1].shape[0], image.shape[0])
        self.assertFalse(image.any())
