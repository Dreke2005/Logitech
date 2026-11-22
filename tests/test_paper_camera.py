import unittest

from paper_camera_check import choose_camera


class CameraChoiceTests(unittest.TestCase):
    def test_logiinsight_rgb_is_preferred_and_a_webcam_is_the_fallback(self):
        names = ["Integrated Camera", "LVNext Camera Hello", "LVNext Camera CPD", "LVNext Camera"]
        self.assertEqual(choose_camera(names), (3, True))
        self.assertEqual(choose_camera(["Integrated IR Camera", "Integrated Camera"]), (1, False))
        self.assertEqual(choose_camera(["OBS Virtual Camera"]), (None, False))
        self.assertEqual(choose_camera(names, 0), (0, False))
        self.assertEqual(choose_camera(names, 9), (None, False))


if __name__ == "__main__":
    unittest.main()
