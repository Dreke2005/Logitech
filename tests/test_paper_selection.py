import hermetic  # noqa: F401
import unittest

import cv2
import numpy as np

from paper_selection import select_paper_region


def equation(page, position, text="A*x = b"):
    cv2.putText(page, text, position, cv2.FONT_HERSHEY_SIMPLEX, 1.1, (22, 22, 22), 2, cv2.LINE_AA)


class PaperSelectionTests(unittest.TestCase):
    def setUp(self):
        self.page = np.full((640, 960, 3), 245, dtype=np.uint8)

    def test_oval_keeps_written_equation_and_omits_nearby_notes(self):
        equation(self.page, (380, 332))
        cv2.ellipse(self.page, (461, 320), (128, 62), 0, 0, 360, (30, 60, 170), 4)
        equation(self.page, (290, 500), "Unrelated note")
        original = self.page.copy()
        selection = select_paper_region(self.page)
        self.assertEqual(selection.kind, "circle")
        left, top, right, bottom = selection.bounds
        self.assertLess(left, 380)
        self.assertLess(top, 304)
        self.assertGreater(right, 510)
        self.assertLess(bottom, 420)
        self.assertTrue(np.any(selection.image < 80))
        self.assertFalse(np.any(np.ptp(selection.image, axis=2) > 30))
        np.testing.assert_array_equal(original, self.page)

    def test_oval_survives_perspective_like_rotation_and_shadows(self):
        lighting = np.linspace(152, 252, 1600, dtype=np.uint8)[None, :, None]
        page = np.broadcast_to(lighting, (900, 1600, 3)).copy()
        equation(page, (624, 470), "x^2 + y^2 = r^2")
        cv2.ellipse(page, (768, 460), (225, 87), 18, 0, 360, (22, 33, 128), 5)
        selection = select_paper_region(page)
        self.assertEqual(selection.kind, "circle")
        self.assertLess(selection.bounds[0], 625)
        self.assertGreater(selection.bounds[2], 845)
        self.assertLess(selection.bounds[3], 605)

    def test_a_small_gap_in_the_oval_can_be_bridged(self):
        equation(self.page, (380, 332))
        cv2.ellipse(self.page, (461, 320), (128, 62), 0, 0, 360, (25, 25, 25), 4)
        self.page[254:264, 461] = 245
        self.assertEqual(select_paper_region(self.page).kind, "circle")

    def test_long_equation_uses_a_wide_oval(self):
        equation(self.page, (340, 333), "A*x = b + c*d")
        cv2.ellipse(self.page, (461, 320), (300, 41), 0, 0, 360, (25, 25, 25), 4)
        self.assertEqual(select_paper_region(self.page).kind, "circle")

    def test_full_frame_used_for_open_circle_or_empty_ring(self):
        equation(self.page, (380, 332))
        cv2.ellipse(self.page, (461, 320), (128, 62), 0, 10, 312, (25, 25, 25), 4)
        cv2.ellipse(self.page, (761, 320), (90, 62), 0, 0, 360, (25, 25, 25), 4)
        selection = select_paper_region(self.page)
        self.assertEqual((selection.kind, selection.bounds), ("full", None))
        self.assertIs(selection.image, self.page)

    def test_multiple_circles_require_an_explicit_selection(self):
        for x in (250, 690):
            equation(self.page, (x - 80, 330))
            cv2.ellipse(self.page, (x, 320), (128, 62), 0, 0, 360, (25, 25, 25), 4)
        selection = select_paper_region(self.page)
        self.assertEqual((selection.kind, selection.reason), ("full", "multiple circles"))

    def test_copy_prefers_dashed_selection_over_existing_solid_circle(self):
        equation(self.page, (570, 340), "y=x^2+2x+1")
        for angle in range(0, 360, 20):
            cv2.ellipse(self.page, (690, 330), (158, 78), 0, angle, angle + 14, (20, 20, 20), 3)
        equation(self.page, (160, 320))
        cv2.ellipse(self.page, (228, 310), (122, 60), 0, 0, 360, (20, 20, 20), 4)
        solid = select_paper_region(self.page)
        dashed = select_paper_region(self.page, allow_dashed=True)
        self.assertLess(solid.bounds[2], 400)
        self.assertEqual(dashed.kind, "circle")
        self.assertGreater(dashed.bounds[0], 500)
        self.assertTrue(np.any(dashed.image < 80))

    def test_manual_selection_is_clipped_and_takes_precedence(self):
        equation(self.page, (380, 332))
        cv2.ellipse(self.page, (461, 320), (128, 62), 0, 0, 360, (25, 25, 25), 4)
        selection = select_paper_region(self.page, (720, 54, 999, -48))
        self.assertEqual((selection.kind, selection.bounds), ("manual", (720, 0, 960, 54)))
        np.testing.assert_array_equal(selection.image, self.page[:54, 720:])
        with self.assertRaisesRegex(ValueError, "larger selection"):
            select_paper_region(self.page, (11, 11, 14, 17))
