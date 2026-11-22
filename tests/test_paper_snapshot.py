"""Snapshot targeting: the sheet of paper, the blocks of writing on it, and the block the pen points at."""
import hermetic  # noqa: F401
import unittest
from pathlib import Path

import cv2
import numpy as np

import paper_bench
from paper_snapshot import find_paper, snapshot_region, writing_blocks

FONT = Path("C:/Windows/Fonts/Inkfree.ttf")
EQUATIONS = (("y = x^2 + 1", (520, 330)), ("f(x) = 3x - 2", (520, 520)), ("E = m c^2", (1000, 420)))


def desk_with_sheet(equations=EQUATIONS, height=26):
    """A 1080p frame: a dark wooden desk with a slightly turned white sheet and handwritten equations."""
    frame = np.full((1080, 1920, 3), (60, 85, 120), np.uint8)  # brownish desk
    sheet = np.float32([[380, 180], [1500, 150], [1530, 900], [400, 930]])
    cv2.fillConvexPoly(frame, sheet.astype(np.int32), (222, 224, 226))
    for text, (x, y) in equations:
        glyphs = paper_bench.write(text, FONT)
        scale = height / glyphs.shape[0] * 1.6
        small = cv2.resize(glyphs, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        h, w = small.shape
        ink = (255 - small.astype(np.float32)) / 255.0
        frame[y:y + h, x:x + w] = np.clip(frame[y:y + h, x:x + w] - ink[:, :, None] * 120, 0, 255).astype(np.uint8)
    noise = np.random.default_rng(0).normal(0, 2.0, frame.shape)
    return np.clip(frame + noise, 0, 255).astype(np.uint8), sheet


@unittest.skipUnless(FONT.is_file(), "needs the Ink Free font")
class SnapshotTests(unittest.TestCase):
    def setUp(self):
        self.frame, self.sheet = desk_with_sheet()

    def test_the_sheet_is_found_on_the_desk(self):
        paper = find_paper(self.frame)
        self.assertTrue(paper.found)
        x0, y0, x1, y1 = paper.bounds
        self.assertTrue(360 <= x0 <= 410 and 130 <= y0 <= 170 and 1500 <= x1 <= 1560 and 900 <= y1 <= 950,
                        paper.bounds)

    def test_each_equation_is_its_own_block(self):
        blocks = writing_blocks(self.frame, find_paper(self.frame))
        self.assertEqual(len(blocks), 3, [block.bounds for block in blocks])

    def test_the_pen_picks_the_equation_it_points_at(self):
        for index, (_, (x, y)) in enumerate(EQUATIONS):
            with self.subTest(equation=index):
                tip = (x + 60, y + 50)  # resting just below the start of that equation
                bounds, reason = snapshot_region(self.frame, tip)
                self.assertEqual(reason, "pointed")
                self.assertTrue(bounds[0] <= x + 5 and bounds[1] <= y + 5 and bounds[3] < y + 90, bounds)
                others = [position for number, (_, position) in enumerate(EQUATIONS) if number != index]
                for ox, oy in others:
                    inside = bounds[0] <= ox + 20 <= bounds[2] and bounds[1] <= oy + 15 <= bounds[3]
                    self.assertFalse(inside, (bounds, (ox, oy)))

    def test_without_a_pen_several_equations_give_the_whole_sheet(self):
        bounds, reason = snapshot_region(self.frame, None)
        self.assertEqual(reason, "paper")
        self.assertTrue(bounds[0] <= 400 and bounds[2] >= 1500)
        bounds, reason = snapshot_region(self.frame, (200, 1000))  # pen far off the writing
        self.assertEqual(reason, "paper")

    def test_a_single_equation_is_read_on_its_own(self):
        frame, _ = desk_with_sheet(EQUATIONS[:1])
        bounds, reason = snapshot_region(frame, None)
        self.assertEqual(reason, "single")
        self.assertLess(bounds[2] - bounds[0], 500)


if __name__ == "__main__":
    unittest.main()
