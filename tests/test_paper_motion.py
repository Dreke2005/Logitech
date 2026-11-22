"""The sheet is followed while it slides and turns, and its outline is calibrated on a clear view."""
import hermetic  # noqa: F401
import unittest

import cv2
import numpy as np

from paper_motion import SheetTracker, SheetWatch, displacement, move_points
from paper_writing import WritingSession

SIZE = (1280, 720)
CORNERS = np.float32([[380, 190], [920, 190], [920, 530], [380, 530]])  # The sheet on the desk at the start.


def sheet_picture(blank=False, seed=0):
    """A sheet lying on a desk, as one picture in desk coordinates: (desk, sheet layer, sheet mask)."""
    rng = np.random.default_rng(seed)
    width, height = SIZE
    desk = np.full((height, width, 3), 150, np.uint8)
    cv2.rectangle(desk, (0, 600), (width, height), (40, 40, 40), -1)  # A laptop in front.
    cv2.line(desk, (150, 0), (120, 600), (120, 120, 120), 3)  # A seam in the desk: it never moves.
    sheet = np.zeros((height, width, 3), np.uint8)
    mask = np.zeros((height, width), np.uint8)
    cv2.fillConvexPoly(mask, CORNERS.astype(np.int32), 255)
    sheet[mask != 0] = 222
    if not blank:
        # A few lines of small handwriting.
        letters = "abcdefghkmnpqrstuvwxyz=+()23"
        for line in range(4):
            text = "".join(letters[int(i)] for i in rng.integers(0, len(letters), 16 + 3 * line))
            cv2.putText(sheet, text, (420, 250 + 62 * line), cv2.FONT_HERSHEY_SCRIPT_SIMPLEX, 0.85, (95, 95, 95), 2,
                        cv2.LINE_AA)
    return desk, sheet, mask


def pose_of(turn=0.0, shift=(0.0, 0.0), centre=(650.0, 360.0)):
    """A turn about the sheet's middle and a slide, as a 3x3 matrix (desk at the start -> frame)."""
    matrix = np.vstack([cv2.getRotationMatrix2D(centre, turn, 1.0), (0, 0, 1)])
    matrix[:2, 2] += shift
    return matrix


def shoot(scene, pose=None, hands=(), seed=0):
    """The camera frame with the sheet at ``pose`` and skin-coloured hands (centre, axes) on top."""
    desk, sheet, mask = scene
    pose = np.eye(3) if pose is None else pose
    moved = cv2.warpAffine(sheet, pose[:2].astype(np.float32), SIZE, flags=cv2.INTER_LINEAR)
    where = cv2.warpAffine(mask, pose[:2].astype(np.float32), SIZE, flags=cv2.INTER_LINEAR)[:, :, None] / 255.0
    frame = desk * (1 - where) + moved * where
    for centre, axes in hands:
        cv2.ellipse(frame, centre, axes, 20, 0, 360, (120, 150, 215), -1)
    noise = np.random.default_rng(seed).normal(0, 1.5, frame.shape)
    return np.clip(frame * (0.99 + 0.02 * (seed % 3) / 2) + noise, 0, 255).astype(np.uint8)


def error(tracker, pose):
    """Largest distance between where the tracker and the truth put the sheet's corners, in pixels."""
    return float(np.linalg.norm(move_points(tracker.pose, CORNERS) - move_points(pose, CORNERS), axis=1).max())


class SheetTrackerTests(unittest.TestCase):
    def tracker(self, scene, area=(360, 170, 940, 550)):
        return SheetTracker(shoot(scene), area)

    def test_a_still_sheet_stays_where_it_is(self):
        scene = sheet_picture()
        tracker = self.tracker(scene)
        for step in range(6):
            self.assertEqual(tracker.update(shoot(scene, seed=step + 1)), "steady")
        self.assertEqual(tracker.moves, 0)
        self.assertTrue(np.array_equal(tracker.pose, np.eye(3)))

    def test_follows_a_sheet_that_slides_and_turns_step_by_step(self):
        scene = sheet_picture()
        tracker = self.tracker(scene)
        for step in range(1, 9):
            pose = pose_of(turn=0.8 * step, shift=(7.0 * step, -3.0 * step))
            state = tracker.update(shoot(scene, pose, seed=step))
            self.assertIn(state, ("moved", "steady"))
            self.assertLess(error(tracker, pose), 1.5)
        self.assertTrue(tracker.tied)

    def test_finds_the_sheet_again_after_a_sudden_jump(self):
        scene = sheet_picture()
        tracker = self.tracker(scene)
        tracker.update(shoot(scene, seed=1))
        pose = pose_of(turn=11.0, shift=(170.0, 40.0))
        self.assertEqual(tracker.update(shoot(scene, pose, seed=2)), "moved")
        self.assertLess(error(tracker, pose), 2.0)
        self.assertEqual(tracker.update(shoot(scene, pose, seed=3)), "steady")

    def test_finds_the_sheet_after_a_large_turn(self):
        scene = sheet_picture()
        tracker = self.tracker(scene)
        tracker.update(shoot(scene, seed=1))
        pose = pose_of(turn=-33.0, shift=(-60.0, 20.0))
        self.assertEqual(tracker.update(shoot(scene, pose, seed=2)), "moved")
        self.assertLess(error(tracker, pose), 2.5)

    def test_hands_on_the_sheet_do_not_move_it(self):
        scene = sheet_picture()
        tracker = self.tracker(scene)
        tracker.update(shoot(scene, seed=1))
        hands = [((560, 300), (170, 110)), ((800, 430), (120, 90))]  # Two hands over most of the writing.
        for step in range(5):
            state = tracker.update(shoot(scene, hands=hands, seed=step + 2))
            self.assertIn(state, ("steady", "held"))
        self.assertLess(error(tracker, np.eye(3)), 1.0)

    def test_a_move_under_the_hands_is_found_when_they_lift(self):
        scene = sheet_picture()
        tracker = self.tracker(scene)
        tracker.update(shoot(scene, seed=1))
        pose = pose_of(turn=4.0, shift=(45.0, -18.0))
        hands = [((600, 330), (230, 150)), ((860, 420), (150, 110))]
        for step in range(3):
            tracker.update(shoot(scene, pose, hands=hands, seed=step + 2))
        for step in range(2):
            tracker.update(shoot(scene, pose, seed=step + 6))
        self.assertLess(error(tracker, pose), 2.0)
        self.assertTrue(tracker.tied)

    def test_a_blank_sheet_is_followed_by_its_outline(self):
        scene = sheet_picture(blank=True)
        tracker = self.tracker(scene)
        tracker.update(shoot(scene, seed=1))
        pose = pose_of(turn=6.0, shift=(70.0, 25.0))
        tracker.update(shoot(scene, pose, seed=2))
        self.assertLess(error(tracker, pose), 5.0)

    def test_says_lost_and_keeps_the_pose_when_the_sheet_is_gone(self):
        scene = sheet_picture()
        tracker = self.tracker(scene)
        tracker.update(shoot(scene, seed=1))
        desk_only = (scene[0], scene[1], np.zeros_like(scene[2]))
        self.assertEqual(tracker.update(shoot(desk_only, seed=2)), "lost")
        self.assertTrue(np.array_equal(tracker.pose, np.eye(3)))
        # The sheet comes back somewhere else: found, and still tied to the reference.
        pose = pose_of(turn=-5.0, shift=(-90.0, 30.0))
        self.assertEqual(tracker.update(shoot(scene, pose, seed=3)), "moved")
        self.assertLess(error(tracker, pose), 2.0)
        self.assertTrue(tracker.tied)

    def test_the_desk_does_not_hold_the_sheet_back(self):
        # The region from the key press is far too large: it takes in the seam of the desk and the laptop.
        scene = sheet_picture()
        tracker = self.tracker(scene, area=(60, 60, 1200, 660))
        tracker.update(shoot(scene, seed=1))
        pose = pose_of(turn=3.0, shift=(-40.0, 22.0))
        tracker.update(shoot(scene, pose, seed=2))
        self.assertLess(error(tracker, pose), 2.0)


class SheetWatchTests(unittest.TestCase):
    def test_calibrates_on_a_clear_view_and_follows_the_sheet(self):
        scene = sheet_picture()
        watch = SheetWatch()
        self.assertEqual(watch.confidence, 0.0)
        self.assertIsNone(watch.outline())
        now = 0.0
        for step in range(3):
            watch.update(shoot(scene, seed=step), None, now)
            now += 0.6
        self.assertTrue(watch.calibrated)
        self.assertLess(float(np.abs(watch.outline() - CORNERS).max()), 6.0)
        self.assertGreater(watch.confidence, 0.7)
        pose = pose_of(turn=7.0, shift=(120.0, -30.0))
        for step in range(2):
            watch.update(shoot(scene, pose, seed=step + 5), None, now)
            now += 0.3
        self.assertEqual(watch.generation, 1)
        self.assertLess(float(np.linalg.norm(watch.outline() - move_points(pose, CORNERS), axis=1).max()), 6.0)
        self.assertGreater(watch.confidence, 0.7)
        x0, y0, x1, y1 = watch.bounds()
        self.assertTrue(0 <= x0 < x1 <= SIZE[0] and 0 <= y0 < y1 <= SIZE[1])

    def test_a_hand_on_the_sheet_does_not_prevent_calibration(self):
        scene = sheet_picture()
        watch = SheetWatch()
        hands = [((640, 250), (120, 90))]  # Over the upper edge and some writing.
        for step in range(3):
            watch.update(shoot(scene, hands=hands, seed=step), None, 0.6 * step)
        self.assertTrue(watch.calibrated)
        self.assertLess(float(np.abs(watch.outline() - CORNERS).max()), 8.0)

    def test_no_sheet_no_outline(self):
        scene = sheet_picture()
        desk_only = (scene[0], scene[1], np.zeros_like(scene[2]))
        watch = SheetWatch()
        for step in range(4):
            watch.update(shoot(desk_only, seed=step), None, 0.6 * step)
        self.assertFalse(watch.calibrated)
        self.assertEqual(watch.confidence, 0.0)
        self.assertIsNone(watch.bounds())

    def test_another_sheet_is_calibrated_anew(self):
        scene, other = sheet_picture(seed=0), sheet_picture(seed=7)
        watch = SheetWatch()
        now = 0.0
        for step in range(3):
            watch.update(shoot(scene, seed=step), None, now)
            now += 0.6
        self.assertEqual(watch.generation, 1)
        pose = pose_of(turn=-8.0, shift=(-150.0, 10.0))
        for step in range(14):  # The first sheet is replaced by one with other writing, somewhere else.
            watch.update(shoot(other, pose, seed=step + 3), None, now)
            now += 0.6
        self.assertEqual(watch.generation, 2)
        self.assertLess(float(np.linalg.norm(watch.outline() - move_points(pose, CORNERS), axis=1).max()), 8.0)


class SessionSheetTests(unittest.TestCase):
    def session(self, scene):
        return WritingSession(shoot(scene), (360, 170, 940, 550), sheet=CORNERS.tolist())

    def test_the_record_moves_with_the_sheet(self):
        scene = sheet_picture()
        session = self.session(scene)
        self.assertTrue(session._sheet_track(shoot(scene, seed=1)))
        # Ink registered before the sheet moves, and where the pen was.
        session.ink_count = np.zeros(SIZE[::-1], np.uint8)
        session.ink_dark = np.zeros(SIZE[::-1], np.uint8)
        cv2.line(session.ink_count, (700, 480), (760, 480), 3, 2)
        cv2.line(session.ink_dark, (700, 480), (760, 480), 200, 2)
        session.path.append((1.0, "Right", (730, 470), 30.0))
        pose = pose_of(turn=5.0, shift=(60.0, -25.0))
        frame = shoot(scene, pose, seed=2)
        self.assertTrue(session._sheet_track(frame))
        start, end = move_points(pose, [(700, 480), (760, 480)])
        middle = np.rint((start + end) / 2).astype(int)
        self.assertGreaterEqual(int(session.ink_count[middle[1] - 2:middle[1] + 3, middle[0] - 2:middle[0] + 3].max()), 3)
        self.assertEqual(int(session.ink_count[480, 730]), 0)
        carried = move_points(pose, [(730, 470)])[0]
        self.assertLess(float(np.linalg.norm(np.subtract(session.path[-1][2], carried))), 2.0)
        # The reference now shows the old writing where the frame has it: nothing of it looks new.
        self.assertFalse(session.realigned)
        new = session._new_ink(frame)
        self.assertLess(int(np.count_nonzero(new)), 40)
        box = session.area
        self.assertLess(displacement(pose, session._sheet_pose, box), 2.0)

    def test_follows_the_calibrated_sheet_of_a_watch(self):
        scene = sheet_picture()
        watch = SheetWatch()
        now = 0.0
        for step in range(3):
            watch.update(shoot(scene, seed=step), None, now)
            now += 0.6
        # The recording starts after the sheet was moved once already, with a hand on it.
        first = pose_of(turn=3.0, shift=(30.0, 10.0))
        hands = [((640, 330), (140, 100))]
        baseline = shoot(scene, first, hands=hands, seed=9)
        watch.locate(baseline)
        session = WritingSession(baseline, watch.bounds(), sheet=watch.outline().tolist())
        self.assertTrue(session._sheet_use(watch))
        second = pose_of(turn=-4.0, shift=(-50.0, 35.0))
        frame = shoot(scene, second, seed=10)
        self.assertTrue(session._sheet_track(frame))
        expected = second @ np.linalg.inv(first)  # The motion since the key press.
        self.assertLess(displacement(session._sheet_pose, expected, (380, 190, 920, 530)), 4.0)


if __name__ == "__main__":
    unittest.main()
