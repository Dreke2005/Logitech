import unittest

import cv2
import hermetic  # noqa: F401
import numpy as np

from paper_pen import (
    OneEuro,
    PenTip,
    PenTipTracker,
    draw_trajectory,
    estimate_pen_tip,
    find_closed_loop,
    find_pen,
    locate_pen,
)
from paper_writing import TipSample, WritingSession, paper_bounds

AXIS = np.array((-0.6, -0.8))  # Pen axis from the hand toward the tip (a right-handed writer).
NORMAL = np.array((-AXIS[1], AXIS[0]))
SKIN = (150, 172, 205)
INK = (25, 25, 25)


def grip(x, y):
    return {"hands": [{"handedness": "Right", "landmarks": [{"pixel": (x - 15, y + 70)}],
                       "fingertips": {"thumb": {"pixel": (x, y + 7)}, "index": {"pixel": (x + 7, y - 10)}}}]}


def hand_for_tip(tip, scale=80.0, side="Right"):
    """A full 21-landmark MediaPipe-style hand whose pen grip places the pen tip at ``tip``."""
    tip = np.asarray(tip, float)
    grip_center = tip - AXIS * 0.28 * scale
    rest = grip_center - AXIS * 0.5 * scale
    wrist = rest - AXIS * 0.6 * scale
    points = {0: wrist, 2: rest - NORMAL * 15, 5: rest + NORMAL * 10,
              4: grip_center + NORMAL * 8, 8: grip_center - NORMAL * 8,
              9: wrist + (AXIS * 0.8 + NORMAL * 0.6) * scale}
    points[1] = (points[0] + points[2]) / 2
    points[3] = (points[2] + points[4]) / 2
    points[6], points[7] = (2 * points[5] + points[8]) / 3, (points[5] + 2 * points[8]) / 3
    points[13] = wrist + (AXIS * 0.75 + NORMAL * 0.85) * scale
    points[17] = wrist + (AXIS * 0.6 + NORMAL * 1.0) * scale
    for joint, base in ((10, 9), (11, 9), (12, 9), (14, 13), (15, 13), (16, 13), (18, 17), (19, 17), (20, 17)):
        points[joint] = points[base] + (AXIS * 0.12 + NORMAL * 0.05) * scale * (joint - base)
    landmarks = [{"pixel": [int(round(v)) for v in points[index]]} for index in range(21)]
    names = {"thumb": 4, "index": 8, "middle": 12, "ring": 16, "pinky": 20}
    return {"handedness": side, "landmarks": landmarks,
            "fingertips": {name: landmarks[index] for name, index in names.items()}}


def with_hand(page, hand):
    frame = page.copy()
    hull = cv2.convexHull(np.asarray([point["pixel"] for point in hand["landmarks"]], np.int32))
    cv2.fillConvexPoly(frame, hull, SKIN)
    return frame


def line(start, end, step=4):
    start, end = np.asarray(start, float), np.asarray(end, float)
    count = max(2, int(np.ceil(np.linalg.norm(end - start) / step)) + 1)
    return [tuple(int(round(v)) for v in start + (end - start) * k / (count - 1)) for k in range(count)]


def loop(center, radius, turns=1.12, count=26, start_angle=0.0):
    angles = start_angle + np.linspace(0, 2 * np.pi * turns, count)
    return [(int(round(center[0] + radius * np.cos(a))), int(round(center[1] + radius * np.sin(a))))
            for a in angles]


def glyph_strokes(x, top=220, height=30):
    """Three handwriting-like glyphs ("1", "+", "2") starting at x."""
    one = line((x, top), (x, top + height))
    plus = line((x + 18, top + height // 2), (x + 42, top + height // 2))
    plus_bar = line((x + 30, top + 4), (x + 30, top + height - 4))
    two = (line((x + 55, top + 6), (x + 68, top), 3) + line((x + 68, top), (x + 78, top + 10), 3)
           + line((x + 78, top + 10), (x + 56, top + height), 3) + line((x + 56, top + height), (x + 82, top + height), 3))
    return [one, plus, plus_bar, two]


def simulate(session, page, strokes, start=1.0, dt=0.06, ink=True, dwell=0.0, lift=0.0):
    """Feed frames with a hand holding a pen whose tip follows ``strokes``.

    Ink appears under the tip on pen-down strokes; the pen hovers in straight lines between
    strokes. ``dwell`` keeps the hand still at the last point; ``lift`` removes the hand.
    Returns the inked page (without the hand) and the final time.
    """
    now, previous = start, None
    for stroke in strokes:
        if previous is not None:
            for point in line(previous, stroke[0], 6)[1:-1]:
                hand = hand_for_tip(point)
                session.observe(with_hand(page, hand), {"hands": [hand]}, now)
                now += dt
        for index, point in enumerate(stroke):
            if ink and index:
                cv2.line(page, stroke[index - 1], point, INK, 2, cv2.LINE_AA)
            hand = hand_for_tip(point)
            session.observe(with_hand(page, hand), {"hands": [hand]}, now)
            now += dt
        previous = stroke[-1]
    hand = hand_for_tip(previous)
    for _ in range(int(round(dwell / dt))):
        session.observe(with_hand(page, hand), {"hands": [hand]}, now)
        now += dt
    for _ in range(int(round(lift / dt))):
        session.observe(page, {"hands": []}, now)
        now += dt
    return page, now


class WritingSessionTests(unittest.TestCase):
    def setUp(self):
        self.blank = np.full((480, 640, 3), 245, dtype=np.uint8)
        cv2.rectangle(self.blank, (32, 45), (610, 415), (110, 110, 110), 3)
        self.sheet = self.blank.copy()
        cv2.putText(self.sheet, "y=x^2+2x+1", (160, 250), 0, 1.25, (24, 24, 24), 3)

    def test_recording_waits_for_lift_and_uses_only_new_ink_near_the_writer(self):
        cv2.putText(self.sheet, "unrelated", (400, 375), 0, 0.8, (24, 24, 24), 2)
        session = WritingSession(self.blank)
        self.assertIsNone(session.ready(self.blank, 1))
        for tick in range(17):
            now = 2 + tick * 0.245
            session.observe(self.blank if tick < 10 else self.sheet, grip(158 + tick * 11, 240), now)
        self.assertIsNone(session.ready(self.sheet, now + 0.8))
        for step in (0.9, 1.3, 1.8):
            session.observe(self.sheet, None, now + step)
        selected = session.ready(self.sheet, now + 1.8)
        self.assertIsNotNone(selected)
        left, top, right, bottom = selected.page.bounds
        self.assertLess(left, 160)
        self.assertGreater(right, 350)
        self.assertLess(top, 225)
        self.assertLess(bottom, 350)
        self.assertTrue(np.any(selected.evidence < 100))

    def test_hand_motion_without_visible_new_ink_does_not_finish(self):
        session = WritingSession(self.sheet)
        for tick in range(20):
            session.observe(self.sheet, grip(158 + tick * 8, 240), 1 + tick * 0.25)
        for now in (6, 7, 8):
            session.observe(self.sheet, None, now)
        self.assertIsNone(session.ready(self.sheet, 8))

    def test_paper_area_keeps_a_moving_keypad_out_of_the_crop(self):
        desk = np.full((480, 640, 3), 138, np.uint8)
        cv2.rectangle(desk, (90, 125), (390, 365), (245, 245, 245), -1)
        area = paper_bounds(desk)
        self.assertIsNotNone(area)
        self.assertLess(area[2], 450)
        session = WritingSession(desk, area)
        moved_keypad = desk.copy()
        cv2.rectangle(moved_keypad, (470, 210), (610, 325), (30, 30, 30), -1)
        for tick in range(17):
            session.observe(moved_keypad, grip(475 + tick * 5, 260), 1 + tick * 0.25)
        self.assertFalse(session.path)
        self.assertIsNone(session.ready(moved_keypad, 8))
        written = moved_keypad.copy()
        cv2.putText(written, "y=x^2+2x+1", (145, 255), 0, 1.1, (20, 20, 20), 2)
        for tick in range(17):
            session.observe(moved_keypad if tick < 10 else written, grip(145 + tick * 10, 244),
                            9 + tick * 0.25)
        for now in (14, 14.5, 15):
            session.observe(written, None, now)
        selected = session.ready(written, 15)
        self.assertIsNotNone(selected)
        self.assertLess(selected.page.bounds[2], area[2])

    def test_pen_tracking_uses_the_exposed_tip_near_the_grip(self):
        frame = self.blank.copy()
        cv2.line(frame, (160, 220), (235, 220), (210, 40, 10), 6)
        tip = locate_pen(frame, (208, 220), (165, 224), 35)
        self.assertIsNotNone(tip)
        self.assertLess(np.linalg.norm(np.subtract(tip, (235, 220))), 8)
        self.assertIsNone(locate_pen(self.blank, (208, 220), (165, 224), 35))

    def test_tip_landmark_tracks_past_the_colored_barrel_to_the_nib(self):
        frame = self.blank.copy()
        cv2.rectangle(frame, (156, 216), (226, 224), (210, 40, 10), -1)
        cv2.fillConvexPoly(frame, np.array([(225, 216), (242, 220), (225, 224)]), (115, 115, 115))
        tip = locate_pen(frame, (207, 220), (165, 224), 42)
        self.assertIsNotNone(tip)
        self.assertLessEqual(np.linalg.norm(np.subtract(tip, (242, 220))), 3)

    def test_recording_with_no_new_ink_or_pen_path_has_no_recognition_frame(self):
        session = WritingSession(self.blank, (35, 50, 600, 410))
        self.assertIsNone(session.select(self.blank, prefer_recent=True))

    def test_recording_crop_excludes_distant_ink_on_same_sheet(self):
        session = WritingSession(self.blank, (35, 50, 600, 410))
        session.path.extend((i * 0.08, "Right", (180 + i * 3, 240), 35) for i in range(24))
        third = [(i * 0.08, (180 + 3 * i, 235 + i)) for i in range(12)]
        fourth = [(1.6 + i * 0.08, (245 + 3 * i, 236 + i)) for i in range(12)]
        session.tips["Right"] = [TipSample(t, point, down=True) for t, point in third + fourth]
        cv2.circle(self.sheet, (490, 120), 17, (10, 10, 10), -1)
        result = session.select(self.sheet, prefer_recent=True)
        self.assertIsNotNone(result)
        x0, y0, x1, y1 = result.page.bounds
        # The writing the pen passed over, glyph by glyph, and not the blob at x 473..507 elsewhere on the sheet.
        self.assertLess(x1, 400)
        self.assertGreater(y0, 190)
        np.testing.assert_array_equal(result.page.image, self.sheet[y0:y1, x0:x1])

    def test_pen_trajectory_keeps_segments_separate(self):
        first = [(i * 0.08, (180 + 3 * i, 200 + i)) for i in range(12)]
        second = [(1.6 + i * 0.08, (245 + 3 * i, 201 + i)) for i in range(12)]
        result = draw_trajectory(first + second, None, (640, 480), grip_samples=28)
        self.assertIsNotNone(result)
        image, bounds, reliable = result
        self.assertTrue(reliable)
        self.assertTrue(np.all(image[:, 229 - bounds[0]:236 - bounds[0]] == 255))


class PenTipTests(unittest.TestCase):
    def setUp(self):
        self.page = np.full((480, 640, 3), 245, dtype=np.uint8)

    def test_tip_landmark_extends_the_grip_along_the_pen_axis(self):
        hand = hand_for_tip((300, 200))
        tip = estimate_pen_tip(hand)
        self.assertIsNotNone(tip)
        self.assertFalse(tip.visual)
        self.assertLessEqual(np.linalg.norm(np.subtract(tip.point, (300, 200))), 2)
        index = hand["fingertips"]["index"]["pixel"]
        self.assertGreater(np.linalg.norm(np.subtract(tip.point, index)), 15)

    def test_open_hand_has_no_pen_tip(self):
        hand = hand_for_tip((300, 200))
        wrist = hand["landmarks"][0]["pixel"]
        hand["fingertips"]["thumb"] = {"pixel": [wrist[0] + 90, wrist[1]]}
        self.assertIsNone(estimate_pen_tip(hand))

    def test_visible_pen_shaft_refines_the_tip(self):
        hand = hand_for_tip((300, 200))
        nib = np.array((300, 200)) + AXIS * 7
        cv2.line(self.page, tuple(np.rint(nib - AXIS * 60).astype(int)), tuple(np.rint(nib).astype(int)),
                 (210, 40, 10), 6)
        tip = estimate_pen_tip(hand, self.page)
        self.assertTrue(tip.visual)
        self.assertLess(np.linalg.norm(np.subtract(tip.point, nib)), 5)

    @staticmethod
    def pen_photo():
        frame = np.full((720, 1280, 3), 245, np.uint8)
        cv2.ellipse(frame, (425, 413), (32, 25), 0, 0, 360, SKIN, -1)
        cv2.rectangle(frame, (290, 379), (490, 391), (220, 45, 25), -1)
        cv2.fillConvexPoly(frame, np.array([(490, 379), (523, 385), (490, 391)], np.int32), (85, 85, 85))
        return frame

    def test_capture_accepts_an_inside_nib_even_when_the_barrel_extends_outside(self):
        pen = find_pen(self.pen_photo(), inside=(420, 350, 570, 430))
        self.assertIsNotNone(pen)
        self.assertTrue(pen.tapered)
        self.assertLess(pen.back[0], 420)
        self.assertLess(np.linalg.norm(np.subtract(pen.point, (523, 385))), 5)

    def test_capture_rejects_an_exterior_or_cut_off_nib(self):
        frame = self.pen_photo()
        for boundary in (510, 523):
            with self.subTest(boundary=boundary):
                self.assertIsNone(find_pen(frame, inside=(270, 350, boundary, 430)))
        self.assertIsNone(find_pen(frame[:, :515].copy()))

    def test_sheet_outline_excludes_a_nib_inside_its_bounding_rectangle(self):
        frame = self.pen_photo()
        sheet = np.zeros(frame.shape[:2], np.uint8)
        outline = np.array([(285, 330), (540, 330), (438, 480), (285, 480)], np.int32)
        cv2.fillConvexPoly(sheet, outline, 1)
        self.assertIsNone(find_pen(frame, inside=(285, 330, 541, 481), surface=sheet))

    def test_exterior_estimate_is_not_clamped_to_the_capture_edge(self):
        hand = hand_for_tip((288, 200))
        self.assertIsNone(estimate_pen_tip(hand, self.page, inside=(300, 160, 400, 300)))

    def test_preview_hides_an_unobserved_tip_and_a_tip_outside_the_capture(self):
        session = WritingSession(area=(260, 160, 380, 300))
        session.current_tips["Right"] = 2, PenTip((280, 240), (300, 245), 10, 20, False)
        session.current_tips["Pen"] = 2, PenTip((250, 240), (300, 245), 10, 20, True)
        np.testing.assert_array_equal(session.draw(self.page, 2), self.page)
        session.current_tips["Pen"] = 2, PenTip((300, 240), (310, 245), 10, 20, True)
        self.assertFalse(np.array_equal(session.draw(self.page, 2), self.page))

    def test_tracker_smooths_jitter_and_learns_a_tip_offset(self):
        rng = np.random.default_rng(3)
        tracker, raw, smooth = PenTipTracker(), [], []
        for step in range(40):
            hand = hand_for_tip((300 + rng.normal(0, 3), 200 + rng.normal(0, 3)))
            raw.append(estimate_pen_tip(hand).point)
            smooth.append(tracker.update("Right", hand, None, step * 0.06).point)
        self.assertLess(np.std(np.asarray(smooth)[10:], axis=0).sum(),
                        0.7 * np.std(np.asarray(raw)[10:], axis=0).sum())
        nib = np.array((300, 200)) + AXIS * 9
        shaft = self.page.copy()
        cv2.line(shaft, tuple(np.rint(nib - AXIS * 60).astype(int)), tuple(np.rint(nib).astype(int)),
                 (210, 40, 10), 6)
        hand = hand_for_tip((300, 200))
        for step in range(30):
            tracker.update("Right", hand, shaft, 10 + step * 0.06)
        hidden = tracker.update("Right", hand, self.page, 20)
        self.assertFalse(hidden.visual)
        self.assertLess(np.linalg.norm(np.subtract(hidden.point, nib)), 5)

    def test_one_euro_follows_fast_motion_with_little_lag(self):
        smooth = OneEuro()
        for step in range(20):
            value = smooth(step * 0.06, (step * 25.0, 0.0))
        self.assertLess(abs(value[0] - 19 * 25), 25)

    def test_closed_loop_detection(self):
        self.assertIsNotNone(find_closed_loop(loop((200, 200), 16), 4, 60))
        lead_in = line((120, 200), (184, 200)) + loop((200, 200), 16, start_angle=np.pi)
        self.assertIsNotNone(find_closed_loop(lead_in, 4, 60))
        self.assertIsNone(find_closed_loop(loop((200, 200), 16, turns=0.7), 4, 60))
        self.assertIsNone(find_closed_loop(line((120, 200), (220, 230)), 4, 60))
        self.assertIsNone(find_closed_loop(loop((200, 200), 2), 4, 60))


class LiveRegistrationTests(unittest.TestCase):
    def setUp(self):
        self.page = np.full((480, 640, 3), 245, dtype=np.uint8)
        cv2.putText(self.page, "old note", (430, 420), 0, 0.9, INK, 2)

    def test_hover_moves_are_not_part_of_the_pen_down_trajectory(self):
        session = WritingSession(self.page)
        page, now = simulate(session, self.page.copy(),
                             [line((150, 240), (220, 240)), line((260, 200), (330, 200))], lift=2.0)
        strokes = session._stroke_samples("Right", cv2.cvtColor(page, cv2.COLOR_BGR2GRAY))
        self.assertEqual(len(strokes), 2)
        hovered = [sample for sample in session.tips["Right"] if sample.down is False]
        self.assertTrue(hovered)
        self.assertFalse([sample for sample in session.tips["Right"] if sample.down and 225 < sample.point[0] < 255])
        selected = session.ready(page, now)
        self.assertIsNotNone(selected)
        self.assertEqual(session.finish_reason, "lifted")
        self.assertEqual(selected.trace.shape[:2], selected.page.image.shape[:2])
        x0, y0 = selected.page.bounds[:2]
        self.assertTrue(np.all(selected.trace[220 - y0, 240 - x0] == 255))  # Middle of the hover move.
        self.assertTrue(np.any(selected.trace[240 - y0, 160 - x0:210 - x0] < 128))

    def test_crop_is_the_written_region_not_the_page(self):
        session = WritingSession(self.page)
        page, now = simulate(session, self.page.copy(), glyph_strokes(160), lift=2.0)
        selected = session.ready(page, now)
        self.assertIsNotNone(selected)
        self.assertTrue(selected.trace_reliable)
        x0, y0, x1, y1 = selected.page.bounds
        self.assertLessEqual(x0, 160)
        self.assertGreaterEqual(x1, 242)
        self.assertLessEqual(y0, 220)
        self.assertGreaterEqual(y1, 250)
        self.assertGreater(x0, 160 - 30)
        self.assertLess(x1, 242 + 30)
        self.assertLess((x1 - x0) * (y1 - y0), 0.1 * 640 * 480)

    def test_crop_includes_new_ink_the_tip_track_missed(self):
        session = WritingSession(self.page)
        page, now = simulate(session, self.page.copy(), glyph_strokes(160), lift=0.3)
        cv2.line(page, (250, 225), (290, 250), INK, 2)  # Ink without a tracked tip, e.g. while occluded.
        for step in range(30):
            session.observe(page, {"hands": []}, now + step * 0.06)
        selected = session.select(page, prefer_recent=True)
        self.assertGreaterEqual(selected.page.bounds[2], 290)
        self.assertEqual(session.finish_reason, "key")

    def test_end_circle_finishes_without_lifting_and_is_not_transcribed(self):
        session = WritingSession(self.page)
        strokes = glyph_strokes(160) + [loop((330, 236), 17)]
        page, now = simulate(session, self.page.copy(), strokes, dwell=1.2)
        self.assertTrue(session.end_marked)
        selected = session.ready(with_hand(page, hand_for_tip(strokes[-1][-1])), now)
        self.assertIsNotNone(selected)
        self.assertTrue(selected.end_marked)
        self.assertEqual(session.finish_reason, "circle")
        x0, y0, x1, y1 = selected.page.bounds
        self.assertLess(x1, 330 - 17)
        self.assertLessEqual(x0, 160)
        self.assertTrue(np.any(selected.trace < 128))
        self.assertLess(np.linalg.norm(np.subtract(session.end_circle.center, (330, 236))), 6)

    def test_circle_ink_inside_the_crop_is_erased(self):
        session = WritingSession(self.page)
        strokes = glyph_strokes(160) + [line((160, 262), (240, 262)), loop((268, 262), 15)]
        page, now = simulate(session, self.page.copy(), strokes, dwell=1.2)
        self.assertTrue(session.end_marked)
        selected = session.ready(with_hand(page, hand_for_tip(strokes[-1][-1])), now)
        x0, y0, x1, y1 = selected.page.bounds
        ring = page[247:278, 253:284]
        self.assertTrue(np.any(ring < 128))
        inside = selected.page.image[max(0, 247 - y0):max(0, 278 - y0), max(0, 253 - x0):max(0, 284 - x0)]
        self.assertTrue(np.all(inside > 200))

    def test_air_circle_without_ink_does_not_end(self):
        session = WritingSession(self.page)
        page, now = simulate(session, self.page.copy(), glyph_strokes(160), dwell=0.3)
        circle = loop((330, 236), 17)
        for point in circle + [circle[-1]] * 20:
            hand = hand_for_tip(point)
            session.observe(with_hand(page, hand), {"hands": [hand]}, now)
            now += 0.06
        self.assertFalse(session.end_marked)

    def test_letter_loop_followed_by_more_writing_does_not_end(self):
        session = WritingSession(self.page)
        strokes = glyph_strokes(160) + [loop((265, 236), 15)] + glyph_strokes(300)
        simulate(session, self.page.copy(), strokes, dwell=0.3)
        self.assertFalse(session.end_marked)

    def test_loop_around_existing_writing_is_not_an_end_mark(self):
        session = WritingSession(self.page)
        strokes = glyph_strokes(160) + [loop((200, 236), 50, count=40)]
        simulate(session, self.page.copy(), strokes, dwell=1.2)
        self.assertFalse(session.end_marked)

    def test_dashed_oval_copy_selection_still_works(self):
        page = self.page.copy()
        cv2.putText(page, "y=x^2+2x+1", (210, 250), 0, 1.0, INK, 2)
        for angle in range(0, 360, 20):
            cv2.ellipse(page, (320, 240), (150, 60), 0, angle, angle + 13, INK, 3)
        selected = WritingSession().select(page, marked_only=True)
        self.assertEqual(selected.page.kind, "circle")
        self.assertLess(selected.page.bounds[0], 210)
        self.assertGreater(selected.page.bounds[2], 400)


def relit(frame, gain=0.82):
    """Auto-exposure drop plus a lighting gradient across the page."""
    ramp = np.linspace(1.0, 0.88, frame.shape[1])[None, :, None]
    return np.clip(frame * gain * ramp, 0, 255).astype(np.uint8)


def resting_hand(frame, tip):
    """The writing hand still on the page with its forearm, pen barrel, and a soft shadow."""
    hand = hand_for_tip(tip)
    points = np.asarray([item["pixel"] for item in hand["landmarks"]], np.int32)
    wrist, middle = points[0].astype(float), points[9].astype(float)
    back = (wrist - middle) / np.linalg.norm(wrist - middle)
    across = np.array((-back[1], back[0])) * 30
    arm = np.asarray([wrist + across, wrist - across, wrist - across + back * 300,
                      wrist + across + back * 300], np.int32)
    shade = np.zeros(frame.shape[:2], np.float32)
    cv2.fillConvexPoly(shade, cv2.convexHull(points + (25, 18)), 1)
    cv2.fillConvexPoly(shade, arm + (25, 18), 1)
    shade = cv2.GaussianBlur(shade, (0, 0), 9)[..., None]
    out = (frame * (1 - 0.35 * shade)).astype(np.uint8)
    cv2.fillConvexPoly(out, cv2.convexHull(points), SKIN)
    cv2.fillConvexPoly(out, arm, (150, 150, 160))
    grip = (np.asarray(hand["fingertips"]["thumb"]["pixel"]) + np.asarray(hand["fingertips"]["index"]["pixel"])) / 2
    cv2.line(out, tuple(np.rint(grip + AXIS * 22).astype(int)), tuple(np.rint(grip - AXIS * 70).astype(int)),
             (150, 60, 30), 7)
    return out, {"hands": [hand]}


def turned_hand(tip):
    """A writer across the desk: the same hand rotated 180 degrees about the pen tip."""
    hand = hand_for_tip(tip)
    for item in hand["landmarks"]:
        item["pixel"] = [2 * tip[0] - item["pixel"][0], 2 * tip[1] - item["pixel"][1]]
    return hand


class TightCropTests(unittest.TestCase):
    """The recognition crop must be the newly written area, never the page or the hand's sweep."""

    def setUp(self):
        self.page = np.full((480, 640, 3), 245, dtype=np.uint8)
        cv2.putText(self.page, "old note", (430, 420), 0, 0.9, INK, 2)
        cv2.putText(self.page, "a+b", (60, 120), 0, 1.0, INK, 2)
        self.strokes = glyph_strokes(160)
        self.truth = (160, 220, 243, 251)  # Bounds of the written "1 + 2".

    def assert_tight(self, bounds, slack=24):
        x0, y0, x1, y1 = bounds
        tx0, ty0, tx1, ty1 = self.truth
        self.assertLessEqual(x0, tx0 + 2)
        self.assertLessEqual(y0, ty0 + 2)
        self.assertGreaterEqual(x1, tx1 - 2)
        self.assertGreaterEqual(y1, ty1 - 2)
        self.assertLess(tx0 - x0, slack)
        self.assertLess(ty0 - y0, slack)
        self.assertLess(x1 - tx1, slack)
        self.assertLess(y1 - ty1, slack)

    def record_with_clutter(self):
        """Write "1 + 2", then finish with the hand resting beside it under changed lighting."""
        session = WritingSession(self.page)
        page, now = simulate(session, self.page.copy(), self.strokes, dwell=0.3)
        final = relit(page)
        cv2.ellipse(final, (420, 300), (90, 40), 20, 0, 360, (60, 60, 60), -1)  # A dark object put down.
        final = cv2.addWeighted(final, 1.0, np.zeros_like(final), 0, 0)
        frame, hands = resting_hand(final, (300, 262))
        for step in range(8):
            session.observe(frame, hands, now + step * 0.12)
        return session, frame, page

    def test_lighting_shadow_and_resting_hand_do_not_grow_the_crop(self):
        session, frame, _ = self.record_with_clutter()
        selected = session.select(frame, prefer_recent=True)
        self.assertIsNotNone(selected)
        self.assert_tight(selected.page.bounds)
        self.assertEqual(selected.rotation, 0)
        self.assertGreater(len(selected.strokes), 2)
        self.assertTrue(18 <= selected.text_height <= 32, selected.text_height)

    def test_without_a_pen_down_trace_new_ink_near_the_tip_is_still_tight(self):
        session, frame, _ = self.record_with_clutter()
        for sample in session.tips["Right"]:
            sample.down = False  # Tip tracking failed to confirm any stroke.
        selected = session.select(frame, prefer_recent=True)
        self.assertIsNotNone(selected)
        self.assert_tight(selected.page.bounds)
        self.assertIsNotNone(selected.evidence)
        # The strokes are traced on the new ink itself, so they exist even without a confirmed pen-down sample.
        ink = session._new_ink(frame)
        points = np.vstack(selected.strokes)
        self.assertTrue(ink[points[:, 1], points[:, 0]].all())

    def test_new_ink_map_keeps_thin_strokes_and_drops_blobs_and_old_ink(self):
        session, frame, page = self.record_with_clutter()
        ink = session._new_ink(frame)
        written = np.zeros(ink.shape, bool)
        written[215:256, 155:248] = True
        self.assertGreater(np.count_nonzero(ink[written]), 60)
        self.assertEqual(np.count_nonzero(ink[~written]), 0)

    def test_writer_across_the_desk_is_read_upside_down(self):
        session = WritingSession(self.page)
        now = 1.0
        page = self.page.copy()
        for stroke in self.strokes:
            for index, point in enumerate(stroke):
                if index:
                    cv2.line(page, stroke[index - 1], point, INK, 2, cv2.LINE_AA)
                hand = turned_hand(point)
                session.observe(page, {"hands": [hand]}, now)
                now += 0.06
        for step in range(10):
            session.observe(page, {"hands": []}, now + step * 0.1)
        selected = session.select(page, prefer_recent=True)
        self.assertEqual(selected.rotation, 180)

    def test_hover_hop_between_close_glyphs_is_cut_out_of_the_trace(self):
        """A hop to the next glyph that stays within the ink radius of both glyphs is not a stroke."""
        page = np.full((480, 640, 3), 245, dtype=np.uint8)
        session = WritingSession(page)
        first, second = line((200, 220), (200, 250)), line((214, 220), (214, 250))
        cv2.polylines(page, [np.asarray(first, np.int32)], False, INK, 2, cv2.LINE_AA)
        cv2.polylines(page, [np.asarray(second, np.int32)], False, INK, 2, cv2.LINE_AA)
        joined = [first + line((200, 250), (214, 220), 5)[1:-1] + second]
        ink = session._new_ink(page)
        strokes = session._match_ink(joined, ink)
        self.assertEqual(len(strokes), 2)
        self.assertTrue(all(abs(x - 200) <= 5 for x, _ in strokes[0]))
        self.assertTrue(all(abs(x - 214) <= 5 for x, _ in strokes[1]))

    def test_trace_offset_from_the_nib_is_registered_on_the_ink(self):
        page = np.full((480, 640, 3), 245, dtype=np.uint8)
        session = WritingSession(page)
        strokes = glyph_strokes(200)
        for stroke in strokes:
            cv2.polylines(page, [np.asarray(stroke, np.int32)], False, INK, 2, cv2.LINE_AA)
        shifted = [[(x + 3, y - 2) for x, y in stroke] for stroke in strokes]
        matched = session._match_ink(shifted, session._new_ink(page))
        self.assertEqual(len(matched), len(strokes))
        error = np.abs(np.asarray(matched[0], float) - np.asarray(strokes[0], float)).max()
        self.assertLessEqual(error, 1.5)

    def test_missing_ink_map_leaves_the_trace_alone(self):
        session = WritingSession(np.full((480, 640, 3), 245, dtype=np.uint8))
        strokes = glyph_strokes(200)
        self.assertEqual(session._match_ink(strokes, np.zeros((480, 640), np.uint8)), strokes)

    def test_end_circle_can_be_disabled_for_diagrams(self):
        strokes = self.strokes + [loop((330, 236), 17)]
        for enabled in (True, False):
            session = WritingSession(self.page, end_circle=enabled)
            simulate(session, self.page.copy(), strokes, dwell=1.2)
            self.assertEqual(session.end_marked, enabled)
