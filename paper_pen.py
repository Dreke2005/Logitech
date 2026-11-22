"""Estimate a pen tip near a tracked writing grip in camera coordinates."""

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass(frozen=True)
class PenEnd:
    point: tuple  # The nib, in frame pixels.
    back: tuple  # The other end of the visible barrel.
    width: float  # Barrel width in pixels.
    colour: tuple = None  # Median hue, saturation, value of the barrel.
    held: bool = False  # True when a detected hand's thumb and index are at the barrel.
    coloured: bool = True  # Found by its colour (not as a dark shape): only then is its colour worth learning.
    tapered: bool = False  # The body was followed to its tapering point: the landmark is on the nib itself.
    cone: float = 0.0  # From the end of the barrel's colour to the point, in pixels, as last measured.


def hue_distance(first, second):
    """Distance between two OpenCV hues (0-180), around the circle: red is both 1 and 178."""
    difference = abs(float(first) - float(second)) % 180.0
    return min(difference, 180.0 - difference)


def blend_colour(known, seen, share=0.1):
    """(hue, saturation, value) moved a share of the way from ``known`` to ``seen``, the hue around its circle."""
    step = (float(seen[0]) - float(known[0]) + 90.0) % 180.0 - 90.0
    hue = (float(known[0]) + share * step) % 180.0
    return (hue,) + tuple((1 - share) * float(a) + share * float(b) for a, b in zip(known[1:], seen[1:]))


def _line(points):
    """Centre, unit axis, extent along the axis, and width of a bar of pixels.

    The width is taken from the middle of the bar (four times the median distance from its line) and the
    line is fitted again without what sticks out sideways: a finger's shadow hanging on to a barrel must
    neither turn it nor make it look too wide to be a pen."""
    kept = points
    for _ in range(2):
        centre = kept.mean(axis=0)
        offset = kept - centre
        _, vectors = np.linalg.eigh(offset.T @ offset)
        axis = vectors[:, 1].astype(np.float32)
        across = (points - centre) @ np.array((-axis[1], axis[0]), np.float32)
        width = 4.0 * float(np.median(np.abs(across)))
        inside = np.abs(across) <= 0.5 * width + 1
        if inside.all():
            break
        kept = points[inside]
    along = (kept - centre) @ axis
    return centre, axis, float(along.min()), float(along.max()), width


def _barrels(mask, hue, saturation, paper_like, ink_width, widest, shortest, reach):
    """Bars in a 0/1 mask that can be a pen's barrel, each as (pixels, centre, axis, low, high, thickness).

    The fingers hold a pen across its barrel, so the camera sees two or three pieces in a row: the long
    back, sometimes a bit between the fingers, and the short front before the point. A piece that lies
    on the line of a bar, is as narrow as the bar, has its colour, and is separated from it by something
    other than plain paper (the fingers) belongs to the same pen. Without this the longest piece alone
    would be taken for the pen and its end at the fingers for the nib."""
    # A glossy barrel has a white highlight along it that would split it into two thin bars: close it first.
    solid = cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
    solid = cv2.morphologyEx(solid, cv2.MORPH_OPEN, np.ones((ink_width + 2,) * 2, np.uint8))
    count, labels, stats, centroids = cv2.connectedComponentsWithStats(solid, connectivity=8)
    if count <= 1:
        return []
    largest = 0.02 * solid.size
    usable = [label for label in range(1, count) if 6 <= stats[label, cv2.CC_STAT_AREA] <= largest]
    pieces = {}

    def pixels(label):
        if label not in pieces:
            x, y, w, h = stats[label, :4]
            ys, xs = np.nonzero(labels[y:y + h, x:x + w] == label)
            pieces[label] = np.c_[xs + x, ys + y].astype(np.float32)
        return pieces[label]

    seeds = []
    for label in usable:
        x, y, w, h, area = stats[label]
        if max(w, h) < 0.7 * shortest or area < 30:
            continue
        centre, axis, low, high, thickness = _line(pixels(label))
        length = high - low
        if length >= shortest and length >= 3.5 * thickness and ink_width + 1.5 <= thickness <= widest:
            seeds.append((length, label, centre, axis, low, high, thickness))
    seeds.sort(key=lambda item: -item[0])
    centres = centroids[usable].astype(np.float32)
    usable = np.asarray(usable)
    rows, cols = mask.shape
    found, taken = [], set()
    for _, label, centre, axis, low, high, thickness in seeds[:6]:
        if label in taken:
            continue
        members, points = {label}, pixels(label)
        tone = float(np.median(hue[points[:, 1].astype(int), points[:, 0].astype(int)])) if hue is not None else None
        for _ in range(4):
            normal = np.array((-axis[1], axis[0]), np.float32)
            offset = centres - centre
            ahead = np.maximum(np.maximum(offset @ axis - high, low - offset @ axis), 0)
            # A short or ragged piece gives its own direction only roughly: allow a quarter of the distance.
            near = (np.abs(offset @ normal) <= 0.5 * thickness + 2 + 0.25 * ahead) & (ahead <= reach + 4)
            added = False
            for other in usable[near]:
                other = int(other)
                if other in members or other in taken:
                    continue
                piece = pixels(other)
                if tone is not None:
                    other_tone = float(np.median(hue[piece[:, 1].astype(int), piece[:, 0].astype(int)]))
                    if hue_distance(tone, other_tone) > 14:
                        continue
                # One straight bar through both: every pixel of the two within the barrel's width of one line.
                joined = np.vstack([points, piece])
                joint_centre, joint_axis, joint_low, joint_high, _ = _line(joined)
                joint_normal = np.array((-joint_axis[1], joint_axis[0]), np.float32)
                # (A finger's red shadow often hangs on to a piece, so its middle counts, not its widest part.)
                if any(abs(float(np.mean(part))) > 0.25 * thickness + 1.5
                       or float(np.median(np.abs(part))) > 0.5 * thickness + 1
                       for part in ((points - joint_centre) @ joint_normal, (piece - joint_centre) @ joint_normal)):
                    continue
                first, second = (points - joint_centre) @ joint_axis, (piece - joint_centre) @ joint_axis
                gap = max(float(second.min() - first.max()), float(first.min() - second.max()))
                if gap > reach:
                    continue
                if gap > 2:
                    start = float(first.max()) if second.min() > first.max() else float(second.max())
                    steps = joint_centre + np.outer(np.arange(start + 1, start + gap), joint_axis)
                    xs = np.clip(np.rint(steps[:, 0]).astype(int), 0, cols - 1)
                    ys = np.clip(np.rint(steps[:, 1]).astype(int), 0, rows - 1)
                    if paper_like[ys, xs].mean() > 0.35:
                        continue  # Plain paper between them: two things in a row, not one pen under a finger.
                members.add(other)
                points = joined
                if joint_axis @ axis < 0:
                    joint_axis, joint_low, joint_high = -joint_axis, -joint_high, -joint_low
                centre, axis, low, high = joint_centre, joint_axis, joint_low, joint_high
                added = True
                break
            if not added:
                break
        taken |= members
        found.append((points, centre, axis, low, high, thickness))
    return found


def capture_contains(point, shape, region=None, mask=None, margin=0):
    """Whether a frame-coordinate point and its margin belong to the capture."""
    height, width = shape[:2]
    left, top, right, bottom = region if region is not None else (0, 0, width, height)
    x, y = point
    if not (max(0, left) + margin <= x < min(width, right) - margin
            and max(0, top) + margin <= y < min(height, bottom) - margin):
        return False
    if mask is None:
        return True
    if mask.shape != (height, width):
        return False
    ix, iy = int(round(x)), int(round(y))
    if not (margin <= ix < width - margin and margin <= iy < height - margin):
        return False
    return bool(mask[iy - margin:iy + margin + 1, ix - margin:ix + margin + 1].all())


def _probe(frame, end, direction, width, cone_length=0.0, capture=None):
    """Follow the pen's body from a barrel end outward to where it stops.

    A strip of the frame along the pen is cut out, straightened. Row by row the body's cross-section
    (anything coloured or much darker than the paper) is followed from the barrel. A pencil's cone and
    graphite or a ballpoint's cone and metal tip narrow steadily to a point lying on paper; the back end
    stops bluntly (an eraser, a cap), disappears into the hand or a sleeve, or leaves the picture. Fingers
    lying across the pen are crossed: beyond them the body is as narrow as before and goes on to its point.
    ``cone_length`` is how long this pen's cone was when last seen whole (0: unknown).
    Returns (tip, kind, measured barrel width, cone) with kind "point", "blunt", "open" (ended on paper, neither
    clearly), "ink" (ended on dark writing), "hand" (went under something wide and did not come out), or
    "edge" (left the picture); cone is the length from where the barrel starts to narrow to the point when
    the point was followed over plain paper to its end, else 0."""
    height, frame_width = frame.shape[:2]
    end = np.asarray(end, np.float32)
    direction = np.asarray(direction, np.float32)
    direction = direction / max(float(np.linalg.norm(direction)), 1e-6)
    normal = np.array((-direction[1], direction[0]), np.float32)
    half = int(max(8, round(1.6 * width)))
    behind = int(max(6, round(1.5 * width)))
    ahead = int(round(11 * width + 40))
    along = np.arange(-behind, ahead + 1, dtype=np.float32)
    across = np.arange(-half, half + 1, dtype=np.float32)
    xs = end[0] + along[:, None] * direction[0] + across[None, :] * normal[0]
    ys = end[1] + along[:, None] * direction[1] + across[None, :] * normal[1]
    strip = cv2.remap(frame, xs, ys, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    outside = (xs < 0) | (xs > frame_width - 1) | (ys < 0) | (ys > height - 1)
    saturation = cv2.cvtColor(strip, cv2.COLOR_BGR2HSV)[:, :, 1]
    gray = cv2.cvtColor(strip, cv2.COLOR_BGR2GRAY)
    plain = (saturation <= 45) & ~outside
    if np.count_nonzero(plain) >= 30:
        paper, tint = float(np.percentile(gray[plain], 80)), float(np.median(saturation[plain]))
    else:
        paper, tint = float(np.percentile(gray, 90)), 15.0
    body = ((saturation > tint + 28) | (gray < 0.62 * paper)).astype(np.uint8)
    # A glossy barrel has a white highlight along it, which must not split the cross-section in two.
    bridge = max(3, int(round(0.45 * width))) | 1
    body = cv2.morphologyEx(body, cv2.MORPH_CLOSE, np.ones((1, bridge), np.uint8))

    def run(row, centre):
        edges = np.flatnonzero(np.diff(np.r_[0, body[row].astype(np.int8), 0]))
        best = None
        for start, stop in zip(edges[::2], edges[1::2]):
            distance = 0.0 if start - 1.5 <= centre <= stop + 0.5 else min(abs(start - centre), abs(stop - 1 - centre))
            if best is None or distance < best[0]:
                best = distance, start, stop
        return best

    barrel = []
    for row in range(behind):
        found = run(row, half)
        if found is not None and found[0] == 0 and found[1] > 0 and found[2] < len(across):
            barrel.append(found[2] - found[1])
    # Fingers beside the barrel widen its cross-section: the narrower rows show the barrel alone, and the
    # measurement only adjusts the given width.
    measured = float(np.clip(np.percentile(barrel, 25), 0.6 * width, 1.1 * width)) if barrel else float(width)
    centre, misses, covered, kind, leaving = float(half), 0, 0, None, 0.0
    steps, widths, middles = [], [], []
    for row in range(behind, len(along)):
        if outside[row, half]:
            kind = "edge"
            break
        found = run(row, centre)
        if found is None or found[0] > 2:
            misses += 1
            if misses >= 3:
                break
            continue
        _, start, stop = found
        # BORDER_REPLICATE is useful for sampling but contributes no real edge support.
        if outside[row, max(0, start - 1):min(len(across), stop + 1)].any():
            kind = "edge"
            break
        if stop - start > 1.4 * measured + 2 or start == 0 or stop == len(across):
            covered += 1
            misses = 0
            leaving = float(along[row])
            continue
        if covered >= 3:
            steps, widths, middles = [], [], []  # Out from under the fingers: the front of the pen starts here.
        misses = covered = 0
        middle = (start + stop - 1) / 2
        centre += float(np.clip(middle - centre, -1.5, 1.5))
        steps.append(float(along[row]))
        widths.append(float(stop - start))
        middles.append(middle - half)
    if kind is None and covered >= 3:
        # Under the fingers and not out again: the point is hidden, at best where the hand ends on this line.
        tip = end + direction * leaving
        return (float(tip[0]), float(tip[1])), "hand", measured, 0.0
    if not steps:
        return tuple(float(v) for v in end), kind or "blunt", measured, 0.0
    steps, widths, middles = np.asarray(steps), np.asarray(widths), np.asarray(middles)
    narrow = np.flatnonzero(widths < 0.8 * measured)  # The cone starts where the barrel first narrows.
    taper_start = float(steps[max(narrow[0] - 1, 0)]) if len(narrow) else float(steps[-1])
    # Past the barrel a pen only gets narrower. Where the cross-section widens again, the nib lies on
    # something dark (writing it points at, a printed line), which is not part of the pen: those rows are
    # left out, and what the walk found beyond them is trusted less than the cone's own shape.
    grown = (widths > np.minimum.accumulate(widths) + 2.5) & (steps > taper_start)
    on_ink = bool(grown.any()) and kind is None
    if on_ink:
        steps, widths, middles = steps[~grown], widths[~grown], middles[~grown]
    last = float(steps[-1])
    taper = last - taper_start
    reach, offset, seen = last, float(middles[-1]), 0.0
    if kind is None:
        kind = "blunt" if taper < 0.35 * measured else "open"
        cone = (steps >= taper_start) & (widths >= 0.25 * measured) & (widths <= 0.85 * measured)
        if capture is not None:
            region, mask = capture
            observed_end = end + direction * last + normal * middles[-1]
            if not capture_contains(observed_end, frame.shape, region, mask, margin=2):
                return tuple(float(v) for v in observed_end), "edge", measured, 0.0
            # Measure the taper only from its two actually visible sides in the capture.
            for index in np.flatnonzero(cone):
                middle = end + direction * steps[index] + normal * middles[index]
                half_section = normal * (0.5 * widths[index] + 1)
                if not all(capture_contains(edge, frame.shape, region, mask)
                           for edge in (middle - half_section, middle + half_section)):
                    cone[index] = False
        fitted = False
        if np.count_nonzero(cone) >= 4:
            slope, intercept = np.polyfit(steps[cone], widths[cone], 1)
            # Writing that touches the cone's side widens single rows: fit again without those.
            clean = cone & (widths - (slope * steps + intercept) <= 1.5)
            if 4 <= np.count_nonzero(clean) < np.count_nonzero(cone):
                cone = clean
                slope, intercept = np.polyfit(steps[cone], widths[cone], 1)
            fitted = slope < -0.08
        if fitted:
            # The cone's edges meet just past the point (a point is a little blunt, about a fifth of the
            # barrel wide): ink or a shadow under the nib cannot move it.
            apex = -intercept / slope
            blunt = apex + 0.18 * measured / slope
            reach = float(min(last, blunt) if on_ink else min(max(last, apex - 0.45 * taper), blunt))
        if on_ink and cone_length > 0 and len(narrow):
            # The point rests on dark writing (it points at it), which hides its outline. Where the barrel
            # starts to narrow is still seen, and the point lies as far from there as it always does.
            reach = taper_start + cone_length
        if fitted and reach - taper_start >= 0.7 * measured:
            kind = "point"
            seen = 0.0 if on_ink else reach - taper_start
        elif on_ink:
            kind = "ink"
        if fitted:
            lean, level = np.polyfit(steps[cone], middles[cone], 1)
            offset = float(lean * reach + level)
    tip = end + direction * reach + normal * offset
    return (float(tip[0]), float(tip[1])), kind, measured, float(seen)


def _pen_masks(frame, region, scale):
    """The part of the frame to search, at ``scale``, with what a barrel is told by: its HSV channels,
    the skin-coloured pixels, plain paper, and the coloured and the black pixels a barrel can be made of."""
    height, width = frame.shape[:2]
    x0, y0, x1, y1 = region or (0, 0, width, height)
    x0, y0, x1, y1 = max(0, int(x0)), max(0, int(y0)), min(width, int(x1)), min(height, int(y1))
    if x1 - x0 < 40 or y1 - y0 < 40:
        return None
    patch = cv2.resize(frame[y0:y1, x0:x1], None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    hue, saturation, value = cv2.split(cv2.cvtColor(patch, cv2.COLOR_BGR2HSV))
    skin = ((hue <= 25) | (hue >= 165)) & (saturation >= 30) & (saturation <= 180) & (value >= 60)
    coloured = (saturation >= 90) & (value >= 45) & ~skin
    black = value < 60
    paper_like = (saturation <= 45) & (value >= float(np.percentile(value[::4, ::4], 90)) - 40)
    return np.array((x0, y0), np.float32), hue, saturation, value, skin, coloured, black, paper_like


INK_WIDTH = 2  # Ink strokes are about this wide at half scale; a barrel is much thicker.


def find_bars(frame, region=None, scale=0.5):
    """Both ends of every bar in the frame that looks like a pen's barrel, held or not: [(end, end)].

    Called on the frame from before the writing, it lists what must not be taken for the pen later: a
    cable, a ruler, a pen left lying on the desk (``find_pen(ignore=...)``)."""
    masks = _pen_masks(frame, region, scale)
    if masks is None:
        return []
    origin, hue, saturation, value, skin, coloured, black, paper_like = masks
    widest = 0.018 * frame.shape[1] * scale
    bars = []
    for mask in (coloured, black):
        for _, centre, axis, low, high, _ in _barrels(
                mask, hue if mask is coloured else None, saturation, paper_like, INK_WIDTH, widest,
                22 * scale * 2, 140 * scale):
            ends = (centre + np.outer((low, high), axis)) / scale + origin
            bars.append((ends[0], ends[1]))
    return bars


def find_pen(frame, region=None, hands=(), scale=0.5, colour=None, previous=None, ignore=(), inside=None, surface=None):
    """The pen in the frame, found by its barrel: an elongated, coloured (or black), non-skin shape.

    The hand model often misses a hand reaching in from the far side of the desk, and its landmarks
    only give the fingers, not the nib. The barrel itself is easy to see: a saturated or very dark bar,
    several times longer than wide and thicker than ink, crossed by the fingers that hold it
    (``_barrels``). Each end of it is then followed outward (``_probe``): the nib is the end that tapers
    to a point on the paper, on the short side of the grip, away from the wrist, and where it was in
    ``previous`` (the PenEnd of the frame before). ``ignore`` lists the ends of bars that lay in the
    picture before anyone held a pen (``find_bars``): a bar still lying exactly there is not the pen.
    ``inside`` and ``surface`` constrain the nib and its observed taper edges. The shaft can extend
    outside the writing area so it can still establish the orientation of a pen at the page margin.
    Returns a PenEnd, or None.
    """
    width = frame.shape[1]
    masks = _pen_masks(frame, region, scale)
    if masks is None:
        return None
    origin, hue, saturation, value, skin, coloured, black, paper_like = masks
    ink_width = INK_WIDTH
    widest = 0.018 * width * scale  # A barrel is narrower than a wrapper or a phone edge.
    near_skin = cv2.dilate(skin.astype(np.uint8), np.ones((9, 9), np.uint8)) != 0  # Next to the hand holding it.
    grips = [np.mean([hand["fingertips"][name]["pixel"] for name in ("thumb", "index")], axis=0)
             for hand in hands or () if hand.get("fingertips") and "thumb" in hand["fingertips"]
             and "index" in hand["fingertips"]]
    wrists = [np.asarray(hand["landmarks"][0]["pixel"], float) for hand in hands or () if hand.get("landmarks")]
    estimated = None
    for mask, weight in ((coloured, 1.0), (black, 0.7)):
        candidates = []
        for points, centre, axis, low, high, thickness in _barrels(
                mask, hue if mask is coloured else None, saturation, paper_like, ink_width, widest,
                22 * scale * 2, 140 * scale):
            ys, xs = points[:, 1].astype(int), points[:, 0].astype(int)
            if near_skin[ys, xs].mean() < 0.03:
                continue
            length = high - low
            ends = (centre + np.outer((low, high), axis)) / scale + origin
            if any(min(max(np.linalg.norm(ends[0] - a), np.linalg.norm(ends[1] - b)),
                       max(np.linalg.norm(ends[0] - b), np.linalg.norm(ends[1] - a))) <= 8 for a, b in ignore):
                continue  # Lying where it lay before the recording: a cable, a ruler, a pen put down.
            score = weight * min(length / (60 * scale * 2), 2.0)
            shade = (float(np.median(hue[ys, xs])), float(np.median(saturation[ys, xs])), float(np.median(value[ys, xs])))
            reds = hue[ys, xs]
            if np.count_nonzero(reds < 12) and np.count_nonzero(reds > 168):
                # Red lies on both ends of the hue scale; the median of the two halves would be no colour at all.
                turned = (reds.astype(np.float32) + 90) % 180
                shade = ((float(np.median(turned)) - 90) % 180,) + shade[1:]
            match = 1.0
            if colour is not None:
                # The pen seen before: a cable or a marker of another colour is not it.
                difference = np.array((hue_distance(shade[0], colour[0]), shade[1] - colour[1],
                                       shade[2] - colour[2])) / (12.0, 45.0, 55.0)
                match = float(np.exp(-0.5 * np.sum(difference ** 2)))
            held = False
            if grips:
                # The barrel the writing hand holds: close to its thumb and index tips.
                near = min(float(np.min(np.linalg.norm(ends - grip, axis=1))) for grip in grips)
                held = near < 0.6 * length / scale
                score *= 1.5 if held else 0.6
            if previous is not None:
                # The pen of the frame before moves on a little; it does not jump to another bar.
                reach = 0.5 * length / scale + 60
                if min(np.linalg.norm(ends - np.asarray(previous.point, float), axis=1).min(),
                       np.linalg.norm(ends - np.asarray(previous.back, float), axis=1).min()) < reach:
                    score *= 1.4
            # Skin along the barrel: where the fingers hold it, as a share of its length from the low end.
            steps = centre + np.outer(np.arange(low, high + 1), axis)
            sx = np.clip(np.rint(steps[:, 0]).astype(int), 0, skin.shape[1] - 1)
            sy = np.clip(np.rint(steps[:, 1]).astype(int), 0, skin.shape[0] - 1)
            covered = np.flatnonzero(skin[sy, sx])
            grip_at = float(np.median(covered)) / max(len(steps) - 1, 1) if len(covered) >= 3 else None
            around = []
            reach = max(6, int(0.35 * length))
            for x, y in (centre + np.outer((low, high), axis)).astype(int):
                around.append(int(skin[max(0, y - reach):y + reach + 1, max(0, x - reach):x + reach + 1].sum()))
            candidates.append((score, match, ends, thickness / scale, shade, held, mask is coloured, grip_at, around))
        # Try the learned colour first, then the other barrels in case the pen has changed.
        candidates.sort(key=lambda item: (item[1] >= 0.45, item[0] * (item[1] if item[1] >= 0.45 else 1.0)),
                        reverse=True)
        for _, _, ends, thickness, shade, held, is_coloured, grip_at, around in candidates:
            result = _evaluate_barrel(frame, ends, thickness, wrists, held, shade, is_coloured, grip_at, around,
                                      previous, inside, surface)
            if result is not None:
                if result.tapered:
                    return result
                if estimated is None:
                    estimated = result
    return estimated


def _evaluate_barrel(frame, ends, thickness, wrists, held, shade, is_coloured, grip_at, around, previous,
                     inside, surface):
    outward = (ends[0] - ends[1]) / max(float(np.linalg.norm(ends[0] - ends[1])), 1.0)
    known = previous.cone if previous is not None else 0.0
    probes = [_probe(frame, ends[0], outward, thickness, known), _probe(frame, ends[1], -outward, thickness, known)]
    votes = []
    for index, (_, kind, _, _) in enumerate(probes):
        vote = {"point": 1.2, "blunt": -0.6, "edge": -1.0}.get(kind, 0.0)
        if grip_at is not None:
            # The fingers hold a pen near its point: the nib is the end on the short side of the grip.
            share = grip_at if index == 0 else 1 - grip_at
            vote += 0.6 if share < 0.45 else (-0.6 if share > 0.55 else 0.0)
        elif around[index] < around[1 - index] and probes[1 - index][1] != "blunt":
            vote += 0.3  # Less skin around it: the other end is in the hand.
        if wrists and held:
            wrist = min(wrists, key=lambda point: float(np.min(np.linalg.norm(ends - point, axis=1))))
            if np.linalg.norm(ends[index] - wrist) > np.linalg.norm(ends[1 - index] - wrist):
                vote += 0.5
        if previous is not None:
            before = np.subtract(previous.point, previous.back)
            if before @ (ends[index] - ends[1 - index]) > 0:
                vote += 0.5
        votes.append(vote)
    nib = int(np.argmax(votes))
    tip, kind, measured, seen = probes[nib]
    if kind == "edge" or not capture_contains(tip, frame.shape, inside, surface, margin=2):
        return None  # An exterior nib does not turn the opposite end into a candidate.
    if kind == "point" and (inside is not None or surface is not None):
        direction = outward if nib == 0 else -outward
        tip, kind, measured, seen = _probe(frame, ends[nib], direction, thickness, known, (inside, surface))
        if kind == "edge" or not capture_contains(tip, frame.shape, inside, surface, margin=2):
            return None
    back = ends[1 - nib]
    # The cone's length as seen whole on plain paper, steadied over the frames (see _probe).
    cone = known if seen <= 0 else (seen if known <= 0 else 0.7 * known + 0.3 * seen)
    return PenEnd(tuple(int(round(v)) for v in tip), tuple(int(round(v)) for v in back),
                  float(max(thickness, 0.8 * measured)), shade, held, is_coloured, kind == "point", cone)


def locate_pen(frame, grip, wrist, radius, inside=None, surface=None):
    """Find a narrow pen shaft aligned with the grip, then estimate its exposed endpoint."""
    height, width = frame.shape[:2]
    x0, y0, x1, y1 = map(int, inside) if inside is not None else (0, 0, width, height)
    x, y = grip
    reach = max(28, round(1.4 * radius))
    left, top = max(0, x0, x - reach), max(0, y0, y - reach)
    right, bottom = min(width, x1, x + reach + 1), min(height, y1, y + reach + 1)
    if right - left < 15 or bottom - top < 15:
        return None
    patch = frame[top:bottom, left:right]
    hue, saturation, value = cv2.split(cv2.cvtColor(patch, cv2.COLOR_BGR2HSV))
    # Saturated blue/green/purple and red shafts stand out from skin and paper.
    colored = (saturation >= 85) & ((hue >= 36) | (hue <= 5)) & (value >= 35)
    dark = value < 85
    origin = np.array((x - left, y - top), dtype=float)
    forward = np.subtract(grip, wrist).astype(float)
    norm = np.linalg.norm(forward)
    if norm < 1:
        return None
    forward /= norm
    best = None
    for mask, color_weight in ((colored, 1.0), (dark, 0.78)):
        strokes = cv2.Canny(mask.astype(np.uint8) * 255, 40, 120)
        lines = cv2.HoughLinesP(strokes, 1, np.pi / 180, max(10, radius // 4),
                                minLineLength=max(15, radius // 2), maxLineGap=max(3, radius // 8))
        if lines is None:
            continue
        for segment in lines.reshape(-1, 4):
            points = segment.reshape(2, 2).astype(float) - origin
            progress = points @ forward
            tip, back = points[np.argmax(progress)], points[np.argmin(progress)]
            if np.linalg.norm(tip) > radius * 1.25 or max(progress) < radius * 0.14:
                continue
            shaft = tip - back
            length = np.linalg.norm(shaft)
            if not length or (shaft @ forward) / length < 0.25:
                continue
            distance_to_grip = abs(shaft[0] * back[1] - shaft[1] * back[0]) / length
            if distance_to_grip > radius * 0.22 or min(progress) > radius * 0.36:
                continue
            score = color_weight * min(1.0, length / radius) * (1 - distance_to_grip / radius)
            if best is None or score > best[0]:
                best = score, tip + origin, shaft / length
        # A visible colored shaft is easier to distinguish from letters, shadows, and the desk.
        if best is not None and best[0] >= 0.47:
            break
    if best is None or best[0] < 0.40:
        return None
    mask = surface[top:bottom, left:right].astype(np.uint8, copy=False) if surface is not None else None
    tip = _centered_endpoint(patch, best[1], best[2], radius, mask)
    if tip is None:
        return None
    result = tuple(np.rint(tip + (left, top)).astype(int))
    return result if capture_contains(result, frame.shape, inside, surface, margin=2) else None


def _centered_endpoint(patch, end, axis, radius, mask=None):
    """Walk across the shaft's center toward the narrow tip until foreground ends."""
    normal = np.array((-axis[1], axis[0]))
    along = np.arange(-max(9, round(radius * 0.38)), max(9, round(radius * 0.50)) + 1)
    half = max(6, min(18, round(radius * 0.28)))
    across = np.arange(-half, half + 1)
    grid = end + along[:, None, None] * axis + across[None, :, None] * normal
    gray = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY)
    strip = cv2.remap(gray, grid[:, :, 0].astype(np.float32), grid[:, :, 1].astype(np.float32),
                      cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=255)
    valid = None if mask is None else cv2.remap(
        mask, grid[:, :, 0].astype(np.float32), grid[:, :, 1].astype(np.float32), cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    background = np.median(np.c_[strip[:, :2], strip[:, -2:]], axis=1)
    foreground = strip < np.minimum(196, background - 20)[:, None]
    center, width, misses, last = 0.0, None, 0, None
    for sample, (step, row) in enumerate(zip(along, foreground)):
        x, y = end + step * axis + center * normal
        if not (0.5 <= x < gray.shape[1] - 1 and 0.5 <= y < gray.shape[0] - 1):
            return None
        if valid is not None and not valid[sample, int(round(center)) + half]:
            return None
        edges = np.flatnonzero(np.diff(np.r_[False, row, False]))
        runs = [(across[start], across[stop - 1]) for start, stop in edges.reshape(-1, 2)]
        nearby = [(first, last_pixel) for first, last_pixel in runs
                  if first > -half and last_pixel < half and first - 3 <= center <= last_pixel + 3
                  and last_pixel - first + 1 <= (half if width is None else max(4, width * 1.7))]
        if not nearby:
            misses += 1
            if last is not None and misses >= 3:
                return last[1] if last[0] >= -max(5, radius * 0.15) else None
            continue
        first, last_pixel = min(nearby, key=lambda run: abs(np.mean(run) - center))
        if valid is not None and not valid[sample, first + half - 1:last_pixel + half + 2].all():
            return None
        center = (first + last_pixel) / 2
        span = last_pixel - first + 1
        width = span if width is None else (0.75 * width + 0.25 * span)
        misses = 0
        last = step, end + step * axis + center * normal
    # The shaft is clipped or still continues at the edge of the search window.
    if last is not None and misses >= 2 and last[0] >= -max(5, radius * 0.15):
        return last[1]
    return None


def draw_trajectory(samples, area, frame_size, grip_samples):
    """Render ordered segments, leaving gaps where a tip was lost or moved suddenly."""
    if len(samples) < 10:
        return None
    paths = [path for path in split_strokes(samples, frame_size[0]) if len(path) >= 3
             and np.sum(np.linalg.norm(np.diff(np.asarray(path, float), axis=0), axis=1)) >= 8]
    tight = stroke_bounds(paths)
    if tight is None:
        return None
    left, top, right, bottom = tight
    if right - left < 15 or bottom - top < 7:
        return None
    margin = max(12, round((bottom - top) * 0.22))
    width, height = frame_size
    x0, y0, x1, y1 = area or (0, 0, width, height)
    bounds = max(x0, left - margin), max(y0, top - margin), min(x1, right + margin), min(y1, bottom + margin)
    if bounds[2] <= bounds[0] or bounds[3] <= bounds[1]:
        return None
    count = sum(len(path) for path in paths)
    reliable = len(paths) >= 2 and count >= 18 and count >= grip_samples * 0.58
    return render_strokes(paths, bounds), bounds, reliable


def split_strokes(samples, frame_width, gap=0.22):
    """Split ordered (time, point[, stroke id]) samples at time gaps, jumps, or stroke changes."""
    strokes, previous = [], None
    for sample in samples:
        now, point = sample[0], tuple(sample[1])
        stroke = sample[2] if len(sample) > 2 else None
        if (previous is None or now - previous[0] > gap or stroke != previous[2]
                or np.linalg.norm(np.subtract(point, previous[1])) > max(18, frame_width * 0.027)):
            strokes.append([])
        strokes[-1].append(point)
        previous = now, point, stroke
    return [stroke for stroke in strokes if len(stroke) >= 2]


def stroke_bounds(strokes):
    """Tight half-open bounds around all stroke points, or None."""
    if not strokes:
        return None
    points = np.vstack([np.asarray(stroke) for stroke in strokes])
    left, top = points.min(axis=0)
    right, bottom = points.max(axis=0)
    return int(left), int(top), int(right) + 1, int(bottom) + 1


def render_strokes(strokes, bounds, thickness=2):
    """Draw pen-down strokes on white paper in the coordinates of a crop's bounds."""
    left, top, right, bottom = bounds
    image = np.full((max(1, bottom - top), max(1, right - left), 3), 255, np.uint8)
    for stroke in strokes:
        cv2.polylines(image, [(np.asarray(stroke) - (left, top)).astype(np.int32)], False,
                      (12, 12, 12), thickness, cv2.LINE_AA)
    return image


def rotate_points(points, rotation):
    """Turn image coordinates clockwise by 0, 90, 180, or 270 degrees (up to a translation)."""
    x, y = np.asarray(points, float).reshape(-1, 2).T
    return {0: np.c_[x, y], 90: np.c_[-y, x], 180: np.c_[-x, -y], 270: np.c_[y, -x]}[rotation % 360]


def rotate_image(image, rotation):
    codes = {90: cv2.ROTATE_90_CLOCKWISE, 180: cv2.ROTATE_180, 270: cv2.ROTATE_90_COUNTERCLOCKWISE}
    return image if image is None or rotation % 360 == 0 else cv2.rotate(image, codes[rotation % 360])


def render_trace(strokes, rotation=0, text_height=None, glyph=80, longest=1400):
    """Clean pen-down strokes, upright for the writer, cropped tight and scaled to ~``glyph`` px glyphs.

    Strokes are drawn black on white in writing order with a light smoothing of the tip
    jitter; nothing else (hand, paper, earlier ink) is in the image.
    """
    strokes = [rotate_points(stroke, rotation) for stroke in strokes if len(stroke)]
    if not strokes:
        return None
    if not text_height:
        heights = [np.ptp(stroke[:, 1]) for stroke in strokes if len(stroke) > 1]
        text_height = float(np.median(heights)) if heights else 20.0
    points = np.vstack(strokes)
    low, high = points.min(axis=0), points.max(axis=0)
    scale = float(np.clip(glyph / max(text_height, 1.0), 0.2, 8.0))
    pad = round(0.45 * glyph)
    size = (high - low) * scale + 2 * pad
    if size.max() > longest:
        shrink = (longest - 2 * pad) / max((high - low).max() * scale, 1)
        scale *= shrink
        size = (high - low) * scale + 2 * pad
    width, height = (max(96, int(np.ceil(v)) + 1) for v in size)
    image = np.full((height, width, 3), 255, np.uint8)
    offset = np.array((width, height)) / 2 - (low + high) / 2 * scale
    thickness = max(2, round(glyph * 0.06))
    for stroke in strokes:
        if len(stroke) >= 3:
            smooth = stroke.copy()
            smooth[1:-1] = (stroke[:-2] + 2 * stroke[1:-1] + stroke[2:]) / 4
            stroke = smooth
        drawn = np.rint(stroke * scale + offset).astype(np.int32)
        if len(drawn) == 1 or np.ptp(drawn, axis=0).max() < 2:
            cv2.circle(image, tuple(int(v) for v in drawn[0]), thickness, (0, 0, 0), -1, cv2.LINE_AA)
        else:
            cv2.polylines(image, [drawn.reshape(-1, 1, 2)], False, (0, 0, 0), thickness, cv2.LINE_AA)
    return image


def find_closed_loop(points, min_radius, max_radius):
    """Find a final closed, roughly round loop in ordered points.

    Returns (start index, center, radius) for the earliest start whose segment to the
    last point turns at least 1.75 full half-turns around its center and closes near
    its start. Later points may only dwell near the ring.
    """
    points = np.asarray(points, float)
    for start in range(max(0, len(points) - 100), len(points) - 4):
        segment = points[start:]
        center = (segment.min(axis=0) + segment.max(axis=0)) / 2
        distances = np.linalg.norm(segment - center, axis=1)
        radius = float(np.median(distances))
        if not min_radius <= radius <= max_radius or np.std(distances) > 0.35 * radius:
            continue
        angles = np.unwrap(np.arctan2(segment[:, 1] - center[1], segment[:, 0] - center[0]))
        turned = np.max(np.abs(angles - angles[0]))
        closing = np.min(np.linalg.norm(segment[len(segment) // 2:] - segment[0], axis=1))
        if turned >= 1.75 * np.pi and closing <= 0.75 * radius:
            return start, tuple(center), radius
    return None


@dataclass(frozen=True)
class PenTip:
    point: tuple
    grip: tuple
    radius: int
    scale: float
    visual: bool


def _landmark(hand, index):
    landmarks = hand.get("landmarks") or ()
    return np.asarray(landmarks[index]["pixel"], float) if index < len(landmarks) else None


def pen_grip(hand):
    """Grip center, pen-axis rest point, grip search radius, and hand scale, or None.

    MediaPipe landmarks: 0 wrist, 2 thumb MCP, 5 index MCP, 9 middle MCP. A pen passes
    between the thumb and index tips and rests in the web near the index knuckle.
    """
    thumb = np.asarray(hand["fingertips"]["thumb"]["pixel"], float)
    index = np.asarray(hand["fingertips"]["index"]["pixel"], float)
    wrist = _landmark(hand, 0)
    size = np.linalg.norm(index - wrist)
    if size < 12 or np.linalg.norm(index - thumb) > 0.7 * size:
        return None
    grip = (thumb + index) / 2
    knuckle, thumb_base, middle = _landmark(hand, 5), _landmark(hand, 2), _landmark(hand, 9)
    rest = 0.6 * knuckle + 0.4 * thumb_base if knuckle is not None and thumb_base is not None else wrist
    if np.linalg.norm(grip - rest) < 4:
        rest = wrist
    scale = np.linalg.norm(middle - wrist) if middle is not None else 0.55 * size
    return grip, rest, int(np.clip(size * 0.7, 24, 100)), max(float(scale), 8.0)


def estimate_pen_tip(hand, frame=None, bias=(0.0, 0.0), inside=None, surface=None):
    """Place a landmark at the pen tip rather than the index fingertip.

    The geometric estimate continues the rest-to-grip pen axis past the grip by about a
    quarter of the hand scale (the exposed pen). A visible shaft found near the grip
    replaces it when the two agree. ``bias`` is a learned (along, across) correction in
    hand-scale units, used when the shaft is hidden.
    """
    found = pen_grip(hand)
    if found is None:
        return None
    grip, rest, radius, scale = found
    axis = (grip - rest) / np.linalg.norm(grip - rest)
    normal = np.array((-axis[1], axis[0]))
    estimate = grip + axis * scale * (0.28 + bias[0]) + normal * scale * bias[1]
    grip_point = tuple(int(v) for v in np.rint(grip))
    if frame is not None:
        seen = locate_pen(frame, grip_point, tuple(int(v) for v in np.rint(rest)), radius, inside, surface)
        if seen is not None and np.linalg.norm(np.subtract(seen, estimate)) <= max(radius * 0.9, scale * 0.35):
            return PenTip(tuple(int(v) for v in seen), grip_point, radius, scale, True)
        if not capture_contains(np.rint(estimate), frame.shape, inside, surface):
            return None
    return PenTip(tuple(int(v) for v in np.rint(estimate)), grip_point, radius, scale, False)


class OneEuro:
    """One Euro filter: steady when the pen is slow, responsive when it moves quickly."""

    def __init__(self, min_cutoff=2.5, beta=0.03, derivative_cutoff=1.5, reset_after=0.35):
        self.min_cutoff, self.beta, self.derivative_cutoff = min_cutoff, beta, derivative_cutoff
        self.reset_after = reset_after
        self.time = self.value = self.speed = None

    @staticmethod
    def _alpha(dt, cutoff):
        return 1 / (1 + 1 / (2 * np.pi * cutoff * dt))

    def __call__(self, now, value):
        value = np.asarray(value, float)
        if self.time is None or now - self.time > self.reset_after or now <= self.time:
            self.time, self.value, self.speed = now, value, np.zeros_like(value)
            return value
        dt = now - self.time
        self.speed = self.speed + self._alpha(dt, self.derivative_cutoff) * ((value - self.value) / dt - self.speed)
        cutoff = self.min_cutoff + self.beta * np.linalg.norm(self.speed)
        self.value = self.value + self._alpha(dt, cutoff) * (value - self.value)
        self.time = now
        return self.value


class PenTipTracker:
    """Smoothed pen-tip landmark per hand, with a grip-to-tip offset learned from visible shafts."""

    def __init__(self):
        self.filters, self.bias = {}, {}

    def update(self, side, hand, frame, now, inside=None, surface=None):
        bias = self.bias.get(side, np.zeros(2))
        tip = estimate_pen_tip(hand, frame, bias, inside, surface)
        if tip is None:
            return None
        if tip.visual:
            plain = np.asarray(estimate_pen_tip(hand).point, float)
            axis = plain - tip.grip
            if np.linalg.norm(axis) > 1:
                axis /= np.linalg.norm(axis)
                offset = (np.asarray(tip.point, float) - plain) / tip.scale
                observed = np.array((offset @ axis, offset @ np.array((-axis[1], axis[0]))))
                self.bias[side] = np.clip(0.8 * bias + 0.2 * observed, -0.3, 0.3)
        smooth = self.filters.setdefault(side, OneEuro())(now, tip.point)
        point = tuple(int(v) for v in np.rint(smooth))
        if frame is not None and not capture_contains(point, frame.shape, inside, surface):
            point = tip.point
        return PenTip(point, tip.grip, tip.radius, tip.scale, tip.visual)
