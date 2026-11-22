"""Follow the sheet of paper when it moves while you write.

A hand pushes the sheet around: it slides and turns during a recording, sometimes suddenly and by a
lot. Compared against the reference frame from the key press, everything already written then looks
new, and new strokes look doubled. ``SheetTracker`` follows the sheet through the whole recording and
gives its pose: the affine map from the reference frame to the current frame.

What it follows are the sheet's landmarks: the writing on it (the best one) and its outline against
the desk, both taken relative to the paper around them so light and exposure cancel. They are kept in
a key picture in the sheet's own coordinates. At every checkpoint:

1. Small patches of the key picture are looked up in the frame around where the pose puts them. The
   patches that agree on one motion give the new pose, to a fraction of a pixel; patches under a hand
   or on anything else that moved are outvoted.
2. When much of the key picture is no longer where it should be (a sudden large move), the part that
   left is searched in the part of the frame that is new, over turns of up to 45 degrees, and the
   patches then refine that.
3. A motion is accepted only when it explains the picture: it must make clearly more of the sheet's
   landmarks line up than it breaks. Otherwise the last pose is kept, and ``state`` says ``lost``
   while the landmarks are visibly somewhere else.

What stays in place in the frame while the sheet moves (a desk seam, another paper) is the desk
and is taken out of the key picture. What a hand covers stays in the key picture as it was, so
the sheet is found again when the hand lifts.

``sheet_motion`` is the older one-shot alignment of two frames, kept for sessions without a tracker.
"""

import cv2
import numpy as np

IDENTITY = np.float32([[1, 0, 0], [0, 1, 0]])


def _ink_map(image, scale, hidden=None):
    """Darkness of the marks against the local paper (lighting cancels), at ``scale``, hands left out."""
    from paper_ink import darkness

    small = cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
    dark, _ = darkness(gray, max(5, round(0.012 * gray.shape[1] / scale * scale)))
    dark[_usable(small) == 0] = 0
    if hidden is not None:
        dark[cv2.resize(hidden, (dark.shape[1], dark.shape[0]), interpolation=cv2.INTER_NEAREST) != 0] = 0
    return cv2.GaussianBlur(np.minimum(dark, 0.4), (0, 0), 1.0)


def sheet_motion(reference, frame, area, hidden=None, scale=0.5, max_angle=25.0):
    """How the sheet moved from ``reference`` to ``frame``: a 2x3 rotation-and-shift (reference -> frame),
    found by aligning the writing already on it. Returns (motion, response); the identity with response
    0 when there is too little writing on the sheet to tell (a blank sheet has nothing to mistake)."""
    x0, y0, x1, y1 = area
    height, width = reference.shape[:2]
    pad = int(0.15 * max(x1 - x0, y1 - y0))
    x0, y0, x1, y1 = max(0, x0 - pad), max(0, y0 - pad), min(width, x1 + pad), min(height, y1 + pad)
    before = _ink_map(reference[y0:y1, x0:x1], scale)
    after = _ink_map(frame[y0:y1, x0:x1], scale, None if hidden is None else hidden[y0:y1, x0:x1])
    if float((before > 0.08).sum()) < 60 or float((after > 0.08).sum()) < 60:
        return IDENTITY.copy(), 0.0
    rows, cols = before.shape
    centre = (cols / 2.0, rows / 2.0)
    window = cv2.createHanningWindow((cols, rows), cv2.CV_32F)
    best = None

    def search(angles):
        nonlocal best
        for angle in angles:
            rotation = cv2.getRotationMatrix2D(centre, float(angle), 1.0)
            turned = cv2.warpAffine(before, rotation, (cols, rows))
            (dx, dy), response = cv2.phaseCorrelate(turned.astype(np.float32), after.astype(np.float32), window)
            if best is None or response > best[0]:
                best = response, angle, rotation, dx, dy

    # Coarse over a sheet turned by up to max_angle, then fine around the best turn.
    search(np.arange(-max_angle, max_angle + 0.01, 2.5))
    search(best[1] + np.arange(-2.0, 2.01, 0.5))
    response, angle, rotation, dx, dy = best
    # Back to full-resolution frame coordinates: crop offset, scale, rotation about the crop centre, shift.
    motion = np.vstack([rotation, (0, 0, 1)])
    motion[:2, 2] += (dx, dy)
    # Only a motion that explains the picture: the writing already there must line up clearly better.
    moved = cv2.warpAffine(before, motion[:2].astype(np.float32), (cols, rows))
    gain = 1 - float(np.abs(after - moved).sum()) / max(float(np.abs(after - before).sum()), 1e-6)
    small = abs(angle) <= 1.0 and np.hypot(dx, dy) <= 6
    if gain < (0.05 if small else 0.2):
        return IDENTITY.copy(), 0.0  # A large move needs clear evidence; a small nudge needs a little.
    to_small = np.float32([[scale, 0, -x0 * scale], [0, scale, -y0 * scale], [0, 0, 1]])
    full = np.linalg.inv(to_small) @ motion @ to_small
    return full[:2].astype(np.float32), float(response)


def _usable(image):
    """0/255 mask of pixels that can be the sheet: not skin, pen, or anything coloured or dark."""
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    return (((hsv[:, :, 1] < 55) & (hsv[:, :, 2] > 90)).astype(np.uint8)) * 255


def displacement(first, second, box):
    """Largest distance in pixels between where two poses (3x3) put the corners of ``box`` (x0, y0, x1, y1)."""
    x0, y0, x1, y1 = box
    corners = np.float64([[x0, y0, 1], [x1, y0, 1], [x1, y1, 1], [x0, y1, 1]]).T
    return float(np.linalg.norm((first @ corners - second @ corners)[:2], axis=0).max())


def move_points(pose, points):
    """Points (n x 2) mapped by a 3x3 affine pose."""
    points = np.asarray(points, float).reshape(-1, 2)
    return points @ pose[:2, :2].T + pose[:2, 2]


class SheetTracker:
    """The sheet's pose through a recording. ``pose`` maps reference pixels to frame pixels (3x3 affine).

    ``state`` after each ``update``:
      steady  the landmarks are where the pose puts them
      moved   the sheet moved and the new pose explains the frame
      held    too little of the sheet is in view to tell; the last pose is kept
      lost    the landmarks are visibly not where the pose puts them and no motion explains the
              frame (the sheet is in motion, or went out of view); the last pose is kept
    """

    SCALE = 0.5  # The landmark pictures are half the frame's size.
    PATCH = 20  # Half the side of a patch, in landmark-picture pixels.
    MATCH = 0.6  # A patch is found when its correlation with the frame is at least this.
    CORNER = 0.0005  # Least structure (smaller eigenvalue of the gradients) for a patch to be told apart.

    def __init__(self, reference, area=None, hidden=None, margin=0.5):
        height, width = reference.shape[:2]
        self.size = (width, height)
        self.small = (int(round(width * self.SCALE)), int(round(height * self.SCALE)))
        x0, y0, x1, y1 = area or (0, 0, width, height)
        region = cv2.cvtColor(reference[y0:y1, x0:x1], cv2.COLOR_BGR2GRAY)
        self.level = float(np.percentile(region, 90)) if region.size else 255.0
        self.box = (x0, y0, x1, y1)
        # Only what is on or near the sheet is followed: the region from the key press with a wide margin
        # (it is often too small or off, with a hand on the sheet), in the sheet's own coordinates.
        pad = margin * max(x1 - x0, y1 - y0)
        self.domain = np.zeros(self.small[::-1], np.uint8)
        left, top, right, bottom = (int(round(v * self.SCALE)) for v in (x0 - pad, y0 - pad, x1 + pad, y1 + pad))
        self.domain[max(0, top):max(0, bottom), max(0, left):max(0, right)] = 1
        self.key, self.known = self._landmarks(reference, hidden)
        self.known &= self.domain
        self.pose = np.eye(3)
        self.state = "steady"
        self.moves = self.rekeyed = self.lost = self.updates = 0
        self.tied = True  # False once the key picture was started again: the pose no longer goes back to the reference.
        self._picture = self._seen = None
        self._failed = False  # The search for a sheet that jumped found nothing on the scene as it is.
        self.report = {}

    # Pictures --------------------------------------------------------------------------------

    def _landmarks(self, image, hidden=None):
        """(picture, seen): the landmarks of a frame at SCALE, 0..1, and where paper is in view (0/1)."""
        from paper_ink import darkness

        small = cv2.resize(image, self.small, interpolation=cv2.INTER_AREA)
        hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)
        # Paper and desk: bright and without colour. Skin, the pen, sleeves, and dark things are not.
        usable = (hsv[:, :, 1] < 55) & (hsv[:, :, 2] >= 0.6 * self.level)
        # Writing itself is dark or blue, but thin: only what is wider than a stroke is not paper.
        usable = cv2.morphologyEx((~usable).astype(np.uint8), cv2.MORPH_OPEN, np.ones((5, 5), np.uint8)) == 0
        coloured = (hsv[:, :, 1] >= 55) & (hsv[:, :, 2] >= 60)  # A hand, the pen.
        if hidden is not None:
            usable &= cv2.resize(hidden, self.small, interpolation=cv2.INTER_NEAREST) == 0
        # Away from the border of what is usable: a hand's outline and its shadow line are not landmarks.
        seen = cv2.erode(usable.astype(np.uint8), np.ones((13, 13), np.uint8))
        gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
        dark, _ = darkness(gray, 11)
        ink = np.clip((dark - 0.04) / 0.2, 0, 1)
        # The sheet's outline: a step in brightness against the desk, however faint, at half this size.
        coarse = cv2.GaussianBlur(cv2.resize(gray, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)
                                  .astype(np.float32), (0, 0), 1.0)
        step = cv2.magnitude(cv2.Sobel(coarse, cv2.CV_32F, 1, 0), cv2.Sobel(coarse, cv2.CV_32F, 0, 1))
        edge = np.clip((step / np.maximum(coarse, 1.0) - 0.06) / 0.25, 0, 1)
        edge = cv2.resize(edge, self.small, interpolation=cv2.INTER_LINEAR)
        self._raw = edge.copy(), usable, coloured, gray  # For finding the sheet's outline (see SheetWatch).
        edge[cv2.erode(seen, np.ones((13, 13), np.uint8)) == 0] = 0  # Coarser, so farther from the border.
        picture = np.maximum(ink, 0.6 * edge)
        picture[seen == 0] = 0
        return cv2.GaussianBlur(picture, (0, 0), 1.0), seen

    def _scaled(self, pose):
        scale = np.diag([self.SCALE, self.SCALE, 1.0])
        return scale @ pose @ np.linalg.inv(scale)

    def _view(self, picture, seen, pose):
        """The frame's landmarks brought into the sheet's coordinates by ``pose`` (at SCALE)."""
        matrix = pose[:2].astype(np.float32)
        view = cv2.warpAffine(picture, matrix, self.small, flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP)
        shown = cv2.warpAffine(seen, matrix, self.small, flags=cv2.INTER_NEAREST | cv2.WARP_INVERSE_MAP)
        return view, shown

    @staticmethod
    def _explained(key, view):
        """How much of each key landmark the frame shows at the same place (a pixel of tolerance)."""
        return np.minimum(key, cv2.dilate(view, np.ones((3, 3), np.uint8)))

    # Patches ---------------------------------------------------------------------------------

    def _corners(self, key, known, cell=24):
        """Places in a key picture with structure in two directions (letters, the sheet's corners): the
        best one of each cell of a grid, so they spread over the sheet."""
        x, y, w, h = cv2.boundingRect(known)
        rows, columns = h // cell, w // cell
        if rows == 0 or columns == 0:
            return np.zeros((0, 2), np.float32)
        strength = cv2.cornerMinEigenVal(key[y:y + rows * cell, x:x + columns * cell], 13)
        strength[known[y:y + rows * cell, x:x + columns * cell] == 0] = 0
        cells = strength.reshape(rows, cell, columns, cell).transpose(0, 2, 1, 3).reshape(rows, columns, cell * cell)
        best = cells.argmax(axis=2)
        ys, xs = np.nonzero(np.take_along_axis(cells, best[:, :, None], axis=2)[:, :, 0] >= self.CORNER)
        inside = best[ys, xs]
        if len(xs) > 90:  # The strongest: enough to tell, and looking them up stays cheap.
            strongest = np.argsort(-np.take_along_axis(cells, best[:, :, None], axis=2)[ys, xs, 0])[:90]
            xs, ys, inside = xs[strongest], ys[strongest], inside[strongest]
        return np.c_[x + xs * cell + inside % cell, y + ys * cell + inside // cell].astype(np.float32)

    def _patches(self, key, corners, view, visible, reach):
        """Look up the key picture's patches in ``view`` within ``reach`` pixels.
        Returns (points, shifts, found): where each usable patch is, how far off the frame shows it, and
        whether it was found at all."""
        half = self.PATCH
        width, height = self.small
        share = cv2.integral(visible)
        points, shifts, found = [], [], []
        for x, y in np.rint(corners).astype(int):
            x0, y0, x1, y1 = x - half, y - half, x + half + 1, y + half + 1
            if x0 - reach < 0 or y0 - reach < 0 or x1 + reach > width or y1 + reach > height:
                continue
            inside = share[y1, x1] - share[y0, x1] - share[y1, x0] + share[y0, x0]
            if inside < 0.8 * (x1 - x0) * (y1 - y0):
                continue  # Under a hand, or out of view.
            patch = key[y0:y1, x0:x1]
            if float(patch.max()) < 0.15:
                continue
            window = view[y0 - reach:y1 + reach, x0 - reach:x1 + reach]
            scores = cv2.matchTemplate(window, patch, cv2.TM_CCOEFF_NORMED)
            scores[~np.isfinite(scores)] = -1
            _, score, _, (px, py) = cv2.minMaxLoc(scores)
            points.append((x, y))
            edge = px in (0, 2 * reach) or py in (0, 2 * reach)
            if score < self.MATCH or edge:
                shifts.append((0.0, 0.0))
                found.append(False)
                continue
            # The peak to a fraction of a pixel, from the scores beside it.
            row, column = scores[py, px - 1:px + 2], scores[py - 1:py + 2, px]
            dx = 0.5 * (row[0] - row[2]) / min(row[0] - 2 * row[1] + row[2], -1e-6)
            dy = 0.5 * (column[0] - column[2]) / min(column[0] - 2 * column[1] + column[2], -1e-6)
            shifts.append((px - reach + float(np.clip(dx, -0.5, 0.5)), py - reach + float(np.clip(dy, -0.5, 0.5))))
            found.append(True)
        return (np.float32(points).reshape(-1, 2), np.float32(shifts).reshape(-1, 2), np.asarray(found, bool))

    @staticmethod
    def _fit(source, target, tolerance=1.0):
        """The motion most pairs agree on: (3x3, inliers). A turn, shift, and scale near one; with many
        pairs spread over the sheet, a full affine map (the camera looks at the desk at an angle, so a
        sheet that slides also changes its shape a little)."""
        none = None, np.zeros(len(source), bool)
        if len(source) < 4:
            return none
        matrix, inliers = cv2.estimateAffinePartial2D(source, target, method=cv2.RANSAC,
                                                      ransacReprojThreshold=2 * tolerance, maxIters=400,
                                                      confidence=0.995, refineIters=10)
        if matrix is None or not 0.9 <= float(np.hypot(matrix[0, 0], matrix[1, 0])) <= 1.1:
            return none
        inliers = inliers.ravel() != 0
        for _ in range(3):
            # With enough pairs that are not all on one line of writing, the affine map through them.
            if inliers.sum() < 10 or np.linalg.eigvalsh(np.cov(source[inliers].T))[0] < 30.0 ** 2:
                break
            ones = np.ones((int(inliers.sum()), 1), np.float32)
            solved = np.linalg.lstsq(np.hstack([source[inliers], ones]), target[inliers], rcond=None)[0].T
            stretch = np.linalg.svd(solved[:, :2], compute_uv=False)
            if not (0.95 <= stretch[1] and stretch[0] <= 1.05):
                break
            matrix = solved
            inliers = np.linalg.norm(source @ matrix[:, :2].T + matrix[:, 2] - target, axis=1) <= tolerance
        else:
            return np.vstack([matrix, (0, 0, 1)]).astype(float), inliers
        inliers = np.linalg.norm(source @ matrix[:, :2].T + matrix[:, 2] - target, axis=1) <= 2 * tolerance
        return np.vstack([matrix, (0, 0, 1)]).astype(float), inliers

    def _measure(self, key, known, corners, picture, seen, pose, reach):
        """Patches of a key picture looked up in the frame at ``pose`` (at SCALE). Returns a dict."""
        view, shown = self._view(picture, seen, pose)
        visible = known & shown
        points, shifts, found = self._patches(key, corners, view, visible, reach)
        # A patch at p that the frame shows at p + shift (sheet coordinates) is at pose (p + shift) in the frame.
        target = move_points(pose, points + shifts).astype(np.float32) if len(points) else points
        return {"view": view, "visible": visible, "points": points, "shifts": shifts, "found": found, "target": target}

    def _patch_motion(self, pose, measured):
        """The pose the patches ask for, None when they are where they should be or do not agree."""
        points, found, target = measured["points"], measured["found"], measured["target"]
        if found.sum() < 6:
            return None
        fitted, inliers = self._fit(points[found], target[found])
        if fitted is None:
            return None
        moved = displacement(fitted, pose, self._extent(points[found]))
        if moved >= 0.5:
            # Most patches moved together: the sheet did. A nudge of a pixel or so needs nearly all of them.
            share = 0.6 if moved >= 2.0 else 0.8
            enough = inliers.sum() >= max(6, share * found.sum()) and (moved >= 2.0 or found.sum() >= 0.7 * len(found))
            return fitted if enough else None
        # Most patches stayed. Others that moved together are the sheet when the rest is the desk.
        rest = np.flatnonzero(found)[~inliers]
        rest = rest[np.linalg.norm(measured["shifts"][rest], axis=1) >= 1.5]
        if len(rest) < 6:
            return None
        fitted, inliers = self._fit(points[rest], target[rest])
        if fitted is None or inliers.sum() < max(6, 0.6 * len(rest)):
            return None
        return fitted

    @staticmethod
    def _extent(points):
        x0, y0 = points.min(axis=0)
        x1, y1 = points.max(axis=0)
        return float(x0), float(y0), float(x1), float(y1)

    # A sudden large move ---------------------------------------------------------------------

    def _search(self, left, new, angles):
        """Where the landmarks that left (``left``, sheet coordinates) are among those that are new in the
        frame (``new``): up to four candidates, each a turn and shift as 3x3 at SCALE, best first. The
        picture of what left is laid over the new one at every place and under turns of up to 45 degrees,
        at an eighth of the frame's size, and the best fits are then sharpened at a quarter. Lines of
        writing repeat, so the best fit at this size is not always the right one: the patches decide."""
        quarter = [cv2.resize(image, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA) for image in (left, new)]
        eighth = [cv2.resize(image, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA) for image in quarter]
        ys, xs = np.nonzero(eighth[0] > 0.05)
        if len(xs) < 12:
            return []
        box = np.array([xs.min(), ys.min(), xs.max() + 1, ys.max() + 1], float)
        centre = (box[:2] + box[2:]) / 2
        radius = float(np.linalg.norm(box[2:] - box[:2])) / 2 + 2  # The box turned any way stays within this.

        def lay(pair, factor, angles, near=None, reach=0):
            """Best fits (score, angle, shift) of the first picture turned and laid over the second."""
            c, r = centre * factor, int(np.ceil(radius * factor))
            pad = r + 1
            first = cv2.copyMakeBorder(pair[0], pad, pad, pad, pad, cv2.BORDER_CONSTANT, value=0)
            second = cv2.copyMakeBorder(pair[1], pad, pad, pad, pad, cv2.BORDER_CONSTANT, value=0)
            x0, y0 = (int(round(v)) + pad - r for v in c)
            size = first.shape[1], first.shape[0]
            fits = []
            for angle in angles:
                rotation = cv2.getRotationMatrix2D((float(c[0] + pad), float(c[1] + pad)), float(angle), 1.0)
                piece = cv2.warpAffine(first, rotation, size)[y0:y0 + 2 * r + 1, x0:x0 + 2 * r + 1]
                if float(piece.sum()) < 1.0:
                    continue
                if near is None:
                    field, ox, oy = second, 0, 0
                else:
                    ox, oy = max(0, x0 + int(round(near[0])) - reach), max(0, y0 + int(round(near[1])) - reach)
                    field = second[oy:oy + 2 * r + 1 + 2 * reach, ox:ox + 2 * r + 1 + 2 * reach]
                if field.shape[0] < piece.shape[0] or field.shape[1] < piece.shape[1]:
                    continue
                scores = cv2.matchTemplate(field, piece, cv2.TM_CCORR_NORMED)
                scores[~np.isfinite(scores)] = 0
                for _ in range(1 if near is not None else 3):
                    _, score, _, (px, py) = cv2.minMaxLoc(scores)
                    fits.append((float(score), float(angle), (px + ox - x0, py + oy - y0)))
                    cv2.circle(scores, (px, py), 4, 0.0, -1)  # The next best place, away from this one.
            return sorted(fits, reverse=True)

        coarse, motions = [], []
        for fit in lay(eighth, 1.0, angles):
            if fit[0] < 0.2 or len(coarse) == 4:
                break
            if all(abs(fit[1] - other[1]) > 5.1 or np.abs(np.subtract(fit[2], other[2])).max() > 3 for other in coarse):
                coarse.append(fit)
        double = np.diag([2.0, 2.0, 1.0])
        for _, turn, shift in coarse:
            fine = lay(quarter, 2.0, turn + np.arange(-4.0, 4.1, 2.0), near=np.multiply(shift, 2), reach=4)
            if fine:
                _, angle, (dx, dy) = fine[0]
                motion = np.vstack([cv2.getRotationMatrix2D((float(2 * centre[0]), float(2 * centre[1])), angle, 1.0),
                                    (0, 0, 1)])
                motion[:2, 2] += (dx, dy)
                motions.append(double @ motion @ np.linalg.inv(double))
        return motions

    # One checkpoint --------------------------------------------------------------------------

    def update(self, frame, hidden=None):
        """Find the sheet in ``frame`` (``hidden``: 0/1 map of what hands cover). Returns the state."""
        picture, seen = self._landmarks(frame, hidden)
        # Is the scene itself at rest? While things move, a sheet that cannot be found is still on its way.
        still = False
        if self._picture is not None:
            shared = (seen & self._seen) != 0
            still = float(np.abs(picture - self._picture)[shared].sum()) < 0.2 * max(float(picture[shared].sum()), 1.0)
        self._picture, self._seen = picture, seen
        self.updates += 1
        corners = self._corners(self.key, self.known)
        before = self._scaled(self.pose)
        here = self._measure(self.key, self.known, corners, picture, seen, before, 12)
        here["explained"] = explained = self._explained(self.key, here["view"])
        visible = here["visible"] != 0
        weight = float(self.key[visible].sum())
        missing = float((self.key - explained)[visible].sum())
        found, gone = int(here["found"].sum()), int((~here["found"]).sum())
        self.report = {"patches": found, "gone": gone, "weight": weight, "missing": missing}
        if weight < 40 and found < 4:
            self._learn(here["view"], here["visible"], visible, explained)
            return self._settle("held")
        frame_side = (corners, picture, seen)
        asked = self._patch_motion(before, here)
        moved = None if asked is None else self._verify(asked, here, frame_side, far=False)
        departed = gone >= max(4, 0.3 * (found + gone)) or missing >= max(120.0, 0.25 * weight)
        if not still:
            self._failed = False
        if moved is None and departed and not self._failed:  # Not again on a scene that has not changed.
            # Much of the sheet is not where it was: look for what left among what is new, and of the places
            # it may have gone to, take the one where most patches are found again.
            left = np.where(visible, self.key - explained, 0).astype(np.float32)
            new = here["view"] - np.minimum(here["view"], cv2.dilate(self.key, np.ones((3, 3), np.uint8)))
            sample = corners[:: max(1, len(corners) // 30)]
            scarce = len(corners) < 16  # A blank sheet, or a line or two: its outline has to tell.
            # Turns of up to 20 degrees first; larger ones are rare.
            for angles in (np.arange(-20.0, 20.1, 5.0), (-45, -40, -35, -30, -25, 25, 30, 35, 40, 45)):
                most, candidate = (0.25 * missing if scarce else 5), None
                for motion in self._search(left, new, angles):
                    if scarce:
                        view, shown = self._view(picture, seen, before @ motion)
                        both = visible & (shown != 0)
                        score = float(np.maximum(self._explained(self.key, view) - explained, 0)[both].sum())
                    else:
                        trial = self._measure(self.key, self.known, sample, picture, seen, before @ motion, 12)
                        score = int(trial["found"].sum())
                    if score > most:
                        most, candidate = score, before @ motion
                if candidate is not None:
                    moved = self._verify(candidate, here, frame_side, far=True)
                    if moved is not None:
                        break
            self._failed = moved is None
        if moved is None:
            if departed and gone >= max(8, 2 * found):
                # Most patches are gone: the sheet is elsewhere (on its way, or where it cannot be found).
                self.lost = self.lost + 1 if still else 0
                if self.lost >= (1 if self.updates <= 4 else 4):  # At the key press a bad picture is likelier.
                    # Nothing moves and still no motion explains it: start again from what the frame shows.
                    # From here on the pose no longer goes back to the reference.
                    self.key = np.where(visible, here["view"], self.key).astype(np.float32)
                    self.known |= here["visible"] & self.domain
                    self.rekeyed += 1
                    self.tied = False
                    return self._settle("held")
                self.state = "lost"
                return self.state
            if departed or asked is not None:
                # Most of the sheet is where it should be. What is missing (a shadow, a sleeve, a hand at the
                # key press) is forgotten when it stays away while nothing moves.
                if still:
                    self._learn(here["view"], here["visible"], visible, explained, decay=departed)
                return self._settle("held")
            self._learn(here["view"], here["visible"], visible, explained, added=True, decay=True)
            return self._settle("steady")
        pose, there, sure = moved
        if sure:
            # What stayed where it was in the frame while the sheet moved is the desk: forget it.
            both = visible & (there["visible"] != 0)
            stayed = both & (self.key > 0.1) & (explained - there["explained"] > 0.5 * self.key)
            if stayed.any():
                self.key[cv2.dilate(stayed.astype(np.uint8), np.ones((5, 5), np.uint8)) != 0] = 0
        scale = np.diag([self.SCALE, self.SCALE, 1.0])
        self.pose = np.linalg.inv(scale) @ pose @ scale
        self.moves += 1
        if sure:
            shows = there["visible"] != 0
            self._learn(there["view"], there["visible"], shows, self._explained(self.key, there["view"]), added=True)
        return self._settle("moved")

    def _verify(self, candidate, here, frame_side, far):
        """Sharpen a candidate pose on the patches and test it against the picture. Returns (pose, what the
        frame shows there, whether most patches agree), or None when the motion does not explain the frame:
        it must make more line up than it breaks, and clearly more patches must be found right where it
        puts them than are still found where they were."""
        corners, picture, seen = frame_side
        for reach in ((12, 8, 5) if far else (5,)):
            there = self._measure(self.key, self.known, corners, picture, seen, candidate, reach)
            if there["found"].sum() >= 6:
                fitted, inliers = self._fit(there["points"][there["found"]], there["target"][there["found"]])
                # Within a wide reach a line of writing is also found on its neighbours: a third will do.
                if fitted is not None and inliers.sum() >= max(6, (0.3 if far else 0.5) * there["found"].sum()):
                    candidate = fitted
        there = self._measure(self.key, self.known, corners, picture, seen, candidate, 3)
        there["explained"] = self._explained(self.key, there["view"])
        both = (here["visible"] != 0) & (there["visible"] != 0)
        gain = float(np.maximum(there["explained"] - here["explained"], 0)[both].sum())
        loss = float(np.maximum(here["explained"] - there["explained"], 0)[both].sum())
        weight = float(self.key[both].sum())
        agree, asked = int(there["found"].sum()), len(there["found"])
        self.report.update(gain=gain, loss=loss, far=far, agree=agree, asked=asked)
        # Patches that are still found where they were: the desk around the sheet, or the sheet itself.
        stayed = int((here["found"] & (np.linalg.norm(here["shifts"], axis=1) < 1.5)).sum())
        accepted = gain + 0.02 * weight >= 1.2 * loss and agree >= max(6, 1.5 * stayed)
        if far:
            accepted = accepted and gain >= 40.0
        sure = agree >= max(8, 2 * stayed)
        if far and not accepted and asked < 16:
            # Too few patches to ask (a blank sheet): the picture alone, when at least half of what was
            # missing lines up again and next to nothing breaks.
            lacking = float((self.key - here["explained"])[both].sum())
            accepted = sure = gain >= max(60.0, 3 * loss, 0.5 * lacking)
        return (candidate, there, sure) if accepted else None

    def _settle(self, state):
        self.lost = 0
        self.state = state
        return state

    def _learn(self, view, shown, visible, explained, added=False, decay=False):
        """Keep the key picture current: what a hand uncovered, with ``added`` what was written since, and
        with ``decay`` less of what is no longer there, such as a shadow or a sleeve at the key press."""
        shown = shown & self.domain
        fresh = (shown != 0) & (self.known == 0)
        self.key[fresh] = view[fresh]
        self.known |= shown
        if added:
            more = visible & (view > self.key + 0.15)
            self.key[more] = view[more]
        if decay:
            faded = visible & (self.key > explained + 0.15)
            self.key[faded] = np.maximum(self.key[faded] - 0.34, view[faded])

    # For the session -------------------------------------------------------------------------

    @property
    def motion(self):
        """The pose as a 2x3 float32 matrix for cv2.warpAffine (reference -> frame)."""
        return self.pose[:2].astype(np.float32)


class SheetWatch:
    """The sheet on the desk, calibrated once and followed from then on, between and during recordings.

    Calibration: whenever no outline is held, the sheet is looked for in the frame: its sides are straight
    edges of bright paper, a hand may cover part of each, and at least three sides must be seen. The same
    outline twice in a row is taken. From then on a ``SheetTracker`` keyed on that frame follows the sheet (its writing
    and its outline), so the outline survives slides, turns, and hands on it; it is dropped and looked
    for again when the tracker has to start over (the sheet was taken away or replaced).

        watch = SheetWatch()                      # at camera start
        watch.update(frame, hand_sample, now)     # every frame; the work is done every ``interval`` seconds
        watch.outline(), watch.bounds(), watch.confidence, watch.pose

    ``confidence`` is 0 without an outline, else the share of the outline's sides that are seen as edges
    in the latest frame, lowered while the tracker holds or has lost the sheet.
    """

    def __init__(self, interval=0.24, search_interval=0.5):
        self.interval, self.search_interval = interval, search_interval
        self.tracker = None  # A SheetTracker keyed on the calibration frame.
        self.corners = None  # The outline at calibration: 4 x 2, clockwise from the top left.
        self.generation = 0  # Counts calibrations: a pose is only comparable within one.
        self.support = 0.0
        self.state = "none"
        self._pending = None
        self._weak = 0
        self._frame = None
        self._last = float("-inf")

    @property
    def calibrated(self):
        return self.tracker is not None

    @property
    def pose(self):
        """The sheet's motion since calibration (3x3, calibration frame -> latest frame)."""
        return np.eye(3) if self.tracker is None else self.tracker.pose

    @property
    def confidence(self):
        if self.tracker is None:
            return 0.0
        return float(self.support * {"steady": 1.0, "moved": 1.0, "held": 0.8}.get(self.state, 0.3))

    def outline(self):
        """The sheet's four corners in the latest frame (4 x 2), or None before calibration."""
        return None if self.tracker is None else move_points(self.tracker.pose, self.corners)

    def bounds(self):
        """The outline's box (x0, y0, x1, y1) within the frame, or None before calibration."""
        if self.tracker is None:
            return None
        outline, (width, height) = self.outline(), self.tracker.size
        return (int(max(0, outline[:, 0].min())), int(max(0, outline[:, 1].min())),
                int(min(width, np.ceil(outline[:, 0].max()))), int(min(height, np.ceil(outline[:, 1].max()))))

    def reset(self):
        """Forget the sheet (the camera was moved, or another desk)."""
        self.tracker = self.corners = self._pending = None
        self.support, self.state, self._weak = 0.0, "none", 0

    def _seen_support(self, tracker, outline):
        edges, judged, _ = self._edges(tracker)
        share, sides, _ = self._support(cv2.dilate(edges, np.ones((7, 7), np.uint8)), judged, outline)
        return share * min(1.0, sides / 3.0)

    def update(self, frame, hands=None, now=None):
        """Call for every frame. Cheap: the sheet is looked at every ``interval`` seconds only."""
        gap = self.interval if self.tracker is not None else self.search_interval
        if now is None or now - self._last >= gap:
            self.locate(frame, hands, now)
        return self.state

    def locate(self, frame, hands=None, now=None):
        """Find the sheet in this very frame now (a recording does so at its checkpoints). Returns the state."""
        if frame is self._frame:
            return self.state
        self._frame = frame
        if now is not None:
            self._last = now
        if self.tracker is not None and self.tracker.size != (frame.shape[1], frame.shape[0]):
            self.reset()
        if self.tracker is None:
            self._calibrate(frame, hands)
            return self.state
        self.state = self.tracker.update(frame)
        if not self.tracker.tied:
            self.reset()  # It had to start over: not the sheet that was calibrated, or not where it can be found.
            return self.state
        if self.state != "lost":
            support = self._seen_support(self.tracker, self.outline())
            self.support = support if self.state == "moved" else 0.5 * self.support + 0.5 * support
            # An outline that is no longer an edge anywhere, for seconds: not the sheet (any more).
            self._weak = self._weak + 1 if support < 0.3 else 0
            if self._weak >= 12:
                self.reset()
        return self.state

    def _calibrate(self, frame, hands):
        found = self._detect(frame, hands)
        if found is None:
            self._pending = None
            return
        corners, support = found
        steady = self._pending is not None and float(np.abs(corners - self._pending).max()) <= 10.0
        self._pending = corners
        if not steady:
            return
        x0, y0 = np.floor(corners.min(axis=0)).astype(int)
        x1, y1 = np.ceil(corners.max(axis=0)).astype(int)
        height, width = frame.shape[:2]
        area = (int(max(0, x0)), int(max(0, y0)), int(min(width, x1)), int(min(height, y1)))
        self.tracker = SheetTracker(frame, area, margin=0.12)
        self.corners, self.support, self.state = corners, support, "steady"
        self.generation += 1
        self._pending = None

    def _detect(self, frame, hands):
        """The sheet in a frame: (corners, support), or None. Its sides are found as straight edges of
        bright paper (a hand may cover part of each), and of the four-sided figures they form the one whose
        outline is best borne out by edges wins; one that holds the pen hand is preferred."""
        probe = SheetTracker(frame, None)
        edges, judged, paper = self._edges(probe)
        lines = self._lines(edges)
        near = cv2.dilate(edges, np.ones((7, 7), np.uint8))
        height, width = edges.shape
        tips = []
        for hand in (hands.get("hands", ()) if isinstance(hands, dict) else (hands or ())):
            marks = hand.get("landmarks") or ()
            if len(marks) > 8:
                tips.append(np.asarray(marks[8]["pixel"], np.float32))  # The index fingertip, next to the nib.
        found = []
        for first in range(len(lines)):
            for second in range(first + 1, len(lines)):
                a, b = lines[first], lines[second]
                if abs(a[1] @ b[1]) < 0.8 or abs((b[0] - a[0]) @ a[2]) < 50:
                    continue  # Two opposite sides run about the same way (the camera looks at an angle), well apart.
                for third in range(len(lines)):
                    for fourth in range(third + 1, len(lines)):
                        c, d = lines[third], lines[fourth]
                        if (len({first, second, third, fourth}) < 4 or abs(c[1] @ d[1]) < 0.8
                                or abs((d[0] - c[0]) @ c[2]) < 50 or abs(a[1] @ c[1]) > 0.65):
                            continue
                        corners = np.float32([self._cross(a, c), self._cross(a, d), self._cross(b, d), self._cross(b, c)])
                        if (corners[:, 0].min() < -0.1 * width or corners[:, 0].max() > 1.1 * width
                                or corners[:, 1].min() < -0.1 * height or corners[:, 1].max() > 1.1 * height):
                            continue
                        size = cv2.contourArea(corners)
                        if not 0.03 * width * height <= size <= 0.6 * width * height:
                            continue
                        corners = self._ordered(corners / probe.SCALE)
                        share, sides, least = self._support(near, judged, corners)
                        if sides < 3 or least < 0.5 or share < 0.7:
                            continue
                        held = any(cv2.pointPolygonTest(corners.astype(np.float32), (float(t[0]), float(t[1])), False) >= 0
                                   for t in tips)
                        # Writing inside tells the sheet from a bare stretch of desk between two things.
                        inside = np.zeros(edges.shape, np.uint8)
                        centre = corners.mean(axis=0)
                        cv2.fillConvexPoly(inside, np.rint((centre + 0.85 * (corners - centre)) * probe.SCALE).astype(np.int32), 1)
                        written = float(probe.key[inside != 0].mean()) if inside.any() else 0.0
                        score = share * (sides / 4.0) ** 0.5 * (1.15 if held else 1.0) * (1.0 + min(1.0, written / 0.02))
                        found.append((score, corners, share * min(1.0, sides / 3.0)))
        for _, corners, support in sorted(found, key=lambda item: -item[0])[:40]:
            # A sheet is paper (or a hand on it) inside, not the dark laptop between four edges, and a
            # surface of its own: brighter than the desk beyond its sides, or darker than it, all around.
            inside = np.zeros(paper.shape, np.uint8)
            cv2.fillConvexPoly(inside, np.rint(corners * probe.SCALE).astype(np.int32), 1)
            if float(paper[inside != 0].mean()) >= 0.7 and self._apart(probe, corners):
                return corners, support
        return None

    @staticmethod
    def _apart(tracker, corners, reach=10, scale=SheetTracker.SCALE):
        """True when the figure differs in brightness from the desk right outside it, the same way on
        every side that has desk beyond it, and there are two such sides. A bare stretch of desk between
        the sheet and something else does not, nor does a figure with a line of writing for a side."""
        _, usable, _, gray = tracker._raw
        height, width = gray.shape
        corners = np.asarray(corners, np.float32) * scale
        centre = corners.mean(axis=0)
        signs, within = [], []
        for start, end in zip(corners, np.roll(corners, -1, axis=0)):
            side = end - start
            normal = np.float32([-side[1], side[0]]) / max(float(np.linalg.norm(side)), 1e-6)
            if (centre - start) @ normal < 0:
                normal = -normal  # Pointing into the figure.
            points = start + side * np.linspace(0.15, 0.85, 30, dtype=np.float32)[:, None]
            values = []
            for offset in (reach, -reach):
                xs = np.rint(points[:, 0] + offset * normal[0]).astype(int)
                ys = np.rint(points[:, 1] + offset * normal[1]).astype(int)
                ok = (xs >= 0) & (xs < width) & (ys >= 0) & (ys < height)
                ok[ok] = usable[ys[ok], xs[ok]]
                values.append(float(np.median(gray[ys[ok], xs[ok]])) if ok.sum() >= 15 else None)
            if values[0] is not None:
                within.append(values[0])
            if None not in values:
                contrast = (values[0] - values[1]) / max(values[0], values[1], 1.0)
                if abs(contrast) < 0.02:
                    return False  # Paper or desk on both sides: a line of writing, a seam, not the sheet's edge.
                signs.append(contrast > 0)
        # One surface inside: as bright along one side as along the others (not sheet here and desk there).
        even = len(within) >= 2 and min(within) >= 0.88 * max(within)
        return even and len(signs) >= 2 and len(set(signs)) == 1

    @staticmethod
    def _edges(tracker):
        """(edges, judged, paper) of the tracker's latest frame at its scale: the steps in brightness that
        border bright paper (0/1), where a hand does not prevent judging them, and what is paper or hand."""
        step, usable, coloured, _ = tracker._raw
        hand = cv2.dilate(coloured.astype(np.uint8), np.ones((15, 15), np.uint8))
        beside = cv2.dilate(usable.astype(np.uint8), np.ones((9, 9), np.uint8))
        edges = ((step >= 0.2) & (beside != 0) & (hand == 0)).astype(np.uint8)
        return edges, (hand == 0).astype(np.uint8), (usable | coloured).astype(np.uint8)

    @staticmethod
    def _lines(edges, least=70.0):
        """Straight edges as (point, direction, normal, length covered), the longest first, at most ten.
        Pieces of one line on either side of a hand are joined."""
        thin = cv2.ximgproc.thinning(edges * 255) if hasattr(cv2, "ximgproc") else edges * 255
        segments = cv2.HoughLinesP(thin, 1, np.pi / 180, 30, minLineLength=40, maxLineGap=20)
        if segments is None:
            return []
        segments = segments.reshape(-1, 4).astype(np.float32)
        lengths = np.linalg.norm(segments[:, 2:] - segments[:, :2], axis=1)
        lines = []
        for index in np.argsort(-lengths):
            start, end = segments[index, :2], segments[index, 2:]
            direction = (end - start) / max(float(lengths[index]), 1e-6)
            for line in lines:
                if abs(direction @ line["direction"]) > 0.9976 and abs(((start + end) / 2 - line["point"]) @ line["normal"]) < 6:
                    line["spans"].append(sorted(((start - line["point"]) @ line["direction"],
                                                 (end - line["point"]) @ line["direction"])))
                    break
            else:
                lines.append({"point": start, "direction": direction, "normal": np.float32([-direction[1], direction[0]]),
                              "spans": [[0.0, float(lengths[index])]]})
        found = []
        for line in lines:
            covered, reached = 0.0, float("-inf")
            for low, high in sorted(line["spans"]):
                covered += max(0.0, high - max(low, reached))
                reached = max(reached, high)
            if covered >= least:
                found.append((line["point"], line["direction"], line["normal"], covered))
        return sorted(found, key=lambda line: -line[3])[:10]

    @staticmethod
    def _cross(first, second):
        """Where two lines (point, direction, ...) meet."""
        solved = np.linalg.solve(np.column_stack([first[1], -second[1]]), second[0] - first[0])
        return first[0] + solved[0] * first[1]

    @staticmethod
    def _ordered(corners):
        """Four corners clockwise in the image, starting with the one nearest the top left."""
        corners = np.asarray(corners, float).reshape(4, 2)
        centre = corners.mean(axis=0)
        corners = corners[np.argsort(np.arctan2(corners[:, 1] - centre[1], corners[:, 0] - centre[0]))]
        return np.roll(corners, -int(np.argmin(corners.sum(axis=1))), axis=0)

    @staticmethod
    def _support(near, judged, corners, scale=SheetTracker.SCALE):
        """How well an outline is borne out by edges: (share, sides, least). ``share`` is the mean, over
        the sides that can be judged (in the frame, not under a hand), of the part of the side that lies
        on an edge; ``sides`` how many could be judged; ``least`` the weakest of them."""
        height, width = near.shape
        corners = np.asarray(corners, np.float32) * scale
        along = np.linspace(0.08, 0.92, 40, dtype=np.float32)[None, :, None]
        points = corners[:, None, :] + (np.roll(corners, -1, axis=0) - corners)[:, None, :] * along
        xs, ys = np.rint(points[:, :, 0]).astype(int), np.rint(points[:, :, 1]).astype(int)
        inside = (xs >= 4) & (xs < width - 4) & (ys >= 4) & (ys < height - 4)
        xs, ys = np.clip(xs, 0, width - 1), np.clip(ys, 0, height - 1)
        can = inside & (judged[ys, xs] != 0)
        on = can & (near[ys, xs] != 0)
        counted = can.sum(axis=1)
        sides = counted >= 12
        if not sides.any():
            return 0.0, 0, 0.0
        shares = on.sum(axis=1)[sides] / counted[sides]
        return float(shares.mean()), int(sides.sum()), float(shares.min())
