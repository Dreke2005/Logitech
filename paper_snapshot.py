"""Where to read for a snapshot: the writing the pen points at, or the whole sheet of paper.

A snapshot (a key pressed twice) reads what is already on the page. A page often holds
several equations or blocks of code, so the region must be chosen:

1. The sheet is found as the large bright, low-texture region of the frame (``find_paper``),
   so the desk, keyboard, and hands around it are never read.
2. The ink on the sheet is grouped into lines, and lines close above each other into blocks
   (``writing_blocks``).
3. With the pen tip in view, the block the tip points at is read: the one under, beside, or
   just above the nib. Without a pen, a single block is read; with several, the whole sheet.
"""

from dataclasses import dataclass

import cv2
import numpy as np

from paper_ink import darkness, glyph_height, pen_marks


@dataclass(frozen=True)
class Paper:
    bounds: tuple  # (x0, y0, x1, y1) in frame pixels
    corners: np.ndarray  # 4 x 2 polygon of the sheet, or the bounds' corners
    found: bool  # False when no sheet was detected and the whole frame stands in for it


def sheet_at(frame, point, work=640):
    """The sheet under a point (the pen's nib): flooded from around the point, up to the sheet's border.

    A desk can hold several sheets, and a white desk is as bright as paper; the sheet being written on
    is the one the nib touches. The flood starts on paper next to the nib (not on the pen or the
    hand), spreads over gradual light changes, and stops at the faint edge around the sheet. Returns a
    Paper, or None when no sheet-shaped region surrounds the point."""
    height, width = frame.shape[:2]
    scale = min(1.0, work / max(height, width))
    small = cv2.resize(frame, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA) if scale < 1 else frame
    gray = cv2.medianBlur(cv2.cvtColor(small, cv2.COLOR_BGR2GRAY), 7)
    rows, cols = gray.shape
    edges = cv2.dilate(cv2.Canny(gray, 8, 20), np.ones((3, 3), np.uint8))
    hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)
    blocked = (edges != 0) | (hsv[:, :, 1] > 60)  # Edges, the pen, the hand.
    cx, cy = point[0] * scale, point[1] * scale
    best = None
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (max(5, round(0.025 * cols)) | 1,) * 2)
    for radius in (6, 12, 20, 30):
        for angle in np.linspace(0, 2 * np.pi, 12, endpoint=False):
            x, y = int(cx + radius * np.cos(angle)), int(cy + radius * np.sin(angle))
            if not (0 <= x < cols and 0 <= y < rows) or blocked[y, x]:
                continue
            if best is not None and best[1][y, x]:
                continue
            mask = np.zeros((rows + 2, cols + 2), np.uint8)
            mask[1:-1, 1:-1][edges != 0] = 1
            cv2.floodFill(gray, mask, (x, y), 255, 2, 2, 4 | cv2.FLOODFILL_MASK_ONLY | (255 << 8))
            region = cv2.morphologyEx((mask[1:-1, 1:-1] == 255).astype(np.uint8), cv2.MORPH_CLOSE, kernel)
            area = int(region.sum())
            if 0.03 * rows * cols <= area <= 0.8 * rows * cols and (best is None or area > best[0]):
                best = area, region
    if best is None:
        return None
    region = best[1]
    contour = max(cv2.findContours(region, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0], key=cv2.contourArea)
    hull = cv2.convexHull(contour)
    if best[0] / max(cv2.contourArea(hull), 1.0) < 0.7:
        return None  # Leaked into the desk around other objects.
    polygon = cv2.approxPolyDP(hull, 0.02 * cv2.arcLength(hull, True), True).reshape(-1, 2)
    if len(polygon) != 4:
        polygon = cv2.boxPoints(cv2.minAreaRect(hull))
    corners = np.clip(polygon.astype(np.float32) / scale, 0, (width - 1, height - 1))
    x, y, w, h = cv2.boundingRect(hull)
    bounds = (int(x / scale), int(y / scale), min(width, int(np.ceil((x + w) / scale))),
              min(height, int(np.ceil((y + h) / scale))))
    return Paper(bounds, corners, True)


def find_paper(frame, work=640, pen=None):
    """The sheet being written on: the one under the pen's nib when the pen is seen (sheet_at),
    else the best sheet-shaped bright region (see below)."""
    if pen is not None:
        sheet = sheet_at(frame, pen, work)
        if sheet is not None:
            return sheet
    return _find_paper(frame, work)


def _find_paper(frame, work=640):
    """The sheet of paper in the frame.

    A sheet is the brightest large region whose outline is a filled four-sided shape. On a
    dark desk any threshold finds it; on a light grey desk only a threshold between the desk
    and the white paper does, so several thresholds over the bright part of the histogram
    are tried, and the region that best fills its rotated rectangle wins. Writing and a pen
    on the sheet are closed over; a hand or keypad touching it is opened away.
    """
    height, width = frame.shape[:2]
    scale = min(1.0, work / max(height, width))
    small = cv2.resize(frame, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA) if scale < 1 else frame
    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY) if small.ndim == 3 else small
    colour = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)[:, :, 1] > 60 if small.ndim == 3 else None
    rows, cols = gray.shape
    size = max(5, round(0.025 * max(rows, cols))) | 1
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (size, size))
    whole = Paper((0, 0, width, height), np.float32([[0, 0], [width, 0], [width, height], [0, height]]), False)
    # Handwriting tells the sheet apart from a desk of the same colour: thin dark marks on it.
    dark, level = darkness(gray, max(7, round(0.02 * cols)))
    writing = pen_marks(dark, level, cols)
    light = np.percentile(cv2.medianBlur(gray, 7), 60)
    best = None
    for region in _sheet_candidates(gray, colour, kernel):
        area = int(region.sum())
        if not 0.04 * rows * cols <= area <= 0.5 * rows * cols:  # A sheet never fills half the view.
            continue
        if float(np.median(gray[region != 0])) < light:
            continue  # Paper is among the brightest things on a desk; a keyboard or a laptop is not.
        contours, _ = cv2.findContours(region, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contour = max(contours, key=cv2.contourArea)
        hull = cv2.convexHull(contour)
        # Seen at an angle a sheet is a trapezoid, so it is measured against its hull, not a rectangle.
        fill = area / max(cv2.contourArea(hull), 1.0)
        corners = len(cv2.approxPolyDP(hull, 0.04 * cv2.arcLength(hull, True), True))
        if fill < 0.85 or not 4 <= corners <= 7:
            continue  # Not a sheet: the desk around objects, or light falling across the scene.
        inside = cv2.erode(region, kernel)
        marks = int((writing & inside).sum())
        score = (marks + 1) * np.sqrt(area) * fill
        if best is None or score > best[0]:
            best = score, region, contour, cv2.minAreaRect(contour)
    if best is None:
        return whole
    _, region, contour, box = best
    hull = cv2.convexHull(contour)
    polygon = cv2.approxPolyDP(hull, 0.02 * cv2.arcLength(hull, True), True).reshape(-1, 2)
    if len(polygon) != 4:
        polygon = cv2.boxPoints(box)
    corners = np.clip(polygon.astype(np.float32) / scale, 0, (width - 1, height - 1))
    x, y, w, h = cv2.boundingRect(region)
    bounds = (int(x / scale), int(y / scale), min(width, int(np.ceil((x + w) / scale))),
              min(height, int(np.ceil((y + h) / scale))))
    return Paper(bounds, corners, True)


def _sheet_candidates(gray, colour, kernel):
    """Regions that may be the sheet (0/1 masks at the working scale).

    Two kinds: bright regions at several thresholds (a white sheet on a darker desk), and regions
    flooded from seeds that stop at faint edges (a sheet on a desk of about the same brightness,
    where only its border shows). Ink and the paper texture are removed first with a median blur."""
    rows, cols = gray.shape
    smooth = cv2.medianBlur(gray, 7)
    for level in sorted({int(v) for v in np.linspace(*np.percentile(smooth, (35, 97)), 10)}, reverse=True):
        bright = (smooth > level).astype(np.uint8)
        if colour is not None:
            bright[colour] = 0  # Paper is white or grey, not skin, wood, or a coloured mat.
        bright = cv2.morphologyEx(cv2.morphologyEx(bright, cv2.MORPH_CLOSE, kernel), cv2.MORPH_OPEN, kernel)
        count, labels = cv2.connectedComponents(bright, connectivity=4)
        for label in range(1, count):
            yield (labels == label).astype(np.uint8)
    edges = cv2.dilate(cv2.Canny(smooth, 8, 20), np.ones((3, 3), np.uint8))
    visited = edges.astype(bool).copy()
    if colour is not None:
        visited |= colour
    step = max(8, cols // 24)
    floor = np.percentile(smooth, 40)
    for y in range(step // 2, rows, step):
        for x in range(step // 2, cols, step):
            if visited[y, x] or smooth[y, x] < floor:
                continue
            mask = np.zeros((rows + 2, cols + 2), np.uint8)
            mask[1:-1, 1:-1][edges != 0] = 1
            cv2.floodFill(smooth, mask, (x, y), 255, 2, 2, 4 | cv2.FLOODFILL_MASK_ONLY | (255 << 8))
            region = (mask[1:-1, 1:-1] == 255).astype(np.uint8)
            visited |= region.astype(bool)
            if region.sum() >= 0.03 * rows * cols:
                yield cv2.morphologyEx(region, cv2.MORPH_CLOSE, kernel)


@dataclass(frozen=True)
class Block:
    bounds: tuple  # (x0, y0, x1, y1) in frame pixels
    lines: int
    text_height: float


def writing_blocks(frame, paper, hidden=None):
    """Blocks of writing on the sheet, in reading order. ``hidden`` masks the hand and pen (frame-sized 0/1)."""
    x0, y0, x1, y1 = paper.bounds
    crop = frame[y0:y1, x0:x1]
    if crop.size == 0:
        return []
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY) if crop.ndim == 3 else crop
    height, width = gray.shape
    dark, level = darkness(gray)
    marks = pen_marks(dark, level, width)
    # Only the sheet: outside its outline (a tilted sheet's corners) and its own edge are not writing.
    outline = np.zeros(gray.shape, np.uint8)
    cv2.fillConvexPoly(outline, np.rint(paper.corners - (x0, y0)).astype(np.int32), 1)
    edge = max(4, round(0.012 * max(height, width)))
    outline = cv2.erode(outline, np.ones((2 * edge + 1, 2 * edge + 1), np.uint8))
    marks &= outline
    if hidden is not None:
        marks &= (hidden[y0:y1, x0:x1] == 0).astype(np.uint8)
    if int(marks.sum()) < 25:
        return []
    text = max(6.0, glyph_height(marks))
    # Lines: marks joined along the writing direction across word gaps, not across lines.
    across = max(3, int(round(1.6 * text)))
    lines = cv2.dilate(marks, cv2.getStructuringElement(cv2.MORPH_RECT, (2 * across + 1, max(1, int(0.35 * text)))))
    count, labels, stats, _ = cv2.connectedComponentsWithStats(lines, connectivity=8)
    boxes = []
    mass = np.bincount(labels[marks != 0], minlength=count)
    for label in range(1, count):
        if mass[label] < 12:
            continue  # Specks and paper texture.
        bx, by, bw, bh = stats[label, :4]
        inner = marks[by:by + bh, bx:bx + bw] & (labels[by:by + bh, bx:bx + bw] == label)
        ys, xs = np.nonzero(inner)
        boxes.append([bx + xs.min(), by + ys.min(), bx + xs.max() + 1, by + ys.max() + 1])
    if not boxes:
        return []
    # Blocks: lines directly under each other (code, a derivation) belong together.
    boxes.sort(key=lambda box: (box[1], box[0]))
    groups = []
    for box in boxes:
        for group in groups:
            gx0, gy0, gx1, gy1 = group["box"]
            overlap = min(gx1, box[2]) - max(gx0, box[0])
            if overlap > -text and 0 <= box[1] - gy1 <= 0.7 * text or (box[1] < gy1 and overlap > 0):
                group["box"] = [min(gx0, box[0]), min(gy0, box[1]), max(gx1, box[2]), max(gy1, box[3])]
                group["lines"] += 1
                break
        else:
            groups.append({"box": list(box), "lines": 1})
    return [Block((int(g["box"][0] + x0), int(g["box"][1] + y0), int(g["box"][2] + x0), int(g["box"][3] + y0)),
                  g["lines"], text) for g in groups]


def pointed_block(blocks, tip):
    """The block the pen tip points at, or None when the tip is not close to any.

    The nib rests on, beside, or just below the writing it points at, so the distance is
    measured to each block's box, with the space below a block counting a little less."""
    best, chosen = None, None
    for block in blocks:
        x0, y0, x1, y1 = block.bounds
        if max(x1 - x0, y1 - y0) < 2 * block.text_height:
            continue  # A speck (the point's shadow, a dot), not a piece of writing to read.
        dx = max(x0 - tip[0], 0, tip[0] - x1)
        dy = max(y0 - tip[1], 0, tip[1] - y1)
        if tip[1] > y1:
            dy *= 0.7  # Pointing from just below.
        distance = float(np.hypot(dx, dy))
        if distance <= 2.5 * block.text_height and (best is None or distance < best):
            best, chosen = distance, block
    return chosen


def snapshot_region(frame, tip=None, hidden=None, ahead=None):
    """(bounds, reason) to read for a snapshot.

    ``ahead`` is the pen's direction (back to nib): a pen points at the writing in front of its nib,
    so the block about two letters ahead of the nib is chosen over one merely beside it.
    reason is "pointed" (the block at the pen tip), "single" (the only block), "paper"
    (several blocks and no pen: the whole sheet), or "frame" (no sheet or writing found)."""
    paper = find_paper(frame, pen=tip)
    blocks = writing_blocks(frame, paper, hidden)
    if not blocks:
        return paper.bounds, "paper" if paper.found else "frame"
    if tip is not None:
        if ahead is not None:
            # A pen points at the writing in front of its point.
            direction = np.asarray(ahead, float) / max(float(np.linalg.norm(ahead)), 1e-6)
            tip = tuple(np.asarray(tip, float) + direction * 1.5 * blocks[0].text_height)
        block = pointed_block(blocks, tip)
        if block is not None:
            return _padded(whole_line(block, blocks), frame.shape), "pointed"
    if len(blocks) == 1:
        return _padded(blocks[0], frame.shape), "single"
    return paper.bounds, "paper" if paper.found else "frame"


def whole_line(block, blocks, gap=5.0):
    """The pointed block joined with the rest of its text line: an equation is one line.

    The pen itself lies across the line it points at and cuts it into pieces (its barrel and the
    hand are left out of the writing), so the piece next to the point is only part of the equation.
    Pieces that overlap the line vertically by half a letter and start within ``gap`` letter heights
    of it are the same line; they are joined until nothing more is added."""
    x0, y0, x1, y1 = block.bounds
    lines, height = block.lines, block.text_height
    rest = [other for other in blocks if other is not block]
    added = True
    while added:
        added = False
        for other in list(rest):
            ox0, oy0, ox1, oy1 = other.bounds
            overlap = min(y1, oy1) - max(y0, oy0)
            apart = max(ox0 - x1, x0 - ox1, 0)
            if overlap >= 0.5 * min(oy1 - oy0, y1 - y0, 2 * height) and apart <= gap * height:
                x0, y0, x1, y1 = min(x0, ox0), min(y0, oy0), max(x1, ox1), max(y1, oy1)
                rest.remove(other)
                added = True
    return Block((x0, y0, x1, y1), lines, height)


def _padded(block, shape):
    height, width = shape[:2]
    pad = int(round(max(6, 0.6 * block.text_height)))
    x0, y0, x1, y1 = block.bounds
    return max(0, x0 - pad), max(0, y0 - pad), min(width, x1 + pad), min(height, y1 + pad)
