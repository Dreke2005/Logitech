"""Clean ink pictures: the handwriting alone, dark on white, ready for the reader.

``extract_ink`` works on a single photo (the snapshot path): it measures how much darker
each pixel is than the paper around it, keeps thin pen-like marks, finds the block of
writing, and returns it without paper texture, shadows, or background. The live path
(paper_writing) produces the same kind of picture from ink registered while writing, so
both paths feed one reader. ``normalize`` brings either picture to the size the reader
works best with.
"""

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass(frozen=True)
class InkPicture:
    image: np.ndarray  # BGR, dark ink on white
    bounds: tuple  # (x0, y0, x1, y1) in the source photo
    text_height: float  # Typical glyph height in source pixels
    lines: int  # Estimated number of text lines


def darkness(gray, stroke=None):
    """How much darker each pixel is than the paper around it (0..1), and the paper brightness.

    A closing removes pen strokes from the brightness estimate, so lighting gradients and
    exposure cancel while thin dark marks keep their full contrast.
    """
    height, width = gray.shape
    size = int(stroke or max(9, round(width * 0.012))) | 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size))
    paper = cv2.GaussianBlur(cv2.morphologyEx(gray, cv2.MORPH_CLOSE, kernel), (0, 0), size / 3)
    paper = np.maximum(paper.astype(np.float32), 1.0)
    return np.clip(1.0 - gray.astype(np.float32) / paper, 0.0, 1.0), paper


def pen_marks(dark, paper, width):
    """0/1 mask of thin, pen-like marks on paper: dark enough, on a bright surface, and not a blob."""
    strong = float(np.percentile(dark, 99.7))
    threshold = max(0.07, min(0.22, 0.42 * strong))
    marks = (dark >= threshold) & (paper >= 0.55 * np.percentile(paper, 90))
    marks = marks.astype(np.uint8)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(marks, connectivity=8)
    if count <= 1:
        return marks
    core = cv2.distanceTransform(marks, cv2.DIST_L2, 3) > max(3.5, width * 0.0045)
    area = stats[:, cv2.CC_STAT_AREA]
    thick = np.bincount(labels[core], minlength=count)
    longest = np.maximum(stats[:, cv2.CC_STAT_WIDTH], stats[:, cv2.CC_STAT_HEIGHT])
    keep = (area >= 3) & (thick <= np.maximum(6, 0.08 * area)) & (longest <= 0.6 * width)
    keep[0] = False
    return keep[labels].astype(np.uint8)


def render(dark, marks, gain=2.6):
    """Dark-on-white picture of the marks, keeping one pixel of soft edge around each stroke."""
    near = cv2.dilate(marks, np.ones((3, 3), np.uint8)) != 0
    level = np.clip(dark * (255 * gain), 0, 255).astype(np.uint8)
    return 255 - np.where(near, level, 0).astype(np.uint8)


def glyph_height(marks):
    """Typical glyph height: the median height of the taller connected marks."""
    count, _, stats, _ = cv2.connectedComponentsWithStats(marks, connectivity=8)
    heights = stats[1:, cv2.CC_STAT_HEIGHT].astype(float)
    heights = heights[stats[1:, cv2.CC_STAT_AREA] >= 6]
    if not heights.size:
        return 0.0
    tall = heights[heights >= 0.5 * np.percentile(heights, 90)]
    return float(np.median(tall))


def _lines(marks):
    """Number of text lines: runs of rows that contain ink, separated by clear gaps."""
    rows = marks.sum(axis=1) > 0
    runs, inside = 0, False
    for value in rows:
        if value and not inside:
            runs += 1
        inside = bool(value)
    return max(1, runs)


def extract_ink(image, roi=None):
    """The block of handwriting in a photo as an InkPicture, or None when no writing is found.

    All blocks of pen marks that hold a real share of the ink are kept, so a page of writing
    gives its lines without the desk, keyboard, shadows, or page edges around them.
    """
    x0, y0 = (roi[0], roi[1]) if roi else (0, 0)
    crop = image[roi[1]:roi[3], roi[0]:roi[2]] if roi else image
    if crop.size == 0:
        return None
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY) if crop.ndim == 3 else crop
    height, width = gray.shape
    dark, paper = darkness(gray)
    marks = pen_marks(dark, paper, width)
    if int(marks.sum()) < 25:
        return None
    text = max(6.0, glyph_height(marks))
    # Marks within about two glyph heights of each other belong to the same block of writing.
    reach = int(round(2.0 * text)) | 1
    blocks = cv2.dilate(marks, cv2.getStructuringElement(cv2.MORPH_RECT, (2 * reach + 1, reach)))
    count, labels, stats, _ = cv2.connectedComponentsWithStats(blocks, connectivity=8)
    mass = np.bincount(labels[marks != 0], minlength=count).astype(float)
    mass[0] = 0
    # Every block that holds a real share of the ink: a page of notes keeps all its lines, while
    # specks and stray marks far from the writing are left out.
    chosen = np.isin(labels, np.flatnonzero(mass >= 0.08 * mass.max()))
    marks = (marks & chosen).astype(np.uint8)
    ys, xs = np.nonzero(marks)
    if len(xs) < 25:
        return None
    margin = int(round(max(4, 0.5 * text)))
    left, top = max(0, xs.min() - margin), max(0, ys.min() - margin)
    right, bottom = min(width, xs.max() + 1 + margin), min(height, ys.max() + 1 + margin)
    picture = render(dark[top:bottom, left:right], marks[top:bottom, left:right])
    return InkPicture(cv2.cvtColor(picture, cv2.COLOR_GRAY2BGR), (x0 + left, y0 + top, x0 + right, y0 + bottom),
                      glyph_height(marks) or text, _lines(marks[top:bottom, left:right]))


def looks_drawn(picture):
    """True when the ink holds a long straight stroke (a branch line, a box side, an arrow shaft).

    Handwritten words never run straight for several letter heights, so this tells a drawing
    from plain text without asking a model. The letter height is taken from the small marks in
    the picture itself: in a drawing the connected lines would make any overall estimate too large.
    """
    gray = picture if picture.ndim == 2 else picture[:, :, 0]
    mask = (gray < 170).astype(np.uint8)
    if int(mask.sum()) < 20:
        return False
    count, _, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    heights = stats[1:, cv2.CC_STAT_HEIGHT][stats[1:, cv2.CC_STAT_AREA] >= 8]
    if not heights.size:
        return False
    letter = float(np.clip(np.median(heights), 6, 0.25 * max(gray.shape)))
    reach = max(40.0, 4.5 * letter)
    # Pieces of the stroke skeleton between crossings: a letter's pieces are short or curved; a drawn
    # line gives one long piece whose extent is nearly its length and which stays close to its axis.
    skeleton = (cv2.ximgproc.thinning(mask * 255) != 0).astype(np.uint8)
    neighbours = cv2.filter2D(skeleton, -1, np.ones((3, 3), np.float32), borderType=cv2.BORDER_CONSTANT) - skeleton
    pieces = skeleton & (neighbours <= 2)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(pieces.astype(np.uint8), connectivity=8)
    for label in np.flatnonzero(np.maximum(stats[:, cv2.CC_STAT_WIDTH], stats[:, cv2.CC_STAT_HEIGHT]) >= reach):
        if label == 0:
            continue
        ys, xs = np.nonzero(labels == label)
        points = np.c_[xs, ys].astype(np.float32)
        points -= points.mean(axis=0)
        _, _, axes = np.linalg.svd(points, full_matrices=False)
        along, across = points @ axes[0], points @ axes[1]
        extent = float(along.max() - along.min())
        if extent >= reach and extent >= 0.85 * len(points) and float(np.abs(across).max()) <= 0.6 * letter:
            return True
    return False


def straight_stroke(strokes, text_height):
    """True when a pen-down stroke runs straight for several letter heights: a line was drawn.

    The trace backs up ``looks_drawn`` for lines the ink record left out (a branch line longer
    than the ink filter accepts, or one still under the hand)."""
    reach = max(40.0, 4.5 * max(text_height, 6.0))
    for stroke in strokes:
        points = np.asarray(stroke, float).reshape(-1, 2)
        if len(points) < 4:
            continue
        length = float(np.linalg.norm(np.diff(points, axis=0), axis=1).sum())
        chord = float(np.linalg.norm(points[-1] - points[0]))
        if chord >= reach and chord >= 0.85 * length:
            return True
    return False


def normalize(picture, text_height, glyph=64, longest=1280, rotation=0):
    """Upright picture scaled so glyphs are about ``glyph`` pixels tall, with a clean white border."""
    codes = {90: cv2.ROTATE_90_CLOCKWISE, 180: cv2.ROTATE_180, 270: cv2.ROTATE_90_COUNTERCLOCKWISE}
    if rotation % 360:
        picture = cv2.rotate(picture, codes[rotation % 360])
    factor = float(np.clip(glyph / max(text_height, 1.0), 0.25, 6.0))
    height, width = picture.shape[:2]
    factor = min(factor, longest / max(height, width))
    if abs(factor - 1) > 0.05:
        interpolation = cv2.INTER_CUBIC if factor > 1 else cv2.INTER_AREA
        picture = cv2.resize(picture, None, fx=factor, fy=factor, interpolation=interpolation)
    gray = picture if picture.ndim == 2 else cv2.cvtColor(picture, cv2.COLOR_BGR2GRAY)
    if int(gray.min()) < 245:
        # The SDK's contrast stretch: the faintest ballpoint becomes as dark as a firm stroke, the paper stays white.
        from example_IR import contrast_stretch

        gray = contrast_stretch(gray, 0.5, 99.5)
    if factor > 1.5:
        # Upscaled camera ink is soft: restore a firm stroke edge.
        gray = np.clip((gray.astype(np.float32) - 60) * (255 / 150), 0, 255).astype(np.uint8)
    picture = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    pad = glyph // 3
    return cv2.copyMakeBorder(picture, pad, pad, pad, pad, cv2.BORDER_CONSTANT, value=(255, 255, 255))
