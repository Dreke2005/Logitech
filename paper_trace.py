"""Pen strokes traced on the ink itself.

The pen-tip landmark comes from the hand and sits a few pixels off the nib, jitters, and
cannot tell a hover move from a stroke on its own. The ink that appeared during a recording
has none of these faults, but no order. ``ink_strokes`` combines the two: the ink's skeleton
gives the exact shape, split into pieces at crossings; each piece takes its time and direction
from the tip samples that passed over it; the pieces are then chained in writing order.
The result lies on the ink, covers every stroke, and has no hover lines.
"""

import cv2
import numpy as np

_NEIGHBOURS = np.ones((3, 3), np.float32)


_STEPS = ((1, 0), (-1, 0), (0, 1), (0, -1), (1, 1), (1, -1), (-1, 1), (-1, -1))  # Straight steps first.


def _ordered(points):
    """The pixels of one skeleton piece (an 8-connected simple path) from one end to the other."""
    if len(points) <= 2:
        return points
    x0, y0 = points.min(axis=0)
    local = points - (x0, y0)
    grid = np.zeros(tuple(local.max(axis=0)[::-1] + 1), np.uint8)
    grid[local[:, 1], local[:, 0]] = 1
    padded = np.pad(grid, 1)
    degree = cv2.filter2D(padded, -1, _NEIGHBOURS, borderType=cv2.BORDER_CONSTANT)[1:-1, 1:-1] - grid
    ends = np.flatnonzero(degree[local[:, 1], local[:, 0]] <= 1)
    x, y = local[ends[0] if len(ends) else 0]
    order = []
    height, width = grid.shape
    while True:
        order.append((x, y))
        grid[y, x] = 0
        for dx, dy in _STEPS:
            nx, ny = x + dx, y + dy
            if 0 <= nx < width and 0 <= ny < height and grid[ny, nx]:
                x, y = nx, ny
                break
        else:
            break
    if len(order) < len(points):
        # A loop or a stray pixel: the rest continues from the nearest remaining pixel.
        rest = np.c_[np.nonzero(grid)[::-1]]
        while len(rest):
            last = np.asarray(order[-1])
            nearest = int(np.argmin(np.abs(rest - last).sum(axis=1)))
            order.append(tuple(rest[nearest]))
            rest = np.delete(rest, nearest, axis=0)
    return np.asarray(order, np.int64) + (x0, y0)


_RING = ((-1, 0), (-1, 1), (0, 1), (1, 1), (1, 0), (1, -1), (0, -1), (-1, -1))  # (dy, dx) around a pixel


def crossings(skeleton):
    """0/1 map of skeleton pixels where three or more branches meet.

    The crossing number (0 -> 1 changes around the pixel) counts branches; a plain neighbour
    count would also flag the corners of every diagonal staircase and cut strokes apart."""
    padded = np.pad(skeleton, 1)
    height, width = skeleton.shape
    ring = [padded[1 + dy:1 + dy + height, 1 + dx:1 + dx + width] for dy, dx in _RING]
    changes = sum(((ring[index] == 0) & (ring[(index + 1) % 8] != 0)).astype(np.uint8) for index in range(8))
    return (skeleton != 0) & (changes >= 3)


def skeleton_pieces(mask, min_length=1, spur=0):
    """Pieces of the mask's skeleton between crossings and ends, each as an ordered (n, 2) x, y array.

    The crossing pixels come back as small pieces of their own, so the pieces cover the whole
    skeleton. A piece shorter than ``spur`` that hangs off a crossing with a free end is a
    thinning artefact of a ragged stroke edge (JPEG blocks, blur), not a stroke, and is left out."""
    skeleton = (cv2.ximgproc.thinning((mask != 0).astype(np.uint8) * 255) != 0).astype(np.uint8)
    junctions = crossings(skeleton)
    near_junction = cv2.dilate(junctions.astype(np.uint8), np.ones((3, 3), np.uint8)) != 0
    found = []
    for part, is_junction in (((skeleton != 0) & ~junctions, False), (junctions, True)):
        count, labels, stats, _ = cv2.connectedComponentsWithStats(part.astype(np.uint8), connectivity=8)
        for label in range(1, count):
            if stats[label, cv2.CC_STAT_AREA] < min_length:
                continue
            x, y, w, h = stats[label, :4]
            ys, xs = np.nonzero(labels[y:y + h, x:x + w] == label)
            piece = _ordered(np.c_[xs + x, ys + y])
            if not is_junction and len(piece) < spur:
                attached = near_junction[piece[[0, -1], 1], piece[[0, -1], 0]]
                if attached.sum() == 1:
                    continue
            found.append(piece)
    return found


def ink_strokes(mask, samples, join=3.5, min_length=1, reach=24.0, spur=0):
    """Strokes in writing order, traced on ``mask`` (0/1 ink in frame coordinates).

    ``samples`` are (time, (x, y)) pen-tip samples. Each skeleton piece takes the median time
    of the samples nearest to its pixels (within ``reach``), runs in the direction the tip
    moved, and joins the previous piece into one stroke when they touch (``join`` px). Pieces
    no sample came near are kept, ordered from left to right after the timed ones.
    """
    pieces = skeleton_pieces(mask, min_length, spur)
    if not pieces:
        return []
    timed = []
    if samples:
        times = np.asarray([time for time, _ in samples], float)
        points = np.asarray([point for _, point in samples], float)
        height, width = mask.shape
        seeds = np.ones((height, width), np.uint8)
        xs = np.clip(np.rint(points[:, 0]).astype(int), 0, width - 1)
        ys = np.clip(np.rint(points[:, 1]).astype(int), 0, height - 1)
        seeds[ys, xs] = 0
        index = np.full((height, width), -1, np.int64)
        index[ys, xs] = np.arange(len(points))
        distance, labels = cv2.distanceTransformWithLabels(seeds, cv2.DIST_L2, 3, labelType=cv2.DIST_LABEL_PIXEL)
        # Label of each seed pixel -> sample index (the last sample at a pixel wins, which is fine).
        lookup = np.full(labels.max() + 1, -1, np.int64)
        lookup[labels[ys, xs]] = index[ys, xs]
    for piece in pieces:
        when = None
        if samples:
            near = distance[piece[:, 1], piece[:, 0]] <= reach
            if near.any():
                nearest = times[lookup[labels[piece[:, 1], piece[:, 0]]]]
                when = float(np.median(nearest[near]))
                # Run in the direction the tip moved over it.
                head, tail = nearest[: max(1, len(piece) // 3)], nearest[-max(1, len(piece) // 3):]
                if np.median(head) > np.median(tail):
                    piece = piece[::-1]
        timed.append((when, piece))
    late = max((when for when, _ in timed if when is not None), default=0.0)
    timed.sort(key=lambda item: (item[0] if item[0] is not None else late + 1 + item[1][:, 0].min() * 1e-6))
    strokes = []
    for _, piece in timed:
        if strokes:
            end = strokes[-1][-1]
            if np.abs(piece[0] - end).max() <= join:
                strokes[-1] = np.vstack([strokes[-1], piece])
                continue
            if np.abs(piece[-1] - end).max() <= join:
                strokes[-1] = np.vstack([strokes[-1], piece[::-1]])
                continue
        strokes.append(piece)
    return [stroke.astype(np.int32) for stroke in strokes]
