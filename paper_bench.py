"""Benchmark the handwriting reader on synthetic camera pictures with known text.

    uv run paper_bench.py                  # all readers, 3 fonts
    uv run paper_bench.py --reader fast    # one reader
    uv run paper_bench.py --save folder    # also save the pictures the reader sees

Each sample is written in a handwriting font, placed on a page as a camera would see it
(small glyphs, grey ballpoint ink, uneven light, noise, blur), passed through the same
ink extraction as the snapshot path, and read. Reported per reader: exact matches,
character accuracy, and seconds per reading.
"""

import argparse
import difflib
import re
import time
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

FONTS = {"inkfree": "Inkfree.ttf", "segoeprint": "segoepr.ttf", "bradley": "BRADHITC.TTF", "mvboli": "mvboli.ttf",
         "comic": "comic.ttf"}
# (kind, text). Kinds follow the keypad keys: what the reader is told to expect.
SAMPLES = [
    ("equation", "y = x^3 + 3x + 2"), ("equation", "f(x) = 2x^2 - 5x + 7"), ("equation", "E = m c^2"),
    ("equation", "a^2 + b^2 = c^2"), ("equation", "v = u + a t"), ("equation", "y = sin(x) + 2cos(x)"),
    ("plot", "y = x^2 - 4"), ("plot", "y = 3sin(2x)"),
    ("code", "for i in range(10):"), ("code", "total = total + x[i]"), ("code", "if n > 0: return n"),
    ("code", "def area(w, h):"), ("code", "A = [1 2; 3 4]"),
    ("text", "add login page"), ("text", "fix crop on light desk"), ("text", "Literature review 2w"),
    ("table", "name | mass | charge"), ("table", "R1 | 1k | in out"),
]
# Several lines, as code and plans are usually written. Only in --hard runs.
HARD_SAMPLES = [("code", "for i in range(3):\n    print(i)"), ("code", "def f(x):\n    return 2*x"),
                ("text", "fix login\nadd tests")]


def canonical(text):
    """Comparable form: no spacing, LaTeX niceties, or case differences that do not change the meaning."""
    text = text.strip().strip("$").replace("\\left", "").replace("\\right", "").replace("\\,", "")
    text = re.sub(r"\\(?:cdot|times)", "*", text)
    text = re.sub(r"\\(sin|cos|tan|log|ln|exp)", r"\1", text)
    text = re.sub(r"\^\{([^{}]*)\}", r"^\1", text)
    text = re.sub(r"_\{([^{}]*)\}", r"_\1", text)
    text = text.replace("**", "^").replace("×", "*").replace("−", "-").replace("·", "*")
    return re.sub(r"\s+", "", text).lower()


def accuracy(expected, got):
    return difflib.SequenceMatcher(None, canonical(expected), canonical(got)).ratio()


def _glyph(char, font, rng):
    """One character as a dark-on-white patch, distorted like a hand-written letter when ``rng`` is given."""
    box = font.getbbox(char)
    size = font.size
    patch = Image.new("L", (int(box[2] + size), int(size * 2.2)), 255)
    ImageDraw.Draw(patch).text((size // 2, size // 2), char, font=font, fill=0)
    if rng is None:
        return patch
    patch = patch.rotate(float(rng.uniform(-7, 7)), resample=Image.BICUBIC, fillcolor=255)
    factor = float(rng.uniform(0.86, 1.14))
    return patch.resize((max(1, int(patch.width * factor)), max(1, int(patch.height * factor * rng.uniform(0.92, 1.08)))),
                        Image.BICUBIC)


def write(text, font_path, size=120, rng=None):
    """The text as dark strokes on white, with ^ raising what follows (one token) like handwritten powers.

    Several lines are written below each other. With ``rng`` every letter gets its own size,
    tilt, baseline, and spacing, and the line a slant, as in real handwriting."""
    font = ImageFont.truetype(str(font_path), size)
    small = ImageFont.truetype(str(font_path), int(size * 0.62))
    lines = text.split("\n")
    longest = max(len(line) for line in lines)
    image = Image.new("L", (size * (longest + 6), int(size * (1.6 * len(lines) + 2))), 255)
    for row, line in enumerate(lines):
        x, raised = size + size * 0.55 * (len(line) - len(line.lstrip())), False
        baseline = size * (1 + 1.6 * row)
        for char in line.lstrip():
            if char == "^":
                raised = True
                continue
            if raised and not (char.isalnum()):
                raised = False
            use = small if raised else font
            y = baseline - size * 0.45 if raised else baseline
            if rng is not None:
                y += size * float(rng.uniform(-0.08, 0.08))
            if char != " ":
                patch = _glyph(char, use, rng)
                image.paste(0, (int(x - use.size // 2), int(y - use.size // 2)), Image.eval(patch, lambda v: 255 - v))
            advance = ImageDraw.Draw(image).textlength(char, font=use) + size * 0.03
            x += advance * (float(rng.uniform(0.9, 1.15)) if rng is not None else 1.0)
    array = np.asarray(image)
    if rng is not None:
        shear = float(rng.uniform(-0.18, 0.12))  # slanted handwriting
        height, width = array.shape
        matrix = np.float32([[1, shear, -shear * height / 2], [0, 1, 0]])
        array = cv2.warpAffine(array, matrix, (width, height), borderValue=255)
    ys, xs = np.nonzero(array < 128)
    return array[max(0, ys.min() - 4):ys.max() + 5, max(0, xs.min() - 4):xs.max() + 5]


def on_page(glyphs, text_height=20, seed=0, page=(720, 1280), hard=False, lines=1, empty=False, truth=None):
    """A camera frame of a desk page with the writing at ``text_height`` px: grey ink, uneven light, noise, blur.

    ``hard`` adds what a real desk camera adds: a hand shadow across the page, a slightly
    tilted view, lighter and less even ink, motion blur, and JPEG compression. ``empty`` gives
    the same page, with the same distortions, before anything was written on it."""
    rng = np.random.default_rng(seed)
    # Ascender-to-descender box vs glyph body; a block of several lines has one such box per line.
    scale = text_height / max(1, glyphs.shape[0] / lines) * (1.6 if lines == 1 else 2.56)
    small = cv2.resize(glyphs, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    height, width = page
    yy, xx = np.mgrid[0:height, 0:width].astype(np.float32)
    paper = 205 + 22 * (xx / width - 0.5) - 14 * (yy / height - 0.5)  # light falls off across the page
    frame = paper.copy()
    top, left = 300, 260
    h, w = small.shape
    w = min(w, width - left - 10)
    h = min(h, height - top - 10)
    ink = (255 - small[:h, :w].astype(np.float32)) / 255.0
    strength = 105
    if hard:
        # Ballpoint pressure varies along the line: some letters are much fainter.
        pressure = cv2.resize(rng.uniform(0.55, 1.0, (1, max(2, w // 25))).astype(np.float32), (w, h),
                              interpolation=cv2.INTER_LINEAR)
        ink *= pressure
        strength = 85
    if empty:
        strength = 0
    frame[top:top + h, left:left + w] -= ink * strength
    exact = np.zeros((height, width), np.uint8)  # Where the pen really left ink, for scoring a trace.
    exact[top:top + h, left:left + w] = small[:h, :w] < 128
    frame = np.clip(frame, 0, 255).astype(np.uint8)
    cv2.rectangle(frame, (40, 40), (width - 40, height - 40), 120, 3)  # page edge
    cv2.putText(frame, "notes 12/3", (820, 600), cv2.FONT_HERSHEY_SIMPLEX, 1.0, 90, 2)  # other writing far away
    region = (left - 30, top - 30, left + w + 30, top + h + 30)
    if hard:
        # A soft shadow over part of the writing, as from the hand or a lamp.
        angle = rng.uniform(0, np.pi)
        edge = (xx - left - w * rng.uniform(0.3, 0.7)) * np.cos(angle) + (yy - top - h / 2) * np.sin(angle)
        frame = frame.astype(np.float32) * (1 - 0.32 / (1 + np.exp(edge / 25)))
        # The camera looks at the page at a slight angle.
        source = np.float32([[0, 0], [width, 0], [width, height], [0, height]])
        target = (source + rng.uniform(-0.025, 0.025, (4, 2)) * (width, height)).astype(np.float32)
        matrix = cv2.getPerspectiveTransform(source, target)
        frame = cv2.warpPerspective(frame, matrix, (width, height), borderMode=cv2.BORDER_REPLICATE)
        exact = cv2.warpPerspective(exact, matrix, (width, height), flags=cv2.INTER_NEAREST)
        corners = cv2.perspectiveTransform(np.float32([[[region[0], region[1]], [region[2], region[1]],
                                                        [region[2], region[3]], [region[0], region[3]]]]), matrix)[0]
        region = (int(corners[:, 0].min()), int(corners[:, 1].min()), int(corners[:, 0].max()), int(corners[:, 1].max()))
        # Hand or camera motion during the exposure.
        length = int(rng.integers(2, 5))
        kernel = np.zeros((length, length), np.float32)
        cv2.line(kernel, (0, int(rng.integers(0, length))), (length - 1, int(rng.integers(0, length))), 1, 1)
        frame = cv2.filter2D(frame, -1, kernel / max(kernel.sum(), 1))
    frame = cv2.GaussianBlur(frame.astype(np.float32), (0, 0), 0.8) + rng.normal(0, 3.5 if hard else 2.5, frame.shape)
    frame = np.clip(frame, 0, 255).astype(np.uint8)
    if hard:
        frame = cv2.imdecode(cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 60])[1], cv2.IMREAD_GRAYSCALE)
    if truth is not None:
        truth.append(exact)
    return cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR), region


# Which keypad key each kind of sample stands for: (app, workflow).
KEYS = {"equation": ("LaTeX", "equation"), "plot": ("MATLAB", "graph"), "code": ("VS Code", "default"),
        "text": ("Git", "git_commit"), "table": ("LaTeX", "table")}


def _hand(tip, scale=110.0):
    """MediaPipe-style landmarks of a right hand holding a pen whose tip is at ``tip``."""
    axis, normal = np.array((-0.6, -0.8)), np.array((0.8, -0.6))
    tip = np.asarray(tip, float)
    grip = tip - axis * 0.28 * scale
    rest = grip - axis * 0.5 * scale
    wrist = rest - axis * 0.6 * scale
    points = {0: wrist, 2: rest - normal * 15, 5: rest + normal * 10, 4: grip + normal * 8, 8: grip - normal * 8,
              9: wrist + (axis * 0.8 + normal * 0.6) * scale}
    points[1], points[3] = (points[0] + points[2]) / 2, (points[2] + points[4]) / 2
    points[6], points[7] = (2 * points[5] + points[8]) / 3, (points[5] + 2 * points[8]) / 3
    points[13] = wrist + (axis * 0.75 + normal * 0.85) * scale
    points[17] = wrist + (axis * 0.6 + normal * 1.0) * scale
    for joint, base in ((10, 9), (11, 9), (12, 9), (14, 13), (15, 13), (16, 13), (18, 17), (19, 17), (20, 17)):
        points[joint] = points[base] + (axis * 0.12 + normal * 0.05) * scale * (joint - base)
    landmarks = [{"pixel": [int(round(v)) for v in points[index]]} for index in range(21)]
    names = {"thumb": 4, "index": 8, "middle": 12, "ring": 16, "pinky": 20}
    return {"handedness": "Right", "landmarks": landmarks,
            "fingertips": {name: landmarks[index] for name, index in names.items()}}


def nib_path(ink, step=6.0):
    """The path a writer's nib takes over ``ink`` (0/1 mask): letter by letter from left to right,
    along each letter's skeleton, lifted between pieces. Returns [(point, down)] at ``step`` px."""
    skeleton = cv2.ximgproc.thinning(ink.astype(np.uint8) * 255) != 0
    count, labels = cv2.connectedComponents(skeleton.astype(np.uint8), connectivity=8)
    pieces = []
    for label in range(1, count):
        ys, xs = np.nonzero(labels == label)
        if len(xs) >= 3:
            pieces.append(np.c_[xs, ys].astype(float))
    pieces.sort(key=lambda piece: piece[:, 0].min())
    path, last = [], None
    for piece in pieces:
        # Greedy walk from the leftmost pixel; a jump of more than two pixels is a short lift.
        order, left = [], list(range(len(piece)))
        current = int(np.argmin(piece[:, 0]))
        while left:
            left.remove(current)
            order.append(current)
            if not left:
                break
            rest = np.asarray(left)
            current = int(rest[np.argmin(np.linalg.norm(piece[rest] - piece[current], axis=1))])
        walk = piece[order]
        for point in walk:
            if last is None:
                path.append((point, True))
            else:
                gap = float(np.linalg.norm(point - last))
                down = gap <= 2.5
                if gap >= step or not down:
                    steps = max(1, int(gap // step))
                    path += [(last + (point - last) * k / steps, down) for k in range(1, steps + 1)]
                    last = point
                continue
            last = point
        if path and not np.array_equal(path[-1][0], last):
            path.append((last, True))
    return path


def live_recording(text, font_path, text_height=20, seed=0, offset=(9, -9), lift=True, hard=False, truth=False):
    """Simulate writing ``text`` in front of the camera and return (selection, seconds of processing, session).

    The nib walks the letters' skeletons from left to right, lifting between pieces, and the
    ink appears where it has been. The hand covers part of the page in every frame, the tracked
    tip is ``offset`` pixels away from the real nib with a few pixels of landmark jitter, and
    every frame has its own noise and a slightly different exposure. With ``truth`` a fourth
    value is returned: the real ink mask and the nib path, to score a trace against.
    """
    from paper_writing import WritingSession

    rng = np.random.default_rng(seed)
    lines = text.count("\n") + 1
    glyphs = write(text, font_path, rng=rng if hard else None)
    exact = []
    written, region = on_page(glyphs, text_height, seed, hard=hard, lines=lines, truth=exact)
    blank, _ = on_page(glyphs, text_height, seed, hard=hard, lines=lines, empty=True)
    real = exact[0]
    path = nib_path(real)
    session = WritingSession(blank.copy())
    spent, now, frame = 0.0, 1.0, None
    noise = [rng.normal(0, 1.5, blank.shape).astype(np.float32) for _ in range(4)]

    def shoot(page, hand):
        nonlocal spent, now, frame
        frame = page.copy()
        if hand is not None:
            hull = cv2.convexHull(np.asarray([point["pixel"] for point in hand["landmarks"]], np.int32))
            cv2.fillConvexPoly(frame, hull, (150, 172, 205))
        noisy = frame.astype(np.float32) * (0.98 + 0.04 * rng.random()) + noise[int(rng.integers(0, 4))]
        frame = np.clip(noisy, 0, 255).astype(np.uint8)
        start = time.perf_counter()
        session.observe(frame, {"hands": [hand] if hand is not None else []}, now)
        spent += time.perf_counter() - start
        now += 0.075  # about 13 frames per second, as on a laptop camera

    revealed = np.zeros(real.shape, np.uint8)
    reach = max(3, round(text_height * 0.2))
    for point, down in path:
        if down:
            cv2.circle(revealed, tuple(int(v) for v in point), reach, 1, -1)
        page = np.where(revealed[:, :, None] != 0, written, blank)
        jitter = rng.normal(0, 1.2, 2)
        shoot(page, _hand(point + np.asarray(offset) + jitter))
    resting = _hand(path[-1][0] + np.asarray(offset)) if path else None
    for _ in range(6):
        shoot(written, resting)
    if lift:
        for _ in range(8):
            shoot(written, None)
    start = time.perf_counter()
    selection = session.select(frame, prefer_recent=True)
    result = selection, spent + time.perf_counter() - start, session
    return (*result, {"ink": real, "path": path}) if truth else result


def trace_scores(strokes, truth, tolerance=2.5):
    """(precision, recall, hover) of a pen trace against the real ink.

    precision: share of the trace (sampled every pixel) lying on real ink.
    recall: share of the real ink's skeleton that the trace passes within ``tolerance`` px of.
    hover: length of trace segments over clean paper, in pixels per 100 px of trace.
    """
    ink = truth["ink"]
    height, width = ink.shape
    drawn = np.zeros(ink.shape, np.uint8)
    points = []
    for stroke in strokes:
        stroke = np.asarray(stroke, float).reshape(-1, 2)
        if len(stroke) == 1:
            points.append(stroke)
        for start, end in zip(stroke, stroke[1:]):
            count = max(2, int(np.ceil(np.linalg.norm(end - start))) + 1)
            points.append(start + (end - start) * np.linspace(0, 1, count)[:, None])
        cv2.polylines(drawn, [np.rint(stroke).astype(np.int32).reshape(-1, 1, 2)], False, 1, 1)
    if not points:
        return 0.0, 0.0, 0.0
    points = np.vstack(points)
    xs = np.clip(np.rint(points[:, 0]).astype(int), 0, width - 1)
    ys = np.clip(np.rint(points[:, 1]).astype(int), 0, height - 1)
    to_ink = cv2.distanceTransform((ink == 0).astype(np.uint8), cv2.DIST_L2, 3)
    on_ink = to_ink[ys, xs] <= tolerance
    skeleton = cv2.ximgproc.thinning(ink.astype(np.uint8) * 255) != 0
    to_trace = cv2.distanceTransform((drawn == 0).astype(np.uint8), cv2.DIST_L2, 3)
    recall = float(np.mean(to_trace[skeleton] <= tolerance)) if skeleton.any() else 0.0
    return float(np.mean(on_ink)), recall, float(100 * np.mean(~on_ink))


def draw_pen(frame, nib, direction=(0.55, 0.835), width=11.0, length=250.0, grip=True, colour=(18, 10, 145)):
    """Paint a held pencil into a camera frame: a glossy barrel, a wooden cone, a graphite point at ``nib``,
    and the fingers across the barrel, as the desk camera sees a writer on the far side of the desk.
    ``direction`` points from the back of the pen to its point."""
    nib = np.asarray(nib, float)
    axis = np.asarray(direction, float) / np.linalg.norm(direction)
    normal = np.array((-axis[1], axis[0]))
    cone, lead = 2.2 * width, 0.5 * width
    base = nib - axis * cone  # Where the paint ends and the cone starts.
    back = nib - axis * length
    layer = frame.copy()

    def polygon(points, shade):
        cv2.fillConvexPoly(layer, np.rint(np.asarray(points) * 8).astype(np.int32), shade, cv2.LINE_AA, 3)

    polygon([back + normal * width / 2, back - normal * width / 2, base - normal * width / 2,
             base + normal * width / 2], colour)
    stripe = normal * width * 0.15
    cv2.line(layer, tuple(int(round(v)) for v in back + stripe), tuple(int(round(v)) for v in base + stripe),
             (190, 185, 220), 1, cv2.LINE_AA)
    polygon([base + normal * width / 2, base - normal * width / 2, nib - axis * lead - normal * width * 0.14,
             nib - axis * lead + normal * width * 0.14], (95, 120, 150))
    polygon([nib - axis * lead + normal * width * 0.14, nib - axis * lead - normal * width * 0.14,
             nib - normal * 0.9, nib + normal * 0.9], (62, 60, 64))
    if grip:
        skin = (92, 118, 172)
        hold = nib - axis * (cone + 5.5 * width)  # Thumb and index, lying across the barrel.
        angle = float(np.degrees(np.arctan2(normal[1], normal[0])))
        cv2.ellipse(layer, tuple(int(round(v)) for v in hold - normal * 1.2 * width), (int(3.6 * width), int(1.5 * width)),
                    angle, 0, 360, skin, -1, cv2.LINE_AA)
        palm = nib - axis * (cone + 10 * width) - normal * 5.5 * width
        cv2.ellipse(layer, tuple(int(round(v)) for v in palm), (int(6.5 * width), int(4.5 * width)), angle + 25, 0, 360,
                    skin, -1, cv2.LINE_AA)
    x0, y0 = (max(0, int(v)) for v in np.minimum(nib, back) - 12 * width)
    x1, y1 = (int(v) for v in np.maximum(nib, back) + 12 * width)
    layer[y0:y1, x0:x1] = cv2.GaussianBlur(layer[y0:y1, x0:x1], (0, 0), 0.7)
    frame[:] = layer
    return frame


def pen_recording(text, font_path, text_height=20, seed=0, hard=False, write_text=True, gestures=True, shift=(0.0, 0.0)):
    """Simulate a recording with a visible pencil (no hand landmarks: the hand model usually misses the
    hand) and return (selection, seconds of processing, session, truth).

    The pen comes in hovering, points at the old note on the page and moves along it in the air, then
    writes ``text`` (unless ``write_text`` is False), lifting between the pieces; afterwards it hovers
    away, comes back, rests beside the last letter, and passes over what it wrote without touching.
    ``shift`` moves the whole page by that many pixels once the pen arrives, as a sheet creeps under a
    hand. ``truth`` holds the real ink, the true nib of every frame as (time, point, phase, down), so
    the landmark, the pen-down decisions, and the capture frame can be scored (``pen_scores``)."""
    from paper_writing import WritingSession

    rng = np.random.default_rng(seed)
    lines = text.count("\n") + 1
    glyphs = write(text, font_path, rng=rng if hard else None)
    exact = []
    written, region = on_page(glyphs, text_height, seed, hard=hard, lines=lines, truth=exact)
    blank, _ = on_page(glyphs, text_height, seed, hard=hard, lines=lines, empty=True)
    real = exact[0]
    path = nib_path(real) if write_text else []
    saved = WritingSession.pen_colour
    WritingSession.pen_colour = None
    session = WritingSession(blank.copy())
    spent, now, frame = 0.0, 1.0, None
    noise = [rng.normal(0, 1.5, blank.shape).astype(np.float32) for _ in range(4)]
    log = []
    moved = np.float32([[1, 0, shift[0]], [0, 1, shift[1]]])
    creep = bool(shift[0] or shift[1])
    if creep:
        real = cv2.warpAffine(real, moved, real.shape[::-1], flags=cv2.INTER_NEAREST)

    def shoot(page, nib, phase, down=False):
        nonlocal spent, now, frame
        frame = page.copy()
        if creep and phase != "before":
            frame = cv2.warpAffine(frame, moved, frame.shape[1::-1], borderMode=cv2.BORDER_REPLICATE)
        if nib is not None:
            nib = np.asarray(nib, float) + (shift if phase != "before" else (0, 0))
            draw_pen(frame, nib + rng.normal(0, 0.4, 2))
            log.append((now, (float(nib[0]), float(nib[1])), phase, bool(down)))
        noisy = frame.astype(np.float32) * (0.98 + 0.04 * rng.random()) + noise[int(rng.integers(0, 4))]
        frame = np.clip(noisy, 0, 255).astype(np.uint8)
        start = time.perf_counter()
        session.observe(frame, {"hands": []}, now)
        spent += time.perf_counter() - start
        now += 0.075

    def glide(page, start, end, phase, step=9.0):
        start, end = np.asarray(start, float), np.asarray(end, float)
        for share in np.linspace(0, 1, max(2, int(np.linalg.norm(end - start) / step))):
            shoot(page, start + (end - start) * share, phase)

    first = np.asarray(path[0][0], float) if path else np.array((300.0, 330.0))
    note = np.array((830.0, 590.0))  # The old note ("notes 12/3") on the page.
    if gestures:
        shoot(blank, None, "before")
        glide(blank, (700, 360), note, "approach")
        for _ in range(10):
            shoot(blank, note + (12, -4), "point")  # Resting on the old note.
        glide(blank, note, note + (150, 0), "underline", step=6.0)  # Along it, in the air.
        glide(blank, note + (150, 0), first, "approach")
    revealed = np.zeros(real.shape, np.uint8)
    reach = max(3, round(text_height * 0.2))
    # The nib is on the paper where it stands on ink and came there in a short step (nib_path's own flag
    # compares with the last point it kept, which marks nearly every step as a lift).
    to_ink = cv2.distanceTransform((exact[0] == 0).astype(np.uint8), cv2.DIST_L2, 3)
    before = None
    for point, _ in path:
        x, y = int(round(point[0])), int(round(point[1]))
        down = bool(to_ink[y, x] <= 1.5 and (before is None or np.linalg.norm(np.subtract(point, before)) <= 8))
        before = point
        if down:
            cv2.circle(revealed, (x, y), reach, 1, -1)
        page = np.where(revealed[:, :, None] != 0, written, blank)
        shoot(page, point, "write", down)
    last = np.asarray(path[-1][0], float) if path else first
    page = written if path else blank
    if gestures:
        glide(page, last, last + (120, -90), "leave")
        glide(page, last + (120, -90), last + (14, -3), "return")
        for _ in range(10):
            shoot(page, last + (14, -3), "rest")  # Beside the last letter, not writing.
        if path:
            glide(page, last, first, "retrace", step=7.0)  # Back over the line, in the air.
    for _ in range(4):
        shoot(page, last + (40, -30), "rest")
    for _ in range(8):
        shoot(page, None, "after")
    start = time.perf_counter()
    selection = session.select(frame, prefer_recent=True)
    spent += time.perf_counter() - start
    WritingSession.pen_colour = saved
    if not path:
        real = np.zeros_like(real)
    return selection, spent, session, {"ink": real, "path": path, "log": log}


def pen_scores(selection, session, truth, reach=6.0):
    """How well a pen recording was followed, as a dict.

    nib: median and largest distance between the landmark and the true nib (frames where the point was seen);
    seen: share of the frames with the pen in view that gave a landmark on its point;
    hover_down: share of the samples taken while the pen was in the air that count as pen-down;
    write_down: share of the samples taken while the nib was on the paper that count as pen-down;
    extra / missed: pixels by which the capture frame exceeds the real ink's box (plus its margin), and by
    which it cuts into it; ``frame`` is None when no frame was made."""
    log = truth["log"]
    times = np.asarray([item[0] for item in log])
    errors, hover, writing, seen = [], [], [], 0
    for sample in (sample for samples in session.tips.values() for sample in samples):
        index = int(np.argmin(np.abs(times - sample.time)))
        if abs(times[index] - sample.time) > 0.01:
            continue
        _, nib, phase, down = log[index]
        if sample.visual:
            seen += 1
            errors.append(float(np.hypot(sample.point[0] - nib[0], sample.point[1] - nib[1])))
        (writing if down else hover).append(bool(sample.down))
    scores = {"nib": (float(np.median(errors)), float(np.max(errors))) if errors else None,
              "seen": seen / max(len(log), 1),
              "hover_down": float(np.mean(hover)) if hover else 0.0,
              "write_down": float(np.mean(writing)) if writing else None,
              "frame": None, "extra": 0, "missed": 0}
    ink = truth["ink"]
    if selection is not None and selection.page.bounds:
        scores["frame"] = tuple(selection.page.bounds)
        x0, y0, x1, y1 = selection.page.bounds
        if ink.any():
            ys, xs = np.nonzero(ink)
            margin = max(6, round(0.5 * selection.text_height)) if selection.text_height else 12
            scores["extra"] = int(max(0, xs.min() - margin - x0, x1 - xs.max() - 1 - margin,
                                      ys.min() - margin - y0, y1 - ys.max() - 1 - margin))
            scores["missed"] = int(max(0, x0 - xs.min(), xs.max() + 1 - x1, y0 - ys.min(), ys.max() + 1 - y1))
        else:
            scores["extra"] = int(max(x1 - x0, y1 - y0))
    elif ink.any():
        ys, xs = np.nonzero(ink)
        scores["missed"] = int(max(np.ptp(xs), np.ptp(ys)))
    return scores


def main():
    from paper_conversion import convert_transcription, read_recording, read_static
    from paper_ink import extract_ink
    from paper_reader import READERS, read_picture

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reader", action="append", choices=sorted(READERS), help="Reader(s) to test")
    parser.add_argument("--fonts", default="inkfree,segoeprint,bradley")
    parser.add_argument("--height", type=int, default=20, help="Glyph height in camera pixels")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--save", type=Path)
    parser.add_argument("--pipeline", action="store_true",
                        help="Whole snapshot path per key: photo, ink, reading, check, generated result")
    parser.add_argument("--hard", action="store_true",
                        help="Handwriting-like letters (size, tilt, slant), faint ink, a shadow, a tilted view, "
                             "motion blur, JPEG, and multi-line samples")
    parser.add_argument("--live", action="store_true",
                        help="Whole live path: simulated writing under a moving hand, registration, reading, result")
    args = parser.parse_args()
    fonts = [name for name in args.fonts.split(",") if name in FONTS]
    samples = SAMPLES + HARD_SAMPLES if args.hard else SAMPLES
    samples = samples[:args.limit] if args.limit else samples
    if args.pipeline or args.live:
        rows = []
        for index, (kind, text) in enumerate(samples):
            target, workflow = KEYS[kind]
            for font in fonts:
                font_path = Path("C:/Windows/Fonts") / FONTS[font]
                start = time.perf_counter()
                trace = None
                if args.live:
                    selection, tracking, _, truth = live_recording(text, font_path, args.height, index, hard=args.hard,
                                                                   truth=True)
                    if selection is not None:
                        trace = trace_scores(selection.strokes, truth)[:2]
                    start = time.perf_counter()
                    reading = None if selection is None else read_recording(selection, target, "Python", workflow,
                                                                            "qwen3-vl:2b-instruct")
                    if args.save and selection is not None and selection.ink is not None:
                        args.save.mkdir(parents=True, exist_ok=True)
                        cv2.imwrite(str(args.save / f"live_{index:02d}_{font}.png"), selection.ink)
                else:
                    rng = np.random.default_rng(index) if args.hard else None
                    frame, region = on_page(write(text, font_path, rng=rng), args.height, index, hard=args.hard,
                                            lines=text.count("\n") + 1)
                    tracking = 0.0
                    reading = read_static(frame[region[1]:region[3], region[0]:region[2]], target, "Python", workflow)
                source = "" if reading is None else reading[0]
                read_time = time.perf_counter() - start
                start = time.perf_counter()
                draft = None
                if source and kind in ("equation", "code"):
                    draft = convert_transcription(source, (), target, "Python", "qwen3-vl:2b-instruct", workflow)
                elif source and kind == "plot":
                    from paper_transfer import graph
                    import tempfile
                    with tempfile.TemporaryDirectory() as folder:
                        try:
                            graph(source, Path(folder))
                            draft = True
                        except (ValueError, ArithmeticError):
                            draft = None
                produce = time.perf_counter() - start
                done = bool(draft) if kind in ("equation", "code", "plot") else bool(source)
                rows.append((kind, text, source, accuracy(text, source), read_time, produce, tracking, done, trace))
        label = "live" if args.live else "snapshot"
        exact = sum(1 for row in rows if canonical(row[1]) == canonical(row[2]))
        print(f"\n== {label} pipeline: exact {exact}/{len(rows)}, character accuracy "
              f"{np.mean([row[3] for row in rows]):.3f}, result produced {sum(row[7] for row in rows)}/{len(rows)}")
        print(f"   reading {np.median([row[4] for row in rows]):.2f} s median ({np.max([row[4] for row in rows]):.2f} max), "
              f"generating the result {np.median([row[5] for row in rows]):.2f} s median "
              f"({np.max([row[5] for row in rows]):.2f} max)"
              + (f", tracking {1000 * np.mean([row[6] for row in rows]):.0f} ms per recording" if args.live else ""))
        traces = [row[8] for row in rows if row[8] is not None]
        if traces:
            print(f"   pen trace: {np.mean([t[0] for t in traces]):.3f} of it on the ink, "
                  f"{np.mean([t[1] for t in traces]):.3f} of the ink covered")
        for kind in sorted({row[0] for row in rows}):
            part = [row for row in rows if row[0] == kind]
            print(f"   {kind:9s} exact {sum(canonical(r[1]) == canonical(r[2]) for r in part)}/{len(part)}  "
                  f"accuracy {np.mean([r[3] for r in part]):.3f}  total "
                  f"{np.median([r[4] + r[5] for r in part]):.2f} s median")
        for row in rows:
            if canonical(row[1]) != canonical(row[2]):
                print(f"   miss [{row[0]}] {row[1]!r} -> {row[2]!r}")
        return
    readers = args.reader or sorted(READERS)
    results = {name: [] for name in readers}
    for index, (kind, text) in enumerate(samples):
        for font in fonts:
            frame, region = on_page(write(text, Path("C:/Windows/Fonts") / FONTS[font]), args.height, index)
            picture = extract_ink(frame, region)
            if picture is None:
                for name in readers:
                    results[name].append((kind, text, "", 0.0, 0.0))
                continue
            if args.save:
                args.save.mkdir(parents=True, exist_ok=True)
                cv2.imwrite(str(args.save / f"{index:02d}_{font}.png"), picture.image)
            for name in readers:
                start = time.perf_counter()
                try:
                    got = read_picture(picture.image, picture.text_height, kind, reader=name)
                except (RuntimeError, ValueError) as error:
                    got = f"<{error}>"
                elapsed = time.perf_counter() - start
                results[name].append((kind, text, got, accuracy(text, got), elapsed))
    for name, rows in results.items():
        exact = sum(1 for _, text, got, _, _ in rows if canonical(text) == canonical(got))
        print(f"\n== {name}: exact {exact}/{len(rows)}, character accuracy "
              f"{np.mean([row[3] for row in rows]):.3f}, {np.mean([row[4] for row in rows]):.2f} s per reading "
              f"(median {np.median([row[4] for row in rows]):.2f})")
        for kind in sorted({row[0] for row in rows}):
            part = [row for row in rows if row[0] == kind]
            print(f"   {kind:9s} exact {sum(canonical(t) == canonical(g) for _, t, g, _, _ in part)}/{len(part)}"
                  f"  accuracy {np.mean([row[3] for row in part]):.3f}")
        for kind, text, got, score, _ in rows:
            if canonical(text) != canonical(got):
                print(f"   miss [{kind}] {text!r} -> {got!r}")


if __name__ == "__main__":
    main()
