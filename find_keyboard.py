import numpy as np
import cv2

def order_corners(pts):
    pts = np.asarray(pts, dtype=np.float32).reshape(-1, 2)
    s = pts.sum(axis=1)
    d = np.diff(pts, axis=1).ravel()
    return np.array([pts[np.argmin(s)], pts[np.argmin(d)],
                     pts[np.argmax(s)], pts[np.argmax(d)]], dtype=np.float32)

def enclosing_quad(hull):
    """Smallest quad (edges parallel to the extreme-point quad) that contains every hull point."""
    pts = hull.reshape(-1, 2).astype(np.float64)
    ext = order_corners(pts).astype(np.float64)
    centroid = ext.mean(axis=0)

    lines = []                                   # (normal, c) with n·x = c
    for i in range(4):
        p, q = ext[i], ext[(i + 1) % 4]
        dx, dy = q - p
        n = np.array([dy, -dx]) / (np.hypot(dx, dy) + 1e-9)
        if np.dot(n, centroid - p) > 0:          # make the normal point outward
            n = -n
        c = np.dot(n, p) + max(0.0, np.max((pts - p) @ n))   # push the edge out to the farthest point
        lines.append((n, c))

    quad = []
    for i in range(4):                           # corner i = edge (i-1) ∩ edge i
        (n1, c1), (n2, c2) = lines[i - 1], lines[i]
        quad.append(np.linalg.solve(np.array([n1, n2]), np.array([c1, c2])))
    return np.array(quad, dtype=np.float32)      # TL, TR, BR, BL

def find_keyboard_corners(img, blur_frac=0.025, thr=0.75, debug_path=None):
    """Corners (TL, TR, BR, BL) of the whole printed keyboard (largest dark region not touching the border).
    blur_frac: blur size as a fraction of image width (bigger = more merging)
    thr: dark threshold as a fraction of the paper brightness (raise it to grow the region)"""
    if isinstance(img, str):
        img = cv2.imread(img)
    h, w = img.shape[:2]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Heavy blur: the black frame + white keys average out to "dark" compared with the paper
    k = max(5, int(w * blur_frac)) | 1
    smooth = cv2.blur(gray, (k, k))

    paper = np.percentile(gray, 90)              # brightness of the white paper
    dark = (smooth < thr * paper).astype(np.uint8) * 255

    dark = cv2.morphologyEx(dark, cv2.MORPH_CLOSE,
                            cv2.getStructuringElement(cv2.MORPH_RECT, (k, k)))
    dark = cv2.morphologyEx(dark, cv2.MORPH_OPEN,
                            cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)))

    contours, _ = cv2.findContours(dark, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    margin = 3
    best, best_area = None, 0
    for c in contours:
        x, y, bw, bh = cv2.boundingRect(c)
        if x <= margin or y <= margin or x + bw >= w - margin or y + bh >= h - margin:
            continue                             # shirt, real keyboard, black pad...
        area = cv2.contourArea(c)
        if area > best_area:
            best, best_area = c, area
    if best is None:
        return None

    corners = enclosing_quad(cv2.convexHull(best))

    if debug_path:
        vis = img.copy()
        cv2.polylines(vis, [corners.astype(np.int32)], True, (0, 255, 0), 2)
        for name, p in zip(("TL", "TR", "BR", "BL"), corners):
            cv2.circle(vis, tuple(int(v) for v in p), 4, (0, 0, 255), -1)
            cv2.putText(vis, name, (int(p[0]) + 6, int(p[1]) - 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
        cv2.imwrite(debug_path, vis)
        cv2.imwrite(debug_path.replace(".png", "_mask.png"), dark)   # inspect the mask too
    return corners