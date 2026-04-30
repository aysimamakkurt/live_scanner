import numpy as np


def order_points(pts):
    pts = np.array(pts, dtype=np.float32)
    rect = np.zeros((4, 2), dtype=np.float32)
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    return rect


def point_inside_image(pt, w, h, margin=180):
    x, y = pt
    return (-margin <= x <= w + margin) and (-margin <= y <= h + margin)


def line_from_two_points(p1, p2):
    x1, y1 = float(p1[0]), float(p1[1])
    x2, y2 = float(p2[0]), float(p2[1])
    a = y1 - y2
    b = x2 - x1
    c = x1 * y2 - x2 * y1
    return (a, b, c)


def line_from_fitline_output(vx, vy, x0, y0):
    a = float(vy)
    b = float(-vx)
    c = float(vx * y0 - vy * x0)
    return (a, b, c)


def normalize_line(abc):
    a, b, c = abc
    norm = np.sqrt(a * a + b * b) + 1e-8
    return (a / norm, b / norm, c / norm)


def intersect_abc(l1, l2):
    a1, b1, c1 = l1
    a2, b2, c2 = l2
    det = a1 * b2 - a2 * b1
    if abs(det) < 1e-8:
        return None
    x = (b1 * c2 - b2 * c1) / det
    y = (c1 * a2 - c2 * a1) / det
    return np.array([x, y], dtype=np.float32)


def point_line_distance(points, abc):
    a, b, c = abc
    denom = np.sqrt(a * a + b * b) + 1e-8
    return np.abs(a * points[:, 0] + b * points[:, 1] + c) / denom


def polygon_area(pts):
    pts = np.asarray(pts, dtype=np.float32)
    x = pts[:, 0]
    y = pts[:, 1]
    return 0.5 * abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))


def expand_quad_from_center(corners, scale=1.01):
    corners = order_points(corners).astype(np.float32)
    center = np.mean(corners, axis=0, keepdims=True)
    expanded = center + (corners - center) * scale
    return order_points(expanded)


def point_to_segment_distance(pt, a, b):
    pt = np.asarray(pt, dtype=np.float32)
    a = np.asarray(a, dtype=np.float32)
    b = np.asarray(b, dtype=np.float32)
    ab = b - a
    denom = np.dot(ab, ab)
    if denom < 1e-8:
        return np.linalg.norm(pt - a)
    t = np.dot(pt - a, ab) / denom
    t = np.clip(t, 0.0, 1.0)
    proj = a + t * ab
    return np.linalg.norm(pt - proj)


def project_t_on_segment_axis(points, a, b):
    a = np.asarray(a, dtype=np.float32)
    b = np.asarray(b, dtype=np.float32)
    ab = b - a
    denom = np.dot(ab, ab) + 1e-8
    return ((points - a) @ ab) / denom
