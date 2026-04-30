import cv2
import numpy as np

from .geometry import (
    normalize_line,
    line_from_two_points,
    line_from_fitline_output,
    point_line_distance,
    point_to_segment_distance,
    project_t_on_segment_axis,
)


def robust_fit_line(points, max_iter=5, inlier_thresh=4.0):
    if points is None or len(points) < 2:
        return None, None
    pts = points.astype(np.float32)
    inliers = pts.copy()
    model = None
    for _ in range(max_iter):
        if len(inliers) < 2:
            break
        line = cv2.fitLine(inliers.reshape(-1, 1, 2), cv2.DIST_L2, 0, 0.01, 0.01)
        vx, vy, x0, y0 = line.flatten()
        model = normalize_line(line_from_fitline_output(vx, vy, x0, y0))
        d = point_line_distance(pts, model)
        new_inliers = pts[d < inlier_thresh]
        if len(new_inliers) < 2:
            break
        if len(new_inliers) == len(inliers):
            inliers = new_inliers
            break
        inliers = new_inliers
    return model, inliers


def angle_of_line_deg(line):
    a, b, _ = line
    dx = b
    dy = -a
    return np.degrees(np.arctan2(dy, dx))


def angle_diff_deg(a, b):
    d = abs(a - b) % 180.0
    return min(d, 180.0 - d)


def build_contour_edge_image(clean_mask_img):
    edges = cv2.Canny(clean_mask_img, 50, 150)
    kernel = np.ones((3, 3), np.uint8)
    return cv2.dilate(edges, kernel, iterations=1)


def detect_hough_segments(clean_mask_img):
    edges = build_contour_edge_image(clean_mask_img)
    linesP = cv2.HoughLinesP(
        edges, rho=1, theta=np.pi / 180, threshold=60,
        minLineLength=max(40, int(0.18 * min(clean_mask_img.shape[:2]))),
        maxLineGap=18,
    )
    segments = []
    if linesP is not None:
        for l in linesP:
            x1, y1, x2, y2 = l[0]
            length = np.hypot(x2 - x1, y2 - y1)
            if length >= 25:
                segments.append((x1, y1, x2, y2))
    return segments, edges


def segment_length(seg):
    x1, y1, x2, y2 = seg
    return float(np.hypot(x2 - x1, y2 - y1))


def segment_midpoint(seg):
    x1, y1, x2, y2 = seg
    return np.array([(x1 + x2) / 2.0, (y1 + y2) / 2.0], dtype=np.float32)


def segment_to_line(seg):
    x1, y1, x2, y2 = seg
    return normalize_line(line_from_two_points((x1, y1), (x2, y2)))


def trim_side_points(points, a, b, trim_ratio=0.08):
    if points is None or len(points) < 2:
        return points
    t = project_t_on_segment_axis(points, a, b)
    keep = (t >= trim_ratio) & (t <= 1.0 - trim_ratio)
    trimmed = points[keep]
    return trimmed if len(trimmed) >= 2 else points


def assign_contour_points_to_rect_sides(contour_points, rect_pts, dist_pad=0.06):
    tl, tr, br, bl = rect_pts
    sides = {"top": (tl, tr), "right": (tr, br), "bottom": (br, bl), "left": (bl, tl)}
    width = max(np.linalg.norm(tr - tl), np.linalg.norm(br - bl))
    height = max(np.linalg.norm(bl - tl), np.linalg.norm(br - tr))
    base_thresh = max(6.0, 0.03 * min(width, height))
    loose_thresh = max(base_thresh, dist_pad * min(width, height))
    groups = {k: [] for k in sides.keys()}

    for p in contour_points:
        dists = {name: point_to_segment_distance(p, a, b) for name, (a, b) in sides.items()}
        best_side = min(dists, key=dists.get)
        if dists[best_side] <= loose_thresh:
            groups[best_side].append(p)

    for k in groups:
        groups[k] = np.array(groups[k], dtype=np.float32) if len(groups[k]) > 0 else np.empty((0, 2), dtype=np.float32)
    for k, (a, b) in sides.items():
        groups[k] = trim_side_points(groups[k], a, b, trim_ratio=0.08)
    return groups, sides


def fallback_line_from_rect_side(rect_pts, side_name):
    tl, tr, br, bl = rect_pts
    mapping = {"top": (tl, tr), "right": (tr, br), "bottom": (br, bl), "left": (bl, tl)}
    a, b = mapping[side_name]
    return normalize_line(line_from_two_points(a, b))


def side_expected_angle(rect_pts, side_name):
    tl, tr, br, bl = rect_pts
    mapping = {"top": (tl, tr), "right": (tr, br), "bottom": (br, bl), "left": (bl, tl)}
    a, b = mapping[side_name]
    line = normalize_line(line_from_two_points(a, b))
    return angle_of_line_deg(line)


def segment_score_for_side(seg, side_name, rect_pts, side_points):
    seg_line = segment_to_line(seg)
    seg_mid = segment_midpoint(seg)
    seg_len = segment_length(seg)
    exp_ang = side_expected_angle(rect_pts, side_name)
    seg_ang = angle_of_line_deg(seg_line)
    ang_err = angle_diff_deg(exp_ang, seg_ang)
    tl, tr, br, bl = rect_pts
    mapping = {"top": (tl, tr), "right": (tr, br), "bottom": (br, bl), "left": (bl, tl)}
    a, b = mapping[side_name]
    mid_dist = point_to_segment_distance(seg_mid, a, b)
    contour_dist = np.median(point_line_distance(side_points, seg_line)) if side_points is not None and len(side_points) > 0 else 9999.0
    score = 3.0 * ang_err + 2.0 * mid_dist + 1.4 * contour_dist - 0.025 * seg_len
    return score, seg_line, seg_len, ang_err, mid_dist, contour_dist


def choose_best_hough_line_for_side(segments, side_name, rect_pts, side_points, img_w, img_h):
    if segments is None or len(segments) == 0:
        return None, None, None
    best_score = 1e18
    best_line = None
    best_seg = None
    best_meta = None
    min_dim = min(img_w, img_h)
    for seg in segments:
        score, seg_line, seg_len, ang_err, mid_dist, contour_dist = segment_score_for_side(seg, side_name, rect_pts, side_points)
        if seg_len < 0.20 * min_dim or ang_err > 12:
            continue
        if score < best_score:
            best_score = score
            best_line = seg_line
            best_seg = seg
            best_meta = {"score": score, "seg_len": seg_len, "ang_err": ang_err, "mid_dist": mid_dist, "contour_dist": contour_dist}
    return best_line, best_seg, best_meta


def line_fit_score_on_side(line, side_points):
    if line is None or side_points is None or len(side_points) == 0:
        return 1e18
    d = point_line_distance(side_points, line)
    return float(np.median(d))


def choose_final_side_line(side_name, rect_pts, side_points, fit_line, hough_line, hough_meta, img_w, img_h):
    fallback = fallback_line_from_rect_side(rect_pts, side_name)
    fit_score = line_fit_score_on_side(fit_line, side_points) if fit_line is not None else 1e18
    hough_score = line_fit_score_on_side(hough_line, side_points) if hough_line is not None else 1e18
    fallback_score = line_fit_score_on_side(fallback, side_points)
    chosen = fit_line if fit_line is not None and fit_score < 1e18 else fallback
    source = "fit" if fit_line is not None and fit_score < 1e18 else "fallback"
    if hough_line is not None and hough_meta is not None:
        min_dim = min(img_w, img_h)
        long_enough = hough_meta["seg_len"] >= 0.28 * min_dim
        angle_good = hough_meta["ang_err"] <= 8
        clearly_better = hough_score < fit_score * 0.78
        if long_enough and angle_good and clearly_better:
            chosen = hough_line
            source = "hough"
    return chosen or fallback, source, fit_score, hough_score, fallback_score
