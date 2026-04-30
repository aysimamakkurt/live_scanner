import cv2
import numpy as np

from .geometry import order_points, point_inside_image, intersect_abc, polygon_area, expand_quad_from_center
from .segmentation import clean_mask, get_largest_contour
from .edge_detection import (
    detect_hough_segments,
    robust_fit_line,
    assign_contour_points_to_rect_sides,
    fallback_line_from_rect_side,
    choose_best_hough_line_for_side,
    choose_final_side_line,
)


def is_reasonable_quad(corners, img_w, img_h):
    if corners is None or len(corners) != 4:
        return False
    for p in corners:
        if not point_inside_image(p, img_w, img_h, margin=220):
            return False
    area = polygon_area(corners)
    if area < 0.10 * img_w * img_h:
        return False
    tl, tr, br, bl = order_points(corners)
    top_w = np.linalg.norm(tr - tl)
    bot_w = np.linalg.norm(br - bl)
    left_h = np.linalg.norm(bl - tl)
    right_h = np.linalg.norm(br - tr)
    if min(top_w, bot_w, left_h, right_h) < 20:
        return False
    return True


def extract_document_corners_safe_hybrid(mask_uint8):
    h, w = mask_uint8.shape[:2]
    clean = clean_mask(mask_uint8)
    cnt = get_largest_contour(clean)
    if cnt is None or len(cnt) < 4:
        return None
    contour_points = cnt.reshape(-1, 2).astype(np.float32)
    rect = cv2.minAreaRect(cnt)
    rect_pts = order_points(cv2.boxPoints(rect).astype(np.float32))
    groups, _ = assign_contour_points_to_rect_sides(contour_points, rect_pts)
    segments, _ = detect_hough_segments(clean)
    final_lines = {}
    for side_name in ["top", "right", "bottom", "left"]:
        pts = groups[side_name]
        if pts is not None and len(pts) >= 20:
            fit_line, _ = robust_fit_line(pts, max_iter=5, inlier_thresh=4.0)
        elif pts is not None and len(pts) >= 2:
            fit_line, _ = robust_fit_line(pts, max_iter=3, inlier_thresh=5.0)
        else:
            fit_line = None
        if fit_line is None:
            fit_line = fallback_line_from_rect_side(rect_pts, side_name)
        hough_line, _, hough_meta = choose_best_hough_line_for_side(segments, side_name, rect_pts, pts, w, h)
        final_line, _, _, _, _ = choose_final_side_line(side_name, rect_pts, pts, fit_line, hough_line, hough_meta, w, h)
        final_lines[side_name] = final_line
    tl = intersect_abc(final_lines["top"], final_lines["left"])
    tr = intersect_abc(final_lines["top"], final_lines["right"])
    br = intersect_abc(final_lines["bottom"], final_lines["right"])
    bl = intersect_abc(final_lines["bottom"], final_lines["left"])
    if any(p is None for p in [tl, tr, br, bl]):
        return None
    rough_corners = order_points(np.array([tl, tr, br, bl], dtype=np.float32))
    expanded_corners = expand_quad_from_center(rough_corners, scale=1.01)
    rough_area = polygon_area(rough_corners)
    expanded_area = polygon_area(expanded_corners)
    final_corners = rough_corners if expanded_area > rough_area * 1.20 else expanded_corners
    final_corners = order_points(final_corners)
    if not is_reasonable_quad(final_corners, w, h):
        if is_reasonable_quad(rough_corners, w, h):
            final_corners = rough_corners
        else:
            return None
    valid = [point_inside_image(p, w, h, margin=220) for p in final_corners]
    if not all(valid):
        if is_reasonable_quad(rough_corners, w, h):
            final_corners = rough_corners
        else:
            return None
    return final_corners
