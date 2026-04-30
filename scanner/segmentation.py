import cv2
import numpy as np


def clean_mask(mask_uint8):
    kernel_close = np.ones((7, 7), np.uint8)
    kernel_open = np.ones((3, 3), np.uint8)
    clean = cv2.morphologyEx(mask_uint8, cv2.MORPH_CLOSE, kernel_close, iterations=2)
    clean = cv2.morphologyEx(clean, cv2.MORPH_OPEN, kernel_open, iterations=1)
    _, clean = cv2.threshold(clean, 127, 255, cv2.THRESH_BINARY)
    return clean


def get_largest_contour(mask_uint8):
    contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        return None
    return max(contours, key=cv2.contourArea)


def choose_best_mask_from_results(results, w, h):
    if results[0].masks is None or len(results[0].masks.data) == 0:
        return None

    masks = results[0].masks.data.cpu().numpy()
    best_mask = None
    best_area = -1
    for m in masks:
        m_resized = cv2.resize(m, (w, h), interpolation=cv2.INTER_NEAREST)
        m_uint8 = (m_resized * 255).astype(np.uint8)
        area = np.count_nonzero(m_uint8 > 127)
        if area > best_area:
            best_area = area
            best_mask = m_uint8
    return best_mask


def run_yolo_segmentation(model, img, lock=None):
    h, w = img.shape[:2]

    if lock is not None:
        with lock:
            results = model(img, verbose=False)
    else:
        results = model(img, verbose=False)
    best_mask = choose_best_mask_from_results(results, w, h)

    if best_mask is None:
        return None

    cleaned_mask = clean_mask(best_mask)
    return cleaned_mask