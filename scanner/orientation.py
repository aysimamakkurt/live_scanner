import logging
import re
import cv2
import numpy as np
import pytesseract

logger = logging.getLogger(__name__)
ROTATIONS = {0: None, 90: cv2.ROTATE_90_CLOCKWISE, 180: cv2.ROTATE_180, 270: cv2.ROTATE_90_COUNTERCLOCKWISE}
ANGLE_180_KEEP_UPRIGHT_DELTA = 35.0


def rotate(img, angle):
    return img.copy() if angle == 0 else cv2.rotate(img, ROTATIONS[angle])


def resize_for_ocr(img, max_dim=1200):
    h, w = img.shape[:2]
    scale = min(max_dim / max(h, w), 1.0)
    if scale == 1.0:
        return img
    return cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)


def preprocess_for_ocr(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img.copy()
    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    return cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(gray)


def quick_ocr_score(img):
    try:
        proc = preprocess_for_ocr(img)
        data = pytesseract.image_to_data(proc, config="--psm 3", output_type=pytesseract.Output.DICT)
        confs, text_len = [], 0
        raw_token_count = 0
        valid_token_count = 0
        for txt, conf in zip(data["text"], data["conf"]):
            txt = txt.strip()
            try:
                conf = float(conf)
            except Exception:
                conf = -1
            if txt:
                raw_token_count += 1
            if txt and conf > 0:
                if is_valid_ocr_token(txt):
                    valid_token_count += 1
                    confs.append(conf)
                text_len += len(txt)
        mean_conf = float(np.mean(confs)) if confs else 0.0
        valid_ratio = valid_token_count / max(raw_token_count, 1)
        garbage_ratio = 1.0 - valid_ratio
        score = 3.0 * mean_conf + 1.2 * np.log1p(text_len) + 18.0 * valid_ratio - 12.0 * garbage_ratio
        return score, text_len
    except Exception:
        return 0.0, 0


def is_valid_ocr_token(token: str) -> bool:
    token = token.strip()
    if not token:
        return False
    if not re.search(r"[A-Za-z0-9]", token):
        return False
    if re.fullmatch(r"[\W_]+", token):
        return False
    if re.fullmatch(r"[Il1|]{4,}", token):
        return False
    return True


def component_horizontal_score(binary_img):
    num_labels, _, stats, _ = cv2.connectedComponentsWithStats(binary_img, connectivity=8)
    score = 0.0
    for idx in range(1, num_labels):
        _, _, width, height, area = stats[idx]
        if area < 40 or height <= 0:
            continue
        aspect_ratio = width / float(height)
        if 1.6 <= aspect_ratio <= 25.0:
            score += min(area, 1200)
    return float(score)


def row_projection_score(binary_img):
    row_sums = np.sum(binary_img > 0, axis=1).astype(np.float32)
    col_sums = np.sum(binary_img > 0, axis=0).astype(np.float32)
    row_var = float(np.std(row_sums))
    col_var = float(np.std(col_sums))
    return row_var - 0.35 * col_var


def image_heuristic_score(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img.copy()
    bw = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 31, 15)
    h, w = bw.shape[:2]
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (max(15, w // 30), 5))
    merged = cv2.morphologyEx(bw, cv2.MORPH_CLOSE, kernel)
    cc_score = component_horizontal_score(merged)
    projection_score = row_projection_score(merged)
    return 0.012 * cc_score + projection_score


def auto_orient_document(img):
    all_candidates = []
    for angle in [0, 90, 180, 270]:
        rotated = rotate(img, angle)
        small = resize_for_ocr(rotated, max_dim=900)
        heuristic_score = image_heuristic_score(small)
        all_candidates.append({
            "angle": angle,
            "image": rotated,
            "small_image": small,
            "heuristic_score": heuristic_score,
            "ocr_conf": 0.0,
            "text_len": 0,
            "final_score": heuristic_score,
        })

    horizontal_pair = [candidate for candidate in all_candidates if candidate["angle"] in {0, 180}]
    vertical_pair = [candidate for candidate in all_candidates if candidate["angle"] in {90, 270}]

    horizontal_pair_score = max(candidate["heuristic_score"] for candidate in horizontal_pair)
    vertical_pair_score = max(candidate["heuristic_score"] for candidate in vertical_pair)

    if horizontal_pair_score >= vertical_pair_score:
        candidates = horizontal_pair
        override_reason = "Selected horizontal orientation pair (0/180) from heuristic pre-check."
    else:
        candidates = vertical_pair
        override_reason = "Selected vertical orientation pair (90/270) from heuristic pre-check."

    shortlisted_angles = {candidate["angle"] for candidate in candidates}

    for candidate in candidates:
        ocr_conf, text_len = quick_ocr_score(candidate["small_image"])
        candidate["ocr_conf"] = ocr_conf
        candidate["text_len"] = text_len
        candidate["final_score"] = candidate["heuristic_score"] + 3.0 * ocr_conf + 1.5 * text_len

    best = max(candidates, key=lambda candidate: candidate["final_score"])
    zero_candidate = next((candidate for candidate in candidates if candidate["angle"] == 0), None)

    if (
        best["angle"] == 180
        and zero_candidate is not None
        and (best["final_score"] - zero_candidate["final_score"]) <= ANGLE_180_KEEP_UPRIGHT_DELTA
    ):
        best = zero_candidate
        override_reason = "Preferred 0 degrees because 180-degree score advantage was too small."

    return best["image"], {
        "best_angle": best["angle"],
        "override_reason": override_reason,
        "candidates": [{
            "angle": c["angle"],
            "heuristic_score": c["heuristic_score"],
            "ocr_conf": c["ocr_conf"],
            "text_len": c["text_len"],
            "used_ocr": True,
            "final_score": c["final_score"],
        } for c in candidates]
    }
