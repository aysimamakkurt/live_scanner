import cv2
import numpy as np
import re

try:
    import pytesseract
except ImportError:  # pragma: no cover - environment-dependent fallback
    pytesseract = None


OCR_CANDIDATE_BRANCHES = [
    "24_gray_whitened_scan",
    "32_bw_shadow_adaptive",

]

VISUAL_CANDIDATE_BRANCHES = [
    "10_color_natural_scan",
    "11_color_strong_scan",
    "00_original_bgr",
]


def _prepare_for_tesseract(img):
    if img is None:
        return None

    if len(img.shape) == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img.copy()

    h, w = gray.shape[:2]
    max_dim = 1000
    scale = min(max_dim / max(h, w), 1.0)

    if scale < 1.0:
        gray = cv2.resize(
            gray,
            (int(w * scale), int(h * scale)),
            interpolation=cv2.INTER_AREA,
        )

    return gray


def quick_branch_score(img) -> float:
    proc = _prepare_for_tesseract(img)
    if proc is None:
        return float("-inf")

    lap_var = float(cv2.Laplacian(proc, cv2.CV_64F).var())
    stddev = float(np.std(proc))
    mean_val = float(np.mean(proc))

    contrast_score = stddev
    sharpness_score = np.log1p(max(lap_var, 0.0))
    exposure_penalty = abs(mean_val - 180.0) / 30.0

    return sharpness_score + 0.35 * contrast_score - exposure_penalty


def quick_visual_score(img) -> float:
    if img is None:
        return float("-inf")

    if len(img.shape) == 2:
        bgr = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    else:
        bgr = img

    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    lap_var = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    stddev = float(np.std(gray))
    mean_val = float(np.mean(gray))

    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    sat_mean = float(np.mean(hsv[:, :, 1]))

    sharpness_score = np.log1p(max(lap_var, 0.0))
    contrast_score = 0.25 * stddev
    exposure_penalty = abs(mean_val - 190.0) / 35.0
    saturation_bonus = min(sat_mean / 65.0, 1.1)
    saturation_penalty = max(sat_mean - 70.0, 0.0) / 45.0

    return sharpness_score + contrast_score + saturation_bonus - exposure_penalty - saturation_penalty


def is_valid_ocr_token(token: str) -> bool:
    token = token.strip()
    if not token:
        return False

    # It must contain at least one alphanumeric character to be considered valid
    if not re.search(r"[A-Za-z0-9]", token):
        return False

    # If the token is made up of only non-alphanumeric characters, it's likely garbage
    if re.fullmatch(r"[\W_]+", token):
        return False

    # If the token consists of only repeated characters, it's likely garbage
    if re.fullmatch(r"[Il1|]{4,}", token):
        return False

    return True


def ocr_readability_score(img):
    try:
        if pytesseract is None:
            return 0.0, {
                "mean_conf": 0.0,
                "text_len": 0,
                "token_count": 0,
                "raw_token_count": 0,
                "valid_token_count": 0,
                "valid_ratio": 0.0,
                "garbage_ratio": 1.0,
                "density_score": 0.0,
                "length_score": 0.0,
                "token_score": 0.0,
                "ocr_available": False,
            }

        proc = _prepare_for_tesseract(img)
        if proc is None:
            return 0.0, {
                "mean_conf": 0.0,
                "text_len": 0,
                "token_count": 0,
                "valid_token_count": 0,
                "valid_ratio": 0.0,
                "garbage_ratio": 1.0,
                "density_score": 0.0,
                "length_score": 0.0,
                "token_score": 0.0,
                "ocr_available": pytesseract is not None,
            }

        data = pytesseract.image_to_data(
            proc,
            config="--psm 3",
            output_type=pytesseract.Output.DICT,
        )

        confs = []
        text_len = 0
        token_count = 0
        raw_token_count = 0
        valid_token_count = 0

        for txt, conf in zip(data["text"], data["conf"]):
            txt = txt.strip()

            try:
                conf = float(conf)
            except Exception:
                conf = -1.0

            if txt:
                raw_token_count += 1

            if txt and conf > 0:
                token_count += 1
                text_len += len(txt)

                if is_valid_ocr_token(txt):
                    valid_token_count += 1
                    confs.append(conf)

        mean_conf = float(np.mean(confs)) if confs else 0.0

        valid_ratio = (
            valid_token_count / max(raw_token_count, 1)
        )
        garbage_ratio = 1.0 - valid_ratio

        img_area = float(proc.shape[0] * proc.shape[1])
        density = text_len / max(img_area, 1.0)
        density_score = density * 1e6

        # Logarithmic scaling for length and token counts to prevent extreme values from dominating the score
        length_score = np.log1p(text_len)
        token_score = np.log1p(valid_token_count)

        # Final score
        score = (
            3.5 * mean_conf
            + 12.0 * valid_ratio
            + 1.5 * length_score
            + 2.0 * token_score
            + 0.8 * density_score
            - 10.0 * garbage_ratio
        )

        return float(score), {
            "mean_conf": mean_conf,
            "text_len": text_len,
            "token_count": token_count,
            "raw_token_count": raw_token_count,
            "valid_token_count": valid_token_count,
            "valid_ratio": valid_ratio,
            "garbage_ratio": garbage_ratio,
            "density_score": density_score,
            "length_score": float(length_score),
            "token_score": float(token_score),
            "ocr_available": True,
        }

    except Exception:
        return 0.0, {
            "mean_conf": 0.0,
            "text_len": 0,
            "token_count": 0,
            "raw_token_count": 0,
            "valid_token_count": 0,
            "valid_ratio": 0.0,
            "garbage_ratio": 1.0,
            "density_score": 0.0,
            "length_score": 0.0,
            "token_score": 0.0,
            "ocr_available": pytesseract is not None,
        }


def auto_select_enhance(enhance_outputs: dict):
    best_branch = None
    best_score = float("-inf")
    debug = {}
    branch_rankings = []

    for branch in OCR_CANDIDATE_BRANCHES:
        if branch not in enhance_outputs:
            continue
        quick_score = quick_branch_score(enhance_outputs[branch])
        branch_rankings.append((quick_score, branch))

    branch_rankings.sort(reverse=True)
    shortlisted_branches = [branch for _, branch in branch_rankings[:2]]

    for branch in shortlisted_branches:
        img = enhance_outputs[branch]
        score, metrics = ocr_readability_score(img)
        if not metrics.get("ocr_available", True):
            score = quick_branch_score(img)

        debug[branch] = {
            "score": score,
            "quick_score": next(q for q, b in branch_rankings if b == branch),
            **metrics,
        }

        if score > best_score:
            best_score = score
            best_branch = branch

    if best_branch is None:
        raise ValueError("No valid enhance branch found.")

    original_branch = "00_original_bgr"
    original_score = None
    if original_branch in enhance_outputs:
        original_score, original_metrics = ocr_readability_score(enhance_outputs[original_branch])
        debug[original_branch] = {
            "score": original_score,
            "quick_score": quick_branch_score(enhance_outputs[original_branch]),
            **original_metrics,
        }
        if original_score >= best_score:
            best_branch = original_branch
            best_score = original_score

    return {
        "ocr_image": enhance_outputs[best_branch],
        "visual_image": enhance_outputs[original_branch],
    }, {
        "selected_ocr_branch": best_branch,
        "selected_visual_branch": original_branch,
        "visual_scores": {
            original_branch: float(quick_visual_score(enhance_outputs[original_branch]))
        } if original_branch in enhance_outputs else {},
        "scores": debug,
    }



