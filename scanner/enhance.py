import cv2
import numpy as np


# =========================================================
# BASIC HELPERS
# =========================================================

def gray_world_white_balance(img):
    """Apply Gray-World white balance on BGR image."""
    img_f = img.astype(np.float32) + 1e-6
    b, g, r = cv2.split(img_f)

    mean_b = np.mean(b)
    mean_g = np.mean(g)
    mean_r = np.mean(r)
    mean_gray = (mean_b + mean_g + mean_r) / 3.0

    b *= mean_gray / mean_b
    g *= mean_gray / mean_g
    r *= mean_gray / mean_r

    out = cv2.merge([b, g, r])
    return np.clip(out, 0, 255).astype(np.uint8)


def illumination_normalization(img_or_gray, sigma=35, target=180.0):
    """Normalize low-frequency illumination variation."""
    if len(img_or_gray.shape) == 3:
        gray = cv2.cvtColor(img_or_gray, cv2.COLOR_BGR2GRAY)
    else:
        gray = img_or_gray.copy()

    bg = cv2.GaussianBlur(gray, (0, 0), sigmaX=sigma, sigmaY=sigma)
    gray_f = gray.astype(np.float32)
    bg_f = bg.astype(np.float32) + 1.0

    norm = (gray_f / bg_f) * target
    return np.clip(norm, 0, 255).astype(np.uint8)


def contrast_enhancement_clahe(gray, clip=2.5, tile=(8, 8)):
    """Apply CLAHE on grayscale image."""
    clahe = cv2.createCLAHE(clipLimit=clip, tileGridSize=tile)
    return clahe.apply(gray)


def shadow_reduction(gray, kernel_size=25, target=200.0):
    """Reduce shadow by estimating background using morphology."""
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size))
    bg = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, kernel)

    gray_f = gray.astype(np.float32)
    bg_f = bg.astype(np.float32) + 1.0

    out = (gray_f / bg_f) * target
    return np.clip(out, 0, 255).astype(np.uint8)


def denoise_document(gray, h=10):
    """Lightweight document denoising to keep preprocessing fast."""
    kernel = 3 if h <= 7 else 5
    return cv2.GaussianBlur(gray, (kernel, kernel), 0)


def denoise_document_colored(img, h=7, h_color=7):
    """Lightweight color denoising for faster scan generation."""
    return cv2.GaussianBlur(img, (3, 3), 0)


def detect_glare_regions(img):
    """Detect glare / specular reflection regions."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    _, bright = cv2.threshold(gray, 245, 255, cv2.THRESH_BINARY)

    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    _, s, v = cv2.split(hsv)

    low_sat = cv2.threshold(s, 40, 255, cv2.THRESH_BINARY_INV)[1]
    high_val = cv2.threshold(v, 240, 255, cv2.THRESH_BINARY)[1]

    glare = cv2.bitwise_and(low_sat, high_val)
    glare = cv2.bitwise_or(glare, bright)

    kernel = np.ones((5, 5), np.uint8)
    glare = cv2.morphologyEx(glare, cv2.MORPH_CLOSE, kernel, iterations=2)
    glare = cv2.morphologyEx(glare, cv2.MORPH_OPEN, kernel, iterations=1)
    return glare


def inpaint_glare(img, glare_mask):
    """Inpaint glare areas."""
    if glare_mask is None or np.count_nonzero(glare_mask) == 0:
        return img.copy()
    return cv2.inpaint(img, glare_mask, 5, cv2.INPAINT_TELEA)


def sharpen_image(gray, sigma=1.2, alpha=1.6, beta=-0.6):
    """Unsharp-mask style sharpening for grayscale."""
    blur = cv2.GaussianBlur(gray, (0, 0), sigmaX=sigma)
    sharp = cv2.addWeighted(gray, alpha, blur, beta, 0)
    return np.clip(sharp, 0, 255).astype(np.uint8)


def sharpen_color_image(img, sigma=1.0, alpha=1.35, beta=-0.35):
    """Mild sharpening for color image."""
    blur = cv2.GaussianBlur(img, (0, 0), sigmaX=sigma)
    sharp = cv2.addWeighted(img, alpha, blur, beta, 0)
    return np.clip(sharp, 0, 255).astype(np.uint8)


def adaptive_thresholding(gray, block_size=31, C=15):
    """Adaptive Gaussian threshold for document binarization."""
    return cv2.adaptiveThreshold(
        gray,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        block_size,
        C,
    )


def sauvola_binarization(gray, window_size=25, k=0.2, R=128):
    """Sauvola local thresholding."""
    gray_f = gray.astype(np.float32)

    mean = cv2.boxFilter(
        gray_f, ddepth=-1, ksize=(window_size, window_size), normalize=True
    )
    mean_sq = cv2.boxFilter(
        gray_f * gray_f, ddepth=-1, ksize=(window_size, window_size), normalize=True
    )

    variance = mean_sq - (mean * mean)
    variance = np.maximum(variance, 0)
    std = np.sqrt(variance)

    thresh = mean * (1 + k * ((std / R) - 1))
    return ((gray_f > thresh).astype(np.uint8) * 255)


# =========================================================
# NEW HELPERS FOR SCAN-LIKE RENDERING
# =========================================================

def normalize_paper_background(gray, sigma=45, white_target=245.0):
    """
    Flatten document background so paper looks more uniform and scan-like.
    This is useful for 'clean scanned page' rendering.
    """
    bg = cv2.GaussianBlur(gray, (0, 0), sigmaX=sigma, sigmaY=sigma)
    gray_f = gray.astype(np.float32)
    bg_f = bg.astype(np.float32) + 1.0

    norm = (gray_f / bg_f) * white_target
    return np.clip(norm, 0, 255).astype(np.uint8)


def stretch_contrast(gray, low_perc=2, high_perc=98):
    """Apply percentile-based contrast stretching."""
    lo = np.percentile(gray, low_perc)
    hi = np.percentile(gray, high_perc)

    if hi <= lo:
        return gray.copy()

    out = (gray.astype(np.float32) - lo) * (255.0 / (hi - lo))
    return np.clip(out, 0, 255).astype(np.uint8)


def enhance_l_channel_in_lab(img_bgr):
    """
    Enhance only luminance in LAB, preserving color information.
    This gives a more natural 'scanned document' look than grayscale-only processing.
    """
    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)

    l1 = illumination_normalization(l, sigma=31, target=190.0)
    l2 = contrast_enhancement_clahe(l1, clip=2.2, tile=(8, 8))
    l3 = denoise_document(l2, h=7)
    l4 = sharpen_image(l3, sigma=1.0, alpha=1.35, beta=-0.35)

    merged = cv2.merge([l4, a, b])
    out = cv2.cvtColor(merged, cv2.COLOR_LAB2BGR)
    return out


def whiten_background_preserve_text(gray):
    """
    Make the paper look cleaner without destroying dark text strokes too much.
    """
    norm = normalize_paper_background(gray, sigma=45, white_target=240.0)
    stretched = stretch_contrast(norm, low_perc=1, high_perc=99)
    return stretched


# =========================================================
# MAIN ENHANCEMENT PIPELINE
# =========================================================

def enhance_document(warped_bgr: np.ndarray) -> dict:
    """
    Produce multiple scan-like renderings from a single document image.

    Outputs include:
    - natural color scan
    - strong color scan
    - clean grayscale scan
    - paper-whitened scan
    - OCR-friendly black/white outputs
    """
    outputs = {}

    # -----------------------------------------------------
    # Stage 0: original
    # -----------------------------------------------------
    outputs["00_original_bgr"] = warped_bgr.copy()

    # -----------------------------------------------------
    # Stage 1: base cleanup on color image
    # -----------------------------------------------------
    wb = gray_world_white_balance(warped_bgr)
    glare_mask = detect_glare_regions(wb)

    glare_fixed = inpaint_glare(wb, glare_mask)

    color_denoised = denoise_document_colored(glare_fixed, h=6, h_color=6)

    base_gray = cv2.cvtColor(color_denoised, cv2.COLOR_BGR2GRAY)

    # -----------------------------------------------------
    # Stage 2: color scan branches
    # -----------------------------------------------------
    # Natural-looking color scan
    color_natural = enhance_l_channel_in_lab(color_denoised)
    color_natural = sharpen_color_image(color_natural, sigma=1.0, alpha=1.15, beta=-0.15)
    outputs["10_color_natural_scan"] = color_natural

    # Stronger color document branch
    color_lab = cv2.cvtColor(color_denoised, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(color_lab)
    l = normalize_paper_background(l, sigma=41, white_target=235.0)
    l = contrast_enhancement_clahe(l, clip=2.8, tile=(8, 8))
    l = sharpen_image(l, sigma=1.0, alpha=1.4, beta=-0.4)
    color_strong = cv2.cvtColor(cv2.merge([l, a, b]), cv2.COLOR_LAB2BGR)
    outputs["11_color_strong_scan"] = color_strong

    # -----------------------------------------------------
    # Stage 3: grayscale scan branches
    # -----------------------------------------------------
    gray_illum = illumination_normalization(base_gray, sigma=35, target=185.0)
    outputs["20_gray_illumination"] = gray_illum

    gray_clahe = contrast_enhancement_clahe(base_gray, clip=2.5, tile=(8, 8))
    outputs["22_gray_clahe"] = gray_clahe

    # Paper-whitened branch: more scanner-like visual output
    gray_white = whiten_background_preserve_text(base_gray)
    gray_white = denoise_document(gray_white, h=6)
    gray_white = sharpen_image(gray_white, sigma=1.0, alpha=1.3, beta=-0.3)
    outputs["24_gray_whitened_scan"] = gray_white

    # -----------------------------------------------------
    # Stage 4: OCR / binary branches
    # -----------------------------------------------------
    gray_shadow = shadow_reduction(base_gray, kernel_size=25, target=200.0)
    bw_shadow = adaptive_thresholding(gray_shadow, block_size=31, C=10)
    outputs["32_bw_shadow_adaptive"] = bw_shadow

    # -----------------------------------------------------
    # Stage 5: recommended outputs
    # -----------------------------------------------------
    outputs["90_recommended_visual"] = color_natural
    outputs["91_recommended_gray"] = gray_white
    outputs["92_recommended_ocr"] = bw_shadow

    return outputs
