import cv2
import numpy as np


DEFAULT_LAPLACIAN_THRESHOLD = 140.0
DEFAULT_TENENGRAD_THRESHOLD = 1800.0


def _prepare_grayscale(image, max_dimension=1200):
    if image is None:
        return None

    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()

    height, width = gray.shape[:2]
    longest_side = max(height, width)

    if longest_side > max_dimension:
        scale = max_dimension / float(longest_side)
        gray = cv2.resize(
            gray,
            (max(1, int(width * scale)), max(1, int(height * scale))),
            interpolation=cv2.INTER_AREA,
        )

    return gray


def compute_blur_score(image, max_dimension=1200):
    """Return a Laplacian-variance sharpness score for the given image."""
    gray = _prepare_grayscale(image, max_dimension=max_dimension)
    if gray is None:
        return 0.0

    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def compute_tenengrad_score(image, max_dimension=1200):
    """Return a Tenengrad sharpness score based on Sobel gradient energy."""
    gray = _prepare_grayscale(image, max_dimension=max_dimension)
    if gray is None:
        return 0.0

    gx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    return float(np.mean(gx ** 2 + gy ** 2))


def assess_image_quality(
    image,
    laplacian_threshold=DEFAULT_LAPLACIAN_THRESHOLD,
    tenengrad_threshold=DEFAULT_TENENGRAD_THRESHOLD,
):
    blur_score = compute_blur_score(image)
    tenengrad_score = compute_tenengrad_score(image)
    laplacian_pass = blur_score >= laplacian_threshold
    tenengrad_pass = tenengrad_score >= tenengrad_threshold

    return {
        "is_blurry": not (laplacian_pass and tenengrad_pass),
        "blur_score": blur_score,
        "blur_threshold": float(laplacian_threshold),
        "laplacian_score": blur_score,
        "laplacian_threshold": float(laplacian_threshold),
        "laplacian_pass": laplacian_pass,
        "tenengrad_score": tenengrad_score,
        "tenengrad_threshold": float(tenengrad_threshold),
        "tenengrad_pass": tenengrad_pass,
    }
