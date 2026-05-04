from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import cv2

from scanner.auto_enhance import (
    OCR_CANDIDATE_BRANCHES,
    VISUAL_CANDIDATE_BRANCHES,
    ocr_readability_score,
    quick_branch_score,
    quick_visual_score,
)
from scanner.enhance import enhance_document
from scanner.orientation import auto_orient_document


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def load_image(path: Path):
    return cv2.imread(str(path), cv2.IMREAD_COLOR)


def save_image(path: Path, image):
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), image)


def discover_images(input_path: Path) -> list[Path]:
    if input_path.is_file():
        return [input_path]
    return sorted(
        path for path in input_path.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )


def evaluate_image(image_path: Path, output_dir: Path, apply_orientation: bool) -> dict:
    image = load_image(image_path)
    if image is None:
        raise ValueError(f"Could not read image: {image_path}")

    working = image.copy()
    orientation_info = None
    if apply_orientation:
        working, orientation_info = auto_orient_document(working)

    outputs = enhance_document(working)

    image_output_dir = output_dir / image_path.stem
    image_output_dir.mkdir(parents=True, exist_ok=True)

    branch_scores = {}

    for branch_name, branch_image in outputs.items():
        extension = ".png" if len(branch_image.shape) == 2 else ".jpg"
        save_image(image_output_dir / f"{branch_name}{extension}", branch_image)

        entry = {}
        if branch_name in OCR_CANDIDATE_BRANCHES or branch_name == "00_original_bgr":
            quick_score = quick_branch_score(branch_image)
            ocr_score, ocr_metrics = ocr_readability_score(branch_image)
            entry["quick_ocr_score"] = float(quick_score)
            entry["ocr_score"] = float(ocr_score)
            entry["ocr_metrics"] = ocr_metrics

        if branch_name in VISUAL_CANDIDATE_BRANCHES:
            entry["visual_score"] = float(quick_visual_score(branch_image))

        branch_scores[branch_name] = entry

    ranked_ocr = sorted(
        (
            (branch_name, data["ocr_score"])
            for branch_name, data in branch_scores.items()
            if "ocr_score" in data
        ),
        key=lambda item: item[1],
        reverse=True,
    )
    ranked_visual = sorted(
        (
            (branch_name, data["visual_score"])
            for branch_name, data in branch_scores.items()
            if "visual_score" in data
        ),
        key=lambda item: item[1],
        reverse=True,
    )

    result = {
        "image": str(image_path),
        "orientation": orientation_info,
        "selected_ocr_branch": ranked_ocr[0][0] if ranked_ocr else None,
        "selected_visual_branch": ranked_visual[0][0] if ranked_visual else None,
        "ocr_ranking": [
            {"branch": branch_name, "score": float(score)}
            for branch_name, score in ranked_ocr
        ],
        "visual_ranking": [
            {"branch": branch_name, "score": float(score)}
            for branch_name, score in ranked_visual
        ],
        "branch_scores": branch_scores,
    }

    with (image_output_dir / "scores.json").open("w", encoding="utf-8") as handle:
        json.dump(result, handle, ensure_ascii=True, indent=2)

    return result


def build_summary(results: list[dict]) -> dict:
    visual_counter = Counter(
        result["selected_visual_branch"]
        for result in results
        if result["selected_visual_branch"]
    )
    ocr_counter = Counter(
        result["selected_ocr_branch"]
        for result in results
        if result["selected_ocr_branch"]
    )

    return {
        "image_count": len(results),
        "selected_visual_counts": dict(visual_counter),
        "selected_ocr_counts": dict(ocr_counter),
        "recommended_visual_branch": visual_counter.most_common(1)[0][0] if visual_counter else None,
        "recommended_ocr_branch": ocr_counter.most_common(1)[0][0] if ocr_counter else None,
        "results": results,
    }


def main():
    parser = argparse.ArgumentParser(description="Evaluate document enhancement branches on sample images.")
    parser.add_argument("input", help="Image file or directory containing sample images.")
    parser.add_argument(
        "--output-dir",
        default="enhance_eval",
        help="Directory where branch renders and score reports are written.",
    )
    parser.add_argument(
        "--no-orientation",
        action="store_true",
        help="Skip auto orientation before enhancement.",
    )
    args = parser.parse_args()

    input_path = Path(args.input)
    output_dir = Path(args.output_dir)
    images = discover_images(input_path)

    if not images:
        raise SystemExit(f"No images found in {input_path}")

    results = []
    for image_path in images:
        print(f"Evaluating {image_path} ...")
        results.append(
            evaluate_image(
                image_path=image_path,
                output_dir=output_dir,
                apply_orientation=not args.no_orientation,
            )
        )

    summary = build_summary(results)
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=True, indent=2)

    print(json.dumps(
        {
            "image_count": summary["image_count"],
            "recommended_visual_branch": summary["recommended_visual_branch"],
            "recommended_ocr_branch": summary["recommended_ocr_branch"],
            "selected_visual_counts": summary["selected_visual_counts"],
            "selected_ocr_counts": summary["selected_ocr_counts"],
        },
        ensure_ascii=True,
        indent=2,
    ))


if __name__ == "__main__":
    main()
