from fastapi import FastAPI, UploadFile, File, Form
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from ultralytics import YOLO
import numpy as np
import cv2
import base64

from scanner.corners import extract_document_corners_safe_hybrid
from scanner.segmentation import run_yolo_segmentation
from scanner.quality import assess_image_quality
from scanner.warp import warp_from_corners
from scanner.orientation import auto_orient_document
from scanner.enhance import enhance_document
from scanner.auto_enhance import auto_select_enhance

app = FastAPI()

# Load YOLO model once at startup
model = YOLO("weights/best.pt")

app.mount("/static", StaticFiles(directory="static"), name="static")


@app.get("/")
def home():
    return FileResponse("static/index.html")


def decode_upload_to_image(image_bytes: bytes):
    np_arr = np.frombuffer(image_bytes, np.uint8)
    return cv2.imdecode(np_arr, cv2.IMREAD_COLOR)


def encode_image_to_data_url(image, mime_type="image/jpeg", quality=95):
    extension = ".jpg" if mime_type == "image/jpeg" else ".png"
    encode_params = []
    if mime_type == "image/jpeg":
      encode_params = [cv2.IMWRITE_JPEG_QUALITY, quality]

    ok, buffer = cv2.imencode(extension, image, encode_params)
    if not ok:
        return None

    encoded = base64.b64encode(buffer.tobytes()).decode("ascii")
    return f"data:{mime_type};base64,{encoded}"


def detect_document_geometry(image):
    if image is None:
        return None

    mask_result = run_yolo_segmentation(model, image, conf=0.35, return_results=True)
    best_mask, results = mask_result
    if best_mask is None:
        return None

    corners = extract_document_corners_safe_hybrid(best_mask)
    if corners is None:
        return None

    corners = corners.astype(np.float32)
    xs = corners[:, 0]
    ys = corners[:, 1]
    bbox = [
        float(np.min(xs)),
        float(np.min(ys)),
        float(np.max(xs)),
        float(np.max(ys)),
    ]

    box_result = results[0].boxes[0] if results and results[0].boxes is not None and len(results[0].boxes) else None
    conf = float(box_result.conf[0]) if box_result is not None else 1.0
    cls = int(box_result.cls[0]) if box_result is not None else 0
    label = model.names.get(cls, str(cls))

    return {
        "bbox": bbox,
        "polygon": corners.tolist(),
        "quad": corners.tolist(),
        "confidence": conf,
        "class_id": cls,
        "label": label,
    }


@app.post("/detect-frame")
async def detect_frame(file: UploadFile = File(...)):
    image_bytes = await file.read()
    image = decode_upload_to_image(image_bytes)

    if image is None:
        return {"success": False, "error": "Invalid image"}

    detection = detect_document_geometry(image)

    return {
        "success": True,
        "detections": [detection] if detection is not None else []
    }


@app.post("/capture-document")
async def capture_document(
    file: UploadFile = File(...),
    apply_orientation: bool = Form(True),
    apply_enhance: bool = Form(True),
):
    image_bytes = await file.read()
    image = decode_upload_to_image(image_bytes)

    if image is None:
        return {"success": False, "error": "Invalid image"}

    detection = detect_document_geometry(image)
    if detection is None:
        return {"success": False, "error": "Document boundaries not found"}

    warped = warp_from_corners(image, np.array(detection["quad"], dtype=np.float32))
    quality = assess_image_quality(warped)

    if quality["is_blurry"]:
        return {
            "success": False,
            "error": "Photo is too blurry. Please retake it.",
            "error_code": "BLURRY_IMAGE",
            "quality": quality,
        }

    raw_warped_data_url = encode_image_to_data_url(warped, mime_type="image/jpeg", quality=95)
    if raw_warped_data_url is None:
        return {"success": False, "error": "Warp encoding failed"}

    processed = warped.copy()
    orientation = None
    enhanced_ocr_data_url = None
    enhancement = None

    if apply_orientation:
        processed, orientation = auto_orient_document(processed)

    if apply_enhance:
        enhanced_outputs = enhance_document(processed, include_visual=False)
        selected_outputs, enhancement = auto_select_enhance(enhanced_outputs)

        enhanced_ocr_data_url = encode_image_to_data_url(selected_outputs["ocr_image"], mime_type="image/png")

        if enhanced_ocr_data_url is None:
            return {"success": False, "error": "Enhanced image encoding failed"}

    processed_data_url = encode_image_to_data_url(processed, mime_type="image/jpeg", quality=95)
    if processed_data_url is None:
        return {"success": False, "error": "Processed image encoding failed"}

    return {
        "success": True,
        "detection": detection,
        "quality": quality,
        "raw_warped_image": raw_warped_data_url,
        "warped_image": processed_data_url,
        "orientation": orientation,
        "enhancement": enhancement,
        "enhanced_ocr_image": enhanced_ocr_data_url,
    }
