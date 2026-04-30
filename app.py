from fastapi import FastAPI, UploadFile, File
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from ultralytics import YOLO
import numpy as np
import cv2
import base64

from scanner.corners import extract_document_corners_safe_hybrid
from scanner.segmentation import choose_best_mask_from_results
from scanner.warp import warp_from_corners

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

    results = model(image, conf=0.35, verbose=False)
    h, w = image.shape[:2]
    best_mask = choose_best_mask_from_results(results, w, h)

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
async def capture_document(file: UploadFile = File(...)):
    image_bytes = await file.read()
    image = decode_upload_to_image(image_bytes)

    if image is None:
        return {"success": False, "error": "Invalid image"}

    detection = detect_document_geometry(image)
    if detection is None:
        return {"success": False, "error": "Document boundaries not found"}

    warped = warp_from_corners(image, np.array(detection["quad"], dtype=np.float32))
    warped_data_url = encode_image_to_data_url(warped, mime_type="image/jpeg", quality=95)

    if warped_data_url is None:
        return {"success": False, "error": "Warp encoding failed"}

    return {
        "success": True,
        "detection": detection,
        "warped_image": warped_data_url,
    }
