from train.cost_estimation import estimate_cost
import os
import sys
from pathlib import Path
from functools import lru_cache

from ultralytics import YOLO
from PIL import Image, UnidentifiedImageError
from fastapi import FastAPI, File, HTTPException, UploadFile

app = FastAPI()
MODEL_VERSION = "1.0.0"
BASE_DIR = Path(__file__).resolve().parent
REPO_ROOT = BASE_DIR.parent

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from train.severity import generate_severity_report

MODEL_PATH = REPO_ROOT / "runs" / "damage" / "weights" / "best.pt"
PART_MODEL_PATH = REPO_ROOT / "runs" / "parts" / "weights" / "best.pt"


@lru_cache(maxsize=1)
def get_model() -> YOLO:
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Damage model not found at: {MODEL_PATH}")
    return YOLO(str(MODEL_PATH))


@lru_cache(maxsize=1)
def get_part_model() -> YOLO | None:
    if not PART_MODEL_PATH.exists():
        return None
    return YOLO(str(PART_MODEL_PATH))


def boxes_to_rows(boxes, names) -> list[dict]:
    """Convert YOLO boxes to list of detection dicts."""
    rows = []
    for box in boxes:
        class_id = int(box.cls[0])
        rows.append({
            "class": names[class_id],
            "confidence": float(box.conf[0]),
            "bbox": [float(v) for v in box.xyxy[0].tolist()],
        })
    return rows


async def read_image_upload(file: UploadFile) -> tuple[Image.Image, bytes]:
    content = await file.read()
    if not content:
        raise HTTPException(status_code=400, detail="Uploaded file is empty")

    try:
        from io import BytesIO

        image = Image.open(BytesIO(content)).convert("RGB")
    except UnidentifiedImageError as exc:
        raise HTTPException(status_code=400, detail="Invalid image file") from exc

    return image, content

@app.get("/")
def home():
    return "Welcome to Insurance Premium Prediction API"

@app.get("/health")
def health_check():
    return {"status": "healthy",
            "version": MODEL_VERSION,
            }
    
@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    content_type = file.content_type or ""
    if content_type not in ["image/jpeg", "image/png", "image/webp"]:
        raise HTTPException(status_code=400, detail="Unsupported file type. Please upload a JPEG, PNG, or WEBP image.")

    content = await file.read()
    if not content:
        raise HTTPException(status_code=400, detail="Uploaded file is empty")

    return {
        "file_info": {
            "filename": file.filename,
            "content_type": content_type,
            "size": len(content),
        }
    }

@app.post("/upload/predict")
async def upload_and_predict(file: UploadFile = File(...)):
    content_type = file.content_type or ""
    if content_type not in ["image/jpeg", "image/png", "image/webp"]:
        raise HTTPException(status_code=400, detail="Unsupported file type. Please upload a JPEG, PNG, or WEBP image.")

    image, content = await read_image_upload(file)

    model = get_model()
    results = model.predict(source=image, conf=0.25)
    detections = boxes_to_rows(results[0].boxes, model.names)

    return {
        "predictions": detections,
        "count": len(detections),
    }
    
@app.post("/upload/severity")
async def upload_and_predict_severity(file: UploadFile = File(...)):
    content_type = file.content_type or ""
    if content_type not in ["image/jpeg", "image/png", "image/webp"]:
        raise HTTPException(status_code=400, detail="Unsupported file type. Please upload a JPEG, PNG, or WEBP image.")

    image, content = await read_image_upload(file)

    # ---- Damage detection ----
    model = get_model()
    results = model.predict(source=image, conf=0.25, imgsz=640)
    detections = boxes_to_rows(results[0].boxes, model.names)

    # ---- Part detection ----
    part_detections = []
    part_model = get_part_model()
    if part_model is not None:
        part_results = part_model.predict(source=image, conf=0.25, imgsz=640)
        part_detections = boxes_to_rows(part_results[0].boxes, part_model.names)

    # ---- Severity report ----
    severity_report = generate_severity_report(
        detections, image.width, image.height, part_detections
    )
    severity_report.pop("damage_table", None)

    return {
        "severity_report": severity_report,
        "count": len(detections),
    }

@app.post("/upload/cost-estimation")
async def upload_and_estimate_cost(file: UploadFile = File(...)):
    """
    Upload + auto-detect damage → severity → damage_table → cost_estimation.
    Returns:
      - severity_report
      - cost_estimation (breakdown per part)
      - total_estimated_cost
    """
    content_type = file.content_type or ""
    if content_type not in ["image/jpeg", "image/png", "image/webp"]:
        raise HTTPException(
            status_code=400,
            detail="Unsupported file type. Please upload a JPEG, PNG, or WEBP image.",
        )

    image, content = await read_image_upload(file)

    # ---- Damage detection ----
    model = get_model()
    results = model.predict(source=image, conf=0.25, imgsz=640)
    detections = boxes_to_rows(results[0].boxes, model.names)

    # ---- Part detection ----
    part_detections = []
    part_model = get_part_model()
    if part_model is not None:
        part_results = part_model.predict(source=image, conf=0.25, imgsz=640)
        part_detections = boxes_to_rows(part_results[0].boxes, part_model.names)

    # ---- Severity + part severity ----
    severity_report = generate_severity_report(
        detections, image.width, image.height, part_detections
    )

    part_severity = severity_report.get("part_severity", {})
    if not part_severity:
        cost_report = {"line_items": [], "parts_total": 0.0, "labor_total": 0.0, "grand_total": 0.0, "skipped_parts": []}
    else:
        cost_report = estimate_cost(part_severity)

    # ---- Final response ----
    response = {
       #S "severity_report": severity_report,
        "cost_estimation": cost_report,
    }

    return response