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


@lru_cache(maxsize=1)
def get_model() -> YOLO:
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model not found at: {MODEL_PATH}")
    return YOLO(str(MODEL_PATH))


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
    boxes = results[0].boxes

    detections = []
    for box in boxes:
        class_id = int(box.cls[0])
        detections.append(
            {
                "class": model.names[class_id],
                "confidence": float(box.conf[0]),
                "bbox": [float(value) for value in box.xyxy[0].tolist()],
            }
        )

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

    model = get_model()
    results = model.predict(source=image, conf=0.25)
    boxes = results[0].boxes

    detections = []
    for box in boxes:
        class_id = int(box.cls[0])
        detections.append(
            {
                "class": model.names[class_id],
                "confidence": float(box.conf[0]),
                "bbox": [float(value) for value in box.xyxy[0].tolist()],
            }
        )

    severity_report = generate_severity_report(detections, image.width, image.height)

    return {
        # "predictions": detections,
        "severity_report": severity_report,
        "count": len(detections),
    }