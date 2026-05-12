import os
import sys
from pathlib import Path
from functools import lru_cache

from ultralytics import YOLO
from PIL import Image, UnidentifiedImageError, ImageDraw, ImageFont
from fastapi import FastAPI, File, Form, HTTPException, UploadFile, Response

app = FastAPI()

from fastapi.middleware.cors import CORSMiddleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://localhost:5174",
        "http://127.0.0.1:5173",
        "http://127.0.0.1:5174",
    ],
    allow_methods=["*"],
    allow_headers=["*"],
)
 

MODEL_VERSION = "1.0.0"
BASE_DIR = Path(__file__).resolve().parent
REPO_ROOT = BASE_DIR.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from train.cost_estimation import estimate_cost
from train.severity import generate_severity_report
from train.vehicle_catalog import (
    ALL_MAKES,
    MAKE_MODEL_MAP,
    VEHICLE_CATALOG,
    get_vehicle_info,
    SEVERITY_PART_MAP,
    get_models_for_make,
    lookup_vehicle_price,
)
from train.report import generate_report
from inference.detection_pipeline import DamageDetectionPipeline

MODEL_PATH = REPO_ROOT / "runs" / "damage" / "weights" / "best.pt"
PART_MODEL_PATH = REPO_ROOT / "runs" / "parts" / "weights" / "best.pt"

# ─── Initialize Unified Pipeline ──────────────────────────────────────────────
# @lru_cache(maxsize=1)
# def get_pipeline() -> DamageDetectionPipeline:
#     """Initialize the damage detection pipeline with GPU optimization."""
#     return DamageDetectionPipeline(
#         damage_model_path=str(MODEL_PATH),
#         part_model_path=str(PART_MODEL_PATH),
#         confidence_threshold=0.25,
#         iou_threshold=0.1,
#         device="0",  # GPU device ID
#         use_parallel_inference=True,  # ✅ Run both models in parallel
#         use_half_precision=True,  # ✅ FP16 precision for faster inference
#     )


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


def draw_themed_annotations(image: Image.Image, detections: list[dict]) -> Image.Image:
    """Draw boxes and labels that match the frontend theme (#60176F)."""
    # Create a copy so we don't mutate the original
    img_copy = image.copy().convert("RGBA")
    draw = ImageDraw.Draw(img_copy)
    
    try:
        # Use a standard font if possible
        font = ImageFont.truetype("arial.ttf", 20)
    except:
        font = ImageFont.load_default()

    brand_color = (152, 66, 22, 255)  # #984216
    
    for det in detections:
        x1, y1, x2, y2 = det["bbox"]
        label = det["class"].replace("-", " ").upper()
        conf = det["confidence"]
        
        # 1. Draw Bounding Box (Border only, no fill)
        draw.rectangle([x1, y1, x2, y2], outline=brand_color, width=3)
        
        # 2. Draw Label Background
        text_str = f" {label} {conf:.0%} "
        
        # Calculate text size using textbbox (modern PIL)
        try:
            t_l, t_t, t_r, t_b = draw.textbbox((x1, y1), text_str, font=font)
            text_w = t_r - t_l
            text_h = t_b - t_t
        except AttributeError:
            # Fallback for very old PIL
            text_w, text_h = draw.textsize(text_str, font=font)
            t_l, t_t, t_r, t_b = x1, y1, x1 + text_w, y1 + text_h

        # Position label above the box if there's space, else inside
        label_y = y1 - (text_h + 8) if y1 > 30 else y1
        draw.rectangle([x1, label_y, x1 + text_w, label_y + text_h + 8], fill=brand_color)
        
        # 3. Draw Text
        draw.text((x1, label_y + 2), text_str, fill=(255, 255, 255, 255), font=font)
        
    return img_copy.convert("RGB")

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

    return {
        "severity_report": severity_report,
        "count": len(detections),
    }

@app.post("/upload/cost-estimation")
async def upload_and_estimate_cost(
    file:  UploadFile = File(...),
    make:  str = Form(default=""),
    model: str = Form(default=""),
    year:  int = Form(default=0),
):
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
    damage_model = get_model()
    results = damage_model.predict(source=image, conf=0.25, imgsz=640)
    detections = boxes_to_rows(results[0].boxes, damage_model.names)

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
        cost_report = {"line_items": [], "parts_total": 0.0, "labor_total": 0.0, "grand_total": 0.0, "skipped_parts": [], "vehicle_info": {}}
    else:
        vehicle_info = None
        if make and model and make.lower() != "unknown" and model.lower() != "unknown":
            vehicle_info = {"make": make, "model": model, "year": year}
        cost_report = estimate_cost(part_severity, vehicle_info=vehicle_info)

    # ---- Final response ----
    response = {
       #S "severity_report": severity_report,
        "cost_estimation": cost_report,
    }

    return response 


@app.post("/upload/report")
async def upload_and_generate_report(
    file:  UploadFile = File(...),
    make:  str = Form(default=""),
    model: str = Form(default=""),
    year:  int = Form(default=0),
):
    """
    Complete pipeline: Detect → Severity → Cost → PDF Report.
    Returns: PDF binary stream.
    """
    image, content = await read_image_upload(file)

    # 1. Damage Detection
    damage_model = get_model()
    damage_results = damage_model.predict(source=image, conf=0.25, imgsz=640)
    detections = boxes_to_rows(damage_results[0].boxes, damage_model.names)

    # 2. Part Detection
    part_detections = []
    part_model = get_part_model()
    if part_model is not None:
        part_results = part_model.predict(source=image, conf=0.25, imgsz=640)
        part_detections = boxes_to_rows(part_results[0].boxes, part_model.names)

    # 3. Severity Analysis
    severity_report = generate_severity_report(
        detections, image.width, image.height, part_detections
    )

    # 4. Cost Estimation
    vehicle_info = None
    if make and model and make.lower() != "unknown" and model.lower() != "unknown":
        vehicle_info = {"make": make, "model": model, "year": year}
    
    cost_report = estimate_cost(severity_report.get("part_severity", {}), vehicle_info=vehicle_info)

    # 5. Save Annotated Image temporarily for PDF
    import tempfile
    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
        annot_path = tmp.name
    
    # Use our custom themed annotations instead of YOLO default
    themed_img = draw_themed_annotations(image, detections)
    themed_img.save(annot_path, quality=95)

    # 6. Generate PDF
    try:
        pdf_bytes = generate_report(
            severity_result=severity_report,
            cost_result=cost_report,
            annotated_image_path=annot_path,
            vehicle_info=vehicle_info,
        )
    finally:
        if os.path.exists(annot_path):
            os.remove(annot_path)

    return Response(
        content=pdf_bytes,
        media_type="application/pdf",
        headers={"Content-Disposition": "attachment; filename=damage_report.pdf"}
    )


@app.post("/upload/full-scan")
async def upload_full_scan(
    files: list[UploadFile] = File(...),
    make:  str = Form(default=""),
    model: str = Form(default=""),
    year:  int = Form(default=0),
):
    """
    Multi-angle scan: Process 4-5 images and return a single aggregated report.
    """
    if len(files) < 4:
        raise HTTPException(
            status_code=400,
            detail="Full Scan requires 4 images: Front, Rear, Left, and Right.",
        )

    all_detections = []
    all_damage_rows = []
    aggregated_part_severity = {}
    total_detections_count = 0
    
    # We'll use these to build an aggregated cost report
    combined_part_severity_for_cost = {}

    for i, file in enumerate(files):
        image, content = await read_image_upload(file)
        
        # 1. Damage Detection
        damage_model = get_model()
        damage_results = damage_model.predict(source=image, conf=0.25, imgsz=640)
        detections = boxes_to_rows(damage_results[0].boxes, damage_model.names)
        
        # Attach image index to each detection for multi-view correlation
        for det in detections:
            det["image_index"] = i
            
        total_detections_count += len(detections)
        all_detections.extend(detections)

        # 2. Part Detection
        part_detections = []
        part_model = get_part_model()
        if part_model is not None:
            part_results = part_model.predict(source=image, conf=0.25, imgsz=640)
            part_detections = boxes_to_rows(part_results[0].boxes, part_model.names)

        # 3. Severity Analysis for THIS image
        severity_report = generate_severity_report(
            detections, image.width, image.height, part_detections
        )
        
        # 4. Merge part_severity
        # If a part exists in multiple images, we take the one with the higher severity score
        for key, info in severity_report.get("part_severity", {}).items():
            if key not in aggregated_part_severity or info["severity_score"] > aggregated_part_severity[key]["severity_score"]:
                aggregated_part_severity[key] = info
                combined_part_severity_for_cost[key] = info

        # 5. Preserve detailed per-image rows for reporting
        for row in severity_report.get("damage_table", []):
            enriched_row = dict(row)
            enriched_row["image_index"] = i
            all_damage_rows.append(enriched_row)

    # 6. Cost Estimation for ALL aggregated parts
    vehicle_info = None
    if make and model and make.lower() != "unknown" and model.lower() != "unknown":
        vehicle_info = {"make": make, "model": model, "year": year}
    
    cost_report = estimate_cost(combined_part_severity_for_cost, vehicle_info=vehicle_info)

    # 7. Final aggregated severity summary
    final_severity_score = 0
    if aggregated_part_severity:
        scores = [info["severity_score"] for info in aggregated_part_severity.values()]
        # Simple aggregation: 70% of max + 30% average
        final_severity_score = 0.7 * max(scores) + 0.3 * (sum(scores) / len(scores))
        final_severity_score = min(100.0, final_severity_score)

    return {
        "severity_report": {
            "severity_score": round(final_severity_score, 2),
            "severity_level": generate_severity_report([], 0, 0).get("severity_level", "Low") if not aggregated_part_severity else (
                "Low" if final_severity_score < 25 else "Medium" if final_severity_score < 50 else "High" if final_severity_score < 75 else "Critical"
            ),
            "detected_parts": list(set([info["part"] for info in aggregated_part_severity.values()])),
            "part_severity": aggregated_part_severity,
            "damage_table": all_damage_rows,  # Detailed rows for reports
            "detection_boxes": all_detections,  # Raw boxes for visual overlays
        },
        "cost_estimation": cost_report,
        "count": total_detections_count,
    }


@app.get("/vehicle-catalog")
def get_vehicle_catalog_legacy():
    """Returns the vehicle catalog in the format expected by the frontend."""
    catalog_list = []
    # Sort makes alphabetically for a better UI experience
    sorted_makes = sorted(VEHICLE_CATALOG.items())
    
    for make_name, models in sorted_makes:
        make_item = {
            "name": make_name.title(),
            "models": []
        }
        # Sort models alphabetically
        sorted_models = sorted(models.items())
        for model_name, model_data in sorted_models:
            make_item["models"].append({
                "name": model_name.replace("-", " ").title(),
                "year_start": model_data["years"][0],
                "year_end": model_data["years"][1],
                "segment": model_data["segment"].title()
            })
        catalog_list.append(make_item)
        
    return {"catalog": catalog_list}


# ─────────────────────────────────────────────────────────────────────────────
#  VEHICLE CATALOG ENDPOINTS
# ─────────────────────────────────────────────────────────────────────────────

@app.get("/catalog/makes")
def get_makes():
    """Returns a list of all available vehicle makes."""
    return {"makes": ALL_MAKES}

@app.get("/catalog/models")
def get_models(make: str):
    """Returns a list of models for a given make."""
    # Normalize to Title Case as used in MAKE_MODEL_MAP keys
    models = get_models_for_make(make.title())
    return {"make": make, "models": models}

@app.get("/catalog/vehicle-info")
def get_info(make: str, model: str):
    """Returns segment and year range for a specific make and model."""
    info = get_vehicle_info(make, model)
    if not info:
        raise HTTPException(status_code=404, detail="Vehicle not found in catalog")
    return {"make": make, "model": model, **info}

@app.get("/catalog/data")
def get_full_catalog():
    """Returns the full internal vehicle catalog mapping."""
    return VEHICLE_CATALOG

@app.get("/catalog/map")
def get_catalog_map():
    """Returns the human-readable make-to-model mapping."""
    return MAKE_MODEL_MAP

@app.get("/catalog/parts")
def get_parts(make: str, model: str):
    """Returns the full part price list for a specific vehicle."""
    # We use the same normalization as the internal catalog
    make_key = make.lower().replace(" ", "-")
    model_key = model.lower().replace(" ", "-")
    vehicle = VEHICLE_CATALOG.get(make_key, {}).get(model_key)
    if not vehicle:
        raise HTTPException(status_code=404, detail="Vehicle not found in catalog")
    return {
        "make": make,
        "model": model,
        "segment": vehicle["segment"],
        "parts": vehicle["parts"]
    }

@app.get("/catalog/lookup-price")
def get_part_price(make: str, model: str, year: int, part: str):
    """Looks up the price for a specific part on a specific vehicle."""
    price = lookup_vehicle_price(make, model, year, part)
    if price is None:
        raise HTTPException(status_code=404, detail="Price not found for given vehicle/part")
    return {"make": make, "model": model, "year": year, "part": part, "price": price}

@app.get("/catalog/severity-map")
def get_severity_map():
    """Returns the mapping from detection labels to catalog parts."""
    return SEVERITY_PART_MAP


# # ─────────────────────────────────────────────────────────────────────────────
# #  UNIFIED DAMAGE DETECTION PIPELINE ENDPOINTS (NEW)
# # ─────────────────────────────────────────────────────────────────────────────

# @app.post("/pipeline/detect")
# async def pipeline_detect(file: UploadFile = File(...)):
#     """
#     Unified pipeline: Detect damages AND parts with overlap analysis.
    
#     Returns:
#       - damage_detections: List of detected damages
#       - part_detections: List of detected car parts
#       - damage_part_associations: Damage-part overlaps with IoU scores
#       - summary: Detection counts and statistics
#       - processing_time_ms: Inference time
#     """
#     image, _ = await read_image_upload(file)
    
#     pipeline = get_pipeline()
#     result = pipeline.process(image, imgsz=640)
    
#     return result.to_dict()


# @app.post("/pipeline/severity-with-parts")
# async def pipeline_severity_with_parts(file: UploadFile = File(...)):
#     """
#     Pipeline + Severity: Detect damages/parts and return severity per part.
    
#     Returns:
#       - all detection data
#       - part-specific severity scores
#     """
#     image, _ = await read_image_upload(file)
    
#     pipeline = get_pipeline()
#     result = pipeline.process(image, imgsz=640)
    
#     # Map associations to part-severity
#     part_severity_map = {}
#     for assoc in result.associations:
#         part_name = assoc.part_detection.class_name
#         if part_name not in part_severity_map:
#             part_severity_map[part_name] = {
#                 "part": part_name,
#                 "confidence": assoc.part_detection.confidence,
#                 "damages": [],
#                 "max_overlap": 0.0,
#             }
        
#         part_severity_map[part_name]["damages"].append({
#             "type": assoc.damage_detection.class_name,
#             "confidence": assoc.damage_detection.confidence,
#             "iou": assoc.iou,
#             "overlap_percentage": assoc.overlap_percentage,
#         })
#         part_severity_map[part_name]["max_overlap"] = max(
#             part_severity_map[part_name]["max_overlap"],
#             assoc.overlap_percentage,
#         )
    
#     return {
#         "detection_results": result.to_dict(),
#         "part_severity": part_severity_map,
#         "total_affected_parts": len(part_severity_map),
#     }


# @app.post("/pipeline/cost-estimation-v2")
# async def pipeline_cost_estimation_v2(
#     file: UploadFile = File(...),
#     make: str = Form(default=""),
#     model: str = Form(default=""),
#     year: int = Form(default=0),
# ):
#     """
#     Pipeline-based cost estimation using damage-part associations.
    
#     More accurate than previous version as it uses IoU-weighted damage severity.
#     """
#     image, _ = await read_image_upload(file)
    
#     pipeline = get_pipeline()
#     result = pipeline.process(image, imgsz=640)
    
#     # Build part-severity from associations
#     part_severity_map = {}
#     for assoc in result.associations:
#         part_name = assoc.part_detection.class_name
#         if part_name not in part_severity_map:
#             part_severity_map[part_name] = {
#                 "part": part_name,
#                 "severity_score": 0.0,
#                 "damage_types": set(),
#             }
        
#         # Weight severity by overlap percentage and damage type
#         damage_type = assoc.damage_detection.class_name
#         iou_weight = assoc.iou
        
#         # Damage severity scoring (customize based on domain knowledge)
#         damage_scores = {
#             "tire flat": 80,
#             "lamp broken": 70,
#             "glass shatter": 65,
#             "crack": 50,
#             "scratch": 30,
#             "dent": 40,
#         }
#         base_score = damage_scores.get(damage_type.lower(), 40)
#         weighted_score = base_score * iou_weight
        
#         part_severity_map[part_name]["severity_score"] = max(
#             part_severity_map[part_name]["severity_score"],
#             weighted_score,
#         )
#         part_severity_map[part_name]["damage_types"].add(damage_type)
    
#     # Convert damage_types set to list for serialization
#     for part_info in part_severity_map.values():
#         part_info["damage_types"] = list(part_info["damage_types"])
    
#     # Use existing cost estimation
#     vehicle_info = None
#     if make and model and make.lower() != "unknown" and model.lower() != "unknown":
#         vehicle_info = {"make": make, "model": model, "year": year}
    
#     cost_report = estimate_cost(part_severity_map, vehicle_info=vehicle_info)
    
#     return {
#         "detection_results": result.to_dict(),
#         "part_severity": part_severity_map,
#         "cost_estimation": cost_report,
#     }


# @app.post("/pipeline/debug")
# async def pipeline_debug(file: UploadFile = File(...)):
#     """
#     Debug endpoint: Return all intermediate detection data.
    
#     Useful for model validation and hyperparameter tuning.
#     """
#     image, _ = await read_image_upload(file)
    
#     pipeline = get_pipeline()
#     result = pipeline.process(image, imgsz=640)
    
#     return {
#         "raw_damages": [d.to_dict() for d in result.damage_detections],
#         "raw_parts": [p.to_dict() for p in result.part_detections],
#         "associations": [a.to_dict() for a in result.associations],
#         "summary": result.to_dict()["summary"],
#     }

