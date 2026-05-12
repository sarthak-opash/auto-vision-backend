"""
Detection Pipeline: Unified Damage & Parts Detection with IoU-based Overlap
Purpose: Core inference engine for damage-part associations
Author: Senior Computer Vision Engineer
Version: 2.0.0
"""

import sys
import logging
import threading
import numpy as np
from pathlib import Path
from functools import lru_cache
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass, field, asdict
from concurrent.futures import ThreadPoolExecutor

from PIL import Image
from ultralytics import YOLO

# ─── Setup logging ─────────────────────────────────────────────────────────
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ─── Check CUDA Availability ──────────────────────────────────────────────────
try:
    import torch
    CUDA_AVAILABLE = torch.cuda.is_available()
    CUDA_DEVICE_COUNT = torch.cuda.device_count() if CUDA_AVAILABLE else 0
    logger.info(f"🎮 CUDA Available: {CUDA_AVAILABLE} | Device Count: {CUDA_DEVICE_COUNT}")
except ImportError:
    CUDA_AVAILABLE = False
    CUDA_DEVICE_COUNT = 0
    logger.warning("⚠️  PyTorch not available for CUDA check")
@dataclass
class BoundingBox:
    """Normalized bounding box representation."""
    x1: float  # top-left x
    y1: float  # top-left y
    x2: float  # bottom-right x
    y2: float  # bottom-right y

    def area(self) -> float:
        """Calculate box area."""
        return (self.x2 - self.x1) * (self.y2 - self.y1)

    def to_list(self) -> List[float]:
        """Convert to list format."""
        return [self.x1, self.y1, self.x2, self.y2]


@dataclass
class Detection:
    """Single detection result."""
    class_name: str
    class_id: int
    confidence: float
    bbox: BoundingBox
    mask: Optional[np.ndarray] = None  # For segmentation models

    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            "class": self.class_name,
            "class_id": self.class_id,
            "confidence": float(self.confidence),
            "bbox": self.bbox.to_list(),
        }


@dataclass
class DamagePartAssociation:
    """Association between damage and car part."""
    damage_detection: Detection
    part_detection: Detection
    iou: float  # Intersection over Union
    overlap_percentage: float  # Percentage of damage overlapping with part

    def to_dict(self) -> Dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "damage": {
                "class": self.damage_detection.class_name,
                "confidence": float(self.damage_detection.confidence),
                "bbox": self.damage_detection.bbox.to_list(),
            },
            "part": {
                "class": self.part_detection.class_name,
                "confidence": float(self.part_detection.confidence),
                "bbox": self.part_detection.bbox.to_list(),
            },
            "iou": float(self.iou),
            "overlap_percentage": float(self.overlap_percentage),
        }


@dataclass
class PipelineOutput:
    """Structured output from detection pipeline."""
    damage_detections: List[Detection] = field(default_factory=list)
    part_detections: List[Detection] = field(default_factory=list)
    associations: List[DamagePartAssociation] = field(default_factory=list)
    image_shape: Tuple[int, int] = (0, 0)  # (height, width)
    processing_time_ms: float = 0.0

    def to_dict(self) -> Dict:
        """Convert to dictionary for API responses."""
        return {
            "damage_detections": [d.to_dict() for d in self.damage_detections],
            "part_detections": [p.to_dict() for p in self.part_detections],
            "damage_part_associations": [a.to_dict() for a in self.associations],
            "summary": {
                "total_damages": len(self.damage_detections),
                "total_parts_detected": len(self.part_detections),
                "total_associations": len(self.associations),
                "image_size": list(self.image_shape),
            },
            "processing_time_ms": self.processing_time_ms,
        }


# ─── Utility Functions ───────────────────────────────────────────────────────
def calculate_iou(box1: BoundingBox, box2: BoundingBox) -> float:
    """
    Calculate Intersection over Union (IoU) between two bounding boxes.
    
    Args:
        box1: First bounding box
        box2: Second bounding box
    
    Returns:
        IoU value (0.0 to 1.0)
    """
    # Calculate intersection area
    inter_x1 = max(box1.x1, box2.x1)
    inter_y1 = max(box1.y1, box2.y1)
    inter_x2 = min(box1.x2, box2.x2)
    inter_y2 = min(box1.y2, box2.y2)

    if inter_x2 < inter_x1 or inter_y2 < inter_y1:
        return 0.0  # No overlap

    inter_area = (inter_x2 - inter_x1) * (inter_y2 - inter_y1)
    
    # Calculate union area
    box1_area = box1.area()
    box2_area = box2.area()
    union_area = box1_area + box2_area - inter_area

    if union_area == 0:
        return 0.0

    return inter_area / union_area


def calculate_overlap_percentage(damage_box: BoundingBox, part_box: BoundingBox) -> float:
    """
    Calculate percentage of damage box overlapping with part box.
    
    Args:
        damage_box: Damage bounding box
        part_box: Part bounding box
    
    Returns:
        Percentage (0.0 to 100.0) of damage overlapping with part
    """
    # Calculate intersection area
    inter_x1 = max(damage_box.x1, part_box.x1)
    inter_y1 = max(damage_box.y1, part_box.y1)
    inter_x2 = min(damage_box.x2, part_box.x2)
    inter_y2 = min(damage_box.y2, part_box.y2)

    if inter_x2 < inter_x1 or inter_y2 < inter_y1:
        return 0.0  # No overlap

    inter_area = (inter_x2 - inter_x1) * (inter_y2 - inter_y1)
    damage_area = damage_box.area()

    if damage_area == 0:
        return 0.0

    return (inter_area / damage_area) * 100.0


def parse_yolo_results(results, model_names: Dict[int, str]) -> List[Detection]:
    """
    Parse YOLO detection results into Detection objects.
    
    Args:
        results: YOLO prediction results
        model_names: Mapping of class_id to class_name
    
    Returns:
        List of Detection objects
    """
    detections = []
    
    if results.boxes is None or len(results.boxes) == 0:
        return detections

    for box in results.boxes:
        class_id = int(box.cls[0])
        class_name = model_names.get(class_id, f"unknown_{class_id}")
        confidence = float(box.conf[0])
        
        # Normalize coordinates (0-1 range)
        img_height, img_width = results.orig_shape
        x1, y1, x2, y2 = box.xyxy[0].tolist()
        
        bbox = BoundingBox(
            x1=x1 / img_width,
            y1=y1 / img_height,
            x2=x2 / img_width,
            y2=y2 / img_height,
        )
        
        # Extract mask if available (segmentation models)
        mask = None
        if results.masks is not None and hasattr(box, 'masks'):
            mask = box.masks
        
        detection = Detection(
            class_name=class_name,
            class_id=class_id,
            confidence=confidence,
            bbox=bbox,
            mask=mask,
        )
        detections.append(detection)

    return detections


# ─── Main Pipeline Class ──────────────────────────────────────────────────────
class DamageDetectionPipeline:
    """
    Unified pipeline for damage and parts detection with IoU-based association.
    
    Workflow:
        1. Load both models (damage & parts)
        2. Run inference on image
        3. Calculate IoU between all damage-part pairs
        4. Return structured results with associations
    """

    DAMAGE_MODEL_NAMES = {
        0: "dent",
        1: "scratch",
        2: "crack",
        3: "glass shatter",
        4: "lamp broken",
        5: "tire flat",
    }

    PART_MODEL_NAMES = {
        0: "Hood",
        1: "Headlight",
        2: "Gas_clip",
        3: "Vent Grill",
        4: "Window",
        5: "Taillight",
        6: "Windshield",
        7: "Rear Bumper",
        8: "Front Bumper",
        9: "Fender",
        10: "Front Fender",
        11: "Rear Fender",
        12: "License Plate",
        13: "Tire",
        14: "Door",
        15: "Side Mirror",
        16: "Wheel",
        17: "Gas Cap",
        18: "Trunk Lid",
        19: "Roof",
        20: "Rear Glass",
        21: "Front Glass",
        22: "Emblem",
        23: "Fog Lamp",
        24: "Tailgate",
    }

    def __init__(
        self,
        damage_model_path: str,
        part_model_path: str,
        confidence_threshold: float = 0.25,
        iou_threshold: float = 0.1,  # Minimum IoU to consider association
        min_overlap_percentage: float = 50.0,  # Strict cutoff: only overlap > 50% is accepted
        device: str = "0",  # GPU device ID or "cpu"
        use_parallel_inference: bool = True,
        use_half_precision: bool = True,  # FP16 for faster inference
    ):
        """
        Initialize the detection pipeline with GPU optimization.
        
        Args:
            damage_model_path: Path to damage detection model weights
            part_model_path: Path to parts detection model weights
            confidence_threshold: Minimum confidence for detections
            iou_threshold: Minimum IoU to consider damage-part association
            min_overlap_percentage: Minimum accepted overlap percentage
            device: GPU device ID (e.g., "0", "1") or "cpu"
            use_parallel_inference: Run both models concurrently
            use_half_precision: Use FP16 precision for faster inference
        """
        self.damage_model_path = Path(damage_model_path)
        self.part_model_path = Path(part_model_path)
        self.confidence_threshold = confidence_threshold
        self.iou_threshold = iou_threshold
        self.min_overlap_percentage = min_overlap_percentage
        
        # Convert and validate device
        if isinstance(device, str):
            if device.lower() == "cpu":
                self.device = "cpu"
            else:
                # Try to convert to int (e.g., "0" -> 0)
                try:
                    device_id = int(device)
                    # Check if CUDA is available
                    if not CUDA_AVAILABLE:
                        logger.warning(
                            f"⚠️  CUDA not available. Falling back to CPU (requested device={device_id})"
                        )
                        self.device = "cpu"
                    elif device_id >= CUDA_DEVICE_COUNT:
                        logger.warning(
                            f"⚠️  GPU device {device_id} not available (only {CUDA_DEVICE_COUNT} devices). "
                            f"Falling back to CPU"
                        )
                        self.device = "cpu"
                    else:
                        self.device = device_id
                except ValueError:
                    logger.error(f"❌ Invalid device: {device}. Using CPU.")
                    self.device = "cpu"
        else:
            # Integer device
            if not CUDA_AVAILABLE:
                logger.warning(
                    f"⚠️  CUDA not available. Falling back to CPU (requested device={device})"
                )
                self.device = "cpu"
            elif device >= CUDA_DEVICE_COUNT:
                logger.warning(
                    f"⚠️  GPU device {device} not available (only {CUDA_DEVICE_COUNT} devices). "
                    f"Falling back to CPU"
                )
                self.device = "cpu"
            else:
                self.device = device
        
        self.use_parallel_inference = use_parallel_inference
        self.use_half_precision = use_half_precision

        self._damage_model = None
        self._part_model = None
        self._executor = ThreadPoolExecutor(max_workers=2) if use_parallel_inference else None

        device_type = "🎮 GPU" if isinstance(self.device, int) else "💻 CPU"
        logger.info(
            f"Pipeline initialized: "
            f"{device_type} device={self.device}, "
            f"confidence={confidence_threshold}, "
            f"iou_threshold={iou_threshold}, "
            f"parallel={use_parallel_inference}, "
            f"half_precision={use_half_precision}"
        )

    @property
    def damage_model(self) -> YOLO:
        """Lazy load damage model."""
        if self._damage_model is None:
            if not self.damage_model_path.exists():
                raise FileNotFoundError(
                    f"Damage model not found: {self.damage_model_path}"
                )
            logger.info(f"Loading damage model from {self.damage_model_path}")
            self._damage_model = YOLO(str(self.damage_model_path))
            device_type = "🎮 GPU" if isinstance(self.device, int) else "💻 CPU"
            logger.info(f"✅ Damage model ready ({device_type} device={self.device})")
        return self._damage_model

    @property
    def part_model(self) -> Optional[YOLO]:
        """Lazy load parts model."""
        if self._part_model is None:
            if not self.part_model_path.exists():
                logger.warning(f"Parts model not found: {self.part_model_path}")
                return None
            logger.info(f"Loading parts model from {self.part_model_path}")
            self._part_model = YOLO(str(self.part_model_path))
            device_type = "🎮 GPU" if isinstance(self.device, int) else "💻 CPU"
            logger.info(f"✅ Parts model ready ({device_type} device={self.device})")
        return self._part_model

    def detect_damages(
        self, image: Image.Image, imgsz: int = 640
    ) -> List[Detection]:
        """
        Detect damage types in image with optimized inference.
        
        Args:
            image: PIL Image
            imgsz: Input size for model
        
        Returns:
            List of Detection objects
        """
        # Only use FP16 on GPU (not beneficial on CPU)
        use_half = self.use_half_precision and isinstance(self.device, int)
        
        results = self.damage_model.predict(
            source=image,
            conf=self.confidence_threshold,
            imgsz=imgsz,
            device=self.device,
            half=use_half,
            verbose=False,
            augment=False,  # Disable for speed
            agnostic_nms=False,
        )
        
        detections = parse_yolo_results(results[0], self.damage_model.names)
        logger.info(f"🔴 Damages: {len(detections)} detected")
        return detections

    def detect_parts(
        self, image: Image.Image, imgsz: int = 640
    ) -> List[Detection]:
        """
        Detect car parts in image with optimized inference.
        
        Args:
            image: PIL Image
            imgsz: Input size for model
        
        Returns:
            List of Detection objects
        """
        if self.part_model is None:
            logger.warning("Parts model not available")
            return []

        # Only use FP16 on GPU (not beneficial on CPU)
        use_half = self.use_half_precision and isinstance(self.device, int)
        
        results = self.part_model.predict(
            source=image,
            conf=self.confidence_threshold,
            imgsz=imgsz,
            device=self.device,
            half=use_half,
            verbose=False,
            augment=False,  # Disable for speed
            agnostic_nms=False,
        )
        
        detections = parse_yolo_results(results[0], self.part_model.names)
        logger.info(f"🔵 Parts: {len(detections)} detected")
        return detections

    def associate_damages_to_parts(
        self,
        damages: List[Detection],
        parts: List[Detection],
    ) -> List[DamagePartAssociation]:
        """
        Find damage-part associations using IoU.
        
        Args:
            damages: List of damage detections
            parts: List of part detections
        
        Returns:
            List of associations sorted by IoU (descending)
        """
        associations = []

        for damage in damages:
            for part in parts:
                iou = calculate_iou(damage.bbox, part.bbox)
                
                if iou < self.iou_threshold:
                    continue

                overlap_pct = calculate_overlap_percentage(
                    damage.bbox, part.bbox
                )

                if overlap_pct <= self.min_overlap_percentage:
                    continue

                association = DamagePartAssociation(
                    damage_detection=damage,
                    part_detection=part,
                    iou=iou,
                    overlap_percentage=overlap_pct,
                )
                associations.append(association)

        # Sort by IoU (descending) - strongest associations first
        associations.sort(key=lambda x: x.iou, reverse=True)
        logger.info(f"Found {len(associations)} damage-part associations")
        return associations

    def process(
        self, image: Image.Image, imgsz: int = 640
    ) -> PipelineOutput:
        """
        Run complete pipeline with PARALLEL model inference.
        
        Workflow:
            1. Detect damages (GPU)
            2. Detect parts (GPU) - PARALLEL with damage detection
            3. Associate them using IoU
        
        Args:
            image: PIL Image (RGB)
            imgsz: Input size for models
        
        Returns:
            PipelineOutput with all detections and associations
        """
        import time

        start_time = time.time()

        # Ensure image is RGB
        if image.mode != "RGB":
            image = image.convert("RGB")

        # ─── RUN MODELS IN PARALLEL ──────────────────────────────────────
        if self.use_parallel_inference and self._executor:
            # Submit both inference tasks to thread pool
            damage_future = self._executor.submit(self.detect_damages, image, imgsz)
            parts_future = self._executor.submit(self.detect_parts, image, imgsz)
            
            # Wait for both to complete
            damages = damage_future.result()
            parts = parts_future.result()
            
            logger.info("✅ Parallel inference completed")
        else:
            # Fallback to sequential (slower)
            damages = self.detect_damages(image, imgsz=imgsz)
            parts = self.detect_parts(image, imgsz=imgsz)
            logger.info("⚠️ Sequential inference (parallel disabled)")

        # Associate damages to parts
        associations = self.associate_damages_to_parts(damages, parts)

        # Calculate processing time
        elapsed_ms = (time.time() - start_time) * 1000

        output = PipelineOutput(
            damage_detections=damages,
            part_detections=parts,
            associations=associations,
            image_shape=image.size[::-1],  # Convert (W, H) to (H, W)
            processing_time_ms=elapsed_ms,
        )

        logger.info(
            f"🚀 Pipeline completed in {elapsed_ms:.0f}ms | "
            f"Damages: {len(damages)} | Parts: {len(parts)} | "
            f"Associations: {len(associations)}"
        )

        return output

    def process_from_path(self, image_path: str) -> PipelineOutput:
        """
        Process image from file path.
        
        Args:
            image_path: Path to image file
        
        Returns:
            PipelineOutput
        """
        image = Image.open(image_path).convert("RGB")
        return self.process(image)


# ─── Convenience Functions ────────────────────────────────────────────────────
def get_pipeline(
    damage_model_path: str,
    part_model_path: str,
    confidence_threshold: float = 0.25,
    iou_threshold: float = 0.1,
    min_overlap_percentage: float = 50.0,
    device: str = "0",
    use_parallel_inference: bool = True,
    use_half_precision: bool = True,
) -> DamageDetectionPipeline:
    """
    Factory function to create an optimized pipeline.
    
    Args:
        damage_model_path: Path to damage model
        part_model_path: Path to parts model
        confidence_threshold: Detection confidence threshold
        iou_threshold: IoU threshold for associations
        device: GPU device ID or "cpu"
        use_parallel_inference: Run both models in parallel
        use_half_precision: Use FP16 precision for speed
    
    Returns:
        Configured DamageDetectionPipeline
    """
    return DamageDetectionPipeline(
        damage_model_path=damage_model_path,
        part_model_path=part_model_path,
        confidence_threshold=confidence_threshold,
        iou_threshold=iou_threshold,
        min_overlap_percentage=min_overlap_percentage,
        device=device,
        use_parallel_inference=use_parallel_inference,
        use_half_precision=use_half_precision,
    )
