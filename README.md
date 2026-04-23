# AutoClaim Vision - AI Model Implementation Plan

## Executive Summary

**Project**: AutoClaim Vision - AI-Powered Vehicle Damage Detection & Insurance Cost Estimation System  
**Duration**: 10 Weeks  
**Team Size**: 3 Interns  
**Core Technology**: Computer Vision (YOLOv8, EfficientNet, ResNet) + Full-Stack Web Application  
**Primary Goal**: Reduce insurance claim processing time from 7 days to 3 seconds

---

## 📋 Table of Contents

1. [Project Overview](#project-overview)
2. [AI/ML Models Architecture](#aiml-models-architecture)
3. [Phase-by-Phase Implementation](#phase-by-phase-implementation)
4. [Workflow Diagram](#workflow-diagram)
5. [How to Start the Project](#how-to-start-the-project)
6. [Team Roles & Responsibilities](#team-roles--responsibilities)
7. [Critical Success Factors](#critical-success-factors)
8. [Risk Mitigation Strategy](#risk-mitigation-strategy)

---

## 1. Project Overview

### The Problem
- Traditional insurance claims take **3-7 business days** for assessor visit
- Total settlement time: **2-8 weeks**
- High operational cost: **Rs. 2,000-5,000 per assessment**
- Fraud leakage: **10-15% of claims**
- Inconsistent evaluation standards

### The Solution
Upload vehicle damage photos → AI processes in **<3 seconds** → Generate detailed damage report with:
- Localized bounding boxes with pixel-precise masks
- 30+ car part identification (hood, bumper, doors, etc.)
- 8 damage type classifications (scratch, dent, crack, etc.)
- 4-level severity scoring (minor/moderate/severe/total loss)
- Cost estimation range (Rs. X,XXX - Rs. Y,YYY)
- Fraud detection signal (0.00 - 1.00 score)
- Auto-generated PDF report

---

## 2. AI/ML Models Architecture

### 2.1 Seven AI Models in the Pipeline

| Model # | Name | Architecture | Purpose | Input | Output | Target Metric |
|---------|------|--------------|---------|-------|--------|---------------|
| 1 | **Vehicle Detector** | YOLOv8m | Locate vehicle in image | 640×640 RGB | BBox + Confidence | mAP50 > 0.90 |
| 2 | **Damage Segmentor** | YOLOv8x-seg | Core damage detection | 800×800 RGB | Masks + BBox + Class | mAP50 > 0.75 |
| 3 | **Part Classifier** | EfficientNet-B4 | Identify car parts | 224×224 crop | Part label + prob | Top-1 Acc > 80% |
| 4 | **Severity Estimator** | ResNet-50 + custom head | Damage severity | Segment crop | 4-class + score | Weighted F1 > 0.74 |
| 5 | **Make/Model Recognizer** | MobileNetV3 | Vehicle identification | Vehicle crop | Make/Model/Year | Top-1 Acc > 80% |
| 6 | **License Plate OCR** | PaddleOCR | Extract plate text | Plate region | Text string | Word Acc > 95% |
| 7 | **Cost Predictor** | XGBoost | Estimate repair cost | Damage features | Min/Max cost | MAPE < 30% |

### 2.2 Inference Pipeline Flow

```
┌──────────────────────┐
│  User Uploads Photos │
│  (4-6 images)        │
└──────────┬───────────┘
           │
           ▼
┌──────────────────────┐
│  Preprocessing       │
│  (Resize, Normalize) │
└──────────┬───────────┘
           │
           ▼
┌──────────────────────┐
│  Vehicle Detection   │
│  (YOLOv8m)          │ ◄── Model 1
└──────────┬───────────┘
           │
           ▼
┌──────────────────────┐
│  Damage Segmentation │
│  (YOLOv8x-seg)      │ ◄── Model 2
└──────────┬───────────┘
           │
           ├─────────────────────────┐
           │                         │
           ▼                         ▼
┌──────────────────┐      ┌──────────────────┐
│  Part Classifier │      │  Severity Score  │
│  (EfficientNet)  │      │  (ResNet-50)     │
│  Model 3         │      │  Model 4         │
└────────┬─────────┘      └────────┬─────────┘
         │                         │
         └────────┬────────────────┘
                  │
                  ├─────────────────────────┐
                  │                         │
                  ▼                         ▼
        ┌──────────────────┐      ┌──────────────────┐
        │  Vehicle ID      │      │  License Plate   │
        │  (MobileNetV3)   │      │  (PaddleOCR)     │
        │  Model 5         │      │  Model 6         │
        └────────┬─────────┘      └────────┬─────────┘
                 │                         │
                 └────────┬────────────────┘
                          │
                          ▼
                ┌──────────────────────┐
                │  Cost Estimation     │
                │  (Rule Engine +      │
                │   XGBoost Model 7)   │
                └──────────┬───────────┘
                           │
                           ▼
                ┌──────────────────────┐
                │  Fraud Detection     │
                │  (Feature Engineering│
                │   + XGBoost)         │
                └──────────┬───────────┘
                           │
                           ▼
                ┌──────────────────────┐
                │  PDF Report          │
                │  Generation          │
                └──────────────────────┘
```

---

## 3. Phase-by-Phase Implementation

### **PHASE 1: Foundation & Data Collection** (Weeks 1-2)

#### Week 1: Environment Setup & Data Collection

**Goals:**
- Set up complete development environment
- Download and audit public datasets
- Begin custom Indian vehicle photo collection

**Tasks Breakdown:**

**Day 1-2: Development Environment**
```bash
# All team members complete:
1. Install Python 3.10, PyTorch 2.2, CUDA (if GPU available)
2. Install Node.js 20 LTS
3. Install PostgreSQL 15
4. Install Redis 7.2
5. Install Docker Desktop
6. Install Git and configure SSH keys
7. Set up IDE (VS Code / PyCharm)
```

**Day 3-4: Project Structure**
```
autoclaim-vision/
├── backend/              # FastAPI application
│   ├── app/
│   │   ├── api/         # REST endpoints
│   │   ├── core/        # Config, security
│   │   ├── models/      # SQLAlchemy models
│   │   ├── services/    # Business logic
│   │   └── workers/     # Celery tasks
│   ├── tests/
│   └── requirements.txt
├── frontend/             # React application
│   ├── src/
│   │   ├── components/
│   │   ├── pages/
│   │   ├── hooks/
│   │   └── services/
│   └── package.json
├── ml/                   # ML models & training
│   ├── datasets/        # Data management
│   ├── models/          # Model definitions
│   ├── training/        # Training scripts
│   ├── inference/       # Inference pipeline
│   └── notebooks/       # Jupyter notebooks
├── docker/              # Docker configs
│   ├── backend/
│   ├── frontend/
│   └── nginx/
├── scripts/             # Utility scripts
└── docker-compose.yml
```

**Day 5-7: Dataset Acquisition**

Download all public datasets:

1. **CarDD Dataset** (4,000 images)
   - Source: Kaggle / GitHub
   - Content: Car damage, 6 classes, bbox + segmentation
   - Usage: Primary training data for damage detection

2. **VCoR Dataset** (2,000+ images)
   - Source: Kaggle
   - Content: Vehicle collision damage, severity labeled
   - Usage: Severity estimation training

3. **Stanford Cars** (16,185 images)
   - Source: Stanford AI Lab
   - Content: 196 car classes, make/model/year
   - Usage: Vehicle make/model recognition

4. **CarParts-1K** (~5,000 images)
   - Source: Roboflow Universe
   - Content: 30 car part categories
   - Usage: Part classifier training

5. **Open Images V7** (subset ~3,000 images)
   - Source: Google
   - Content: Vehicles with part annotations
   - Usage: Additional part detection data

**Custom Data Collection (Ongoing throughout Week 1):**
- Target: 300-500 images of Indian vehicles
- Focus: Maruti Alto/Swift, Hyundai i20/Creta, Tata Nexon, Mahindra Scorpio, Honda City
- Capture all 8 damage types
- Record metadata: make, model, year, damage type, severity, city

#### Week 2: Annotation & Preprocessing Pipeline

**Day 1-2: Label Studio Setup**
```bash
# Deploy Label Studio
docker run -it -p 8080:8080 \
  -v $(pwd)/ml/datasets:/label-studio/data \
  heartexai/label-studio:latest

# Create annotation project:
# - 8 damage classes: Scratch, Dent, Crack, Shatter, 
#   Deformation, Corrosion, Paint Damage, Missing Part
# - 30 part labels: hood, bumper_front, bumper_rear, 
#   door_fl, door_fr, door_rl, door_rr, etc.
# - Segmentation masks + bounding boxes
```

**Day 3-7: Annotation Work**
- Each intern: 100-150 images (total ~400 annotated)
- Cross-validate 20% of annotations
- Inter-annotator agreement check (IoU > 0.8)
- Quality control: check label distribution, mask validity

**Data Preprocessing Pipeline:**
```python
# ml/datasets/preprocessing.py

import albumentations as A
from sklearn.model_selection import train_test_split

# Augmentation pipeline
train_transform = A.Compose([
    A.RandomRotate90(p=0.3),
    A.HorizontalFlip(p=0.5),
    A.OneOf([
        A.MotionBlur(p=0.2),
        A.GaussNoise(p=0.2),
        A.ISONoise(p=0.2),
    ], p=0.3),
    A.OneOf([
        A.RandomBrightnessContrast(p=0.3),
        A.RandomGamma(p=0.3),
    ], p=0.3),
    A.OneOf([
        A.RainDrops(p=0.2),
        A.ImageCompression(quality_lower=70, p=0.3),
    ], p=0.2),
    A.Resize(800, 800),
    A.Normalize(mean=[0.485, 0.456, 0.406], 
                std=[0.229, 0.224, 0.225]),
])

# Train/Val/Test split: 70/15/15
def split_dataset(annotations, test_size=0.15, val_size=0.15):
    train_val, test = train_test_split(
        annotations, test_size=test_size, random_state=42
    )
    train, val = train_test_split(
        train_val, test_size=val_size/(1-test_size), random_state=42
    )
    return train, val, test

# Export to YOLOv8 format
def export_yolo_format(annotations, output_dir):
    # Create YAML config
    # Copy images and labels to proper structure
    pass
```

**Deliverables:**
- ✅ 400+ annotated images in Label Studio
- ✅ Data quality report
- ✅ Train/Val/Test splits (70/15/15)
- ✅ YOLOv8 format export ready

---

### **PHASE 2: AI/ML Pipeline Development** (Weeks 3-5)

#### Week 3: Damage Detection Model (YOLOv8)

**Day 1-2: Baseline Training**
```python
# ml/training/train_damage_detector.py

from ultralytics import YOLO
import mlflow

# Start MLflow tracking
mlflow.set_experiment("damage_detection")

with mlflow.start_run(run_name="yolov8s_baseline"):
    # Train baseline YOLOv8s
    model = YOLO('yolov8s-seg.pt')
    
    results = model.train(
        data='datasets/damage/data.yaml',
        epochs=50,
        imgsz=640,
        batch=16,
        name='damage_yolov8s_baseline',
        patience=10,
        save=True,
        device=0  # GPU
    )
    
    # Log metrics to MLflow
    mlflow.log_metrics({
        'mAP50': results.results_dict['metrics/mAP50(B)'],
        'mAP50-95': results.results_dict['metrics/mAP50-95(B)'],
        'precision': results.results_dict['metrics/precision(B)'],
        'recall': results.results_dict['metrics/recall(B)']
    })
    
    # Target: mAP50 > 0.55 on validation
```

**Day 3-5: Full YOLOv8x-seg Training**
```python
# Train production model
with mlflow.start_run(run_name="yolov8x_seg_800px"):
    model = YOLO('yolov8x-seg.pt')
    
    results = model.train(
        data='datasets/damage/data.yaml',
        epochs=100,
        imgsz=800,  # Higher resolution
        batch=8,    # Smaller batch for larger model
        name='damage_yolov8x_800',
        optimizer='AdamW',
        lr0=0.001,
        weight_decay=0.0005,
        patience=15,
        save_period=10,
        device=0
    )
    
    # Target: mAP50 > 0.75
```

**Day 6-7: Augmentation Experiments**
- Test different augmentation strategies
- Add rain/blur/JPEG compression overlays
- Measure impact on mAP
- Finalize best augmentation set

**Evaluation:**
```python
# ml/training/evaluate_detector.py

from ultralytics import YOLO

model = YOLO('runs/detect/damage_yolov8x_800/weights/best.pt')

# Validate on test set
metrics = model.val(data='datasets/damage/data.yaml', split='test')

# Generate detailed analysis
print(f"mAP50: {metrics.box.map50:.4f}")
print(f"mAP50-95: {metrics.box.map:.4f}")
print(f"Precision: {metrics.box.mp:.4f}")
print(f"Recall: {metrics.box.mr:.4f}")

# Per-class analysis
for i, class_name in enumerate(metrics.names.values()):
    print(f"{class_name}: mAP50 = {metrics.box.maps[i]:.4f}")

# Confusion matrix
confusion_matrix = metrics.confusion_matrix.matrix
# Identify weak classes for Week 5 improvement
```

#### Week 4: Part Classifier, Severity Model, OCR

**Part Classifier (EfficientNet-B4):**
```python
# ml/training/train_part_classifier.py

import torch
import torch.nn as nn
from torchvision import models
import mlflow

# Load pretrained EfficientNet-B4
model = models.efficientnet_b4(pretrained=True)

# Modify classifier head for 30 car parts
num_parts = 30
model.classifier = nn.Sequential(
    nn.Dropout(p=0.3),
    nn.Linear(model.classifier[1].in_features, num_parts)
)

# Training loop
optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()

for epoch in range(50):
    # Training
    model.train()
    for batch in train_loader:
        images, labels = batch
        outputs = model(images)
        loss = criterion(outputs, labels)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
    # Validation
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for batch in val_loader:
            images, labels = batch
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    accuracy = 100 * correct / total
    print(f'Epoch {epoch+1}: Val Accuracy = {accuracy:.2f}%')
    
    # Target: Top-1 Accuracy > 80%
```

**Severity Estimator (ResNet-50):**
```python
# ml/training/train_severity_model.py

import torch
import torch.nn as nn
from torchvision import models

# 4 severity classes: Minor, Moderate, Severe, Total Loss
num_classes = 4

model = models.resnet50(pretrained=True)
model.fc = nn.Sequential(
    nn.Dropout(p=0.4),
    nn.Linear(model.fc.in_features, 256),
    nn.ReLU(),
    nn.Dropout(p=0.3),
    nn.Linear(256, num_classes)
)

# Use weighted loss for imbalanced data
class_weights = torch.tensor([1.0, 1.5, 2.0, 3.0])  # Adjust based on data
criterion = nn.CrossEntropyLoss(weight=class_weights)

# Training similar to part classifier
# Target: Weighted F1 > 0.75
```

**OCR Integration:**
```python
# ml/inference/ocr_service.py

from paddleocr import PaddleOCR
import cv2

class PlateOCR:
    def __init__(self):
        self.ocr = PaddleOCR(
            use_angle_cls=True,
            lang='en',
            use_gpu=True
        )
    
    def extract_plate(self, image_path):
        result = self.ocr.ocr(image_path, cls=True)
        
        # Extract text from result
        texts = []
        for line in result:
            for word_info in line:
                text = word_info[1][0]
                confidence = word_info[1][1]
                if confidence > 0.8:
                    texts.append(text)
        
        plate_number = ''.join(texts).replace(' ', '')
        return plate_number
```

**Vehicle Make/Model Recognition:**
```python
# ml/training/train_vehicle_classifier.py

from torchvision import models
import torch.nn as nn

# Train on Stanford Cars dataset (196 classes)
model = models.mobilenet_v3_large(pretrained=True)
model.classifier = nn.Sequential(
    nn.Linear(model.classifier[0].in_features, 512),
    nn.Hardswish(),
    nn.Dropout(p=0.2),
    nn.Linear(512, 196)  # 196 car models
)

# Target: Top-1 Accuracy > 80%
```

#### Week 5: End-to-End Pipeline + Cost & Fraud Models

**Inference Pipeline Integration:**
```python
# ml/inference/pipeline.py

import torch
from ultralytics import YOLO
import onnxruntime as ort
import numpy as np

class DamageAssessmentPipeline:
    def __init__(self):
        # Load all models
        self.vehicle_detector = YOLO('models/vehicle_detector.pt')
        self.damage_segmentor = YOLO('models/damage_yolov8x.pt')
        
        # Load ONNX models for faster inference
        self.part_classifier = ort.InferenceSession(
            'models/part_classifier.onnx'
        )
        self.severity_model = ort.InferenceSession(
            'models/severity_model.onnx'
        )
        self.vehicle_classifier = ort.InferenceSession(
            'models/vehicle_classifier.onnx'
        )
        
        self.ocr = PlateOCR()
        self.cost_engine = CostEstimationEngine()
        self.fraud_detector = FraudDetector()
    
    def process_claim(self, image_paths):
        """
        Process a full claim with 4-6 images
        Returns: Complete damage assessment
        """
        results = []
        
        for image_path in image_paths:
            # Step 1: Detect vehicle
            vehicle_boxes = self.vehicle_detector(image_path)
            
            if len(vehicle_boxes) == 0:
                continue
            
            # Step 2: Damage segmentation
            damage_results = self.damage_segmentor(image_path)
            
            # Step 3: For each damage region
            for damage in damage_results:
                bbox = damage.boxes.xyxy[0]
                mask = damage.masks.xy[0]
                damage_type = damage.boxes.cls[0]
                confidence = damage.boxes.conf[0]
                
                # Crop damage region
                x1, y1, x2, y2 = map(int, bbox)
                damage_crop = image[y1:y2, x1:x2]
                
                # Step 4: Classify part
                part_label = self.classify_part(damage_crop)
                
                # Step 5: Estimate severity
                severity = self.estimate_severity(damage_crop)
                
                results.append({
                    'image': image_path,
                    'bbox': bbox.tolist(),
                    'mask': mask.tolist(),
                    'damage_type': damage_type,
                    'car_part': part_label,
                    'severity': severity,
                    'confidence': confidence
                })
        
        # Step 6: Extract plate (try all images)
        plate_number = None
        for image_path in image_paths:
            plate = self.ocr.extract_plate(image_path)
            if plate:
                plate_number = plate
                break
        
        # Step 7: Identify vehicle
        vehicle_info = self.identify_vehicle(image_paths[0])
        
        # Step 8: Cost estimation
        cost_estimate = self.cost_engine.estimate_cost(results)
        
        # Step 9: Fraud detection
        fraud_score = self.fraud_detector.analyze(
            results, vehicle_info, claim_metadata
        )
        
        return {
            'damages': results,
            'plate_number': plate_number,
            'vehicle_info': vehicle_info,
            'cost_min': cost_estimate['min'],
            'cost_max': cost_estimate['max'],
            'fraud_score': fraud_score
        }
```

**ONNX Export for Production:**
```python
# ml/export/export_to_onnx.py

import torch
from models import PartClassifier, SeverityModel

# Export part classifier
model = PartClassifier()
model.load_state_dict(torch.load('checkpoints/part_classifier_best.pth'))
model.eval()

dummy_input = torch.randn(1, 3, 224, 224)

torch.onnx.export(
    model,
    dummy_input,
    "models/part_classifier.onnx",
    export_params=True,
    opset_version=14,
    do_constant_folding=True,
    input_names=['input'],
    output_names=['output'],
    dynamic_axes={
        'input': {0: 'batch_size'},
        'output': {0: 'batch_size'}
    }
)

# Similarly export severity model and vehicle classifier
```

**Cost Estimation Engine:**
```python
# ml/inference/cost_engine.py

import sqlite3
import xgboost as xgb
import numpy as np

class CostEstimationEngine:
    def __init__(self):
        # Load pricing database
        self.conn = sqlite3.connect('data/pricing.db')
        
        # Load XGBoost refinement model
        self.xgb_model = xgb.Booster()
        self.xgb_model.load_model('models/cost_xgboost.json')
    
    def estimate_cost(self, damages):
        """
        Estimate repair cost based on detected damages
        """
        total_min = 0
        total_max = 0
        
        for damage in damages:
            part = damage['car_part']
            severity = damage['severity']
            damage_type = damage['damage_type']
            
            # Rule-based pricing lookup
            base_cost = self.get_base_cost(part, damage_type, severity)
            
            # Feature engineering for XGBoost
            features = self.engineer_features(damage)
            
            # XGBoost refinement
            dmatrix = xgb.DMatrix(np.array([features]))
            adjustment = self.xgb_model.predict(dmatrix)[0]
            
            # Calculate min/max range
            cost_min = base_cost * 0.8 * adjustment
            cost_max = base_cost * 1.2 * adjustment
            
            total_min += cost_min
            total_max += cost_max
        
        return {
            'min': round(total_min, 2),
            'max': round(total_max, 2)
        }
    
    def get_base_cost(self, part, damage_type, severity):
        """Query pricing database"""
        cursor = self.conn.cursor()
        query = """
            SELECT base_cost, labor_cost 
            FROM part_pricing 
            WHERE part=? AND damage_type=? AND severity=?
        """
        result = cursor.fetchone()
        
        if result:
            return result[0] + result[1]
        else:
            # Fallback estimation
            return self.fallback_estimate(part, severity)
```

**Fraud Detection:**
```python
# ml/inference/fraud_detector.py

import xgboost as xgb
import numpy as np

class FraudDetector:
    def __init__(self):
        self.model = xgb.Booster()
        self.model.load_model('models/fraud_xgboost.json')
    
    def analyze(self, damages, vehicle_info, claim_metadata):
        """
        Calculate fraud risk score (0.0 - 1.0)
        """
        features = []
        
        # Feature 1: Damage location consistency
        front_damages = sum(1 for d in damages 
                           if d['car_part'].startswith(('bumper_front', 'hood')))
        rear_damages = sum(1 for d in damages 
                          if d['car_part'].startswith(('bumper_rear', 'trunk')))
        
        claimed_type = claim_metadata.get('incident_type')
        if claimed_type == 'front_collision' and rear_damages > front_damages:
            features.append(1.0)  # Suspicious
        else:
            features.append(0.0)
        
        # Feature 2: Cost vs historical average
        estimated_cost = (damages['cost_min'] + damages['cost_max']) / 2
        avg_cost = self.get_historical_average(vehicle_info)
        cost_ratio = estimated_cost / avg_cost if avg_cost > 0 else 1.0
        features.append(min(cost_ratio, 5.0))  # Cap at 5x
        
        # Feature 3: Number of damages
        features.append(len(damages) / 10.0)  # Normalize
        
        # Feature 4: Severity distribution
        severe_count = sum(1 for d in damages if d['severity'] in ['severe', 'total_loss'])
        features.append(severe_count / max(len(damages), 1))
        
        # Feature 5: Photo quality issues
        features.append(0.0)  # Placeholder - implement blur/tampering detection
        
        # Feature 6: Time since last claim
        features.append(0.5)  # Placeholder - query from database
        
        # XGBoost prediction
        dmatrix = xgb.DMatrix(np.array([features]))
        fraud_score = self.model.predict(dmatrix)[0]
        
        return min(max(fraud_score, 0.0), 1.0)
```

**MLflow Model Registry:**
```python
# ml/mlflow/register_models.py

import mlflow
from mlflow.tracking import MlflowClient

client = MlflowClient()

# Register all production models
models_to_register = {
    'damage_detector': 'runs/detect/best_run_id',
    'part_classifier': 'runs/classify/best_run_id',
    'severity_model': 'runs/severity/best_run_id',
    'vehicle_classifier': 'runs/vehicle/best_run_id',
    'cost_predictor': 'runs/cost/best_run_id',
    'fraud_detector': 'runs/fraud/best_run_id'
}

for model_name, run_id in models_to_register.items():
    model_uri = f"runs:/{run_id}/model"
    
    # Register model
    result = mlflow.register_model(
        model_uri=model_uri,
        name=model_name
    )
    
    # Transition to production
    client.transition_model_version_stage(
        name=model_name,
        version=result.version,
        stage="Production"
    )
```

**Deliverables:**
- ✅ All 7 models trained and validated
- ✅ ONNX exported models for CPU inference
- ✅ End-to-end pipeline tested on 50 images
- ✅ Pipeline latency < 3 seconds (CPU)
- ✅ All models registered in MLflow

---

### **PHASE 3: Backend API Development** (Week 6)

#### Week 6: FastAPI Backend + Celery Workers

**Project Structure:**
```
backend/
├── app/
│   ├── __init__.py
│   ├── main.py              # FastAPI app entry
│   ├── api/
│   │   ├── v1/
│   │   │   ├── auth.py      # Auth endpoints
│   │   │   ├── claims.py    # Claims endpoints
│   │   │   └── admin.py     # Admin endpoints
│   ├── core/
│   │   ├── config.py        # Settings
│   │   ├── security.py      # JWT, password hashing
│   │   └── deps.py          # Dependencies
│   ├── models/
│   │   ├── user.py          # User model
│   │   ├── claim.py         # Claim model
│   │   └── damage.py        # Damage detection model
│   ├── schemas/
│   │   ├── auth.py          # Pydantic schemas
│   │   ├── claim.py
│   │   └── damage.py
│   ├── services/
│   │   ├── auth_service.py
│   │   ├── claim_service.py
│   │   └── report_service.py
│   ├── workers/
│   │   ├── celery_app.py    # Celery configuration
│   │   └── tasks.py         # Celery tasks
│   └── utils/
│       ├── s3.py            # MinIO client
│       └── websocket.py     # WebSocket manager
├── alembic/                 # Database migrations
├── tests/
└── requirements.txt
```

**Day 1: FastAPI Setup**
```python
# backend/app/main.py

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.api.v1 import auth, claims, admin
from app.core.config import settings

app = FastAPI(
    title="AutoClaim Vision API",
    description="AI-Powered Vehicle Damage Assessment",
    version="1.0.0"
)

# CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(auth.router, prefix="/api/v1/auth", tags=["auth"])
app.include_router(claims.router, prefix="/api/v1/claims", tags=["claims"])
app.include_router(admin.router, prefix="/api/v1/admin", tags=["admin"])

@app.get("/api/v1/health")
async def health_check():
    return {"status": "healthy"}
```

**Configuration:**
```python
# backend/app/core/config.py

from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    # Database
    DATABASE_URL: str = "postgresql://user:pass@localhost/autoclaim"
    
    # Redis
    REDIS_URL: str = "redis://localhost:6379/0"
    
    # MinIO / S3
    MINIO_ENDPOINT: str = "localhost:9000"
    MINIO_ACCESS_KEY: str = "minioadmin"
    MINIO_SECRET_KEY: str = "minioadmin"
    MINIO_BUCKET: str = "autoclaim"
    
    # JWT
    SECRET_KEY: str = "your-secret-key-change-in-production"
    ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 30
    
    # Celery
    CELERY_BROKER_URL: str = "redis://localhost:6379/1"
    CELERY_RESULT_BACKEND: str = "redis://localhost:6379/2"
    
    # CORS
    CORS_ORIGINS: list = ["http://localhost:3000"]
    
    # ML Models
    MODELS_PATH: str = "/app/models"
    
    class Config:
        env_file = ".env"

settings = Settings()
```

**Day 2: Database Models**
```python
# backend/app/models/claim.py

from sqlalchemy import Column, String, Integer, Float, DateTime, Boolean, Enum
from sqlalchemy.dialects.postgresql import UUID, JSONB
from sqlalchemy.orm import relationship
import uuid
from datetime import datetime
from app.db.base_class import Base

class Claim(Base):
    __tablename__ = "claims"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    customer_id = Column(UUID(as_uuid=True), ForeignKey('users.id'))
    assessor_id = Column(UUID(as_uuid=True), ForeignKey('users.id'), nullable=True)
    
    status = Column(Enum(
        'uploaded', 'processing', 'ai_complete',
        'under_review', 'approved', 'rejected', 'fraud_review',
        name='claim_status'
    ), default='uploaded')
    
    vehicle_reg = Column(String(20))
    vehicle_make = Column(String(100))
    vehicle_model = Column(String(100))
    vehicle_year = Column(Integer)
    
    incident_type = Column(String(50))  # collision, theft, weather, fire
    incident_date = Column(DateTime)
    incident_city = Column(String(100))
    
    ai_cost_min = Column(Float)
    ai_cost_max = Column(Float)
    final_cost = Column(Float, nullable=True)
    
    fraud_score = Column(Float)  # 0.0 to 1.0
    fraud_flagged = Column(Boolean, default=False)
    
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationships
    customer = relationship("User", foreign_keys=[customer_id])
    assessor = relationship("User", foreign_keys=[assessor_id])
    images = relationship("ClaimImage", back_populates="claim")
    damages = relationship("DamageDetection", back_populates="claim")


class DamageDetection(Base):
    __tablename__ = "damage_detections"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    claim_id = Column(UUID(as_uuid=True), ForeignKey('claims.id'))
    image_id = Column(UUID(as_uuid=True), ForeignKey('claim_images.id'))
    
    car_part = Column(String(60))  # hood, door_fl, bumper_front, etc.
    damage_type = Column(String(30))  # scratch, dent, crack, etc.
    
    severity = Column(Enum(
        'minor', 'moderate', 'severe', 'total_loss',
        name='severity_level'
    ))
    
    confidence = Column(Float)
    
    bbox_x = Column(Integer)
    bbox_y = Column(Integer)
    bbox_w = Column(Integer)
    bbox_h = Column(Integer)
    
    mask_polygon = Column(JSONB)
    area_cm2 = Column(Float)
    
    part_cost_min = Column(Float)
    part_cost_max = Column(Float)
    labor_cost = Column(Float)
    
    # Relationships
    claim = relationship("Claim", back_populates="damages")
    image = relationship("ClaimImage", back_populates="damages")
```

**Day 3: Authentication**
```python
# backend/app/api/v1/auth.py

from fastapi import APIRouter, Depends, HTTPException
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from sqlalchemy.orm import Session
from app.core import security
from app.schemas import auth as schemas
from app.models.user import User
from app.core.deps import get_db

router = APIRouter()

@router.post("/register", response_model=schemas.UserResponse)
def register(user_data: schemas.UserCreate, db: Session = Depends(get_db)):
    # Check if user exists
    existing_user = db.query(User).filter(User.email == user_data.email).first()
    if existing_user:
        raise HTTPException(status_code=400, detail="Email already registered")
    
    # Hash password
    hashed_password = security.get_password_hash(user_data.password)
    
    # Create user
    db_user = User(
        email=user_data.email,
        hashed_password=hashed_password,
        full_name=user_data.full_name,
        role=user_data.role or "customer"
    )
    db.add(db_user)
    db.commit()
    db.refresh(db_user)
    
    return db_user


@router.post("/login", response_model=schemas.Token)
def login(
    form_data: OAuth2PasswordRequestForm = Depends(),
    db: Session = Depends(get_db)
):
    # Authenticate user
    user = db.query(User).filter(User.email == form_data.username).first()
    
    if not user or not security.verify_password(form_data.password, user.hashed_password):
        raise HTTPException(status_code=401, detail="Incorrect email or password")
    
    # Create access token
    access_token = security.create_access_token(subject=str(user.id))
    
    return {
        "access_token": access_token,
        "token_type": "bearer"
    }
```

**Day 4: Claims API**
```python
# backend/app/api/v1/claims.py

from fastapi import APIRouter, Depends, HTTPException, File, UploadFile
from sqlalchemy.orm import Session
from typing import List
from app.core.deps import get_db, get_current_user
from app.models.claim import Claim
from app.models.user import User
from app.schemas import claim as schemas
from app.workers.tasks import process_claim_async
from app.utils.s3 import upload_file_to_s3
import uuid

router = APIRouter()

@router.post("/", response_model=schemas.ClaimResponse)
async def create_claim(
    claim_data: schemas.ClaimCreate,
    images: List[UploadFile] = File(...),
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Create new claim and upload images"""
    
    # Validate number of images (4-6 required)
    if len(images) < 4 or len(images) > 6:
        raise HTTPException(
            status_code=400, 
            detail="Please upload 4-6 images"
        )
    
    # Create claim record
    claim = Claim(
        customer_id=current_user.id,
        vehicle_reg=claim_data.vehicle_reg,
        incident_type=claim_data.incident_type,
        incident_date=claim_data.incident_date,
        incident_city=claim_data.incident_city,
        status='uploaded'
    )
    db.add(claim)
    db.commit()
    db.refresh(claim)
    
    # Upload images to MinIO
    image_urls = []
    for image in images:
        image_id = str(uuid.uuid4())
        file_path = f"claims/{claim.id}/{image_id}.jpg"
        
        url = await upload_file_to_s3(
            file=image.file,
            filename=file_path,
            content_type=image.content_type
        )
        
        # Save image record
        claim_image = ClaimImage(
            id=image_id,
            claim_id=claim.id,
            file_path=file_path,
            url=url
        )
        db.add(claim_image)
        image_urls.append(url)
    
    db.commit()
    
    # Trigger async processing
    process_claim_async.delay(str(claim.id))
    
    return claim


@router.get("/", response_model=List[schemas.ClaimListItem])
def list_claims(
    skip: int = 0,
    limit: int = 20,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """List user's claims"""
    claims = db.query(Claim).filter(
        Claim.customer_id == current_user.id
    ).offset(skip).limit(limit).all()
    
    return claims


@router.get("/{claim_id}", response_model=schemas.ClaimDetail)
def get_claim(
    claim_id: str,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Get claim details with all damage detections"""
    claim = db.query(Claim).filter(Claim.id == claim_id).first()
    
    if not claim:
        raise HTTPException(status_code=404, detail="Claim not found")
    
    # Authorization check
    if claim.customer_id != current_user.id and current_user.role not in ['assessor', 'admin']:
        raise HTTPException(status_code=403, detail="Not authorized")
    
    return claim
```

**Day 5: Celery Workers**
```python
# backend/app/workers/celery_app.py

from celery import Celery
from app.core.config import settings

celery_app = Celery(
    "autoclaim_workers",
    broker=settings.CELERY_BROKER_URL,
    backend=settings.CELERY_RESULT_BACKEND
)

celery_app.conf.task_routes = {
    "app.workers.tasks.process_claim_async": "inference_queue"
}

celery_app.conf.update(
    task_serializer='json',
    accept_content=['json'],
    result_serializer='json',
    timezone='UTC',
    enable_utc=True,
)
```

```python
# backend/app/workers/tasks.py

from celery import Task
from app.workers.celery_app import celery_app
from app.db.session import SessionLocal
from app.models.claim import Claim, DamageDetection
from ml.inference.pipeline import DamageAssessmentPipeline
from app.services.report_service import generate_pdf_report
from app.utils.websocket import send_status_update
import logging

logger = logging.getLogger(__name__)

# Initialize ML pipeline once (shared across workers)
pipeline = DamageAssessmentPipeline()

@celery_app.task(bind=True)
def process_claim_async(self: Task, claim_id: str):
    """
    Async task to process claim with AI pipeline
    """
    db = SessionLocal()
    
    try:
        # Update status
        claim = db.query(Claim).filter(Claim.id == claim_id).first()
        claim.status = 'processing'
        db.commit()
        
        # Send WebSocket update
        send_status_update(claim_id, 'processing')
        
        # Get image URLs
        image_paths = [img.file_path for img in claim.images]
        
        # Run AI pipeline
        logger.info(f"Processing claim {claim_id} with {len(image_paths)} images")
        result = pipeline.process_claim(image_paths)
        
        # Save damage detections
        for damage in result['damages']:
            detection = DamageDetection(
                claim_id=claim_id,
                image_id=damage['image_id'],
                car_part=damage['car_part'],
                damage_type=damage['damage_type'],
                severity=damage['severity'],
                confidence=damage['confidence'],
                bbox_x=damage['bbox'][0],
                bbox_y=damage['bbox'][1],
                bbox_w=damage['bbox'][2],
                bbox_h=damage['bbox'][3],
                mask_polygon=damage['mask'],
                part_cost_min=damage['cost_min'],
                part_cost_max=damage['cost_max'],
                labor_cost=damage['labor_cost']
            )
            db.add(detection)
        
        # Update claim with results
        claim.vehicle_make = result['vehicle_info']['make']
        claim.vehicle_model = result['vehicle_info']['model']
        claim.vehicle_year = result['vehicle_info']['year']
        claim.ai_cost_min = result['cost_min']
        claim.ai_cost_max = result['cost_max']
        claim.fraud_score = result['fraud_score']
        claim.fraud_flagged = result['fraud_score'] > 0.75
        claim.status = 'ai_complete'
        
        db.commit()
        
        # Generate PDF report
        pdf_path = generate_pdf_report(claim_id)
        
        # Send completion update
        send_status_update(claim_id, 'ai_complete', {
            'cost_min': result['cost_min'],
            'cost_max': result['cost_max'],
            'damages_count': len(result['damages']),
            'fraud_score': result['fraud_score']
        })
        
        logger.info(f"Claim {claim_id} processed successfully")
        
    except Exception as e:
        logger.error(f"Error processing claim {claim_id}: {str(e)}")
        claim.status = 'error'
        db.commit()
        send_status_update(claim_id, 'error', {'message': str(e)})
        raise
    
    finally:
        db.close()
```

**Day 6: PDF Report Generation**
```python
# backend/app/services/report_service.py

from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.lib.units import inch
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer, Image
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from datetime import datetime
from app.db.session import SessionLocal
from app.models.claim import Claim
import os

def generate_pdf_report(claim_id: str) -> str:
    """
    Generate PDF report for claim
    Returns: PDF file path
    """
    db = SessionLocal()
    claim = db.query(Claim).filter(Claim.id == claim_id).first()
    
    # Create PDF
    pdf_filename = f"claim_{claim_id}_report.pdf"
    pdf_path = f"/tmp/{pdf_filename}"
    
    doc = SimpleDocTemplate(pdf_path, pagesize=A4)
    story = []
    styles = getSampleStyleSheet()
    
    # Title
    title_style = ParagraphStyle(
        'CustomTitle',
        parent=styles['Heading1'],
        fontSize=24,
        textColor=colors.HexColor('#1B3A6B'),
        spaceAfter=30,
    )
    title = Paragraph("AutoClaim Vision - Damage Assessment Report", title_style)
    story.append(title)
    story.append(Spacer(1, 0.2*inch))
    
    # Claim Details
    claim_data = [
        ['Claim ID:', str(claim.id)],
        ['Status:', claim.status.upper()],
        ['Date:', claim.created_at.strftime('%Y-%m-%d %H:%M:%S')],
        ['Vehicle:', f"{claim.vehicle_make} {claim.vehicle_model} ({claim.vehicle_year})"],
        ['Registration:', claim.vehicle_reg or 'N/A'],
        ['Incident Type:', claim.incident_type],
        ['Incident Date:', claim.incident_date.strftime('%Y-%m-%d')],
        ['Location:', claim.incident_city],
    ]
    
    claim_table = Table(claim_data, colWidths=[2*inch, 4*inch])
    claim_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (0, -1), colors.HexColor('#EBF3FB')),
        ('TEXTCOLOR', (0, 0), (-1, -1), colors.HexColor('#1e2535')),
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('FONTNAME', (0, 0), (0, -1), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, -1), 10),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 12),
        ('GRID', (0, 0), (-1, -1), 1, colors.HexColor('#D6DCE4')),
    ]))
    story.append(claim_table)
    story.append(Spacer(1, 0.3*inch))
    
    # Cost Estimate
    story.append(Paragraph("Cost Estimate", styles['Heading2']))
    cost_data = [
        ['Estimated Repair Cost (Min):', f"Rs. {claim.ai_cost_min:,.2f}"],
        ['Estimated Repair Cost (Max):', f"Rs. {claim.ai_cost_max:,.2f}"],
        ['Fraud Risk Score:', f"{claim.fraud_score:.3f}"],
        ['Fraud Flagged:', 'YES' if claim.fraud_flagged else 'NO'],
    ]
    
    cost_table = Table(cost_data, colWidths=[2.5*inch, 3.5*inch])
    cost_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (0, -1), colors.HexColor('#EBF3FB')),
        ('TEXTCOLOR', (0, 0), (-1, -1), colors.HexColor('#1e2535')),
        ('FONTNAME', (0, 0), (0, -1), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, -1), 11),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 12),
        ('GRID', (0, 0), (-1, -1), 1, colors.HexColor('#D6DCE4')),
    ]))
    story.append(cost_table)
    story.append(Spacer(1, 0.3*inch))
    
    # Damage Detections
    story.append(Paragraph("Detected Damages", styles['Heading2']))
    
    damage_header = ['#', 'Part', 'Damage Type', 'Severity', 'Confidence', 'Est. Cost']
    damage_rows = [damage_header]
    
    for i, damage in enumerate(claim.damages, 1):
        row = [
            str(i),
            damage.car_part.replace('_', ' ').title(),
            damage.damage_type.title(),
            damage.severity.title(),
            f"{damage.confidence:.2%}",
            f"Rs. {damage.part_cost_min:.0f} - {damage.part_cost_max:.0f}"
        ]
        damage_rows.append(row)
    
    damage_table = Table(damage_rows, colWidths=[0.4*inch, 1.3*inch, 1.2*inch, 1*inch, 0.9*inch, 1.7*inch])
    damage_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#1B3A6B')),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, 0), 10),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
        ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
        ('GRID', (0, 0), (-1, -1), 1, colors.black),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor('#F5F9FC')]),
    ]))
    story.append(damage_table)
    
    # Build PDF
    doc.build(story)
    
    # Upload to MinIO
    s3_path = f"reports/{claim_id}/{pdf_filename}"
    upload_file_to_s3(pdf_path, s3_path)
    
    db.close()
    return s3_path
```

**Day 7: Testing**
```python
# backend/tests/test_claims.py

import pytest
from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)

def test_create_claim():
    # Register and login
    register_response = client.post("/api/v1/auth/register", json={
        "email": "test@example.com",
        "password": "testpass123",
        "full_name": "Test User"
    })
    assert register_response.status_code == 200
    
    login_response = client.post("/api/v1/auth/login", data={
        "username": "test@example.com",
        "password": "testpass123"
    })
    token = login_response.json()["access_token"]
    
    # Create claim
    files = [
        ("images", ("front.jpg", open("test_images/front.jpg", "rb"), "image/jpeg")),
        ("images", ("rear.jpg", open("test_images/rear.jpg", "rb"), "image/jpeg")),
        ("images", ("left.jpg", open("test_images/left.jpg", "rb"), "image/jpeg")),
        ("images", ("right.jpg", open("test_images/right.jpg", "rb"), "image/jpeg")),
    ]
    
    data = {
        "vehicle_reg": "MH01AB1234",
        "incident_type": "collision",
        "incident_date": "2024-04-15",
        "incident_city": "Mumbai"
    }
    
    response = client.post(
        "/api/v1/claims/",
        files=files,
        data=data,
        headers={"Authorization": f"Bearer {token}"}
    )
    
    assert response.status_code == 200
    assert response.json()["status"] == "uploaded"

# Target: 80% code coverage
```

**Deliverables:**
- ✅ Complete FastAPI backend with auth, claims, admin endpoints
- ✅ Celery workers for async AI inference
- ✅ PDF report generation
- ✅ WebSocket real-time updates
- ✅ 80%+ test coverage

---

### **PHASE 4: Frontend Development** (Weeks 7-8)

#### Week 7: Core Customer UI

**Project Structure:**
```
frontend/
├── src/
│   ├── components/
│   │   ├── ui/              # shadcn/ui components
│   │   ├── auth/
│   │   │   ├── LoginForm.tsx
│   │   │   └── RegisterForm.tsx
│   │   ├── claims/
│   │   │   ├── ClaimWizard.tsx
│   │   │   ├── PhotoUpload.tsx
│   │   │   ├── StatusTracker.tsx
│   │   │   └── DamageViewer.tsx
│   │   └── layout/
│   │       ├── Navbar.tsx
│   │       └── Sidebar.tsx
│   ├── pages/
│   │   ├── Home.tsx
│   │   ├── Login.tsx
│   │   ├── Dashboard.tsx
│   │   ├── CreateClaim.tsx
│   │   └── ClaimDetail.tsx
│   ├── hooks/
│   │   ├── useAuth.ts
│   │   ├── useClaims.ts
│   │   └── useWebSocket.ts
│   ├── services/
│   │   └── api.ts
│   ├── stores/
│   │   └── authStore.ts
│   ├── types/
│   │   └── index.ts
│   └── App.tsx
├── package.json
└── tailwind.config.js
```

**Setup:**
```bash
# Initialize React project
npm create vite@latest frontend -- --template react-ts
cd frontend

# Install dependencies
npm install react-router-dom zustand @tanstack/react-query
npm install axios react-dropzone react-hook-form zod
npm install konva react-konva recharts
npm install lucide-react framer-motion
npm install -D tailwindcss postcss autoprefixer
npm install -D @types/node

# Install shadcn/ui
npx shadcn-ui@latest init
npx shadcn-ui@latest add button input card form
npx shadcn-ui@latest add dropdown-menu avatar badge
npx shadcn-ui@latest add dialog alert progress
```

**API Client:**
```typescript
// src/services/api.ts

import axios from 'axios';

const api = axios.create({
  baseURL: import.meta.env.VITE_API_URL || 'http://localhost:8000/api/v1',
  headers: {
    'Content-Type': 'application/json',
  },
});

// Request interceptor to add auth token
api.interceptors.request.use((config) => {
  const token = localStorage.getItem('access_token');
  if (token) {
    config.headers.Authorization = `Bearer ${token}`;
  }
  return config;
});

// Response interceptor for error handling
api.interceptors.response.use(
  (response) => response,
  (error) => {
    if (error.response?.status === 401) {
      localStorage.removeItem('access_token');
      window.location.href = '/login';
    }
    return Promise.reject(error);
  }
);

export default api;
```

**Auth Store:**
```typescript
// src/stores/authStore.ts

import { create } from 'zustand';
import { persist } from 'zustand/middleware';

interface User {
  id: string;
  email: string;
  full_name: string;
  role: string;
}

interface AuthState {
  user: User | null;
  token: string | null;
  setAuth: (user: User, token: string) => void;
  logout: () => void;
}

export const useAuthStore = create<AuthState>()(
  persist(
    (set) => ({
      user: null,
      token: null,
      setAuth: (user, token) => {
        localStorage.setItem('access_token', token);
        set({ user, token });
      },
      logout: () => {
        localStorage.removeItem('access_token');
        set({ user: null, token: null });
      },
    }),
    {
      name: 'auth-storage',
    }
  )
);
```

**Claim Wizard:**
```typescript
// src/components/claims/ClaimWizard.tsx

import React, { useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { useForm } from 'react-hook-form';
import { zodResolver } from '@hookform/resolvers/zod';
import * as z from 'zod';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import PhotoUpload from './PhotoUpload';
import api from '@/services/api';

const claimSchema = z.object({
  vehicle_reg: z.string().min(1, 'Registration number required'),
  incident_type: z.enum(['collision', 'theft', 'weather', 'fire']),
  incident_date: z.string(),
  incident_city: z.string().min(1, 'City required'),
});

type ClaimFormData = z.infer<typeof claimSchema>;

export default function ClaimWizard() {
  const [step, setStep] = useState(1);
  const [images, setImages] = useState<File[]>([]);
  const [isSubmitting, setIsSubmitting] = useState(false);
  const navigate = useNavigate();
  
  const { register, handleSubmit, formState: { errors } } = useForm<ClaimFormData>({
    resolver: zodResolver(claimSchema),
  });
  
  const onSubmit = async (data: ClaimFormData) => {
    if (images.length < 4 || images.length > 6) {
      alert('Please upload 4-6 images');
      return;
    }
    
    setIsSubmitting(true);
    
    try {
      const formData = new FormData();
      formData.append('vehicle_reg', data.vehicle_reg);
      formData.append('incident_type', data.incident_type);
      formData.append('incident_date', data.incident_date);
      formData.append('incident_city', data.incident_city);
      
      images.forEach((image) => {
        formData.append('images', image);
      });
      
      const response = await api.post('/claims/', formData, {
        headers: { 'Content-Type': 'multipart/form-data' },
      });
      
      navigate(`/claims/${response.data.id}`);
    } catch (error) {
      console.error('Error creating claim:', error);
      alert('Failed to create claim. Please try again.');
    } finally {
      setIsSubmitting(false);
    }
  };
  
  return (
    <div className="max-w-4xl mx-auto p-6">
      <Card>
        <CardHeader>
          <CardTitle>File New Claim - Step {step} of 3</CardTitle>
        </CardHeader>
        <CardContent>
          {step === 1 && (
            <div className="space-y-4">
              <h3 className="text-lg font-semibold">Incident Details</h3>
              
              <div>
                <label className="block text-sm font-medium mb-1">Incident Type</label>
                <select
                  {...register('incident_type')}
                  className="w-full p-2 border rounded-md"
                >
                  <option value="collision">Collision</option>
                  <option value="theft">Theft</option>
                  <option value="weather">Weather Damage</option>
                  <option value="fire">Fire</option>
                </select>
                {errors.incident_type && (
                  <p className="text-red-500 text-sm mt-1">{errors.incident_type.message}</p>
                )}
              </div>
              
              <div>
                <label className="block text-sm font-medium mb-1">Incident Date</label>
                <Input type="date" {...register('incident_date')} />
                {errors.incident_date && (
                  <p className="text-red-500 text-sm mt-1">{errors.incident_date.message}</p>
                )}
              </div>
              
              <div>
                <label className="block text-sm font-medium mb-1">City</label>
                <Input {...register('incident_city')} placeholder="Mumbai" />
                {errors.incident_city && (
                  <p className="text-red-500 text-sm mt-1">{errors.incident_city.message}</p>
                )}
              </div>
              
              <Button onClick={() => setStep(2)} className="w-full">
                Next: Vehicle Information
              </Button>
            </div>
          )}
          
          {step === 2 && (
            <div className="space-y-4">
              <h3 className="text-lg font-semibold">Vehicle Information</h3>
              
              <div>
                <label className="block text-sm font-medium mb-1">Registration Number</label>
                <Input {...register('vehicle_reg')} placeholder="MH01AB1234" />
                {errors.vehicle_reg && (
                  <p className="text-red-500 text-sm mt-1">{errors.vehicle_reg.message}</p>
                )}
              </div>
              
              <div className="flex gap-2">
                <Button variant="outline" onClick={() => setStep(1)}>Back</Button>
                <Button onClick={() => setStep(3)} className="flex-1">
                  Next: Upload Photos
                </Button>
              </div>
            </div>
          )}
          
          {step === 3 && (
            <div className="space-y-4">
              <h3 className="text-lg font-semibold">Upload Photos (4-6 images)</h3>
              
              <PhotoUpload images={images} setImages={setImages} />
              
              <div className="flex gap-2">
                <Button variant="outline" onClick={() => setStep(2)}>Back</Button>
                <Button
                  onClick={handleSubmit(onSubmit)}
                  disabled={images.length < 4 || isSubmitting}
                  className="flex-1"
                >
                  {isSubmitting ? 'Submitting...' : 'Submit Claim'}
                </Button>
              </div>
            </div>
          )}
        </CardContent>
      </Card>
    </div>
  );
}
```

**Photo Upload Component:**
```typescript
// src/components/claims/PhotoUpload.tsx

import React, { useCallback } from 'react';
import { useDropzone } from 'react-dropzone';
import { X, Upload } from 'lucide-react';
import { Button } from '@/components/ui/button';

interface PhotoUploadProps {
  images: File[];
  setImages: (images: File[]) => void;
}

export default function PhotoUpload({ images, setImages }: PhotoUploadProps) {
  const onDrop = useCallback((acceptedFiles: File[]) => {
    const newImages = [...images, ...acceptedFiles].slice(0, 6);
    setImages(newImages);
  }, [images, setImages]);
  
  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop,
    accept: { 'image/*': ['.jpeg', '.jpg', '.png'] },
    maxFiles: 6 - images.length,
  });
  
  const removeImage = (index: number) => {
    setImages(images.filter((_, i) => i !== index));
  };
  
  return (
    <div className="space-y-4">
      <div
        {...getRootProps()}
        className={`border-2 border-dashed rounded-lg p-8 text-center cursor-pointer transition-colors ${
          isDragActive ? 'border-blue-500 bg-blue-50' : 'border-gray-300 hover:border-gray-400'
        }`}
      >
        <input {...getInputProps()} />
        <Upload className="mx-auto h-12 w-12 text-gray-400 mb-4" />
        {isDragActive ? (
          <p className="text-blue-600">Drop the images here...</p>
        ) : (
          <div>
            <p className="text-gray-600 mb-2">Drag & drop images here, or click to select</p>
            <p className="text-sm text-gray-400">
              {images.length}/6 images uploaded | Recommended angles: Front, Rear, Left, Right, Close-ups
            </p>
          </div>
        )}
      </div>
      
      {images.length > 0 && (
        <div className="grid grid-cols-3 gap-4">
          {images.map((image, index) => (
            <div key={index} className="relative group">
              <img
                src={URL.createObjectURL(image)}
                alt={`Upload ${index + 1}`}
                className="w-full h-32 object-cover rounded-lg"
              />
              <Button
                size="sm"
                variant="destructive"
                className="absolute top-2 right-2 opacity-0 group-hover:opacity-100 transition-opacity"
                onClick={() => removeImage(index)}
              >
                <X className="h-4 w-4" />
              </Button>
              <div className="absolute bottom-2 left-2 bg-black bg-opacity-60 text-white text-xs px-2 py-1 rounded">
                Image {index + 1}
              </div>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}
```

**WebSocket Hook:**
```typescript
// src/hooks/useWebSocket.ts

import { useEffect, useRef, useState } from 'react';

export function useWebSocket(claimId: string | undefined) {
  const [status, setStatus] = useState<string>('');
  const [data, setData] = useState<any>(null);
  const ws = useRef<WebSocket | null>(null);
  
  useEffect(() => {
    if (!claimId) return;
    
    const wsUrl = `ws://localhost:8000/ws/claims/${claimId}`;
    ws.current = new WebSocket(wsUrl);
    
    ws.current.onmessage = (event) => {
      const message = JSON.parse(event.data);
      setStatus(message.status);
      setData(message.data);
    };
    
    ws.current.onerror = (error) => {
      console.error('WebSocket error:', error);
    };
    
    return () => {
      ws.current?.close();
    };
  }, [claimId]);
  
  return { status, data };
}
```

**Deliverables:**
- ✅ Complete customer UI (login, register, create claim, view claims)
- ✅ Multi-step claim wizard
- ✅ Photo upload with drag & drop
- ✅ Real-time status tracking via WebSocket
- ✅ Responsive design

#### Week 8: Assessor + Admin + Analytics UI

**Assessor Dashboard:**
```typescript
// src/pages/AssessorDashboard.tsx

import React, { useState } from 'react';
import { useQuery } from '@tanstack/react-query';
import api from '@/services/api';
import DamageViewer from '@/components/claims/DamageViewer';
import { Button } from '@/components/ui/button';
import { Card } from '@/components/ui/card';

export default function AssessorDashboard() {
  const [selectedClaim, setSelectedClaim] = useState<string | null>(null);
  
  const { data: claims } = useQuery({
    queryKey: ['assessor-claims'],
    queryFn: async () => {
      const response = await api.get('/admin/claims?status=ai_complete');
      return response.data;
    },
  });
  
  const handleApprove = async (claimId: string) => {
    await api.patch(`/claims/${claimId}/review`, {
      action: 'approve',
      final_cost: null, // Use AI estimate
    });
    // Refresh claims list
  };
  
  const handleOverride = async (claimId: string, overrides: any) => {
    await api.patch(`/claims/${claimId}/review`, {
      action: 'override',
      overrides: overrides,
    });
  };
  
  return (
    <div className="p-6">
      <h1 className="text-2xl font-bold mb-6">Claims Review Queue</h1>
      
      <div className="grid grid-cols-3 gap-6">
        <div className="col-span-1 space-y-4">
          {claims?.map((claim: any) => (
            <Card
              key={claim.id}
              className={`p-4 cursor-pointer ${
                selectedClaim === claim.id ? 'ring-2 ring-blue-500' : ''
              }`}
              onClick={() => setSelectedClaim(claim.id)}
            >
              <div className="flex justify-between items-start mb-2">
                <div>
                  <p className="font-semibold">{claim.vehicle_make} {claim.vehicle_model}</p>
                  <p className="text-sm text-gray-500">{claim.vehicle_reg}</p>
                </div>
                {claim.fraud_flagged && (
                  <span className="bg-red-100 text-red-800 text-xs px-2 py-1 rounded">
                    FRAUD FLAG
                  </span>
                )}
              </div>
              <p className="text-sm">
                Est: Rs. {claim.ai_cost_min.toLocaleString()} - Rs. {claim.ai_cost_max.toLocaleString()}
              </p>
              <p className="text-xs text-gray-500 mt-2">
                {claim.damages?.length || 0} damages detected
              </p>
            </Card>
          ))}
        </div>
        
        <div className="col-span-2">
          {selectedClaim && (
            <DamageViewer claimId={selectedClaim} onApprove={handleApprove} onOverride={handleOverride} />
          )}
        </div>
      </div>
    </div>
  );
}
```

**Damage Viewer with Konva.js:**
```typescript
// src/components/claims/DamageViewer.tsx

import React, { useState } from 'react';
import { Stage, Layer, Image as KonvaImage, Rect, Text } from 'react-konva';
import { useQuery } from '@tanstack/react-query';
import api from '@/services/api';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Button } from '@/components/ui/button';

interface DamageViewerProps {
  claimId: string;
  onApprove: (claimId: string) => void;
  onOverride: (claimId: string, overrides: any) => void;
}

export default function DamageViewer({ claimId, onApprove, onOverride }: DamageViewerProps) {
  const [selectedImage, setSelectedImage] = useState(0);
  const [hoveredDamage, setHoveredDamage] = useState<number | null>(null);
  
  const { data: claim } = useQuery({
    queryKey: ['claim', claimId],
    queryFn: async () => {
      const response = await api.get(`/claims/${claimId}`);
      return response.data;
    },
  });
  
  if (!claim) return <div>Loading...</div>;
  
  const currentImage = claim.images[selectedImage];
  const imageDamages = claim.damages.filter((d: any) => d.image_id === currentImage.id);
  
  return (
    <Card>
      <CardHeader>
        <CardTitle>Damage Assessment Review</CardTitle>
      </CardHeader>
      <CardContent>
        <div className="grid grid-cols-2 gap-6">
          {/* Image Viewer */}
          <div>
            <Stage width={600} height={400}>
              <Layer>
                {/* Image would be loaded here */}
                {imageDamages.map((damage: any, index: number) => (
                  <React.Fragment key={damage.id}>
                    <Rect
                      x={damage.bbox_x}
                      y={damage.bbox_y}
                      width={damage.bbox_w}
                      height={damage.bbox_h}
                      stroke={hoveredDamage === index ? '#FF0000' : '#00FF00'}
                      strokeWidth={3}
                      opacity={0.7}
                      onMouseEnter={() => setHoveredDamage(index)}
                      onMouseLeave={() => setHoveredDamage(null)}
                    />
                    <Text
                      x={damage.bbox_x}
                      y={damage.bbox_y - 20}
                      text={`${damage.car_part} - ${damage.severity}`}
                      fontSize={12}
                      fill="#FFFFFF"
                      stroke="#000000"
                      strokeWidth={0.5}
                    />
                  </React.Fragment>
                ))}
              </Layer>
            </Stage>
            
            {/* Image thumbnails */}
            <div className="flex gap-2 mt-4">
              {claim.images.map((img: any, index: number) => (
                <img
                  key={img.id}
                  src={img.url}
                  alt={`Image ${index + 1}`}
                  className={`w-16 h-16 object-cover cursor-pointer rounded ${
                    selectedImage === index ? 'ring-2 ring-blue-500' : ''
                  }`}
                  onClick={() => setSelectedImage(index)}
                />
              ))}
            </div>
          </div>
          
          {/* Damage List */}
          <div>
            <h3 className="font-semibold mb-4">Detected Damages ({claim.damages.length})</h3>
            <div className="space-y-2 max-h-96 overflow-y-auto">
              {claim.damages.map((damage: any, index: number) => (
                <Card
                  key={damage.id}
                  className={`p-3 ${hoveredDamage === index ? 'ring-2 ring-blue-500' : ''}`}
                  onMouseEnter={() => setHoveredDamage(index)}
                  onMouseLeave={() => setHoveredDamage(null)}
                >
                  <div className="flex justify-between items-start">
                    <div>
                      <p className="font-medium">{damage.car_part.replace('_', ' ').toUpperCase()}</p>
                      <p className="text-sm text-gray-600">{damage.damage_type} - {damage.severity}</p>
                      <p className="text-xs text-gray-500">Confidence: {(damage.confidence * 100).toFixed(1)}%</p>
                    </div>
                    <div className="text-right">
                      <p className="text-sm font-semibold">
                        Rs. {damage.part_cost_min.toFixed(0)} - {damage.part_cost_max.toFixed(0)}
                      </p>
                      <p className="text-xs text-gray-500">+ Rs. {damage.labor_cost.toFixed(0)} labor</p>
                    </div>
                  </div>
                </Card>
              ))}
            </div>
            
            {/* Action Buttons */}
            <div className="mt-6 space-y-2">
              <Button
                className="w-full"
                onClick={() => onApprove(claimId)}
              >
                Approve as-is
              </Button>
              <Button
                variant="outline"
                className="w-full"
                onClick={() => {/* Open override modal */}}
              >
                Override & Adjust
              </Button>
              <Button
                variant="destructive"
                className="w-full"
              >
                Reject Claim
              </Button>
            </div>
          </div>
        </div>
      </CardContent>
    </Card>
  );
}
```

**Analytics Dashboard:**
```typescript
// src/pages/Analytics.tsx

import React from 'react';
import { useQuery } from '@tanstack/react-query';
import api from '@/services/api';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, PieChart, Pie, Cell } from 'recharts';

export default function Analytics() {
  const { data: analytics } = useQuery({
    queryKey: ['analytics'],
    queryFn: async () => {
      const response = await api.get('/admin/analytics');
      return response.data;
    },
  });
  
  if (!analytics) return <div>Loading...</div>;
  
  const COLORS = ['#1B3A6B', '#2E75B6', '#0D7A8A', '#1E7E4A', '#D46900'];
  
  return (
    <div className="p-6 space-y-6">
      <h1 className="text-2xl font-bold">Analytics Dashboard</h1>
      
      {/* Summary Cards */}
      <div className="grid grid-cols-4 gap-4">
        <Card>
          <CardHeader className="pb-2">
            <CardTitle className="text-sm font-medium text-gray-600">Total Claims</CardTitle>
          </CardHeader>
          <CardContent>
            <p className="text-3xl font-bold">{analytics.total_claims}</p>
          </CardContent>
        </Card>
        
        <Card>
          <CardHeader className="pb-2">
            <CardTitle className="text-sm font-medium text-gray-600">Avg Processing Time</CardTitle>
          </CardHeader>
          <CardContent>
            <p className="text-3xl font-bold">{analytics.avg_processing_time}s</p>
          </CardContent>
        </Card>
        
        <Card>
          <CardHeader className="pb-2">
            <CardTitle className="text-sm font-medium text-gray-600">Fraud Detected</CardTitle>
          </CardHeader>
          <CardContent>
            <p className="text-3xl font-bold text-red-600">{analytics.fraud_count}</p>
          </CardContent>
        </Card>
        
        <Card>
          <CardHeader className="pb-2">
            <CardTitle className="text-sm font-medium text-gray-600">Total Est. Cost</CardTitle>
          </CardHeader>
          <CardContent>
            <p className="text-3xl font-bold">Rs. {analytics.total_cost.toLocaleString()}</p>
          </CardContent>
        </Card>
      </div>
      
      {/* Charts */}
      <div className="grid grid-cols-2 gap-6">
        <Card>
          <CardHeader>
            <CardTitle>Claims by City</CardTitle>
          </CardHeader>
          <CardContent>
            <ResponsiveContainer width="100%" height={300}>
              <BarChart data={analytics.claims_by_city}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="city" />
                <YAxis />
                <Tooltip />
                <Bar dataKey="count" fill="#2E75B6" />
              </BarChart>
            </ResponsiveContainer>
          </CardContent>
        </Card>
        
        <Card>
          <CardHeader>
            <CardTitle>Damage Type Distribution</CardTitle>
          </CardHeader>
          <CardContent>
            <ResponsiveContainer width="100%" height={300}>
              <PieChart>
                <Pie
                  data={analytics.damage_distribution}
                  cx="50%"
                  cy="50%"
                  labelLine={false}
                  label={(entry) => entry.name}
                  outerRadius={80}
                  fill="#8884d8"
                  dataKey="value"
                >
                  {analytics.damage_distribution.map((entry: any, index: number) => (
                    <Cell key={`cell-${index}`} fill={COLORS[index % COLORS.length]} />
                  ))}
                </Pie>
                <Tooltip />
              </PieChart>
            </ResponsiveContainer>
          </CardContent>
        </Card>
      </div>
    </div>
  );
}
```

**Deliverables:**
- ✅ Assessor review dashboard
- ✅ Interactive damage overlay with Konva.js
- ✅ Admin panel with user management
- ✅ Analytics dashboard with Recharts
- ✅ PDF viewer and download
- ✅ Mobile-responsive design
- ✅ Dark mode support

---

### **PHASE 5: Integration, Testing & Demo** (Weeks 9-10)

#### Week 9: Integration Testing & Deployment

**Docker Compose Setup:**
```yaml
# docker-compose.yml

version: '3.8'

services:
  postgres:
    image: postgres:15
    environment:
      POSTGRES_USER: autoclaim
      POSTGRES_PASSWORD: autoclaim123
      POSTGRES_DB: autoclaim_db
    ports:
      - "5432:5432"
    volumes:
      - postgres_data:/var/lib/postgresql/data
  
  redis:
    image: redis:7.2-alpine
    ports:
      - "6379:6379"
  
  minio:
    image: minio/minio:latest
    command: server /data --console-address ":9001"
    environment:
      MINIO_ROOT_USER: minioadmin
      MINIO_ROOT_PASSWORD: minioadmin123
    ports:
      - "9000:9000"
      - "9001:9001"
    volumes:
      - minio_data:/data
  
  backend:
    build:
      context: ./backend
      dockerfile: Dockerfile
    environment:
      DATABASE_URL: postgresql://autoclaim:autoclaim123@postgres:5432/autoclaim_db
      REDIS_URL: redis://redis:6379/0
      MINIO_ENDPOINT: minio:9000
      CELERY_BROKER_URL: redis://redis:6379/1
    ports:
      - "8000:8000"
    depends_on:
      - postgres
      - redis
      - minio
    volumes:
      - ./ml/models:/app/models
  
  celery_worker:
    build:
      context: ./backend
      dockerfile: Dockerfile
    command: celery -A app.workers.celery_app worker --loglevel=info -Q inference_queue
    environment:
      DATABASE_URL: postgresql://autoclaim:autoclaim123@postgres:5432/autoclaim_db
      REDIS_URL: redis://redis:6379/0
      CELERY_BROKER_URL: redis://redis:6379/1
    depends_on:
      - postgres
      - redis
      - backend
    volumes:
      - ./ml/models:/app/models
  
  frontend:
    build:
      context: ./frontend
      dockerfile: Dockerfile
    ports:
      - "3000:3000"
    depends_on:
      - backend
  
  nginx:
    image: nginx:alpine
    ports:
      - "80:80"
    volumes:
      - ./docker/nginx/nginx.conf:/etc/nginx/nginx.conf
    depends_on:
      - backend
      - frontend

volumes:
  postgres_data:
  minio_data:
```

**Backend Dockerfile:**
```dockerfile
# backend/Dockerfile

FROM python:3.10-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    libpq-dev \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application
COPY . .

# Run migrations and start server
CMD alembic upgrade head && \
    uvicorn app.main:app --host 0.0.0.0 --port 8000 --workers 4
```

**CI/CD Pipeline (GitHub Actions):**
```yaml
# .github/workflows/ci-cd.yml

name: CI/CD Pipeline

on:
  push:
    branches: [ main, develop ]
  pull_request:
    branches: [ main, develop ]

jobs:
  test-backend:
    runs-on: ubuntu-latest
    
    services:
      postgres:
        image: postgres:15
        env:
          POSTGRES_USER: test
          POSTGRES_PASSWORD: test
          POSTGRES_DB: test_db
        ports:
          - 5432:5432
      
      redis:
        image: redis:7.2-alpine
        ports:
          - 6379:6379
    
    steps:
    - uses: actions/checkout@v3
    
    - name: Set up Python
      uses: actions/setup-python@v4
      with:
        python-version: '3.10'
    
    - name: Install dependencies
      run: |
        cd backend
        pip install -r requirements.txt
        pip install pytest pytest-cov
    
    - name: Run tests
      env:
        DATABASE_URL: postgresql://test:test@localhost:5432/test_db
        REDIS_URL: redis://localhost:6379/0
      run: |
        cd backend
        pytest tests/ -v --cov=app --cov-report=xml
    
    - name: Upload coverage
      uses: codecov/codecov-action@v3
      with:
        file: ./backend/coverage.xml
  
  test-frontend:
    runs-on: ubuntu-latest
    
    steps:
    - uses: actions/checkout@v3
    
    - name: Set up Node.js
      uses: actions/setup-node@v3
      with:
        node-version: '20'
    
    - name: Install dependencies
      run: |
        cd frontend
        npm ci
    
    - name: Run tests
      run: |
        cd frontend
        npm run test
    
    - name: Build
      run: |
        cd frontend
        npm run build
  
  deploy:
    needs: [test-backend, test-frontend]
    runs-on: ubuntu-latest
    if: github.ref == 'refs/heads/main'
    
    steps:
    - uses: actions/checkout@v3
    
    - name: Deploy to production
      run: |
        # Deploy commands (Railway, Render, etc.)
        echo "Deploying to production..."
```

**Load Testing:**
```python
# scripts/load_test.py

from locust import HttpUser, task, between
import random

class ClaimUser(HttpUser):
    wait_time = between(1, 3)
    
    def on_start(self):
        # Login
        response = self.client.post("/api/v1/auth/login", data={
            "username": "test@example.com",
            "password": "testpass123"
        })
        self.token = response.json()["access_token"]
    
    @task(3)
    def list_claims(self):
        self.client.get(
            "/api/v1/claims/",
            headers={"Authorization": f"Bearer {self.token}"}
        )
    
    @task(1)
    def create_claim(self):
        files = [
            ('images', ('front.jpg', open('test_images/front.jpg', 'rb'), 'image/jpeg')),
            ('images', ('rear.jpg', open('test_images/rear.jpg', 'rb'), 'image/jpeg')),
            ('images', ('left.jpg', open('test_images/left.jpg', 'rb'), 'image/jpeg')),
            ('images', ('right.jpg', open('test_images/right.jpg', 'rb'), 'image/jpeg')),
        ]
        
        data = {
            'vehicle_reg': f'MH{random.randint(10,99)}AB{random.randint(1000,9999)}',
            'incident_type': random.choice(['collision', 'theft', 'weather']),
            'incident_date': '2024-04-15',
            'incident_city': random.choice(['Mumbai', 'Delhi', 'Bangalore', 'Pune'])
        }
        
        self.client.post(
            "/api/v1/claims/",
            files=files,
            data=data,
            headers={"Authorization": f"Bearer {self.token}"}
        )

# Run: locust -f scripts/load_test.py --host=http://localhost:8000
# Target: 50 concurrent users, <5 sec response time
```

**Performance Profiling:**
```python
# scripts/profile_inference.py

import time
import cProfile
import pstats
from ml.inference.pipeline import DamageAssessmentPipeline

def profile_pipeline():
    pipeline = DamageAssessmentPipeline()
    
    test_images = [
        'test_images/front.jpg',
        'test_images/rear.jpg',
        'test_images/left.jpg',
        'test_images/right.jpg'
    ]
    
    # Warm-up
    pipeline.process_claim(test_images)
    
    # Profile
    profiler = cProfile.Profile()
    profiler.enable()
    
    start = time.time()
    result = pipeline.process_claim(test_images)
    end = time.time()
    
    profiler.disable()
    
    print(f"Total time: {end - start:.2f} seconds")
    print(f"Damages detected: {len(result['damages'])}")
    
    # Print top 20 slowest functions
    stats = pstats.Stats(profiler)
    stats.sort_stats('cumulative')
    stats.print_stats(20)

if __name__ == '__main__':
    profile_pipeline()

# Target: <3 seconds end-to-end
```

#### Week 10: Demo Prep & Documentation

**Demo Scenario Preparation:**
```python
# scripts/seed_demo_data.py

from app.db.session import SessionLocal
from app.models.user import User
from app.models.claim import Claim, ClaimImage, DamageDetection
from app.core.security import get_password_hash
import uuid
from datetime import datetime, timedelta

db = SessionLocal()

# Create demo users
demo_users = [
    {
        'email': 'customer@demo.com',
        'password': 'demo123',
        'full_name': 'Raj Kumar',
        'role': 'customer'
    },
    {
        'email': 'assessor@demo.com',
        'password': 'demo123',
        'full_name': 'Priya Singh',
        'role': 'assessor'
    },
    {
        'email': 'admin@demo.com',
        'password': 'demo123',
        'full_name': 'Admin User',
        'role': 'admin'
    }
]

for user_data in demo_users:
    user = User(
        email=user_data['email'],
        hashed_password=get_password_hash(user_data['password']),
        full_name=user_data['full_name'],
        role=user_data['role']
    )
    db.add(user)

db.commit()

# Seed demo claims
demo_claims = [
    {
        'name': 'Scenario A - Mild Damage',
        'description': '2 scratches, 1 minor dent',
        'damages': [
            {'part': 'bumper_front', 'type': 'scratch', 'severity': 'minor', 'cost': (1500, 2000)},
            {'part': 'door_fl', 'type': 'dent', 'severity': 'minor', 'cost': (1500, 3000)},
        ],
        'fraud_score': 0.02
    },
    {
        'name': 'Scenario B - Moderate Damage',
        'description': 'Cracked headlight, dented hood, broken mirror',
        'damages': [
            {'part': 'headlight_left', 'type': 'crack', 'severity': 'moderate', 'cost': (5000, 8000)},
            {'part': 'hood', 'type': 'dent', 'severity': 'moderate', 'cost': (8000, 12000)},
            {'part': 'mirror_left', 'type': 'shatter', 'severity': 'moderate', 'cost': (3000, 5000)},
        ],
        'fraud_score': 0.11
    },
    {
        'name': 'Scenario C - Severe Damage',
        'description': 'Deformed front, shattered windshield',
        'damages': [
            {'part': 'bumper_front', 'type': 'deformation', 'severity': 'severe', 'cost': (20000, 30000)},
            {'part': 'windshield', 'type': 'shatter', 'severity': 'severe', 'cost': (15000, 20000)},
            {'part': 'hood', 'type': 'deformation', 'severity': 'severe', 'cost': (25000, 35000)},
        ],
        'fraud_score': 0.05
    },
    {
        'name': 'Scenario D - FRAUD FLAGGED',
        'description': 'Claims front collision but rear damage only',
        'damages': [
            {'part': 'bumper_rear', 'type': 'dent', 'severity': 'moderate', 'cost': (8000, 12000)},
            {'part': 'trunk', 'type': 'dent', 'severity': 'minor', 'cost': (4000, 6000)},
        ],
        'fraud_score': 0.87
    }
]

customer = db.query(User).filter(User.email == 'customer@demo.com').first()

for scenario in demo_claims:
    # Create claim
    # Add images and damages
    # ...

db.commit()
print("Demo data seeded successfully!")
```

**College Report Template:**
```markdown
# AutoClaim Vision - Final Year Project Report

## Abstract
AutoClaim Vision is an AI-powered vehicle damage detection and insurance cost estimation system that reduces claim processing time from 7 days to 3 seconds...

## 1. Introduction
### 1.1 Problem Statement
### 1.2 Objectives
### 1.3 Scope

## 2. Literature Review
### 2.1 Computer Vision in Insurance
### 2.2 Object Detection Techniques
### 2.3 Cost Estimation Methods

## 3. System Design
### 3.1 Architecture Overview
### 3.2 Technology Stack
### 3.3 Data Flow Diagram

## 4. AI/ML Methodology
### 4.1 Dataset Collection & Annotation
### 4.2 Model Selection & Training
### 4.3 Inference Pipeline

## 5. Implementation
### 5.1 Backend Development (FastAPI)
### 5.2 Frontend Development (React)
### 5.3 Database Design

## 6. Results & Evaluation
### 6.1 Model Performance Metrics
### 6.2 System Performance
### 6.3 User Testing

## 7. Conclusion
### 7.1 Achievements
### 7.2 Limitations
### 7.3 Future Work

## References
## Appendix
```

**Deliverables:**
- ✅ Docker Compose one-command deployment
- ✅ CI/CD pipeline with GitHub Actions
- ✅ Load testing results (50 concurrent users)
- ✅ Performance profiling report
- ✅ 15 demo claims seeded
- ✅ Demo video recorded (backup)
- ✅ College project report (40+ pages)
- ✅ Final presentation slides

---

## 4. Workflow Diagram

**Daily Workflow (Per Intern):**

```
Morning (9 AM - 12 PM):
├── Daily Standup (15 min)
│   ├── What I did yesterday
│   ├── What I'm doing today
│   └── Any blockers
├── Development Work (2h 45min)
│   ├── Write code
│   ├── Run tests
│   └── Push to feature branch
└── Code Review (30 min)

Afternoon (1 PM - 5 PM):
├── Development/Integration (3h)
│   ├── Continue assigned tasks
│   ├── Debug issues
│   └── Integration testing
├── Documentation (30 min)
│   └── Update README, comments
└── Team Sync (30 min)
    ├── Demo progress
    ├── Discuss tomorrow's plan
    └── Update project board
```

**Git Workflow:**
```
main (production)
  └── develop (integration)
      ├── feature/damage-detection-model
      ├── feature/backend-claims-api
      ├── feature/frontend-dashboard
      └── feature/pdf-report-generation

Process:
1. Create feature branch from develop
2. Work on feature
3. Create Pull Request to develop
4. Code review + approval
5. Merge to develop
6. Test on develop
7. Merge develop → main (weekly)
```

**Model Training Workflow:**
```
1. Data Preparation
   ├── Download datasets
   ├── Annotate custom data (Label Studio)
   ├── Quality check annotations
   └── Export to YOLOv8 format

2. Experimentation (MLflow tracking)
   ├── Baseline model training
   ├── Hyperparameter tuning
   ├── Augmentation experiments
   └── Log all metrics to MLflow

3. Evaluation
   ├── Test on holdout set
   ├── Generate confusion matrix
   ├── Identify weak classes
   └── Per-class mAP analysis

4. Production
   ├── Select best model from MLflow
   ├── Export to ONNX
   ├── Benchmark inference speed
   ├── Register in MLflow Registry
   └── Deploy to backend

5. Monitoring
   ├── Track inference latency
   ├── Log prediction distribution
   ├── Detect model drift
   └── Retrain if needed
```

---

## 5. How to Start the Project

### Step-by-Step Kickoff Guide

**Day 1 - Environment Setup (All Interns)**

1. **Install Core Tools:**
```bash
# Python 3.10
curl https://www.python.org/ftp/python/3.10.14/Python-3.10.14.tgz | tar xz
cd Python-3.10.14
./configure && make && sudo make install

# Node.js 20 LTS
curl -fsSL https://deb.nodesource.com/setup_20.x | sudo -E bash -
sudo apt-get install -y nodejs

# PostgreSQL 15
sudo apt-get install postgresql-15 postgresql-client-15

# Redis
sudo apt-get install redis-server

# Docker
curl -fsSL https://get.docker.com -o get-docker.sh
sudo sh get-docker.sh
```

2. **Clone Repository & Setup:**
```bash
# Clone repo (create first)
git clone https://github.com/your-org/autoclaim-vision.git
cd autoclaim-vision

# Create virtual environment
python3.10 -m venv venv
source venv/bin/activate

# Install Python dependencies
cd backend
pip install -r requirements.txt
cd ..

# Install Node dependencies
cd frontend
npm install
cd ..
```

3. **Download Datasets:**
```bash
# Create datasets directory
mkdir -p ml/datasets

# Download CarDD dataset
cd ml/datasets
kaggle datasets download -d your-dataset/cardd
unzip cardd.zip -d cardd/

# Download VCoR
kaggle datasets download -d your-dataset/vcor
unzip vcor.zip -d vcor/

# Download Stanford Cars
wget http://ai.stanford.edu/~jkrause/car196/car_ims.tgz
tar -xzf car_ims.tgz

# Download CarParts-1K
# (Roboflow Universe link)

# Download Open Images subset
# (Custom script using OID v7)
```

**Day 2-7 - Begin Development**

4. **Set Up Label Studio:**
```bash
docker run -it -p 8080:8080 \
  -v $(pwd)/ml/datasets:/label-studio/data \
  heartexai/label-studio:latest

# Access at http://localhost:8080
# Create project: AutoClaim Damage Annotation
# Configure labels:
#   - Damage types: 8 classes
#   - Car parts: 30 classes
#   - Annotation type: Polygon segmentation + Bounding boxes
```

5. **Start Backend Development:**
```bash
cd backend

# Create database
createdb autoclaim_db

# Run migrations
alembic upgrade head

# Start development server
uvicorn app.main:app --reload --port 8000
```

6. **Start Frontend Development:**
```bash
cd frontend

# Create .env file
echo "VITE_API_URL=http://localhost:8000/api/v1" > .env

# Start dev server
npm run dev
# Access at http://localhost:3000
```

**Week 2 - Annotation Sprint**

7. **Divide Annotation Work:**
```
Intern 1 (AI Lead):
  - Annotate 150 images
  - Focus on: Scratches, Dents, Cracks
  - Quality check: 30 images from Intern 2

Intern 2:
  - Annotate 150 images
  - Focus on: Shatter, Deformation, Corrosion
  - Quality check: 30 images from Intern 3

Intern 3 (Frontend Lead):
  - Annotate 100 images
  - Focus on: Paint Damage, Missing Parts
  - Quality check: 30 images from Intern 1
```

**Week 3+ - Sprint Execution**

8. **Follow Phase Plan:**
- Refer to detailed phase breakdown above
- Daily standups at 9 AM
- Code reviews before merging
- Weekly demos on Fridays
- MLflow tracking for all experiments
- Document everything in Notion/Confluence

---

## 6. Team Roles & Responsibilities

### Intern 1 - AI/ML Lead

**Primary Focus:** Model training, MLflow, Celery workers

**Responsibilities:**
- Train all AI models (Weeks 3-5)
- Set up MLflow experiment tracking
- ONNX model export and optimization
- Celery worker implementation
- Inference pipeline integration
- Docker deployment configuration
- Performance profiling and optimization

**Key Deliverables:**
- 7 trained AI models with mAP > targets
- Complete inference pipeline < 3 sec
- MLflow registry with production models
- Docker Compose deployment working

### Intern 2 - Backend & ML

**Primary Focus:** FastAPI backend, cost estimation, fraud detection

**Responsibilities:**
- FastAPI application structure (Week 6)
- SQLAlchemy models and Alembic migrations
- Authentication & authorization (JWT)
- Claims API endpoints
- Cost estimation engine (rule-based + XGBoost)
- Fraud detection model training
- PDF report generation (ReportLab)
- Database design and optimization

**Key Deliverables:**
- Complete REST API with 80% test coverage
- Cost estimation engine with MAPE < 30%
- Fraud detection model with AUC > 0.75
- Auto-generated PDF reports

### Intern 3 - Frontend Lead

**Primary Focus:** React UI/UX, visualization, user experience

**Responsibilities:**
- React + TypeScript setup (Week 7)
- Auth pages (Login, Register)
- Claim creation wizard (multi-step form)
- Photo upload component (Dropzone)
- Real-time status tracking (WebSocket)
- Damage viewer with Konva.js overlays
- Assessor review dashboard
- Admin panel + Analytics (Recharts)
- PDF viewer integration
- Mobile responsive design + dark mode

**Key Deliverables:**
- Complete customer-facing UI
- Assessor review workflow
- Analytics dashboard
- 100% mobile-responsive
- Dark mode support

---

## 7. Critical Success Factors

### Must-Have Features (Non-Negotiable)

1. **AI Pipeline:**
   - ✅ Damage detection mAP50 > 0.75
   - ✅ Part classifier accuracy > 80%
   - ✅ Inference time < 3 seconds (CPU)
   - ✅ All 7 models working end-to-end

2. **Backend:**
   - ✅ Claims creation API working
   - ✅ Async Celery processing
   - ✅ PDF report generation
   - ✅ WebSocket real-time updates

3. **Frontend:**
   - ✅ Photo upload (4-6 images)
   - ✅ Real-time status tracking
   - ✅ Damage visualization
   - ✅ Results page with cost estimate

4. **Demo:**
   - ✅ 4 demo scenarios prepared
   - ✅ Backup video recorded
   - ✅ Docker Compose one-command startup
   - ✅ 10-minute demo script rehearsed

### Nice-to-Have Features (If Time Permits)

- Email notifications
- SMS updates
- WhatsApp bot integration
- Excel export of damage reports
- Multi-language support
- Advanced analytics (city heatmaps)
- Mobile app (React Native)

---

## 8. Risk Mitigation Strategy

### High-Priority Risks

**Risk 1: Insufficient Data for Rare Damage Types**
- **Mitigation:**
  - Over-collect rare classes (coordinate with body shops)
  - Synthetic data augmentation (copy-paste damaged regions)
  - Transfer learning from similar damage types
  - Weighted sampling during training

**Risk 2: Inference Too Slow on CPU Demo Machine**
- **Mitigation:**
  - ONNX export + INT8 quantization
  - Pre-warm models on server startup
  - Use Google Colab GPU as fallback
  - Reduce image resolution if needed (640px instead of 800px)
  - Model distillation (smaller student models)

**Risk 3: Poor Accuracy on Indian Vehicle Types**
- **Mitigation:**
  - Collect 400+ custom Indian vehicle photos
  - Fine-tune on local data
  - Weighted sampling for Indian models
  - Test extensively on Maruti, Hyundai, Tata

**Risk 4: Demo Deployment Failure**
- **Mitigation:**
  - Record 5-minute backup video (all scenarios)
  - Docker Compose localhost fallback
  - Test deployment 3 days before demo
  - Have backup laptop ready

**Risk 5: Integration Issues Between Components**
- **Mitigation:**
  - Start E2E testing from Week 8
  - Daily integration checks
  - Docker Compose environment for local testing
  - Dedicated integration test suite

---

## Success Metrics

**Technical Metrics:**
- Damage Detector mAP50: > 0.75 ✅
- Part Classifier Accuracy: > 80% ✅
- Severity Model F1: > 0.74 ✅
- Cost Estimation MAPE: < 30% ✅
- Fraud Detection AUC: > 0.75 ✅
- Inference Latency: < 3 sec ✅
- Backend Test Coverage: > 80% ✅

**Business Metrics:**
- Claim processing time: 7 days → 3 seconds (99.9% reduction)
- Operational cost: Rs. 2,000-5,000 → Rs. 10 per claim (95% reduction)
- Fraud detection rate: 10-15% baseline → >85% detection (5x improvement)

**Demo Success:**
- 4/4 scenarios execute successfully
- No critical bugs during demo
- <5 second response time visible to audience
- PDF report downloads successfully
- College committee impressed (subjective but critical!)

---

## Final Checklist (Week 10)

**Code:**
- [ ] All models trained and registered in MLflow
- [ ] Backend API 80%+ test coverage
- [ ] Frontend builds without errors
- [ ] Docker Compose starts all services correctly
- [ ] GitHub Actions CI/CD pipeline passing

**Data:**
- [ ] 15 demo claims seeded in database
- [ ] 4 demo scenario image sets ready
- [ ] 1 fraud-flagged demo claim prepared

**Documentation:**
- [ ] README with setup instructions
- [ ] API documentation (Swagger/OpenAPI)
- [ ] College report written (40+ pages)
- [ ] Presentation slides ready (15-20 slides)

**Demo:**
- [ ] Demo script written and rehearsed
- [ ] Backup video recorded (5 minutes)
- [ ] Deployment tested on demo machine
- [ ] All team members know their demo parts

**Presentation:**
- [ ] Slides cover: Problem, Solution, Tech Stack, AI Models, Results, Demo, Future
- [ ] Live demo laptop ready + backup laptop
- [ ] HDMI/USB-C adapters tested
- [ ] Internet connection tested (if needed)
- [ ] Questions anticipated and answers prepared

---

## Resources & References

**Datasets:**
- CarDD: https://github.com/your-link/cardd
- VCoR: https://www.kaggle.com/your-link/vcor
- Stanford Cars: https://ai.stanford.edu/~jkrause/cars/car_dataset.html
- CarParts-1K: Roboflow Universe

**Tools:**
- Label Studio: https://labelstud.io/
- MLflow: https://mlflow.org/
- YOLOv8: https://github.com/ultralytics/ultralytics
- FastAPI: https://fastapi.tiangolo.com/
- React: https://react.dev/

**Learning Resources:**
- Computer Vision: CS231n Stanford
- Object Detection: PyImageSearch tutorials
- FastAPI Tutorial: Official FastAPI docs
- React + TypeScript: React TypeScript Cheatsheet
- Docker: Docker Curriculum

---

## Contact & Support

**Mentor:**
- Email: mentor@company.com
- Office Hours: Mon/Wed/Fri 2-4 PM
- Slack: #autoclaim-vision

**Team Communication:**
- Daily Standup: 9:00 AM IST (Google Meet)
- Weekly Demo: Friday 4:00 PM IST
- Code Review: GitHub Pull Requests
- Slack Channel: #autoclaim-dev

---

**Good luck with AutoClaim Vision! This is a production-grade project that solves a real business problem. Build it well, and you'll have a startup-ready product on Day 1 of graduation. Focus on data quality first, model performance second, and polish last. You've got this! 🚀**
