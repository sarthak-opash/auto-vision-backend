from ultralytics import YOLO
import torch

def train_model():
    model = YOLO("yolo11s.pt")
 
    model.train(
        data="../yaml/damage.yaml",
        epochs=50,
        imgsz=640,
        batch=12,
        workers = 2,
        nbs=64,
        optimizer='AdamW',
        lr0=0.001,
        cos_lr=True,
        patience=50,
        cls=1.5,           
        label_smoothing=0.1,
        dropout=0.1,
        mosaic=1.0,
        mixup=0.1,
        degrees=10.0,
        scale=0.5,
        fliplr=0.5,
        # The Finish
        close_mosaic=10,   
        device=0,
        project = "D:/Files/My Projects/AutoClaim Vision/AutoClaim Vision/runs/damage_v2",
        name = "",
        exist_ok = True
    )

def test_model():
    model = YOLO("../runs/damage/weights/best.pt") 

    print("📊 Evaluating on Test Set...")
    metrics = model.val(
        data="../yaml/damage.yaml", 
        split='test',
        device=0,
        imgsz=640
    )

    print(f"mAP50-95: {metrics.box.map:.4f}")
    print(f"mAP50: {metrics.box.map50:.4f}")

    print("📸 Running Predictions...")
    results = model.predict(
        source="../datasets/damage/test/images",
        conf=0.25,
        save=True,
        device=0,
        stream=True
    )

    for r in results:
        pass


if __name__ == "__main__":
    torch.cuda.empty_cache()

    print("🚀 Starting Car Damage Detection Training...")

    train_model()

    print("🎉 Training Completed!")

    test_model()

    print("Model test completed")