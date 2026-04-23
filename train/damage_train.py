from ultralytics import YOLO
import torch

def train_model():
    model = YOLO("yolo11s.pt")

    model.train(
        data="../yaml/damage.yaml",
        epochs=100,
        imgsz=640,
        batch=12, 
        device=0,
        workers=2,
        patience=20,
        dropout=0.1,
        optimizer='AdamW',
        amp=True,
        cache=True
    )

def test_model():
    # Load your trained model
    # Note: verify if your folder is 'damage' or the default 'train'
    model = YOLO("../runs/damage/weights/best.pt") 

    print("📊 Evaluating on Test Set...")
    metrics = model.val(
        data="../yaml/damage.yaml", 
        split='test',
        device=0,      # Explicitly use GPU
        imgsz=640      # Match your training size
    )

    print(f"mAP50-95: {metrics.box.map:.4f}")
    print(f"mAP50: {metrics.box.map50:.4f}")

    print("📸 Running Predictions...")
    # Using stream=True is a lifesaver for 4GB VRAM laptops
    results = model.predict(
        source="../datasets/damage/test/images", # Update this to your actual test image path
        conf=0.25,
        save=True,
        device=0,
        stream=True  # Processes images one-by-one to save VRAM
    )
    
    # We must iterate through the generator to actually trigger the saving
    for r in results:
        pass


if __name__ == "__main__":
    # Crucial for laptops to prevent memory fragmentation
    torch.cuda.empty_cache()

    print("🚀 Starting Car Damage Detection Training...")

    # train_model()

    print("🎉 Training Completed!")
    print("Best Model Path:")
    print("runs/detect/train/weights/best.pt")

    test_model()

    print("Model test completed")
    print("result saved in runs/detect/predict/")