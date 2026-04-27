from ultralytics import YOLO
import torch

def train_model():
    model = YOLO("yolo11m.pt")
 
    model.train(
        data="../datasets/damage_v2/data.yaml",
        epochs=50,
        imgsz=640,
        batch=8,           # Try 4, if OOM, drop to 2
        nbs=64,
        optimizer='AdamW',
        lr0=0.001,
        cos_lr=True,
        patience=50,
        # Classification priority
        cls=1.5,           
        label_smoothing=0.1,
        dropout=0.1,       
        # Augmentations for 31 classes
        mosaic=1.0,
        mixup=0.1,         # Slightly lower mixup for the larger dataset
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

    train_model()

    print("🎉 Training Completed!")
    print("Best Model Path:")
    print("runs/detect/train/weights/best.pt")

    # test_model()

    # print("Model test completed")
    # print("result saved in runs/detect/predict/")