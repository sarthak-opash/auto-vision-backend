from ultralytics import YOLO

def train_model():
    model = YOLO("yolo11s.pt")   # Best for RTX 3050 Ti 4GB

    model.train(
        data="yaml/cardd.yaml",
        epochs=100,
        imgsz=640,
        batch=6,
        device=0,
        workers=4,
        patience=20
    )

if __name__ == "__main__":

    print("🚀 Starting Car Damage Detection Training...")

    train_model()

    print("🎉 Training Completed!")
    print("Best Model Path:")
    print("runs/detect/train/weights/best.pt")