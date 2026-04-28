from ultralytics import YOLO


def train_model():
    model = YOLO("yolo11s.pt")

    model.train(
        # Dataset
        data="damage_v3/damage_v3.yaml",

        # Core Training
        epochs=100,          # max epochs (early stopping may stop sooner)
        imgsz=768,           # best balance for your dataset
        batch=10,            # reduce to 8 if GPU memory issue
        workers=2,
        device=0,

        # Optimizer
        optimizer="SGD",
        lr0=0.01,
        lrf=0.01,
        momentum=0.937,
        weight_decay=0.0005,

        # Scheduler
        cos_lr=True,
        warmup_epochs=5,

        # Early Stop
        patience=60,

        # Loss Weights
        box=7.5,
        cls=0.7,
        dfl=1.5,

        # Regularization
        label_smoothing=0.05,
        dropout=0.0,

        # Augmentations
        mosaic=1.0,
        mixup=0.10,
        copy_paste=0.0,
        degrees=5.0,
        translate=0.10,
        scale=0.50,
        shear=1.0,
        fliplr=0.5,
        hsv_h=0.015,
        hsv_s=0.50,
        hsv_v=0.30,

        # Final Training Phase
        close_mosaic=15,

        # Save Path
        project=r"D:/Files/My Projects/AutoClaim Vision/AutoClaim Vision/runs/damage_v3",
        name="",
        exist_ok=True,

        # Useful
        verbose=True,
        plots=True,
        save=True
    )


if __name__ == "__main__":
    print("🚀 Starting Car Damage Detection Training...")

    train_model()

    print("🎉 Training Completed!")