from ultralytics import YOLO

def main():
    model = YOLO("../models/yolo11s.pt")

    model.train(
        data="../yaml/parts.yaml",
        epochs=100,
        imgsz=640,
        batch=12,        # try 12 first, fallback to 10 or 8 if OOM
        device=0,
        workers=2,       # test 2 on Windows
        cache=True,      # cache images in RAM/disk speeds loading
        amp=True,        # mixed precision (usually already on)
        patience=20,
        project="../models",
        name="parts"
    )

if __name__ == "__main__":
    main()