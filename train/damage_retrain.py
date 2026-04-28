from ultralytics import YOLO
import torch

def continue_training():
    model = YOLO("../runs/damage_v2/train/weights/last.pt")

    model.train(
        data="../yaml/damage.yaml",
        epochs=50,
        imgsz=640,
        batch=12,              
        workers=2,
        lr0=0.0001,
        warmup_epochs=0,
        device=0,
        project="D:/Files/My Projects/AutoClaim Vision/AutoClaim Vision/runs/damage_v2",
        name="resume_50_to_100",
        exist_ok=True
    )

if __name__ == "__main__":
    torch.cuda.empty_cache()
    print("🛰️ Continuing training from Epoch 50 baseline...")
    continue_training()