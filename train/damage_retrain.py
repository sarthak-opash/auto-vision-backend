from ultralytics import YOLO
import torch

def continue_training():
    # 1. Load your current weights (the ones that finished 50 epochs)
    model = YOLO("../runs/damage_v2/train/weights/last.pt")
 
    # 2. Train for the remaining 50 epochs
    model.train(
        data="../yaml/damage.yaml",
        epochs=50,             # 50 more epochs to reach your 100 goal
        imgsz=640,
        batch=12,              
        workers=2,             
        # --- RESUME SIMULATION SETTINGS ---
        lr0=0.0001,            # START LOW: Use the final LR from your last run
        warmup_epochs=0,       # SKIP WARMUP: Don't let the model "reset" its brain
        # ----------------------------------
        device=0,
        project="D:/Files/My Projects/AutoClaim Vision/AutoClaim Vision/runs/damage_v2",
        name="resume_50_to_100",
        exist_ok=True
    )

if __name__ == "__main__":
    torch.cuda.empty_cache()
    print("🛰️ Continuing training from Epoch 50 baseline...")
    continue_training()