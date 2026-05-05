from ultralytics import YOLO
import torch

def train_model():
    model = YOLO("yolo11s-seg.pt")

    model.train(
        data="cardd_segmentation_dataset/data.yaml",

        # ── Core Training ──────────────────────────────────────────
        epochs=150,
        imgsz=640,
        batch=12,
        device=0,
        workers=3,

        # ── Optimizer ─────────────────────────────────────────────
        optimizer="AdamW",
        lr0=0.001,
        lrf=0.1,
        weight_decay=0.0005,
        momentum=0.937,

        # ── Scheduler ─────────────────────────────────────────────
        warmup_epochs=5.0,
        warmup_momentum=0.8,
        warmup_bias_lr=0.05,
        cos_lr=True,

        # ── Regularization ────────────────────────────────────────
        patience=50,

        # ── Augmentation ──────────────────────────────────────────
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4,

        degrees=10.0,
        translate=0.1,
        scale=0.5,
        shear=2.0,
        perspective=0.0005,

        fliplr=0.5,
        flipud=0.0,

        mosaic=1.0,
        mixup=0.15,
        copy_paste=0.3,
        close_mosaic=20,

        # ── Segmentation Specific ─────────────────────────────────
        overlap_mask=True,
        mask_ratio=4,

        # ── Hardware ───────────────────────
        amp=True,
        cache="disk",
        fraction=1.0,

        # ── Output ────────────────────────────────────────────────
        project="D:/Files/My Projects/AutoClaim Vision/AutoClaim Vision/runs/damage_seg_v1",
        name="epoch_150",
        save=True,
        save_period=10,
        plots=True,
        val=True,
    )

if __name__ == "__main__":
    torch.cuda.empty_cache()

    print("🚀 Starting Car Damage Detection Training...")

    train_model()

    print("🎉 Training Completed!")