from ultralytics import YOLO, settings
import config
import os

def on_train_epoch_end(trainer):
    print(f"📉 [SafetyMarshal] Epoch {trainer.epoch + 1}/{trainer.epochs} complete.")

def train_model():
    # Privacy check: Turn off external logging
    settings.update({'wandb': False, 'clearml': False, 'comet': False})
    
    print(f"🚀 Starting PRO Training for {config.PROJECT_NAME}...")
    model = YOLO(config.MODEL_NAME)
    model.add_callback("on_train_epoch_end", on_train_epoch_end)

    results = model.train(
        data=str(config.DATA_YAML_PATH),
        imgsz=config.IMAGE_SIZE,
        epochs=config.EPOCHS,
        batch=config.BATCH_SIZE,
        patience=config.PATIENCE,
        project=str(config.PROJECT_ROOT / "models" / "runs"),
        name=config.PROJECT_NAME,
        
        # --- THE PRO AUGMENTATION INJECTION ---
        augment=config.AUGMENT,
        mosaic=config.MOSAIC,
        mixup=config.MIXUP,
        copy_paste=config.COPY_PASTE,
        degrees=config.DEGREES,
        hsv_h=config.HSV_H,
        hsv_s=config.HSV_S,
        hsv_v=config.HSV_V,
        
        optimizer='auto', # YOLOv8 auto-selects AdamW for this size usually
        seed=42,
          # Ensure this is reading your new '4'
        workers=config.WORKERS,   # Set to 1
        
        # --- SAFETY SETTINGS ---
        amp=True,          # Keep True (Mixed Precision uses LESS memory)
        cache=False,       # DISABLE RAM caching. Reads from disk instead (Slower but Safer)
        exist_ok=True,     # Overwrite existing folder so you don't make 'project2', 'project3'
    )

    print(f"✅ Training Complete. Best weights: models/runs/{config.PROJECT_NAME}/weights/best.pt")

if __name__ == "__main__":
    train_model()