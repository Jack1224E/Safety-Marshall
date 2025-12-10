from ultralytics import YOLO
import config

def resume_training():
    print("🔄 Resuming Training from last checkpoint...")
    
    # Point to the 'last.pt' file in your specific project folder
    # ADJUST THE PATH below if your folder name is different (e.g., usually 'runs/detect/PPE_Detection_Pro')
    checkpoint_path = config.PROJECT_ROOT / "models" / "runs" / config.PROJECT_NAME / "weights" / "last.pt"
    
    if not checkpoint_path.exists():
        print(f"❌ Error: Checkpoint not found at {checkpoint_path}")
        return

    # Load the half-trained model
    model = YOLO(checkpoint_path)

    # Resume!
    model.train(resume=True)

if __name__ == "__main__":
    resume_training()