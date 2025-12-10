import yaml
import shutil
import os
from pathlib import Path

# --- CONFIGURATION ---
# We use pathlib for robust cross-platform paths (Windows/Mac/Linux friendly)
CONFIG_PATH = Path('configs/data_cleaning.yaml')
OUTPUT_DIR = Path('datasets/combined_ppe')

def normalize_dataset():
    """
    Main ETL function: Extracts raw data, Transforms labels, Loads into final dataset.
    """
    
    # 1. Load the Rulebook
    if not CONFIG_PATH.exists():
        print(f"❌ Error: Config file not found at {CONFIG_PATH}")
        return

    with open(CONFIG_PATH) as f:
        config = yaml.safe_load(f)
    
    target_classes = config['target_classes']
    print(f"\n🚀 Starting Data Normalization")
    print(f"🎯 Target Schema: {target_classes}")
    print(f"📂 Output Directory: {OUTPUT_DIR}\n")

    # 2. Reset the Output Directory (Clean Slate)
    # This ensures we don't have leftover garbage from previous runs
    if OUTPUT_DIR.exists():
        print("🧹 Cleaning previous output...")
        shutil.rmtree(OUTPUT_DIR)

    # 3. Loop through every source defined in YAML
    for source_name, source_data in config['sources'].items():
        print(f"Processing Source: [{source_name}]")
        
        raw_path = Path(source_data['path'])
        mapping = source_data['mapping'] # The dictionary {old_id: "ClassName"}
        
        # 4. Define the Split Strategy
        # Map source folder names (keys) to final folder names (values)
        # We merge 'test', 'valid', and 'val' all into the standard 'val' folder
        split_map = {
            'train': 'train',
            'valid': 'val',
            'test': 'val',
            'val': 'val'
        }

        # Check every possible split folder in the source
        for src_split_name, dest_split_name in split_map.items():
            
            # Construct full paths
            src_labels_dir = raw_path / src_split_name / 'labels'
            src_images_dir = raw_path / src_split_name / 'images'
            
            # Destination paths
            dest_labels_dir = OUTPUT_DIR / 'labels' / dest_split_name
            dest_images_dir = OUTPUT_DIR / 'images' / dest_split_name

            # Skip if this split doesn't exist in the raw dataset
            if not src_labels_dir.exists():
                continue

            # Create destination folders if they don't exist yet
            dest_labels_dir.mkdir(parents=True, exist_ok=True)
            dest_images_dir.mkdir(parents=True, exist_ok=True)

            print(f"  └── Merging '{src_split_name}' ──> '{dest_split_name}'")

            files_processed = 0

            # 5. Process every Label File
            for label_file in src_labels_dir.glob('*.txt'):
                new_lines = []
                has_valid_data = False
                
                # Read the raw label file
                try:
                    with open(label_file, 'r') as f:
                        lines = f.readlines()
                except Exception as e:
                    print(f"⚠️ Error reading {label_file}: {e}")
                    continue
                
                # Transform lines
                for line in lines:
                    parts = line.strip().split()
                    if not parts: continue # Skip empty lines
                    
                    old_id = int(parts[0])
                    
                    # CHECK: Is this old ID in our mapping dictionary?
                    if old_id in mapping:
                        class_name = mapping[old_id]
                        
                        # Find the new ID based on our Target Class list order
                        # e.g., if target_classes=['Hardhat',...], then Hardhat is index 0
                        new_id = target_classes.index(class_name)
                        
                        # Reconstruct the line with NEW ID and OLD Coordinates
                        new_lines.append(f"{new_id} {' '.join(parts[1:])}\n")
                        has_valid_data = True
                
                # 6. Save Data (Only if we found relevant objects)
                if has_valid_data:
                    # Write the new clean label file
                    (dest_labels_dir / label_file.name).write_text(''.join(new_lines))
                    
                    # Copy the corresponding image
                    # We check for common extensions
                    img_name = label_file.stem
                    image_found = False
                    for ext in ['.jpg', '.jpeg', '.png', '.bmp']:
                        src_img = src_images_dir / (img_name + ext)
                        if src_img.exists():
                            shutil.copy(src_img, dest_images_dir / (img_name + ext))
                            image_found = True
                            break
                    
                    if image_found:
                        files_processed += 1

            print(f"      ✅ Processed {files_processed} images/labels")

    print(f"\n✨ SUCCESS! Combined dataset is ready at: {OUTPUT_DIR}")
    print("Next Step: Update 'data.yaml' to point to this folder and run training.")

if __name__ == "__main__":
    normalize_dataset()