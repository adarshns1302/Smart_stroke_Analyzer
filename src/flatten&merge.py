import os
import shutil

# Path to the base directory
base_dir = r'C:\Users\naray\OneDrive\Desktop\Smart Stroke Analyzer\data\final_dataset'

# Loop over all class folders (like back_foot_punch, cover_drive, etc.)
for stroke_type in os.listdir(base_dir):
    stroke_path = os.path.join(base_dir, stroke_type)

    # If inside stroke_type we have subfolders like _01, _02, etc.
    if os.path.isdir(stroke_path):
        subfolders = [f for f in os.listdir(stroke_path) if os.path.isdir(os.path.join(stroke_path, f))]

        if subfolders:
            # Create a new flat folder if it doesn't exist
            flat_dir = os.path.join(base_dir, f"{stroke_type}_flat")
            os.makedirs(flat_dir, exist_ok=True)

            for sub in subfolders:
                subfolder_path = os.path.join(stroke_path, sub)
                for file in os.listdir(subfolder_path):
                    if file.endswith((".jpg", ".jpeg", ".png")):
                        src = os.path.join(subfolder_path, file)
                        # Rename with subfolder name prefix
                        new_name = f"{sub}_{file}"
                        dst = os.path.join(flat_dir, new_name)
                        shutil.copy2(src, dst)

            print(f"[✔] Merged into {flat_dir}")
