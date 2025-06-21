import os
from ultralytics import YOLO
from PIL import Image
from pathlib import Path

# Load your trained YOLOv8 model
model = YOLO("models/yolov8_ball.pt")  

# Define input and output directories
input_root = Path("data/frames")                 # Unlabeled frames
output_root = Path("data/autolabeled_frames")    # Output location
output_root.mkdir(parents=True, exist_ok=True)   # Create output if not exists

# Loop through each folder 
for subfolder in input_root.iterdir():
    if subfolder.is_dir():
        # Output subfolders (images + labels)
        output_img_dir = output_root / subfolder.name
        output_lbl_dir = output_img_dir / "labels"
        output_img_dir.mkdir(parents=True, exist_ok=True)
        output_lbl_dir.mkdir(parents=True, exist_ok=True)

        # Loop through each image in this folder
        for image_file in subfolder.glob("*.jpg"):
            results = model(image_file)       # Run inference on image
            result = results[0]               # Get prediction result
            boxes = result.boxes              # Get bounding boxes

            # Create YOLO label file
            label_path = output_lbl_dir / f"{image_file.stem}.txt"
            with open(label_path, "w") as f:
                for box in boxes:
                    cls_id = int(box.cls[0].item())           # Class ID
                    xywhn = box.xywhn[0].tolist()             # [x_center, y_center, width, height] (normalized)
                    f.write(f"{cls_id} {' '.join(f'{v:.6f}' for v in xywhn)}\n")

            # Copy image to output directory
            img = Image.open(image_file)
            img.save(output_img_dir / image_file.name)

print(" Auto-labeling complete for all images.")
