import os
import json
from PIL import Image

# Path to your dataset
IMAGE_DIR  = '/home/hj/Desktop/Dad/github/DUP/data/val'
output_DIR = '/home/hj/Desktop/Dad/github/DUP/data'
# Define your classes here in order
CLASSES = ['U', 'D', 'P']  # TODO: Replace with actual class names
category_map = {i: name for i, name in enumerate(CLASSES)}

def convert_yolo_to_coco(image_dir):
    image_id = 0
    annotation_id = 0

    coco = {
        "images": [],
        "annotations": [],
        "categories": [{"id": i, "name": name, "supercategory": "none"} for i, name in category_map.items()]
    }

    for filename in os.listdir(image_dir):
        if filename.endswith('.jpg'):
            image_path = os.path.join(image_dir, filename)
            txt_path = image_path.replace('.jpg', '.txt')

            # Open image to get dimensions
            with Image.open(image_path) as img:
                width, height = img.size

            coco['images'].append({
                "file_name": filename,
                "height": height,
                "width": width,
                "id": image_id
            })

            # Read YOLO annotations
            if os.path.exists(txt_path):
                with open(txt_path, 'r') as f:
                    for line in f:
                        parts = line.strip().split()
                        if len(parts) != 5:
                            continue  # Skip malformed lines

                        class_id = int(parts[0])
                        x_center = float(parts[1])
                        y_center = float(parts[2])
                        bbox_width = float(parts[3])
                        bbox_height = float(parts[4])

                        # Convert YOLO to COCO format
                        x = (x_center - bbox_width / 2) * width
                        y = (y_center - bbox_height / 2) * height
                        w = bbox_width * width
                        h = bbox_height * height

                        coco['annotations'].append({
                            "id": annotation_id,
                            "image_id": image_id,
                            "category_id": class_id,
                            "bbox": [x, y, w, h],
                            "area": w * h,
                            "iscrowd": 0
                        })

                        annotation_id += 1

            image_id += 1

    return coco

# Generate COCO annotation
coco_json = convert_yolo_to_coco(IMAGE_DIR)

# Save to file
output_path = os.path.join(output_DIR, 'annotations.json')
with open(output_path, 'w') as f:
    json.dump(coco_json, f, indent=4)

print(f"COCO annotation saved to {output_path}")
