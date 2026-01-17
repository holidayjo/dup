import os
import cv2  # OpenCV to read image dimensions

def yolo_to_voc(yolo_file_path, voc_file_path, image_path):
    # Read the image to get its dimensions
    image = cv2.imread(image_path)
    image_height, image_width = image.shape[:2]
    
    with open(yolo_file_path, 'r') as yolo_file, open(voc_file_path, 'w') as voc_file:
        for line in yolo_file:
            parts = line.strip().split()
            class_id = parts[0]
            x_center = float(parts[1]) * image_width
            y_center = float(parts[2]) * image_height
            width = float(parts[3]) * image_width
            height = float(parts[4]) * image_height
            
            x_min = int(x_center - width / 2)
            y_min = int(y_center - height / 2)
            x_max = int(x_center + width / 2)
            y_max = int(y_center + height / 2)
            
            voc_file.write(f"{class_id} {x_min} {y_min} {x_max} {y_max}\n")

# Example usage:
# Specify the directory containing YOLO format annotations and the output directory for Pascal VOC format annotations
yolo_annotation_dir = 'path_to_yolo_annotations'
voc_annotation_dir = 'path_to_voc_annotations'
image_dir = 'path_to_images'  # Directory containing images

if not os.path.exists(voc_annotation_dir):
    os.makedirs(voc_annotation_dir)

for yolo_filename in os.listdir(yolo_annotation_dir):
    if yolo_filename.endswith('.txt'):
        yolo_file_path = os.path.join(yolo_annotation_dir, yolo_filename)
        voc_file_path = os.path.join(voc_annotation_dir, yolo_filename)
        
        # Determine the corresponding image file name
        # Assuming the image file has the same name as the annotation file but with an image extension (e.g., .jpg or .png)
        image_filename = os.path.splitext(yolo_filename)[0] + '.jpg'  # Change extension if needed
        image_path = os.path.join(image_dir, image_filename)
        
        if os.path.exists(image_path):
            yolo_to_voc(yolo_file_path, voc_file_path, image_path)
        else:
            print(f"Image file {image_path} not found for annotation {yolo_file_path}")
