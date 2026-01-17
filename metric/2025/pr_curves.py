import os
import cv2
import glob
import numpy as np
import matplotlib.pyplot as plt
from mean_average_precision import MetricBuilder

# Directories
gt_dir = '/home/hj/Desktop/Dad/github/DUP/data/test'
pred_dir = '/home/hj/Desktop/Dad/github/DUP/fasterrcnn_predictions_yolo'

# Supported image extension
img_ext = '.jpg'

# Initialize metric (COCO-style 0.5:0.95 if needed)
metric_fn = MetricBuilder.build_evaluation_metric("map_2d", async_mode=False, num_classes=80)

# Get all ground truth .txt files
gt_txts = sorted(glob.glob(os.path.join(gt_dir, '*.txt')))

for gt_txt in gt_txts:
    # Image file name
    base_name = os.path.splitext(os.path.basename(gt_txt))[0]
    img_path = os.path.join(gt_dir, base_name + img_ext)
    if not os.path.exists(img_path):
        continue

    # Load image to get size
    img = cv2.imread(img_path)
    img_h, img_w = img.shape[:2]

    # Load ground truth
    gt_boxes = []
    with open(gt_txt, 'r') as f:
        for line in f:
            parts = list(map(float, line.strip().split()))
            cls, x_c, y_c, w, h = parts
            x1 = (x_c - w / 2) * img_w
            y1 = (y_c - h / 2) * img_h
            x2 = (x_c + w / 2) * img_w
            y2 = (y_c + h / 2) * img_h
            gt_boxes.append([int(cls), x1, y1, x2, y2])

    # Load predictions
    pred_txt = os.path.join(pred_dir, base_name + '.txt')
    pred_boxes = []
    if os.path.exists(pred_txt):
        with open(pred_txt, 'r') as f:
            for line in f:
                parts = list(map(float, line.strip().split()))
                cls, x_c, y_c, w, h = parts[:5]
                score = parts[5] if len(parts) == 6 else 1.0
                x1 = (x_c - w / 2) * img_w
                y1 = (y_c - h / 2) * img_h
                x2 = (x_c + w / 2) * img_w
                y2 = (y_c + h / 2) * img_h
                pred_boxes.append([int(cls), score, x1, y1, x2, y2])

    # Format for metric_fn: (detections, annotations)
    if pred_boxes and gt_boxes:
        preds_np = np.array(pred_boxes)
        gts_np = np.array(gt_boxes)
        metric_fn.add(preds_np, gts_np)

# Compute results
eval_result = metric_fn.value(iou_thresholds=0.5)

# Print mAP
print(f"mAP@0.5: {eval_result['mAP']:.4f}")

# Extract PR curve for one class (e.g., class 0)
# If you want per-class:
for cls_id in range(80):  # replace 80 if your class count is smaller
    if f'precision@0.5@{cls_id}' not in eval_result:
        continue
    precision = eval_result[f'precision@0.5@{cls_id}']
    recall = eval_result[f'recall@0.5@{cls_id}']
    if len(precision) == 0:
        continue
    plt.plot(recall, precision, label=f'Class {cls_id}')

plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('PR Curve per Class')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()
