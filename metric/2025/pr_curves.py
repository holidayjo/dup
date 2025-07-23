from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
import matplotlib.pyplot as plt
import numpy as np
import json

# Paths to files
annFile = '/home/hj/Desktop/Dad/github/DUP/data/coco_test.json'
resFile = '/home/hj/Desktop/Dad/github/DUP/mmdetection/work_dirs/coco_detection/test_fasterrcnn.bbox.json'

# Load ground truth and detection results
cocoGt = COCO(annFile)
cocoDt = cocoGt.loadRes(resFile)
print(cocoDt)
print(cocoGt)
# Initialize COCOeval object
cocoEval = COCOeval(cocoGt, cocoDt, iouType='bbox')

# Run evaluation
cocoEval.evaluate()
cocoEval.accumulate()
cocoEval.summarize()

# Extract precision matrix: (TxRxKxAxM)
# T: IoU thresholds, R: Recall thresholds, K: category, A: area range, M: maxDets
prec = cocoEval.eval['precision']

# Average precision over IoU thresholds for a given class (e.g., category_id = 1)
catId = 2
areaRngIdx = 0    # 0: all area
maxDetsIdx = 2    # 2: maxDets=100

# Get precision across recalls for the specific class
pr_array = prec[:, :, catId-1, areaRngIdx, maxDetsIdx]

# Compute mean across IoU thresholds (axis 0)
pr_mean = np.mean(pr_array, axis=0)
recalls = cocoEval.params.recThrs

# Plot PR Curve
plt.figure(figsize=(8, 6))
plt.plot(recalls, pr_mean, label=f'Category {catId}')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()
