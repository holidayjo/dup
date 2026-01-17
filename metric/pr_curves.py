import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

def load_data(precision_file, recall_file):
    
    numbers = []
    # Load data from text file
    precision = np.loadtxt(precision_file)
    recall    = np.loadtxt(recall_file)
    
    precision = precision.flatten()
    recall    = recall.flatten()
    return precision, recall

# Load precision and recall values from text files
def load_data1(precision_file, recall_file):
    
    numbers = []
    # Load data from text file
    with open(precision_file, 'r') as precision:
        for line in precision:
            line_numbers = [float(num) for num in line.split()]
            numbers.extend(line_numbers)
    precision_flat = np.array(numbers)
    
    with open(recall_file, 'r') as recall:
        for line in recall:
            line_numbers = [float(num) for num in line.split()]
            numbers.extend(line_numbers)
    recall_flat = np.array(numbers)
    return precision_flat, recall_flat

    
    # precision = np.loadtxt(precision_file)
    # recall    = np.loadtxt(recall_file)
    
    # precision = precision.flatten()
    # recall    = recall.flatten()
    # return precision, recall

# Set font to Times New Roman
plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["font.size"]   = 20  # Adjust the default font size as needed

# Define file paths for each model

# U class.
model_files = {
    'CenterNet'   : ('E:/gdrive/My Drive/IEEE_Access/code/metric/results/centernet/centernet_presicion_U.txt',    'E:/gdrive/My Drive/IEEE_Access/code/metric/results/centernet/centernet_recall_U.txt'),
    'Faster R-CNN': ('E:/gdrive/My Drive/IEEE_Access/code/metric/results/faster_rcnn/fasterRCNN_presicion_U.txt', 'E:/gdrive/My Drive/IEEE_Access/code/metric/results/faster_rcnn/fasterRCNN_recall_U.txt'),
    # 'YOLO v7'     : ('E:/gdrive/My Drive/IEEE_Access/code/metric/results/v7/v7_presicion_U_1row.txt',                  'E:/gdrive/My Drive/IEEE_Access/code/metric/results/v7/v7_recall_U_1row.txt'),
    'YOLO v7-tiny': ('E:/gdrive/My Drive/IEEE_Access/code/metric/results/v7_lite/result_v7_lite/precision_U.txt', 'E:/gdrive/My Drive/IEEE_Access/code/metric/results/v7_lite/result_v7_lite/recall_U.txt'),
    'YOLO v9'     : ('E:/gdrive/My Drive/IEEE_Access/code/metric/results/v9/results_test/precision_U.txt',             'E:/gdrive/My Drive/IEEE_Access/code/metric/results/v9/results_test/recall_U.txt')}

# # D class.
# model_files = {
#     'CenterNet'   : ('E:/gdrive/My Drive/IEEE_Access/code/metric/results/centernet/centernet_presicion_D.txt', 'E:/gdrive/My Drive/IEEE_Access/code/metric/results/centernet/centernet_recall_D.txt'),
#     'Faster R-CNN': ('E:/gdrive/My Drive/IEEE_Access/code/metric/results/faster_rcnn/fasterRCNN_presicion_D.txt', 'E:/gdrive/My Drive/IEEE_Access/code/metric/results/faster_rcnn/fasterRCNN_recall_D.txt'),
#     # 'YOLO v7'     : ('E:/gdrive/My Drive/IEEE_Access/code/metric/results/v7/v7_presicion_D.txt', 'E:/gdrive/My Drive/IEEE_Access/code/metric/results/v7/v7_recall_D.txt'),
#     'YOLO v7-tiny': ('E:/gdrive/My Drive/IEEE_Access/code/metric/results/v7_lite/result_v7_lite/precision_D.txt', 'E:/gdrive/My Drive/IEEE_Access/code/metric/results/v7_lite/result_v7_lite/recall_D.txt'),
#     'YOLO v9'     : ('E:/gdrive/My Drive/IEEE_Access/code/metric/results/v9/results_test/precision_D.txt', 'E:/gdrive/My Drive/IEEE_Access/code/metric/results/v9/results_test/recall_D.txt')}

# # P class.
# model_files = {
#     'CenterNet'   : ('E:/gdrive/My Drive/IEEE_Access/code/metric/results/centernet/centernet_presicion_P.txt',    'E:/gdrive/My Drive/IEEE_Access/code/metric/results/centernet/centernet_recall_P.txt'),
#     'Faster R-CNN': ('E:/gdrive/My Drive/IEEE_Access/code/metric/results/faster_rcnn/fasterRCNN_presicion_P.txt', 'E:/gdrive/My Drive/IEEE_Access/code/metric/results/faster_rcnn/fasterRCNN_recall_P.txt'),
#     # 'YOLO v7'     : ('E:/gdrive/My Drive/IEEE_Access/code/metric/results/v7/v7_presicion_P_1row.txt',                  'E:/gdrive/My Drive/IEEE_Access/code/metric/results/v7/v7_recall_P_1row.txt'),
#     'YOLO v7-tiny': ('E:/gdrive/My Drive/IEEE_Access/code/metric/results/v7_lite/result_v7_lite/precision_P.txt', 'E:/gdrive/My Drive/IEEE_Access/code/metric/results/v7_lite/result_v7_lite/recall_P.txt'),
#     'YOLO v9'     : ('E:/gdrive/My Drive/IEEE_Access/code/metric/results/v9/results_test/precision_P.txt',             'E:/gdrive/My Drive/IEEE_Access/code/metric/results/v9/results_test/recall_P.txt')}


# Plot PR curves for each model using Seaborn
plt.figure(figsize=(9, 5))

for model, (precision_file, recall_file) in model_files.items():
    print(model)
    precision, recall = load_data(precision_file, recall_file)
    sns.lineplot(x=recall, y=precision, label=model)

plt.xlabel('Recall')
plt.ylabel('Precision')
# plt.title('Class: Descending')
plt.title('Class: Ascending')
# plt.title('Class: Passing')
plt.legend()
plt.grid(True)
plt.show()

