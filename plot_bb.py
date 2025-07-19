import os
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from sklearn.cluster import KMeans

def iou(box, clusters):
    """
    Compute the IoU between a box and k cluster centroids.
    box: [w, h]
    clusters: ndarray of shape (k, 2)
    """
    x = np.minimum(clusters[:, 0], box[0])
    y = np.minimum(clusters[:, 1], box[1])

    intersection = x * y
    box_area = box[0] * box[1]
    cluster_area = clusters[:, 0] * clusters[:, 1]

    union = box_area + cluster_area - intersection
    return intersection / union


def avg_iou(boxes, clusters):
    return np.mean([np.max(iou(box, clusters)) for box in boxes])


def kmeans_iou(boxes, k, dist=np.median, seed=7, max_iter=3000):
    """
    KMeans clustering using 1 - IoU as distance metric.
    boxes: ndarray of shape (n, 2), all bounding boxes.
    k: number of clusters
    dist: np.median or np.mean for updating centroids
    """
    np.random.seed(seed)
    n = boxes.shape[0]
    # print("n = ", n)

    # Randomly initialize cluster centers
    indices  = np.random.choice(n, k, replace=False)
    clusters = boxes[indices]
    # print("Initial clusters = ", clusters)

    for _i in range(max_iter):
        for box in boxes:
            # print("box[0] = ", box[0], "clusters[:,0] = ", clusters[:,0])
            break
        distances   = np.array([1 - iou(box, clusters) for box in boxes]) # distances[i] will be a vector of distances from the i-th box to all cluster centers.
        # print("distances.shape = ", distances.shape)   # (10764, 9)
        # print("distances[:,10] = ", distances[:10, :]) # the distance of each 10764 boxes from the 9 anchor boxes.
        assignments = np.argmin(distances, axis=1)     # Now we find the minimum distance index for each box. 

        new_clusters = []
        for i in range(k):
            if np.any(assignments == i):
                cluster_boxes = boxes[assignments == i]
                # print("cluster_boxes.shape = ", cluster_boxes.shape) # ex_1st) cluster_boxes.shape =  (1474, 2) -> repeat 9 times. 1472 boxes are assigned to the first cluster.
                new_cluster   = dist(cluster_boxes, axis=0) # median distance of 1472 boxes.
                # print("new_cluster = ", new_cluster)
                new_clusters.append(new_cluster)
            else:
                # Reinitialize a missing cluster
                new_clusters.append(boxes[np.random.choice(n)])
        new_clusters = np.array(new_clusters)

        if np.allclose(clusters, new_clusters):
            print("Converged at iteration =", _i)
            break
        clusters = new_clusters
        # break
    print("iteration =", _i)
    return clusters, assignments



# Seaborn style and font settings
sns.set(style='whitegrid')
font_size = 28
plt.rcParams.update({
    'font.family': 'Times New Roman',
    'font.size': font_size,
    'axes.titlesize': font_size,
    'axes.labelsize': font_size,
    'xtick.labelsize': font_size -2,     # ← X-axis tick font size
    'ytick.labelsize': font_size -2      # ← Y-axis tick font size
})

input_img_size = 320  # Set your input image size
random_state = 7
# Directory containing YOLO labels
label_dir = r'data/train'

# Collect normalized width and height
widths  = []
heights = []
for filename in os.listdir(label_dir):
    if filename.endswith(".txt"):
        filepath = os.path.join(label_dir, filename)
        with open(filepath, 'r') as f:
            lines = f.readlines()
            for line in lines:
                parts = line.strip().split()
                if len(parts) == 5:  # YOLO format: class x_center y_center width height
                    widths.append(float(parts[3]))
                    heights.append(float(parts[4]))
X = np.column_stack((widths, heights))
print("X.shape =",X.shape) # 10764 all boxes.

# Plot all bounding boxes (scatter plot)
custom_color = (123/255, 196/255, 233/255)  # Normalize RGB
fig, ax      = plt.subplots(figsize=(8, 8))
sns.scatterplot(x=widths, y=heights, s=20, color=custom_color, alpha=0.6, ax=ax)
ax.set_xlabel('Normalized Width')
ax.set_ylabel('Normalized Height')
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
fig.patch.set_alpha(0.0)
plt.savefig('pics/all_bboxes.png', transparent=True, dpi=300)
plt.show()

#%%
# Apply KMeans clustering for k from 2 to 10
# for k in range(2, 11):
# cluster k=9.
k = 9

centers, labels = kmeans_iou(X, k, dist=np.mean, seed=random_state)

# Print centroid information
print(f"\nCluster Centers for k={k} (Pixel values, scaled to {input_img_size}x{input_img_size}):")
for idx, (w, h) in enumerate(centers):
    pw = int(round(w * input_img_size))
    ph = int(round(h * input_img_size))
    print(f"  Anchor {idx}: Width={pw}, Height={ph}")


# Create cluster plot
fig, ax = plt.subplots(figsize=(8, 8))
palette = sns.color_palette("tab10", k)
sns.scatterplot(x=X[:, 0], y=X[:, 1], hue=labels, palette=palette, s=20, ax=ax, legend=False)
# sns.scatterplot(x=centers[:, 0], y=centers[:, 1], marker='X', color='black', s=120, ax=ax)

# ax.set_title(f'K-Means Clustering (k={k})')
ax.set_xlabel('Normalized Width')
ax.set_ylabel('Normalized Height')
ax.set_xlim(0, 0.6)
ax.set_ylim(0, 0.8)
ax.set_aspect('equal')
fig.patch.set_alpha(0.0)
plt.savefig(f'pics/kmeans_k{k}.png', transparent=True, dpi=300, bbox_inches='tight', pad_inches=0.1)
plt.show()
    

    
#%%
# Automatically get anchor boxes from KMeans centers
anchors = centers.tolist()

# Calculate area and assign colors
areas = [w * h for w, h in anchors]
sorted_indices = sorted(range(len(anchors)), key=lambda i: areas[i])  # Ascending area

# Define color mapping
color_map = {}
for i in sorted_indices[:3]:
    color_map[i] = (123/255, 196/255, 233/255)  # Small
for i in sorted_indices[3:6]:
    color_map[i] = (255/255, 0/255, 0/255)      # Medium
for i in sorted_indices[6:]:
    color_map[i] = (38/255, 231/255, 36/255)    # Large

# Sort indices in descending area order for drawing (large drawn first)
draw_order = sorted(range(len(anchors)), key=lambda i: areas[i], reverse=True)

# Plotting
font_size = 32
plt.rcParams.update({
    'font.family': 'Times New Roman',
    'font.size': font_size,
    'axes.titlesize': font_size,
    'axes.labelsize': font_size - 2,
})

fig, ax = plt.subplots(figsize=(8, 8))

for idx in draw_order:
    w, h = anchors[idx]
    x0 = 0.5 - w / 2
    y0 = 0.5 - h / 2
    color = color_map[idx]
    rect = patches.Rectangle(
        (x0, y0), w, h,
        linewidth=3.5,
        edgecolor=color,
        facecolor='none',
        label=f'Anchor {idx}'
    )
    ax.add_patch(rect)
    # Optional: annotate
    # ax.text(x0 + w / 2, y0 + h / 2, str(idx), color='black', ha='center', va='center', fontsize=14)

ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
ax.set_aspect('equal')
ax.set_xlabel('Normalized Width')
ax.set_ylabel('Normalized Height')
plt.grid(True)
fig.patch.set_alpha(0.0)
plt.savefig(f'pics/anchor_boxes_k{k}_colored.png', transparent=True, dpi=300, bbox_inches='tight', pad_inches=0.01)
plt.show()
