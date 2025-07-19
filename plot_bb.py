import os
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from sklearn.cluster import KMeans

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


random_state = 7
# Directory containing YOLO labels
label_dir = r'C:\Users\조희\Downloads\train_1'
# label_dir = r'C:\Users\조희\Downloads\all'

# Collect normalized width and height
widths = []
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

# Plot all bounding boxes (scatter plot)
custom_color = (123/255, 196/255, 233/255)  # Normalize RGB
fig, ax = plt.subplots(figsize=(8, 8))
sns.scatterplot(x=widths, y=heights, s=20, color=custom_color, alpha=0.6, ax=ax)
# ax.set_title('All Bounding Boxes (Normalized Width vs Height)')
ax.set_xlabel('Normalized Width')
ax.set_ylabel('Normalized Height')
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
fig.patch.set_alpha(0.0)
plt.savefig('all_bboxes.png', transparent=True, dpi=300)
plt.show()



#%%
# Apply KMeans clustering for k from 2 to 10
# for k in range(2, 11):
# cluster k=9.
k = 9
kmeans  = KMeans(n_clusters=k, random_state=random_state, max_iter=10000)
labels  = kmeans.fit_predict(X)
centers = kmeans.cluster_centers_

# Print centroid information
print(f"\nCluster Centers for k={k}:")
for idx, (w, h) in enumerate(centers):
    print(f"  Cluster {idx}: Width={w:.4f}, Height={h:.4f}")

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
plt.savefig(f'kmeans_k{k}.png', transparent=True, dpi=300, bbox_inches='tight', pad_inches=0.1)
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
plt.savefig(f'anchor_boxes_k{k}_colored.png', transparent=True, dpi=300, bbox_inches='tight', pad_inches=0.01)
plt.show()
