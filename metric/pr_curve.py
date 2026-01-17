import numpy as np
from sklearn.metrics import precision_recall_curve, average_precision_score
import matplotlib.pyplot as plt
import seaborn as sns

# Generate synthetic data
np.random.seed(0)
y_true = np.random.randint(0, 2, 1000)  # True binary labels
y_scores = np.random.rand(1000)         # Predicted scores

# Calculate precision, recall, and thresholds
precision, recall, thresholds = precision_recall_curve(y_true, y_scores)
average_precision = average_precision_score(y_true, y_scores)

# Plot Precision-Recall curve using seaborn
plt.figure(figsize=(10, 6))
sns.lineplot(x=recall, y=precision, marker='o', linestyle='-', drawstyle='steps-post')

# Add axis labels and title
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title(f'Precision-Recall curve: AP={average_precision:.2f}')

# Show the plot
plt.show()
