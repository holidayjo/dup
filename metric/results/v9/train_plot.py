import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Read the Excel file with the specified sheet name
file_path = 'E:/gdrive/My Drive/IEEE_Access/code/metric/results/v9/v9_train_result.xlsx'  # Update the path with the actual file name
df = pd.read_excel(file_path, sheet_name='final')

# Display the first few rows of the DataFrame to ensure it's read correctly
# print(df.head())

# Set font to Times New Roman
plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["font.size"] = 30  # Adjust the default font size as needed

# Plotting the data using seaborn
plt.figure()#figsize=(14, 8))

# Plot mAP (Train) and mAP (Test)
sns.lineplot(data=df, x='Epoch', y='mAP (Train)', marker='o', label='mAP (Train)',linestyle='-', linewidth=5, color='blue')
sns.lineplot(data=df, x='Epoch', y='mAP (Test)', marker='o', label='mAP (Test)',linestyle='-', linewidth=5, color='green')
plt.legend(loc='lower left', fontsize=20)
plt.ylabel('mAP@.5')

# Plot Class loss (Train) on a secondary y-axis
ax2 = plt.twinx()
sns.lineplot(data=df, x='Epoch', y='Class loss (Train)', marker='o', color='r', label='Class loss (Train)', ax=ax2,linestyle='-', linewidth=5)
ax2.set_ylabel('Class loss (Train)')
# Removing the legend for the secondary y-axis plot as we will combine legends
ax2.legend(loc='lower center', fontsize=20)

# Customize the plot
# plt.title('Training and Testing mAP vs Class loss over Epochs', fontsize=16)
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.grid(True)
plt.xticks(fontsize=30)
plt.yticks(fontsize=30)

# Combine the legends from both plots and place them at the right center
# lines, labels = plt.gca().get_legend_handles_labels()
# lines2, labels2 = ax2.get_legend_handles_labels()
# combined_lines = lines + lines2
# combined_labels = labels + labels2
# plt.legend(combined_lines, combined_labels, loc='center left', bbox_to_anchor=(1, 0.5), fontsize=16)

# Show the plot
plt.show()
