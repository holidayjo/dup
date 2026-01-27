import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

# --- 1. Load Data ---
# UPDATED: Read the file that contains the 'Saved Time' column
file_path = 'output/Station_Door_Analysis_With_SavedTime.xlsx'

try:
    df = pd.read_excel(file_path)
    print(f"Successfully loaded: {file_path}")
except:
    # Fallback/Error handling
    print(f"Could not find '{file_path}'. Trying 'Station_Door_Analysis_With_Filename.xlsx'...")
    try:
        df = pd.read_excel('output/Station_Door_Analysis_With_Filename.xlsx')
    except:
        df = pd.read_csv('Station_Door_Analysis_With_Filename.xlsx - Sheet1.csv')

# ==========================================
# PART A: CLEAN DWELL TIME (For Plot 1)
# ==========================================
# Outlier Removal (IQR Method) for Normal Dwell Time
Q1 = df['Duration (1st Open -> 1st Close)'].quantile(0.25)
Q3 = df['Duration (1st Open -> 1st Close)'].quantile(0.75)
IQR = Q3 - Q1
lower = Q1 - 1.5 * IQR
upper = Q3 + 1.5 * IQR

df_clean = df[(df['Duration (1st Open -> 1st Close)'] >= lower) & 
              (df['Duration (1st Open -> 1st Close)'] <= upper)].copy()

mean_val = df_clean['Duration (1st Open -> 1st Close)'].mean()
std_dev = df_clean['Duration (1st Open -> 1st Close)'].std()

# ==========================================
# PART B: CLEAN RE-OPEN TIME & VARIANCE (For Plot 2 Title)
# ==========================================
# 1. Select only Re-open events
reopen_df = df[df['Door Open Count'] > 1].copy()

# 2. Determine the Re-open Duration
if 'Saved Time (2nd Open -> Final Close)' in df.columns:
    # Use the accurate column
    print("Using accurate 'Saved Time' column.")
    raw_reopen_times = reopen_df['Saved Time (2nd Open -> Final Close)']
else:
    # Fallback: Approximate (Final Close - 1st Close)
    print("Column 'Saved Time' not found. Using approximation.")
    raw_reopen_times = reopen_df['Duration (1st Open -> Final Close)'] - reopen_df['Duration (1st Open -> 1st Close)']

# 3. Remove Outliers from Re-open Times
# This removes extreme values (like the 19,410s delay in row 1)
if not raw_reopen_times.empty:
    Q1_r = raw_reopen_times.quantile(0.25)
    Q3_r = raw_reopen_times.quantile(0.75)
    IQR_r = Q3_r - Q1_r
    lower_r = Q1_r - 1.5 * IQR_r
    upper_r = Q3_r + 1.5 * IQR_r

    clean_reopen_times = raw_reopen_times[(raw_reopen_times >= lower_r) & (raw_reopen_times <= upper_r)]
    
    avg_reopen_time = clean_reopen_times.mean()
    std_reopen_time = clean_reopen_times.std()
    
    print(f"Original re-open events: {len(raw_reopen_times)}")
    print(f"Events after removing outliers: {len(clean_reopen_times)}")
else:
    avg_reopen_time = 0
    std_reopen_time = 0

# Add this after calculating avg_reopen_time
print("\n--- DETAILED STATS FOR RE-OPEN TIMES ---")
print(clean_reopen_times.describe())


print(f"\n--- Analysis Results ---")
print(f"Mean Dwell Time (Normal): {mean_val:.2f} s")
print(f"Avg Re-open Duration: {avg_reopen_time:.2f} s")
print(f"Re-open Std Deviation: {std_reopen_time:.2f} s")

# --- Prepare Counts for Bar Chart ---
counts = df['Door Open Count'].value_counts().sort_index()
summary_counts = {
    'Normal (1)': counts.get(1, 0),
    'Reopen (2)': counts.get(2, 0),
    'Multiple (>2)': counts[counts.index > 2].sum()
}

# ==========================================
# PLOTTING SETTINGS (LARGE FONTS)
# ==========================================
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman']
plt.rcParams.update({
    'font.size': 18,
    'axes.labelsize': 22,
    'xtick.labelsize': 18,
    'ytick.labelsize': 18,
    'legend.fontsize': 18
})

plot_dir = 'plots_split'
os.makedirs(plot_dir, exist_ok=True)

# --- PLOT A: DISTRIBUTION ---
plt.figure(figsize=(8, 6))
sns.histplot(df_clean['Duration (1st Open -> 1st Close)'], binwidth=1, kde=True, 
             color='skyblue', alpha=0.6)

plt.axvline(mean_val, color='red', linestyle='--', linewidth=3, label=f'Mean: {mean_val:.1f} s')
# FIX: Added raw string r'' to fix SyntaxWarning
plt.axvspan(mean_val - std_dev, mean_val + std_dev, color='red', alpha=0.15, label=r'Var (±1$\sigma$)')
plt.axvline(mean_val - std_dev, color='red', linestyle=':', alpha=0.5, linewidth=2)
plt.axvline(mean_val + std_dev, color='red', linestyle=':', alpha=0.5, linewidth=2)

plt.xlabel('Dwell Time (s)')
plt.ylabel('Frequency')
plt.legend(loc='upper right', frameon=True)
plt.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig(f'{plot_dir}/plot_a_distribution.png', dpi=300)
plt.close()

# --- PLOT B: REOPEN TIME DISTRIBUTION (New Logic) ---
plt.figure(figsize=(8, 6))

# 1. Plot the Histogram of Re-open durations
sns.histplot(clean_reopen_times, binwidth=2, kde=True, 
             color='#FF9800', alpha=0.6, label='Re-open Events')

# 2. Add the vertical lines for Mean and Std
plt.axvline(avg_reopen_time, color='red', linestyle='--', linewidth=3, 
            label=f'Mean: {avg_reopen_time:.1f} s')
plt.axvline(avg_reopen_time + std_reopen_time, color='red', linestyle=':', linewidth=2, 
            label=r'Mean + 1$\sigma$')

# 3. Titles and Labels
plt.title(fr"Re-open Duration ($\mu \approx \sigma \approx {avg_reopen_time:.1f}s$)", fontsize=18)
plt.xlabel('Re-open Duration (s)')
plt.ylabel('Frequency')
plt.legend(loc='upper right')
plt.grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig(f'{plot_dir}/plot_b_reopen_dist.png', dpi=300)
plt.close()

print(f"Plots saved to {plot_dir}")