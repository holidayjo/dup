import pandas as pd
import glob
import os
import numpy as np

# --- STEP 1: LOAD DATA WITH FILENAME ---
folder_path = r'data'
csv_files = glob.glob(os.path.join(folder_path, "*.csv"))
dataframes_list = []

print(f"Found {len(csv_files)} files. Loading...")

for filename in csv_files:
    try:
        df = pd.read_csv(filename)
        target_cols = ['Time', '[Tc2(9) LIU1] ??? ?????', 'Next Station code']
        
        if set(target_cols).issubset(df.columns):
            df_subset = df[target_cols].copy()
            df_subset.rename(columns={
                'Time': 'Time',
                '[Tc2(9) LIU1] ??? ?????': 'LIU1',
                'Next Station code': 'Next Station code'
            }, inplace=True)
            
            # Add Source File Name
            df_subset['Source_File'] = os.path.basename(filename)
            
            dataframes_list.append(df_subset)
    except Exception as e:
        print(f"Error reading {os.path.basename(filename)}: {e}")

if not dataframes_list:
    print("No data found!")
    exit()

full_df = pd.concat(dataframes_list, ignore_index=True)

# --- STEP 2: FILTERING ---
# Filter out 0, 128, 192
full_df = full_df[~full_df['Next Station code'].isin([0, 128, 192])].copy()
full_df = full_df.reset_index(drop=True)

# --- STEP 3: PREPARATION ---
full_df['LIU1'] = full_df['LIU1'].astype(int)
full_df['Next Station code'] = full_df['Next Station code'].astype(int)
full_df['Datetime'] = pd.to_datetime(full_df['Time'], format='%H:%M:%S')

# Identify Visits (Break on Station Change OR File Change)
full_df['station_change'] = (full_df['Next Station code'] != full_df['Next Station code'].shift(1))
full_df['file_change'] = (full_df['Source_File'] != full_df['Source_File'].shift(1))
full_df['visit_trigger'] = (full_df['station_change'] | full_df['file_change']).astype(int)
full_df['visit_id'] = full_df['visit_trigger'].cumsum()

# --- STEP 4: CALCULATION ---
print("Calculating door durations...")

full_df['prev_LIU1'] = full_df['LIU1'].shift(1)
open_starts = full_df[(full_df['LIU1'] == 0) & (full_df['prev_LIU1'] == 1)].copy()
close_starts = full_df[(full_df['LIU1'] == 1) & (full_df['prev_LIU1'] == 0)].copy()

opens_grouped = open_starts.groupby('visit_id')['Datetime'].apply(list)
closes_grouped = close_starts.groupby('visit_id')['Datetime'].apply(list)

# Map visit_id to (Station Code, Source File)
visit_info_map = full_df.drop_duplicates('visit_id').set_index('visit_id')[['Next Station code', 'Source_File']].to_dict('index')

results = []
unique_visits = full_df['visit_id'].unique()

for visit_id in unique_visits:
    start_times = opens_grouped.get(visit_id, [])
    end_times = closes_grouped.get(visit_id, [])
    
    if not start_times: continue
    
    # Pair logic
    valid_pairs = []
    start_times.sort()
    end_times.sort()
    
    for t_open in start_times:
        future_closes = [t for t in end_times if t > t_open]
        if future_closes:
            t_close = future_closes[0]
            valid_pairs.append((t_open, t_close))
            
    if not valid_pairs: continue

    info = visit_info_map[visit_id]
    
    # --- LOGIC: Calculate Saved Time (2nd Open -> Final Close) ---
    saved_time = 0.0
    if len(valid_pairs) > 1:
        # valid_pairs[1] is the second tuple (2nd Open, 2nd Close)
        # valid_pairs[1][0] is the 2nd Open Time
        # valid_pairs[-1][1] is the Final Close Time
        second_open_time = valid_pairs[1][0]
        final_close_time = valid_pairs[-1][1]
        saved_time = (final_close_time - second_open_time).total_seconds()

    results.append({
        'visit_id': visit_id,
        'Source File': info['Source_File'],
        'Station Code': info['Next Station code'],
        'Door Open Count': len(valid_pairs),
        'First Open Time': valid_pairs[0][0].strftime('%H:%M:%S'),
        'Final Close Time': valid_pairs[-1][1].strftime('%H:%M:%S'),
        'Duration (1st Open -> 1st Close)': (valid_pairs[0][1] - valid_pairs[0][0]).total_seconds(),
        'Duration (1st Open -> Final Close)': (valid_pairs[-1][1] - valid_pairs[0][0]).total_seconds(),
        'Saved Time (2nd Open -> Final Close)': saved_time
    })

# --- STEP 5: RESULTS & CHECKS ---
results_df = pd.DataFrame(results)

if not results_df.empty:
    print("\nAnalysis Complete.")
    
    # Check 1: ANY Re-open (Count > 1)
    reopened = results_df[results_df['Door Open Count'] > 1]
    
    # Check 2: MORE THAN 1 Re-open (Count > 2)
    multiple_reopens = results_df[results_df['Door Open Count'] > 2]
    
    print(f"Total visits detected: {len(results_df)}")
    print(f"Visits with reopens: {len(reopened)}")
    
    # --- ADDED: Calculate Average Saved Time for Re-opens ---
    total_saved_seconds = results_df['Saved Time (2nd Open -> Final Close)'].sum()
    
    # Filter for only events that actually HAD a re-open to get a meaningful average
    avg_saved_time = results_df[results_df['Door Open Count'] > 1]['Saved Time (2nd Open -> Final Close)'].mean()
    
    print(f"Total Time Saved by System: {total_saved_seconds:.2f} seconds")
    
    if not np.isnan(avg_saved_time):
        print(f"Average Time (2nd Open -> Final Close) for Re-open events: {avg_saved_time:.2f} seconds")
    else:
        print("Average Time (2nd Open -> Final Close): N/A (No re-open events found)")

    # Ensure output directory exists and save
    output_dir = 'output'
    try:
        os.makedirs(output_dir, exist_ok=True)
    except Exception as e:
        print(f"Warning: could not create output directory '{output_dir}': {e}")

    output_path = os.path.join(output_dir, 'Station_Door_Analysis_With_SavedTime.xlsx')
    results_df.to_excel(output_path, index=False)
    print(f"\nSaved to '{output_path}'")
else:
    print("No valid door opening events found.")
    