import subprocess
import numpy as np
import os
from collections import defaultdict

# --- Configuration ---
# Path logic: assumes running inside 'yolov9_udp'
base_path = "runs/train/1_2_2_Euljiro_off_peak_balanced_TrainVal_gelan-c_seed{}/weights/last.pt"
seeds = [0, 1, 2, 3, 4]

weights = [base_path.format(s) for s in seeds]

data_yaml = "data/udp.yaml" 
img_size = 320

# Storage
results_ap50 = defaultdict(list)
results_ap50_95 = defaultdict(list)

print(f"Starting evaluation for {len(weights)} models...")
print(f"Data config: {data_yaml}")
print(f"Image size:  {img_size}\n")

for i, w in enumerate(weights):
    if not os.path.exists(w):
        print(f"[Warning] Weight file not found: {w}")
        continue

    print(f"Testing Seed {seeds[i]}: {w}")
    
    cmd = [
        "python", "-u", "val.py",
        "--data", data_yaml,
        "--img", str(img_size),
        "--batch", "32",
        "--conf", "0.001",
        "--iou", "0.65",
        "--device", "0",
        "--weights", w,
        "--task", "test",
        "--name", f"exp_eval_seed_{seeds[i]}"
    ]
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True)
        full_output = result.stdout + "\n" + result.stderr
        
        stats_found = False
        
        # Robust Parsing: Don't wait for header, just look for data structure
        for line in full_output.split('\n'):
            parts = line.split()
            
            # Data row must have at least 7 columns
            # Structure: Class | Images | Labels | P | R | mAP@.5 | mAP@.5:.95
            if len(parts) >= 7:
                try:
                    # Valid data rows have integers in col 1 (Images) and col 2 (Labels)
                    # and floats in the rest. 
                    # We check col 1 to differentiate from headers or garbage.
                    int(parts[1]) 
                    
                    cls_name = parts[0]
                    ap50 = float(parts[5])
                    ap50_95 = float(parts[6])
                    
                    results_ap50[cls_name].append(ap50)
                    results_ap50_95[cls_name].append(ap50_95)
                    
                    # print(f"   -> Found: {cls_name} {ap50}") # Debug print
                    stats_found = True
                    
                except ValueError:
                    # Not a data line (e.g. header line or progress bar text)
                    continue

        if not stats_found:
            print("   -> [ERROR] Could not parse results.")
            # print(full_output) # Uncomment to debug full log

    except Exception as e:
        print(f"   -> [System Error] {e}")

# --- Calculate and Print Statistics ---
print("\n" + "="*60)
print(f"FINAL RESULTS (Mean ± Std over {len(weights)} seeds)")
print(f"{'Class':<15} | {'mAP@0.5':<22} | {'mAP@0.5:0.95':<22}")
print("-" * 60)

# Sort: 'all' first, then alphabetical
sorted_keys = sorted(results_ap50.keys(), key=lambda x: (x != 'all', x))

for cls in sorted_keys:
    scores = results_ap50[cls]
    scores_95 = results_ap50_95[cls]
    
    if scores:
        mean = np.mean(scores)
        std = np.std(scores)
        
        mean_95 = np.mean(scores_95)
        std_95 = np.std(scores_95)
        
        str_50 = f"{mean:.4f} ± {std:.4f}"
        str_95 = f"{mean_95:.4f} ± {std_95:.4f}"
        
        print(f"{cls:<15} | {str_50:<22} | {str_95:<22}")

print("="*60)