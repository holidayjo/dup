import os
import subprocess
import sys

# Configuration
config_file = "/mnt/Documents/Dad/github/DUP/mmdetection_new/configs/_config_2026/UDP_faster_rcnn_TranVal_test.py"
base_work_dir = "/mnt/Documents/Dad/github/DUP/mmdetection_new/work_dirs/faster_rcnn_r18_TrainVal_17e"
seeds = [0, 1, 2, 3, 4]

# 1. SETUP ENVIRONMENT FOR DETERMINISTIC TRAINING
# This fixes the "RuntimeError: Deterministic behavior..." (CuBLAS error)
my_env = os.environ.copy()
my_env["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

for seed in seeds:
    print(f"\n{'='*50}")
    print(f" STARTING FASTER R-CNN TRAINING | SEED: {seed}")
    print(f"{'='*50}")

    work_dir = os.path.join(base_work_dir, f"seed{seed}")

    # 2. COMMAND CONSTRUCTION
    # In MMDetection v3, we use --cfg-options to change the seed
    cmd = [
        "python", "tools/train.py",
        config_file,
        "--work-dir", work_dir,
        "--cfg-options", f"randomness.seed={seed}" 
    ]

    try:
        # Run with the specific environment variable
        subprocess.run(cmd, check=True, env=my_env)
    except subprocess.CalledProcessError as e:
        print(f"❌ Seed {seed} FAILED with error code {e.returncode}.")
        sys.exit(1)

print("\n🎉 ALL 5 SEEDS COMPLETED SUCCESSFULLY!")