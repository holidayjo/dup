#!/bin/bash

# ====================================================
# CONFIGURATION
# ====================================================
CN_CONFIG="configs/UDP_centernet_TrainVal_test.py"


# ====================================================
# PART 1: CenterNet (5 Runs: Seeds 0-4)
# ====================================================
echo "################################################"
echo "STARTING CENTERNET BATCH (5 Runs)"
echo "################################################"

for seed in {0..4}
do
    echo ">>> Running CenterNet | Seed: $seed"
    
    # FIX: Used --cfg-options randomness.seed instead of --seed
    python tools/train.py $CN_CONFIG \
        --work-dir work_dirs/centernet_seed_$seed \
        --cfg-options randomness.seed=$seed
        
    echo ">>> Finished CenterNet Seed $seed"
    echo "------------------------------------------------"
done

# ====================================================
# PART 2: Faster R-CNN (5 Runs: Seeds 0-4)
# ====================================================
echo "################################################"
echo "STARTING FASTER R-CNN BATCH (5 Runs)"
echo "################################################"

RCNN_CONFIG="/mnt/Documents/Dad/github/DUP/mmdetection_new/configs/_config_2026/UDP_faster_rcnn_resnet50_TrainVal.py"
for seed in {0..4}
do
    echo ">>> Running Faster R-CNN | Seed: $seed"
    
    # FIX: Used --cfg-options randomness.seed instead of --seed
    python tools/train.py $RCNN_CONFIG \
        --work-dir work_dirs/faster_rcnn_r50_TrainVal_epoch_11_seed_$seed \
        --cfg-options randomness.seed=$seed
        
    echo ">>> Finished Faster R-CNN Seed $seed"
    echo "------------------------------------------------"
done

echo "################################################"
echo "ALL 10 TRAINING JOBS COMPLETED. GOOD MORNING!"
echo "################################################"