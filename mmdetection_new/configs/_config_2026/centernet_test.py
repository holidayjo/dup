# configs/my_centernet_test.py

# Inherit from your working training config
_base_ = '/mnt/Documents/Dad/github/DUP/mmdetection_new/configs/UDP_centernet_TrainVal_test.py'

data_root = '/mnt/Documents/Dad/github/DUP/yolo_to_coco/output/'

# Override the test dataloader to point to your TEST data
test_dataloader = dict(
    dataset=dict(
        ann_file='test.json',  # Assumes test.json is in data_root
        data_prefix=dict(img='/mnt/Documents/Dad/github/DUP/DATA/Euljiro/1_balanced_simulation/test/')
    )
)

# Point the evaluator to the test annotations
test_evaluator = dict(
    ann_file=data_root + 'test.json',
    outfile_prefix='/mnt/Documents/Dad/github/DUP/mmdetection_new/work_dirs/centernet_test_results' # Optional: saves detailed results here
)


"""
python tools/test.py \
    /mnt/Documents/Dad/github/DUP/mmdetection_new/configs/_test_2026/centernet_test.py \
    /mnt/Documents/Dad/github/DUP/mmdetection_new/work_dirs/centernet_TrainVal/centernet_seed_0/epoch_74.pth \
    --work-dir work_dirs/centernet_test_output
    """