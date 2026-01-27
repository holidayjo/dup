# =========================================================
# CONFIG: configs/my_centernet_r18_coco.py
# Description: CenterNet ResNet-18 (Pretrained on MS COCO)
# =========================================================

# 1. BASE CONFIG
# Use the standard CenterNet ResNet-18 config provided by MMDetection
_base_ = '/mnt/Documents/Dad/github/DUP/mmdetection_new/configs/centernet/centernet_r18_8xb16-crop512-140e_coco.py'

# 2. PATHS & CLASSES
data_root          = '/mnt/Documents/Dad/github/DUP/yolo_to_coco/output/'
image_folder_train = '/mnt/Documents/Dad/github/DUP/DATA/Euljiro/1_balanced_simulation/train/'
image_folder_val   = '/mnt/Documents/Dad/github/DUP/DATA/Euljiro/1_balanced_simulation/val/'
image_folder_test  = '/mnt/Documents/Dad/github/DUP/DATA/Euljiro/1_balanced_simulation/test/'

metainfo = {
    'classes': ('U', 'D', 'P'),
    'palette': [(220, 20, 60), (119, 11, 32), (0, 0, 142)]
}

# 3. MODEL
model = dict(
    # The backbone (ResNet-18) is already set in _base_
    # We only need to change the head to output 3 classes instead of 80
    bbox_head=dict(num_classes=3),
    test_cfg=dict(topk=100, local_maximum_kernel=3, max_per_img=100)
)

# 4. DATA LOADING 
backend_args = None

# --- TRAIN PIPELINE ---
train_dataloader = dict(
    _delete_=True,
    batch_size=8,
    num_workers=8,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    batch_sampler=dict(type='AspectRatioBatchSampler'),
    dataset=dict(
        type='RepeatDataset',
        times=1,
        dataset=dict(
            type='CocoDataset',
            metainfo=metainfo,
            data_root=data_root,
            ann_file='train.json',
            data_prefix=dict(img=image_folder_train),
            filter_cfg=dict(filter_empty_gt=False, min_size=32),
            backend_args=backend_args,
            pipeline=[
                # Fix: 'to_float32=True' is required for CenterNet's RandomCenterCropPad
                dict(type='LoadImageFromFile', backend_args=backend_args, to_float32=True),
                dict(type='LoadAnnotations', with_bbox=True),
                dict(
                    type='RandomCenterCropPad',
                    crop_size=(320, 320), 
                    ratios=(0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3),
                    mean=[123.675, 116.28, 103.53],
                    std=[58.395, 57.12, 57.375],
                    to_rgb=True,
                    test_pad_mode=None, 
                ),
                # Using 0.0 flip probability for our passenger problem.
                dict(type='RandomFlip', prob=0.0), 
                dict(type='PackDetInputs')
            ]
        )
    )
)

# --- VAL PIPELINE ---
val_dataloader = dict(
    _delete_=True,
    batch_size=8,
    num_workers=8,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type='CocoDataset',
        metainfo=metainfo,
        data_root=data_root,
        ann_file='val.json',
        data_prefix=dict(img=image_folder_val),
        test_mode=True,
        backend_args=backend_args,
        pipeline=[
            dict(type='LoadImageFromFile', backend_args=backend_args, to_float32=True),
            dict(
                type='RandomCenterCropPad',
                crop_size=None, 
                ratios=None,
                border=None,
                mean=[123.675, 116.28, 103.53],
                std=[58.395, 57.12, 57.375],
                to_rgb=True,
                test_mode=True,
                test_pad_mode=['logical_or', 31] 
            ),
            dict(type='LoadAnnotations', with_bbox=True),
            dict(
                type='PackDetInputs',
                meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape', 'scale_factor', 'border'))
        ]
    )
)

# Use the same pipeline for Test as Validation
test_dataloader = dict(
    _delete_=True,
    batch_size=1,
    num_workers=2,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type='CocoDataset',
        metainfo=metainfo,
        data_root=data_root,
        ann_file='test.json', # Assumes test.json exists
        data_prefix=dict(img=image_folder_test), # Uses test folder path defined above
        test_mode=True,
        backend_args=backend_args,
        pipeline=val_dataloader['dataset']['pipeline']
    )
)

# 5. EVALUATOR
val_evaluator = dict(
    type='CocoMetric',
    ann_file=data_root + 'val.json',
    metric='bbox',
    format_only=False,
    backend_args=backend_args
)
test_evaluator = val_evaluator

# 6. SCHEDULE
train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=100, val_interval=1)
val_cfg   = dict(type='ValLoop')
test_cfg  = dict(type='TestLoop')

optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(type='SGD', lr=0.0025, momentum=0.9, weight_decay=0.0001)
)

param_scheduler = [
    dict(type='LinearLR', start_factor=0.001, by_epoch=False, begin=0, end=500),
    dict(type='MultiStepLR', begin=0, end=100, by_epoch=True, milestones=[70, 90], gamma=0.1)
]

default_scope = 'mmdet'
default_hooks = dict(
    timer=dict(type='IterTimerHook'),
    logger=dict(type='LoggerHook', interval=50),
    param_scheduler=dict(type='ParamSchedulerHook'),
    checkpoint=dict(type='CheckpointHook', interval=5),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    visualization=dict(type='DetVisualizationHook')
)

# 7. RUNTIME & WEIGHTS
work_dir = '/mnt/Documents/Dad/github/DUP/mmdetection_new/work_dirs/centernet_r18_coco_udp'

# AUTOMATICALLY DOWNLOAD & LOAD OFFICIAL COCO WEIGHTS
# This guarantees "Pretrained on MS COCO" is a valid claim in your paper.
load_from = 'https://download.openmmlab.com/mmdetection/v2.0/centernet/centernet_resnet18_140e_coco/centernet_resnet18_140e_coco_20210705_093630-bb5b3bf7.pth'