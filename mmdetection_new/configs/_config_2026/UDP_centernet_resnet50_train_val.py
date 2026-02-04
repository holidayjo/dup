# =========================================================
# CONFIG FILE: configs/my_centernet_r50_coco.py
# Model: CenterNet ResNet-50 (Pretrained on MS COCO)
# =========================================================

# 1. BASE CONFIG
# We inherit from the official MMDetection CenterNet ResNet-50 config
_base_ = '/mnt/Documents/Dad/github/DUP/mmdetection_new/configs/centernet/centernet-update_r50-caffe_fpn_ms-1x_coco.py'

# 2. PATHS & CLASSES
data_root          = '/mnt/Documents/Dad/github/DUP/yolo_to_coco/output/'
image_folder_train = '/mnt/Documents/Dad/github/DUP/DATA/Euljiro/1_balanced_simulation/train/'
image_folder_val   = '/mnt/Documents/Dad/github/DUP/DATA/Euljiro/1_balanced_simulation/val/'
image_folder_test  = '/mnt/Documents/Dad/github/DUP/DATA/Euljiro/1_balanced_simulation/test/'

metainfo = {
    'classes': ('U', 'D', 'P'),
    'palette': [(220, 20, 60), (119, 11, 32), (0, 0, 142)]
}

# 3. MODEL SETTINGS
model = dict(
    # The backbone (ResNet-50) is already set in _base_
    # We explicitly change the head to output 3 classes
    bbox_head=dict(num_classes=3),
    test_cfg=dict(topk=100, local_maximum_kernel=3, max_per_img=100)
)

# 4. DATA LOADING PIPELINES
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
                # to_float32=True is CRITICAL for CenterNet cropping
                dict(type='LoadImageFromFile', backend_args=backend_args, to_float32=True),
                dict(type='LoadAnnotations', with_bbox=True),
                dict(
                    type='RandomCenterCropPad',
                    crop_size=(320, 320), # Fixed input size
                    ratios=(0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3),
                    mean=[123.675, 116.28, 103.53],
                    std=[58.395, 57.12, 57.375],
                    to_rgb=True,
                    test_pad_mode=None, 
                ),
                dict(type='RandomFlip', prob=0.0), # No flip
                dict(type='PackDetInputs')
            ]
        )
    )
)

# --- VAL PIPELINE (FIXED) ---
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
            # --- CHANGE START: Use Resize to generate 'scale_factor' ---
            dict(
                type='Resize',
                scale=(1333, 800), # Standard testing scale for ResNet
                keep_ratio=True
            ),
            # --- CHANGE END ---
            dict(type='LoadAnnotations', with_bbox=True),
            dict(
                type='PackDetInputs',
                meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape', 'scale_factor', 'border'))
        ]
    )
)

# Ensure Test Dataloader uses the same pipeline
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
        ann_file='test.json',
        data_prefix=dict(img=image_folder_test),
        test_mode=True,
        backend_args=backend_args,
        # Copy the pipeline we just defined above
        pipeline = val_dataloader['dataset']['pipeline']
    )
)

# 5. EVALUATION
val_evaluator = dict(
    type='CocoMetric',
    ann_file=data_root + 'val.json',
    metric='bbox',
    format_only=False,
    backend_args=backend_args
)
test_evaluator = val_evaluator

# 6. SCHEDULE & OPTIMIZER
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

# 7. RUNTIME
work_dir = '/mnt/Documents/Dad/github/DUP/mmdetection_new/work_dirs/centernet_r50_coco_udp_train_val_epoch'

# --- PRETRAINED WEIGHTS ---
# This URL is for ResNet-50 trained on MS COCO.
# load_from = 'https://download.openmmlab.com/mmdetection/v2.0/centernet/centernet_resnet50_140e_coco/centernet_resnet50_140e_coco_20210705_093630-92d9e725.pth'
# load_from = '/mnt/Documents/Dad/github/DUP/mmdetection_new/configs/centernet/centernet-update_r50_fpn_8xb8-amp-lsj-200e_coco.py'
load_from = 'https://download.openmmlab.com/mmdetection/v3.0/centernet/centernet-update_r50-caffe_fpn_ms-1x_coco/centernet-update_r50-caffe_fpn_ms-1x_coco_20230512_203845-8306baf2.pth'