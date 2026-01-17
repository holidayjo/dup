# Resetting: optim_wrapper, param_scheduler

#%% (1) Defining Dataset --> coco detection.py

# dataset settings
dataset_type = 'CocoDataset' # located in D:\Github\DUP\mmdetection\mmdet\datasets\__init__.py
                             # We selected CocoDataset.
data_root    = 'D:/Github/DUP/dataset/COCO_format/'  # The location of dataset.



# Example to use different file client
# Method 1: simply set the data root and let the file I/O module
# automatically infer from prefix (not support LMDB and Memcache yet)

# data_root = 's3://openmmlab/datasets/detection/coco/'

# Method 2: Use `backend_args`, `file_client_args` in versions before 3.0.0rc6
# backend_args = dict(
#     backend='petrel',
#     path_mapping=dict({
#         './data/': 's3://openmmlab/datasets/detection/',
#         'data/': 's3://openmmlab/datasets/detection/'
#     }))
backend_args = None

train_pipeline = [
    dict(type='LoadImageFromFile', backend_args=backend_args),
    dict(type='LoadAnnotations',   with_bbox=True),
    dict(type='Resize',            scale=(320, 320), keep_ratio=True),
    dict(type='RandomFlip',        prob=0.0),
    dict(type='PackDetInputs')]

test_pipeline = [dict(type='LoadImageFromFile', backend_args=backend_args),
                 dict(type='Resize',            scale=(320, 320), keep_ratio=True),
                 # If you don't have a gt annotation, delete the pipeline
                 dict(type='LoadAnnotations',   with_bbox=True),
                 dict(type      = 'PackDetInputs',
                      meta_keys = ('img_id', 'img_path', 'ori_shape', 'img_shape', 'scale_factor'))]

train_dataloader = dict(batch_size         = 2,
                        num_workers        = 2,
                        persistent_workers = True,
                        sampler            = dict(type         = 'DefaultSampler', shuffle=True),
                        batch_sampler      = dict(type         = 'AspectRatioBatchSampler'),
                        dataset            = dict(type         = dataset_type,
                                                  data_root    = data_root,
                                                  ann_file     = 'D:/Github/DUP/dataset/COCO_format/train/train.json',
                                                  data_prefix  = dict(img='D:/Github/DUP/dataset/COCO_format/train/images/'),
                                                  filter_cfg   = dict(filter_empty_gt=True, min_size=32),
                                                  pipeline     = train_pipeline,
                                                  backend_args = backend_args))

val_dataloader = dict(
    batch_size         = 1,
    num_workers        = 2,
    persistent_workers = True,
    drop_last          = False,
    sampler            = dict(type = 'DefaultSampler', shuffle = False),
    dataset            = dict(type         = dataset_type,
                              data_root    = data_root,
                              ann_file     = 'D:/Github/DUP/dataset/COCO_format/val/val.json',
                              data_prefix  = dict(img='D:/Github/DUP/dataset/COCO_format/val/images/'),
                              test_mode    = True,
                              pipeline     = test_pipeline,
                              backend_args = backend_args))

test_dataloader = val_dataloader

val_evaluator = dict(type         = 'CocoMetric',
                     ann_file     = data_root + 'val/val.json',
                     metric       = 'bbox',
                     format_only  = False,
                     backend_args = backend_args)

test_evaluator = val_evaluator

# inference on test dataset and format the output results for submission.
# test_dataloader = dict(
#     batch_size=1,
#     num_workers=2,
#     persistent_workers=True,
#     drop_last=False,
#     sampler=dict(type='DefaultSampler', shuffle=False),
#     dataset=dict(
#         type=dataset_type,
#         data_root=data_root,
#         ann_file=data_root + 'annotations/image_info_test-dev2017.json',
#         data_prefix=dict(img='test2017/'),
#         test_mode=True,
#         pipeline=test_pipeline))
# test_evaluator = dict(
#     type='CocoMetric',
#     metric='bbox',
#     format_only=True,
#     ann_file=data_root + 'annotations/image_info_test-dev2017.json',
#     outfile_prefix='./work_dirs/coco_detection/test')


#%% (2) Model : cornernet
# _base_ = ['../_base_/datasets/coco_detection.py',
#           '../_base_/schedules/schedule_1x.py', 
#           '../_base_/default_runtime.py']

# model settings
model = dict(
    type='FCOS',
    data_preprocessor=dict(
        type='DetDataPreprocessor',
        mean=[102.9801, 115.9465, 122.7717],
        std=[1.0, 1.0, 1.0],
        bgr_to_rgb=False,
        pad_size_divisor=32),
    backbone=dict(
        type='ResNet',
        depth=50,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        frozen_stages=1,
        norm_cfg=dict(type='BN', requires_grad=False),
        norm_eval=True,
        style='caffe',
        init_cfg=dict(
            type='Pretrained',
            checkpoint='open-mmlab://detectron/resnet50_caffe')),
    neck=dict(
        type='FPN',
        in_channels=[256, 512, 1024, 2048],
        out_channels=256,
        start_level=1,
        add_extra_convs='on_output',  # use P5
        num_outs=5,
        relu_before_extra_convs=True),
    bbox_head=dict(
        type='FCOSHead',
        num_classes=80,
        in_channels=256,
        stacked_convs=4,
        feat_channels=256,
        strides=[8, 16, 32, 64, 128],
        loss_cls=dict(
            type='FocalLoss',
            use_sigmoid=True,
            gamma=2.0,
            alpha=0.25,
            loss_weight=1.0),
        loss_bbox=dict(type='IoULoss', loss_weight=1.0),
        loss_centerness=dict(
            type='CrossEntropyLoss', use_sigmoid=True, loss_weight=1.0)),
    # testing settings
    test_cfg=dict(
        nms_pre=1000,
        min_bbox_size=0,
        score_thr=0.05,
        nms=dict(type='nms', iou_threshold=0.5),
        max_per_img=100))

# learning rate
param_scheduler = [
    dict(type='ConstantLR', factor=1.0 / 3, by_epoch=False, begin=0, end=500),
    dict(
        type='MultiStepLR',
        begin=0,
        end=12,
        by_epoch=True,
        milestones=[8, 11],
        gamma=0.1)
]

# optimizer
optim_wrapper = dict(optimizer=dict(lr=0.01),
                     paramwise_cfg=dict(bias_lr_mult=2., bias_decay_mult=0.),
                     clip_grad=dict(max_norm=35, norm_type=2))



#%% Schedules

# training schedule for 1x
train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=20, val_interval=1) # max_epochs=12 for 8 GPUs, 12*8=96 for a single GPU.
val_cfg   = dict(type='ValLoop')
test_cfg  = dict(type='TestLoop')

# learning rate
# param_scheduler = [
#     dict(
#         type='LinearLR', start_factor=0.001, by_epoch=False, begin=0, end=500),
#     dict(
#         type='MultiStepLR',
#         begin=0,
#         end=12,
#         by_epoch=True,
#         milestones=[8, 11],
#         gamma=0.1)
# ]

# # optimizer
# optim_wrapper = dict(type                        = 'OptimWrapper',
#                      optimizer=dict(type         = 'SGD', 
#                                     lr           = 0.0025, 
#                                     momentum     = 0.9, 
#                                     weight_decay = 0.0001))  # lr=0.02 for 8 GPUs, lr=0.025 for a single gpu.

# Default setting for scaling LR automatically
#   - `enable` means enable scaling LR automatically
#       or not by default.
#   - `base_batch_size` = (8 GPUs) x (2 samples per GPU).
auto_scale_lr = dict(enable=True, base_batch_size=64) # base batch = 16



#%% Original code of default_runtime.py

default_scope = 'mmdet'

default_hooks = dict(timer           = dict(type='IterTimerHook'),
                     logger          = dict(type='LoggerHook', interval=50),
                     param_scheduler = dict(type='ParamSchedulerHook'),
                     checkpoint      = dict(type='CheckpointHook', interval=1), # saving weight after how many epochs ran.
                     sampler_seed    = dict(type='DistSamplerSeedHook'),
                     visualization   = dict(type='DetVisualizationHook'))

env_cfg = dict(cudnn_benchmark = False,
               mp_cfg          = dict(mp_start_method='fork', opencv_num_threads=0),
               dist_cfg        = dict(backend='nccl'))

vis_backends  = [dict(type='LocalVisBackend')]
visualizer    = dict(type         = 'DetLocalVisualizer', 
                     vis_backends = vis_backends, 
                     name         = 'visualizer')
log_processor = dict(type        = 'LogProcessor', 
                     window_size = 50, 
                     by_epoch    = True)

log_level = 'INFO'
load_from = 'https://download.openmmlab.com/mmdetection/v2.0/fcos/fcos_r50_caffe_fpn_gn-head_1x_coco/fcos_r50_caffe_fpn_gn-head_1x_coco-821213aa.pth' # None # 'D:/Github/mmdetection/weights/faster_rcnn_r50_fpn_1x_coco_20200130-047c8118.pth' # None
resume    = False
