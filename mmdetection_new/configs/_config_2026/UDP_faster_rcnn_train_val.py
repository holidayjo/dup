# # =========================================================
# # CONFIG: configs/my_faster_rcnn_r18_train_val.py
# # Description: Faster R-CNN R18 (Stand-Alone) - No External Dependencies
# # =========================================================

# # 1. MODEL ARCHITECTURE (Defined explicitly to avoid missing file errors)
# model = dict(
#     type='FasterRCNN',
#     data_preprocessor=dict(
#         type='DetDataPreprocessor',
#         mean=[123.675, 116.28, 103.53],
#         std=[58.395, 57.12, 57.375],
#         bgr_to_rgb=True,
#         pad_size_divisor=32),
#     backbone=dict(
#         type='ResNet',
#         depth=18,  # ResNet-18
#         num_stages=4,
#         out_indices=(0, 1, 2, 3),
#         frozen_stages=1,
#         norm_cfg=dict(type='BN', requires_grad=True),
#         norm_eval=True,
#         style='pytorch',
#         init_cfg=dict(type='Pretrained', checkpoint='torchvision://resnet18')),
#     neck=dict(
#         type='FPN',
#         in_channels=[64, 128, 256, 512], # Correct channels for ResNet-18
#         out_channels=256,
#         num_outs=5),
#     rpn_head=dict(
#         type='RPNHead',
#         in_channels=256,
#         feat_channels=256,
#         anchor_generator=dict(
#             type='AnchorGenerator',
#             scales=[8],
#             ratios=[0.5, 1.0, 2.0],
#             strides=[4, 8, 16, 32, 64]),
#         bbox_coder=dict(
#             type='DeltaXYWHBBoxCoder',
#             target_means=[.0, .0, .0, .0],
#             target_stds=[1.0, 1.0, 1.0, 1.0]),
#         loss_cls=dict(
#             type='CrossEntropyLoss', use_sigmoid=True, loss_weight=1.0),
#         loss_bbox=dict(type='L1Loss', loss_weight=1.0)),
#     roi_head=dict(
#         type='StandardRoIHead',
#         bbox_roi_extractor=dict(
#             type='SingleRoIExtractor',
#             roi_layer=dict(type='RoIAlign', output_size=7, sampling_ratio=0),
#             out_channels=256,
#             featmap_strides=[4, 8, 16, 32]),
#         bbox_head=dict(
#             type='Shared2FCBBoxHead',
#             in_channels=256,
#             fc_out_channels=1024,
#             roi_feat_size=7,
#             num_classes=3,  # <--- YOUR 3 CLASSES
#             bbox_coder=dict(
#                 type='DeltaXYWHBBoxCoder',
#                 target_means=[0., 0., 0., 0.],
#                 target_stds=[0.1, 0.1, 0.2, 0.2]),
#             reg_class_agnostic=False,
#             loss_cls=dict(
#                 type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0),
#             loss_bbox=dict(type='L1Loss', loss_weight=1.0))),
#     train_cfg=dict(
#         rpn=dict(
#             assigner=dict(
#                 type='MaxIoUAssigner',
#                 pos_iou_thr=0.7,
#                 neg_iou_thr=0.3,
#                 min_pos_iou=0.3,
#                 match_low_quality=True,
#                 ignore_iof_thr=-1),
#             sampler=dict(
#                 type='RandomSampler',
#                 num=256,
#                 pos_fraction=0.5,
#                 neg_pos_ub=-1,
#                 add_gt_as_proposals=False),
#             allowed_border=-1,
#             pos_weight=-1,
#             debug=False),
#         rpn_proposal=dict(
#             nms_pre=2000,
#             max_per_img=1000,
#             nms=dict(type='nms', iou_threshold=0.7),
#             min_bbox_size=0),
#         rcnn=dict(
#             assigner=dict(
#                 type='MaxIoUAssigner',
#                 pos_iou_thr=0.5,
#                 neg_iou_thr=0.5,
#                 min_pos_iou=0.5,
#                 match_low_quality=False,
#                 ignore_iof_thr=-1),
#             sampler=dict(
#                 type='RandomSampler',
#                 num=512,
#                 pos_fraction=0.25,
#                 neg_pos_ub=-1,
#                 add_gt_as_proposals=True),
#             pos_weight=-1,
#             debug=False)),
#     test_cfg=dict(
#         rpn=dict(
#             nms_pre=1000,
#             max_per_img=1000,
#             nms=dict(type='nms', iou_threshold=0.7),
#             min_bbox_size=0),
#         rcnn=dict(
#             score_thr=0.05,
#             nms=dict(type='nms', iou_threshold=0.5),
#             max_per_img=100))
# )

# # 2. PATHS & DATASET
# data_root = '/mnt/Documents/Dad/github/DUP/yolo_to_coco/output/'
# image_folder_train = '/mnt/Documents/Dad/github/DUP/DATA/Euljiro/1_balanced_simulation/train/'
# image_folder_val   = '/mnt/Documents/Dad/github/DUP/DATA/Euljiro/1_balanced_simulation/val/'

# metainfo = {
#     'classes': ('U', 'D', 'P'),
#     'palette': [(220, 20, 60), (119, 11, 32), (0, 0, 142)]
# }

# backend_args = None

# train_pipeline = [
#     dict(type='LoadImageFromFile', backend_args=backend_args),
#     dict(type='LoadAnnotations', with_bbox=True),
#     dict(type='Resize', scale=(320, 320), keep_ratio=True),
#     dict(type='RandomFlip', prob=0.0), # No Flip
#     dict(type='PackDetInputs')
# ]

# test_pipeline = [
#     dict(type='LoadImageFromFile', backend_args=backend_args),
#     dict(type='Resize', scale=(320, 320), keep_ratio=True),
#     dict(type='LoadAnnotations', with_bbox=True),
#     dict(
#         type='PackDetInputs',
#         meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape', 'scale_factor')
#     )
# ]

# train_dataloader = dict(
#     batch_size=16,
#     num_workers=8,
#     persistent_workers=True,
#     sampler=dict(type='DefaultSampler', shuffle=True),
#     batch_sampler=dict(type='AspectRatioBatchSampler'),
#     dataset=dict(
#         type='RepeatDataset',
#         times=1,
#         dataset=dict(
#             type='CocoDataset',
#             metainfo=metainfo,
#             data_root=data_root,
#             ann_file='train.json',
#             data_prefix=dict(img=image_folder_train),
#             filter_cfg=dict(filter_empty_gt=False, min_size=32),
#             backend_args=backend_args,
#             pipeline=train_pipeline
#         )
#     )
# )

# val_dataloader = dict(
#     batch_size=16,
#     num_workers=8,
#     persistent_workers=True,
#     drop_last=False,
#     sampler=dict(type='DefaultSampler', shuffle=False),
#     dataset=dict(
#         type='CocoDataset',
#         metainfo=metainfo,
#         data_root=data_root,
#         ann_file='val.json',
#         data_prefix=dict(img=image_folder_val),
#         test_mode=True,
#         backend_args=backend_args,
#         pipeline=test_pipeline
#     )
# )

# test_dataloader = val_dataloader

# val_evaluator = dict(
#     type='CocoMetric',
#     ann_file=data_root + 'val.json',
#     metric='bbox',
#     format_only=False,
#     backend_args=backend_args
# )
# test_evaluator = val_evaluator

# # 3. SCHEDULE (50 Epochs)
# train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=50, val_interval=1)
# val_cfg   = dict(type='ValLoop')
# test_cfg  = dict(type='TestLoop')

# optim_wrapper = dict(
#     type='OptimWrapper',
#     optimizer=dict(type='SGD', lr=0.005, momentum=0.9, weight_decay=0.0001)
# )

# param_scheduler = [
#     dict(type='LinearLR', start_factor=0.001, by_epoch=False, begin=0, end=500),
#     dict(type='MultiStepLR', begin=0, end=50, by_epoch=True, milestones=[35, 45], gamma=0.1)
# ]

# # 4. RUNTIME
# default_scope = 'mmdet'
# default_hooks = dict(
#     timer=dict(type='IterTimerHook'),
#     logger=dict(type='LoggerHook', interval=50),
#     param_scheduler=dict(type='ParamSchedulerHook'),
#     checkpoint=dict(type='CheckpointHook', interval=50),
#     sampler_seed=dict(type='DistSamplerSeedHook'),
#     visualization=dict(type='DetVisualizationHook')
# )

# env_cfg = dict(
#     cudnn_benchmark=False,
#     mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0),
#     dist_cfg=dict(backend='nccl'),
# )

# vis_backends = [dict(type='LocalVisBackend')]
# visualizer = dict(
#     type='DetLocalVisualizer', vis_backends=vis_backends, name='visualizer')
# log_processor = dict(type='LogProcessor', window_size=50, by_epoch=True)
# log_level = 'INFO'

# # 4. RUNTIME
# work_dir = '/mnt/Documents/Dad/github/DUP/mmdetection_new/work_dirs/faster_rcnn_r18_train_val'

# # CORRECT LINK FOR FASTER R-CNN (ResNet-18) COCO WEIGHTS
# load_from = 'https://download.openmmlab.com/mmdetection/v2.0/faster_rcnn/faster_rcnn_r18_fpn_1x_coco/faster_rcnn_r18_fpn_1x_coco_20200130-047c8118.pth'

# resume = False




# =========================================================
# CONFIG: configs/my_faster_rcnn_r18_train_val.py
# Description: Faster R-CNN R18 (ImageNet Init) | Train->Val
# =========================================================

# 1. MODEL ARCHITECTURE
model = dict(
    type='FasterRCNN',
    data_preprocessor=dict(
        type='DetDataPreprocessor',
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        bgr_to_rgb=True,
        pad_size_divisor=32),
    backbone=dict(
        type='ResNet',
        depth=18,  # ResNet-18
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        frozen_stages=1,
        norm_cfg=dict(type='BN', requires_grad=True),
        norm_eval=True,
        style='pytorch',
        # LOAD IMAGENET WEIGHTS (This works!)
        init_cfg=dict(type='Pretrained', checkpoint='torchvision://resnet18')),
    neck=dict(
        type='FPN',
        in_channels=[64, 128, 256, 512],
        out_channels=256,
        num_outs=5),
    rpn_head=dict(
        type='RPNHead',
        in_channels=256,
        feat_channels=256,
        anchor_generator=dict(
            type='AnchorGenerator',
            scales=[8],
            ratios=[0.5, 1.0, 2.0],
            strides=[4, 8, 16, 32, 64]),
        bbox_coder=dict(
            type='DeltaXYWHBBoxCoder',
            target_means=[.0, .0, .0, .0],
            target_stds=[1.0, 1.0, 1.0, 1.0]),
        loss_cls=dict(
            type='CrossEntropyLoss', use_sigmoid=True, loss_weight=1.0),
        loss_bbox=dict(type='L1Loss', loss_weight=1.0)),
    roi_head=dict(
        type='StandardRoIHead',
        bbox_roi_extractor=dict(
            type='SingleRoIExtractor',
            roi_layer=dict(type='RoIAlign', output_size=7, sampling_ratio=0),
            out_channels=256,
            featmap_strides=[4, 8, 16, 32]),
        bbox_head=dict(
            type='Shared2FCBBoxHead',
            in_channels=256,
            fc_out_channels=1024,
            roi_feat_size=7,
            num_classes=3, # YOUR 3 CLASSES
            bbox_coder=dict(
                type='DeltaXYWHBBoxCoder',
                target_means=[0., 0., 0., 0.],
                target_stds=[0.1, 0.1, 0.2, 0.2]),
            reg_class_agnostic=False,
            loss_cls=dict(
                type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0),
            loss_bbox=dict(type='L1Loss', loss_weight=1.0))),
    train_cfg=dict(
        rpn=dict(
            assigner=dict(
                type='MaxIoUAssigner',
                pos_iou_thr=0.7,
                neg_iou_thr=0.3,
                min_pos_iou=0.3,
                match_low_quality=True,
                ignore_iof_thr=-1),
            sampler=dict(
                type='RandomSampler',
                num=256,
                pos_fraction=0.5,
                neg_pos_ub=-1,
                add_gt_as_proposals=False),
            allowed_border=-1,
            pos_weight=-1,
            debug=False),
        rpn_proposal=dict(
            nms_pre=2000,
            max_per_img=1000,
            nms=dict(type='nms', iou_threshold=0.7),
            min_bbox_size=0),
        rcnn=dict(
            assigner=dict(
                type='MaxIoUAssigner',
                pos_iou_thr=0.5,
                neg_iou_thr=0.5,
                min_pos_iou=0.5,
                match_low_quality=False,
                ignore_iof_thr=-1),
            sampler=dict(
                type='RandomSampler',
                num=512,
                pos_fraction=0.25,
                neg_pos_ub=-1,
                add_gt_as_proposals=True),
            pos_weight=-1,
            debug=False)),
    test_cfg=dict(
        rpn=dict(
            nms_pre=1000,
            max_per_img=1000,
            nms=dict(type='nms', iou_threshold=0.7),
            min_bbox_size=0),
        rcnn=dict(
            score_thr=0.05,
            nms=dict(type='nms', iou_threshold=0.5),
            max_per_img=100))
)

# 2. DATASETS & PATHS
data_root = '/mnt/Documents/Dad/github/DUP/yolo_to_coco/output/'
image_folder_train = '/mnt/Documents/Dad/github/DUP/DATA/Euljiro/1_balanced_simulation/train/'
image_folder_val   = '/mnt/Documents/Dad/github/DUP/DATA/Euljiro/1_balanced_simulation/val/'

metainfo = {
    'classes': ('U', 'D', 'P'),
    'palette': [(220, 20, 60), (119, 11, 32), (0, 0, 142)]
}

backend_args = None

# Pipeline: 320x320 Resize, No Flip (Same as CenterNet)
train_pipeline = [
    dict(type='LoadImageFromFile', backend_args=backend_args),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='Resize', scale=(320, 320), keep_ratio=True),
    dict(type='RandomFlip', prob=0.0),
    dict(type='PackDetInputs')
]

test_pipeline = [
    dict(type='LoadImageFromFile', backend_args=backend_args),
    dict(type='Resize', scale=(320, 320), keep_ratio=True),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='PackDetInputs', meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape', 'scale_factor'))
]

train_dataloader = dict(
    batch_size=16,
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
            pipeline=train_pipeline
        )
    )
)

val_dataloader = dict(
    batch_size=16,
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
        pipeline=test_pipeline
    )
)
test_dataloader = val_dataloader

val_evaluator = dict(
    type='CocoMetric',
    ann_file=data_root + 'val.json',
    metric='bbox',
    format_only=False,
    backend_args=backend_args
)
test_evaluator = val_evaluator

# 3. SCHEDULE (50 Epochs)
train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=50, val_interval=1)
val_cfg   = dict(type='ValLoop')
test_cfg  = dict(type='TestLoop')

optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(type='SGD', lr=0.005, momentum=0.9, weight_decay=0.0001)
)

param_scheduler = [
    dict(type='LinearLR', start_factor=0.001, by_epoch=False, begin=0, end=500),
    dict(type='MultiStepLR', begin=0, end=50, by_epoch=True, milestones=[35, 45], gamma=0.1)
]

# 4. RUNTIME
default_scope = 'mmdet'
default_hooks = dict(
    timer=dict(type='IterTimerHook'),
    logger=dict(type='LoggerHook', interval=50),
    param_scheduler=dict(type='ParamSchedulerHook'),
    checkpoint=dict(type='CheckpointHook', interval=50),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    visualization=dict(type='DetVisualizationHook')
)

work_dir = '/mnt/Documents/Dad/github/DUP/mmdetection_new/work_dirs/faster_rcnn_r18_train_val'

# PLAN B: Use ImageNet weights (via init_cfg) instead of broken COCO link
load_from = None
resume = False