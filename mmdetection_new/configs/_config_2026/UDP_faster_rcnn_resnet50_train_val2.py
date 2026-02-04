# =========================================================
# Faster R-CNN R50-FPN (COCO pretrained) – 3 custom classes
# Compatible with MMDetection 3.x
# =========================================================

default_scope = 'mmdet'

# ---------------------------------------------------------
# 1. META INFO
# ---------------------------------------------------------
metainfo = dict(
    classes=('U', 'D', 'P'),
    palette=[(220, 20, 60), (119, 11, 32), (0, 0, 142)]
)

# ---------------------------------------------------------
# 2. MODEL
# ---------------------------------------------------------
model = dict(
    type='FasterRCNN',
    data_preprocessor=dict(
        type='DetDataPreprocessor',
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        bgr_to_rgb=True,
        pad_size_divisor=32
    ),

    backbone=dict(
        type='ResNet',
        depth=50,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        frozen_stages=1,
        norm_cfg=dict(type='BN', requires_grad=True),
        norm_eval=True,
        style='pytorch'
    ),

    neck=dict(
        type='FPN',
        in_channels=[256, 512, 1024, 2048],
        out_channels=256,
        num_outs=5
    ),

    rpn_head=dict(
        type='RPNHead',
        in_channels=256,
        feat_channels=256,
        anchor_generator=dict(
            type='AnchorGenerator',
            scales=[8],
            ratios=[0.5, 1.0, 2.0],
            strides=[4, 8, 16, 32, 64]
        ),
        bbox_coder=dict(
            type='DeltaXYWHBBoxCoder',
            target_means=[0., 0., 0., 0.],
            target_stds=[1.0, 1.0, 1.0, 1.0]
        ),
        loss_cls=dict(
            type='CrossEntropyLoss',
            use_sigmoid=True,
            loss_weight=1.0
        ),
        loss_bbox=dict(type='L1Loss', loss_weight=1.0)
    ),

    roi_head=dict(
        type='StandardRoIHead',
        bbox_roi_extractor=dict(
            type='SingleRoIExtractor',
            roi_layer=dict(type='RoIAlign', output_size=7, sampling_ratio=0),
            out_channels=256,
            featmap_strides=[4, 8, 16, 32]
        ),
        bbox_head=dict(
            type='Shared2FCBBoxHead',
            in_channels=256,
            fc_out_channels=1024,
            roi_feat_size=7,
            num_classes=3,
            bbox_coder=dict(
                type='DeltaXYWHBBoxCoder',
                target_means=[0., 0., 0., 0.],
                target_stds=[0.1, 0.1, 0.2, 0.2]
            ),
            reg_class_agnostic=False,
            loss_cls=dict(
                type='CrossEntropyLoss',
                use_sigmoid=False,
                loss_weight=1.0
            ),
            loss_bbox=dict(type='L1Loss', loss_weight=1.0)
        )
    )
)

# ---------------------------------------------------------
# 3. TRAIN / VAL / TEST LOOPS
# ---------------------------------------------------------
train_cfg = dict(
    type='EpochBasedTrainLoop',
    max_epochs=100,
    val_interval=1
)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')

model['train_cfg'] = dict(
    rpn=dict(
        assigner=dict(
            type='MaxIoUAssigner',
            pos_iou_thr=0.7,
            neg_iou_thr=0.3,
            min_pos_iou=0.3
        ),
        sampler=dict(
            type='RandomSampler',
            num=256,
            pos_fraction=0.5
        ),
        allowed_border=-1,
        pos_weight=-1,   # REQUIRED
        debug=False
    ),
    rpn_proposal=dict(
        nms_pre=2000,
        max_per_img=1000,
        nms=dict(type='nms', iou_threshold=0.7),
        min_bbox_size=0
    ),
    rcnn=dict(
        assigner=dict(
            type='MaxIoUAssigner',
            pos_iou_thr=0.5,
            neg_iou_thr=0.5,
            min_pos_iou=0.5
        ),
        sampler=dict(
            type='RandomSampler',
            num=512,
            pos_fraction=0.25,
            add_gt_as_proposals=True
        ),
        pos_weight=-1,   # ✅ FIX: REQUIRED IN MMDet 3.x
        debug=False
    )
)

model['test_cfg'] = dict(
    rpn=dict(
        nms_pre=1000,
        max_per_img=1000,
        nms=dict(type='nms', iou_threshold=0.7),
        min_bbox_size=0
    ),
    rcnn=dict(
        score_thr=0.05,
        nms=dict(type='nms', iou_threshold=0.5),
        max_per_img=100
    )
)

# ---------------------------------------------------------
# 4. DATASETS
# ---------------------------------------------------------
data_root = '/mnt/Documents/Dad/github/DUP/yolo_to_coco/output/'

train_dataloader = dict(
    batch_size=4,
    num_workers=8,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=dict(
        type='CocoDataset',
        metainfo=metainfo,
        data_root=data_root,
        ann_file='train.json',
        data_prefix=dict(
            img='/mnt/Documents/Dad/github/DUP/DATA/Euljiro/1_balanced_simulation/train/'
        ),
        filter_cfg=dict(filter_empty_gt=True, min_size=32),
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(type='LoadAnnotations', with_bbox=True),
            dict(type='Resize', scale=(320, 320), keep_ratio=True),
            dict(type='RandomFlip', prob=0.0),
            dict(type='PackDetInputs')
        ]
    )
)

val_dataloader = dict(
    batch_size=4,
    num_workers=8,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type='CocoDataset',
        metainfo=metainfo,
        data_root=data_root,
        ann_file='val.json',
        data_prefix=dict(
            img='/mnt/Documents/Dad/github/DUP/DATA/Euljiro/1_balanced_simulation/val/'
        ),
        test_mode=True,
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(type='Resize', scale=(320, 320), keep_ratio=True),
            dict(type='LoadAnnotations', with_bbox=True),
            dict(type='PackDetInputs')
        ]
    )
)

test_dataloader = val_dataloader

val_evaluator = dict(
    type='CocoMetric',
    ann_file=data_root + 'val.json',
    metric='bbox'
)

test_evaluator = val_evaluator

# ---------------------------------------------------------
# 5. OPTIMIZER & LR
# ---------------------------------------------------------
optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(
        type='SGD',
        lr=0.005,
        momentum=0.9,
        weight_decay=0.0001
    )
)

param_scheduler = [
    dict(
        type='LinearLR',
        start_factor=0.001,
        by_epoch=False,
        begin=0,
        end=500
    ),
    dict(
        type='MultiStepLR',
        by_epoch=True,
        begin=0,
        end=100,
        milestones=[70, 90],
        gamma=0.1
    )
]

# ---------------------------------------------------------
# 6. RUNTIME
# ---------------------------------------------------------
work_dir = '/mnt/Documents/Dad/github/DUP/mmdetection_new/work_dirs/faster_rcnn_r50_final'

load_from = (
    '/mnt/Documents/Dad/github/DUP/mmdetection_new/'
    'faster_rcnn_r50_fpn_1x_coco_20200130-047c8118.pth'
)

resume = False
