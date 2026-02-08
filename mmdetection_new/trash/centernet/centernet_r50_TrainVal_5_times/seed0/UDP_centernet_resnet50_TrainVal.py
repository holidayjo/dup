auto_scale_lr = dict(base_batch_size=16, enable=False)
backend_args = None
data_root = '/mnt/Documents/Dad/github/DUP/yolo_to_coco/output/'
dataset_type = 'CocoDataset'
default_hooks = dict(
    checkpoint=dict(interval=5, max_keep_ckpts=3, type='CheckpointHook'),
    logger=dict(interval=50, type='LoggerHook'),
    param_scheduler=dict(type='ParamSchedulerHook'),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    timer=dict(type='IterTimerHook'),
    visualization=dict(type='DetVisualizationHook'))
default_scope = 'mmdet'
env_cfg = dict(
    cudnn_benchmark=False,
    dist_cfg=dict(backend='nccl'),
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0))
image_folder_test = '/mnt/Documents/Dad/github/DUP/DATA/Euljiro/1_balanced_simulation/test/'
image_folder_train = '/mnt/Documents/Dad/github/DUP/DATA/Euljiro/1_balanced_simulation/TrainVal/'
image_folder_val = '/mnt/Documents/Dad/github/DUP/DATA/Euljiro/1_balanced_simulation/test/'
launcher = 'none'
load_from = '/mnt/Documents/Dad/github/DUP/mmdetection_new/pre_trained_weights/centernet-update_r50-caffe_fpn_ms-1x_coco_20230512_203845-8306baf2.pth'
log_level = 'INFO'
log_processor = dict(by_epoch=True, type='LogProcessor', window_size=50)
metainfo = dict(
    classes=(
        'U',
        'D',
        'P',
    ),
    palette=[
        (
            220,
            20,
            60,
        ),
        (
            119,
            11,
            32,
        ),
        (
            0,
            0,
            142,
        ),
    ])
model = dict(
    backbone=dict(
        depth=50,
        frozen_stages=1,
        init_cfg=dict(
            checkpoint='open-mmlab://detectron2/resnet50_caffe',
            type='Pretrained'),
        norm_cfg=dict(requires_grad=False, type='BN'),
        norm_eval=True,
        num_stages=4,
        out_indices=(
            0,
            1,
            2,
            3,
        ),
        style='caffe',
        type='ResNet'),
    bbox_head=dict(
        feat_channels=256,
        hm_min_overlap=0.8,
        hm_min_radius=4,
        in_channels=256,
        loss_bbox=dict(loss_weight=2.0, type='GIoULoss'),
        loss_cls=dict(
            loss_weight=1.0,
            neg_weight=0.75,
            pos_weight=0.25,
            type='GaussianFocalLoss'),
        more_pos_thresh=0.2,
        more_pos_topk=9,
        num_classes=3,
        soft_weight_on_reg=False,
        stacked_convs=4,
        strides=[
            8,
            16,
            32,
            64,
            128,
        ],
        type='CenterNetUpdateHead'),
    data_preprocessor=dict(
        bgr_to_rgb=False,
        mean=[
            103.53,
            116.28,
            123.675,
        ],
        pad_size_divisor=32,
        std=[
            1.0,
            1.0,
            1.0,
        ],
        type='DetDataPreprocessor'),
    neck=dict(
        add_extra_convs='on_output',
        in_channels=[
            256,
            512,
            1024,
            2048,
        ],
        init_cfg=dict(layer='Conv2d', type='Caffe2Xavier'),
        num_outs=5,
        out_channels=256,
        relu_before_extra_convs=True,
        start_level=1,
        type='FPN'),
    test_cfg=dict(
        max_per_img=100,
        min_bbox_size=0,
        nms=dict(iou_threshold=0.6, type='nms'),
        nms_pre=1000,
        score_thr=0.05),
    train_cfg=None,
    type='CenterNet')
optim_wrapper = dict(
    optimizer=dict(lr=0.01, momentum=0.9, type='SGD', weight_decay=0.0001),
    paramwise_cfg=dict(norm_decay_mult=0.0),
    type='OptimWrapper')
param_scheduler = [
    dict(
        begin=0, by_epoch=False, end=500, start_factor=0.001, type='LinearLR'),
    dict(
        begin=0,
        by_epoch=True,
        end=100,
        gamma=0.1,
        milestones=[
            70,
            90,
        ],
        type='MultiStepLR'),
]
randomness = dict(deterministic=False, seed=0)
resume = False
test_cfg = dict(type='TestLoop')
test_dataloader = dict(
    batch_size=1,
    dataset=dict(
        ann_file='test.json',
        backend_args=None,
        data_prefix=dict(
            img=
            '/mnt/Documents/Dad/github/DUP/DATA/Euljiro/1_balanced_simulation/test/'
        ),
        data_root='/mnt/Documents/Dad/github/DUP/yolo_to_coco/output/',
        metainfo=dict(
            classes=(
                'U',
                'D',
                'P',
            ),
            palette=[
                (
                    220,
                    20,
                    60,
                ),
                (
                    119,
                    11,
                    32,
                ),
                (
                    0,
                    0,
                    142,
                ),
            ]),
        pipeline=[
            dict(to_float32=True, type='LoadImageFromFile'),
            dict(keep_ratio=True, scale=(
                1333,
                800,
            ), type='Resize'),
            dict(type='LoadAnnotations', with_bbox=True),
            dict(
                meta_keys=(
                    'img_id',
                    'img_path',
                    'ori_shape',
                    'img_shape',
                    'scale_factor',
                ),
                type='PackDetInputs'),
        ],
        test_mode=True,
        type='CocoDataset'),
    drop_last=False,
    num_workers=2,
    persistent_workers=True,
    sampler=dict(shuffle=False, type='DefaultSampler'))
test_evaluator = dict(
    ann_file='/mnt/Documents/Dad/github/DUP/yolo_to_coco/output/test.json',
    backend_args=None,
    format_only=False,
    metric='bbox',
    type='CocoMetric')
test_pipeline = [
    dict(to_float32=True, type='LoadImageFromFile'),
    dict(keep_ratio=True, scale=(
        1333,
        800,
    ), type='Resize'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(
        meta_keys=(
            'img_id',
            'img_path',
            'ori_shape',
            'img_shape',
            'scale_factor',
        ),
        type='PackDetInputs'),
]
train_cfg = dict(max_epochs=34, type='EpochBasedTrainLoop', val_interval=1)
train_dataloader = dict(
    batch_sampler=dict(type='AspectRatioBatchSampler'),
    batch_size=4,
    dataset=dict(
        ann_file='TrainVal.json',
        backend_args=None,
        data_prefix=dict(
            img=
            '/mnt/Documents/Dad/github/DUP/DATA/Euljiro/1_balanced_simulation/TrainVal/'
        ),
        data_root='/mnt/Documents/Dad/github/DUP/yolo_to_coco/output/',
        filter_cfg=dict(filter_empty_gt=True, min_size=32),
        metainfo=dict(
            classes=(
                'U',
                'D',
                'P',
            ),
            palette=[
                (
                    220,
                    20,
                    60,
                ),
                (
                    119,
                    11,
                    32,
                ),
                (
                    0,
                    0,
                    142,
                ),
            ]),
        pipeline=[
            dict(to_float32=True, type='LoadImageFromFile'),
            dict(type='LoadAnnotations', with_bbox=True),
            dict(
                keep_ratio=True,
                scales=[
                    (
                        1333,
                        640,
                    ),
                    (
                        1333,
                        672,
                    ),
                    (
                        1333,
                        704,
                    ),
                    (
                        1333,
                        736,
                    ),
                    (
                        1333,
                        768,
                    ),
                    (
                        1333,
                        800,
                    ),
                ],
                type='RandomChoiceResize'),
            dict(prob=0.0, type='RandomFlip'),
            dict(type='PackDetInputs'),
        ],
        type='CocoDataset'),
    num_workers=4,
    persistent_workers=True,
    sampler=dict(shuffle=True, type='DefaultSampler'))
train_pipeline = [
    dict(to_float32=True, type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(
        keep_ratio=True,
        scales=[
            (
                1333,
                640,
            ),
            (
                1333,
                672,
            ),
            (
                1333,
                704,
            ),
            (
                1333,
                736,
            ),
            (
                1333,
                768,
            ),
            (
                1333,
                800,
            ),
        ],
        type='RandomChoiceResize'),
    dict(prob=0.0, type='RandomFlip'),
    dict(type='PackDetInputs'),
]
val_cfg = dict(type='ValLoop')
val_dataloader = dict(
    batch_size=1,
    dataset=dict(
        ann_file='test.json',
        backend_args=None,
        data_prefix=dict(
            img=
            '/mnt/Documents/Dad/github/DUP/DATA/Euljiro/1_balanced_simulation/test/'
        ),
        data_root='/mnt/Documents/Dad/github/DUP/yolo_to_coco/output/',
        metainfo=dict(
            classes=(
                'U',
                'D',
                'P',
            ),
            palette=[
                (
                    220,
                    20,
                    60,
                ),
                (
                    119,
                    11,
                    32,
                ),
                (
                    0,
                    0,
                    142,
                ),
            ]),
        pipeline=[
            dict(to_float32=True, type='LoadImageFromFile'),
            dict(keep_ratio=True, scale=(
                1333,
                800,
            ), type='Resize'),
            dict(type='LoadAnnotations', with_bbox=True),
            dict(
                meta_keys=(
                    'img_id',
                    'img_path',
                    'ori_shape',
                    'img_shape',
                    'scale_factor',
                ),
                type='PackDetInputs'),
        ],
        test_mode=True,
        type='CocoDataset'),
    drop_last=False,
    num_workers=2,
    persistent_workers=True,
    sampler=dict(shuffle=False, type='DefaultSampler'))
val_evaluator = dict(
    ann_file='/mnt/Documents/Dad/github/DUP/yolo_to_coco/output/test.json',
    backend_args=None,
    format_only=False,
    metric='bbox',
    type='CocoMetric')
vis_backends = [
    dict(type='LocalVisBackend'),
]
visualizer = dict(
    name='visualizer',
    type='DetLocalVisualizer',
    vis_backends=[
        dict(type='LocalVisBackend'),
    ])
work_dir = 'work_dirs/centernet_r50_TrainVal_5_times/seed0'
