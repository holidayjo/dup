# =========================================================
# CONFIG: configs/my_faster_rcnn_r50_passenger_train.py
# Description: Faster R-CNN R50 (MS-Train 3x) - For Passenger Dataset
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
        depth=50,  # <--- ResNet-50으로 변경
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        frozen_stages=1,
        norm_cfg=dict(type='BN', requires_grad=True),
        norm_eval=True,
        style='pytorch',
        init_cfg=dict(type='Pretrained', checkpoint='torchvision://resnet50')), # R50 초기화
    neck=dict(
        type='FPN',
        in_channels=[256, 512, 1024, 2048], # <--- ResNet-50의 출력 채널에 맞게 변경
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
            num_classes=3, # 클래스: 'U', 'D', 'P'
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
image_folder_train = '/mnt/Documents/Dad/github/DUP/DATA/Euljiro/1_balanced_simulation/TrainVal/'
image_folder_val   = '/mnt/Documents/Dad/github/DUP/DATA/Euljiro/1_balanced_simulation/test/'

metainfo = {
    'classes': ('U', 'D', 'P'),
    'palette': [(220, 20, 60), (119, 11, 32), (0, 0, 142)]
}

# 훈련 효율을 위해 Multi-scale Resize 적용 (가중치 특성 반영)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(
        type='RandomChoiceResize', # <--- 다운로드하신 가중치(mstrain)의 특성을 살림
        scales=[(1333, 640), (1333, 672), (1333, 704), (1333, 736),
                (1333, 768), (1333, 800)],
        keep_ratio=True),
    dict(type='RandomFlip', prob=0.0),
    dict(type='PackDetInputs')
]

test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='Resize', scale=(1333, 800), keep_ratio=True),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='PackDetInputs', meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape', 'scale_factor'))
]

# Dataloader 설정 (메모리 부족 시 batch_size 조절 필요)
train_dataloader = dict(
    batch_size=4, # R50은 R18보다 메모리를 많이 사용하므로 16에서 4~8 정도로 조정 권장
    num_workers=4,
    dataset=dict(
        type='CocoDataset',
        metainfo=metainfo,
        data_root=data_root,
        ann_file='TrainVal.json',
        data_prefix=dict(img=image_folder_train),
        pipeline=train_pipeline)
)

val_dataloader = dict(
    batch_size=1,
    dataset=dict(
        type='CocoDataset',
        metainfo=metainfo,
        data_root=data_root,
        ann_file='test.json',
        data_prefix=dict(img=image_folder_val),
        pipeline=test_pipeline)
)
test_dataloader = val_dataloader
# 3. EVALUATION & RUNTIME
val_evaluator = dict(type='CocoMetric', ann_file=data_root + 'test.json', metric='bbox')
test_evaluator = val_evaluator

# 학습 스케줄 및 루프 설정 (이 부분을 아래 3줄로 확실히 정의해야 합니다)
train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=11, val_interval=1)
val_cfg   = dict(type='ValLoop')   # <--- 추가: 검증 루프 활성화
test_cfg  = dict(type='TestLoop')  # <--- 추가: 테스트 루프 활성화

# 최적화 설정
optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(type='SGD', lr=0.01, momentum=0.9, weight_decay=0.0001))

# 파라미터 스케줄러 (학습률 조정 정책) 추가 권장
param_scheduler = [
    dict(type='LinearLR', start_factor=0.001, by_epoch=False, begin=0, end=500),
    dict(type='MultiStepLR', begin=0, end=100, by_epoch=True, milestones=[60, 80], gamma=0.1)
]

# 런타임 관련 필수 Hook 및 로그 설정
default_scope = 'mmdet'
default_hooks = dict(
    timer=dict(type='IterTimerHook'),
    logger=dict(type='LoggerHook', interval=50),
    param_scheduler=dict(type='ParamSchedulerHook'),
    checkpoint=dict(type='CheckpointHook', interval=1, max_keep_ckpts=3), # 매 에폭 저장
    sampler_seed=dict(type='DistSamplerSeedHook'),
    visualization=dict(type='DetVisualizationHook')
)

env_cfg = dict(
    cudnn_benchmark=False,
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0),
    dist_cfg=dict(backend='nccl'),
)

vis_backends = [dict(type='LocalVisBackend')]
visualizer = dict(type='DetLocalVisualizer', vis_backends=vis_backends, name='visualizer')
log_processor = dict(type='LogProcessor', window_size=50, by_epoch=True)
log_level = 'INFO'

# 가중치 및 경로
load_from = '/mnt/Documents/Dad/github/DUP/mmdetection_new/pre_trained_weights/faster_rcnn_r50_fpn_mstrain_3x_coco_20210524_110822-e10bd31c.pth'
work_dir = '/mnt/Documents/Dad/github/DUP/mmdetection_new/work_dirs/faster_rcnn_r50_TrainVal_epoch_11_seed'
resume = False