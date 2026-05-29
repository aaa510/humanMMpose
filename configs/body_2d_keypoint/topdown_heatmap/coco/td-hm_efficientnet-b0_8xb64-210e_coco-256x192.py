_base_ = [
    '../_base_/default_runtime.py',
    '../_base_/datasets/coco.py',
    '../_base_/schedules/schedule_1x.py',
]

# model settings
model = dict(
    type='TopDown',
    pretrained=None,
    backbone=dict(
        type='EfficientNet',
        arch='b0',
        out_indices=(0, 1, 2, 3, 4),
        frozen_stages=-1,
        norm_cfg=dict(type='BN', requires_grad=True),
        norm_eval=True,
        with_cp=False),
    keypoint_head=dict(
        type='TopDownHeatmapSimpleHead',
        in_channels=1280,  # EfficientNet-B0 final output channels
        out_channels=17,
        loss_keypoint=dict(type='JointsMSELoss', use_target_weight=True)),
    train_cfg=dict(),
    test_cfg=dict()
)

# optimizer
optimizer = dict(type='AdamW', lr=0.001, weight_decay=0.0001)
optimizer_config = dict(grad_clip=None)

# learning policy
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=0.001,
    step=[170, 200])
total_epochs = 210

# data settings
data_cfg = dict(
    image_size=[256, 192],
    heatmap_size=[64, 48],
    num_output_channels=17,
    num_joints=17,
    dataset_channel=[
        [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16],
    ],
    inference_channel=[
        0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16
    ])

# train pipeline
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='TopDownGetBboxCenterScale', padding=1.25),
    dict(type='TopDownRandomShiftBboxCenter', shift_factor=0.16, prob=0.3),
    dict(type='TopDownRandomFlip', flip_prob=0.5),
    dict(
        type='TopDownHalfBodyTransform',
        num_joints_half_body=8,
        prob_half_body=0.3),
    dict(
        type='TopDownGetRandomScaleRotation',
        rot_factor=40,
        scale_factor=0.5,
        rot_prob=0.6),
    dict(type='TopDownAffine'),
    dict(type='ToTensor'),
    dict(
        type='NormalizeTensor',
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]),
    dict(type='TopDownGenerateTarget', sigma=2),
    dict(
        type='Collect',
        keys=['img', 'target', 'target_weight'],
        meta_keys=[
            'image_file', 'joints_3d', 'joints_3d_visible', 'center', 'scale',
            'rotation', 'bbox_score', 'flip_pairs'
        ]),
]

# test pipeline
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='TopDownGetBboxCenterScale', padding=1.25),
    dict(type='TopDownAffine'),
    dict(type='ToTensor'),
    dict(
        type='NormalizeTensor',
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]),
    dict(
        type='Collect',
        keys=['img'],
        meta_keys=[
            'image_file', 'center', 'scale', 'rotation', 'bbox_score',
            'flip_pairs'
        ]),
]

data = dict(
    samples_per_gpu=64,
    workers_per_gpu=4,
    train=dict(
        type='TopDownCocoDataset',
        ann_file='data/coco/annotations/person_keypoints_train2017.json',
        img_prefix='data/coco/train2017/',
        data_cfg=data_cfg,
        pipeline=train_pipeline),
    val=dict(
        type='TopDownCocoDataset',
        ann_file='data/coco/annotations/person_keypoints_val2017.json',
        img_prefix='data/coco/val2017/',
        data_cfg=data_cfg,
        pipeline=test_pipeline),
    test=dict(
        type='TopDownCocoDataset',
        ann_file='data/coco/annotations/person_keypoints_val2017.json',
        img_prefix='data/coco/val2017/',
        data_cfg=data_cfg,
        pipeline=test_pipeline),
) 