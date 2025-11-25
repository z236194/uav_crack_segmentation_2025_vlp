custom_imports = dict(
    imports=['mmseg.datasets.uav_crack_dataset'],
    allow_failed_imports=False
)

_base_ = [
    '../_base_/models/deeplabv3plus_r50-d8.py',
    '../_base_/datasets/uav_crack.py',
    '../_base_/default_runtime.py',
    '../_base_/schedules/schedule_40k.py'
]

crop_size = (378, 672)
data_preprocessor = dict(size=crop_size)


model = dict(
    data_preprocessor=data_preprocessor,


    decode_head=dict(
        c1_channels=96,

        num_classes=2,
        loss_decode=[
            dict(type='CrossEntropyLoss', loss_weight=1.0),
            dict(type='DiceLoss', loss_weight=3.0)
        ]
    ),

    auxiliary_head=dict(
        num_classes=2,
        loss_decode=dict(
            type='CrossEntropyLoss', loss_weight=0.4
        )
    )
)

train_dataloader = dict(batch_size=2, num_workers=2)


default_hooks = dict(
    checkpoint=dict(
        type='CheckpointHook',
        interval=1000,
        by_epoch=False,
        save_best='mIoU',
        max_keep_ckpts=40,
    )
)

log_config = dict(
    interval=100,
    hooks=[
        dict(type='TextLoggerHook'),
        dict(type='TensorboardLoggerHook')
    ]
)
