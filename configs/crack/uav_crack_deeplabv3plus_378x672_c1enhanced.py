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

# 关键：修改 decode_head 的 C1 分支容量
model = dict(
    data_preprocessor=data_preprocessor,

    # ---------- 只动 decode_head 这一段 ----------
    decode_head=dict(
        # 在 base 模型中，deeplabv3plus head 的 C1 默认是：
        # c1_in_channels = 256
        # c1_channels = 48        <-- 我们增强这个
        c1_channels=96,           # <<< 提升低层特征容量（48 → 96）

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

# ------ 完整保持你原本的 checkpoint hook 配置 ------
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
