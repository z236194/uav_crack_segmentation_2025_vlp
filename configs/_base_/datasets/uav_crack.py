# _base_/datasets/uav_crack.py — 修复后的完整版本
dataset_type = 'UAVCrackDataset'
data_root = 'dataset/'

# 我们按 H, W 的形式定义 crop_size（更直观：height, width）
crop_hw = (378, 672)  # (H, W)

# mmseg/mmcv 的 Resize scale 一般按 (w, h) 解释，因此在 Resize 中传入 (W, H)
resize_scale = (crop_hw[1], crop_hw[0])  # (W, H) == (672, 378)

# --------------------------
# 训练集数据增强
# --------------------------
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations'),          # 加载掩码
    dict(type='Resize', scale=resize_scale, keep_ratio=False),
    dict(type='RandomFlip', prob=0.5),
    dict(type='PackSegInputs')
]

# --------------------------
# 验证 pipeline（必须加载 annotations）
# --------------------------
val_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations'),
    dict(type='Resize', scale=resize_scale, keep_ratio=False),
    dict(type='PackSegInputs')
]

# --------------------------
# 测试 pipeline（无标签，用于 inference）
# --------------------------
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='Resize', scale=resize_scale, keep_ratio=False),
    dict(type='PackSegInputs')
]

# --------------------------
# 训练 dataloader
# --------------------------
train_dataloader = dict(
    batch_size=2,
    num_workers=2,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        data_prefix=dict(
            img_path='images/train',
            seg_map_path='masks/train'
        ),
        pipeline=train_pipeline
    )
)

# --------------------------
# 验证 dataloader（用于计算 IoU / Fscore）
# --------------------------
val_dataloader = dict(
    batch_size=1,
    num_workers=2,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        data_prefix=dict(
            img_path='images/val',
            seg_map_path='masks/val'
        ),
        pipeline=val_pipeline
    )
)

# --------------------------
# 测试 dataloader（用于 final inference）——指向 images/test（无 masks）
# 如果没有 test 文件夹，改成 images/val 且 seg_map_path 指向 masks/val（仅用于 dev）
# --------------------------
test_dataloader = dict(
    batch_size=1,
    num_workers=2,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        # 推理时通常不提供 seg_map_path：
        data_prefix=dict(
            img_path='images/val',   # 若无 test，请改为 'images/val' 并添加 seg_map_path；若有，'images/test'
            seg_map_path='masks/val'
        ),
        pipeline=test_pipeline
    )
)

# --------------------------
# evaluator: 使用二分类的核心指标
# --------------------------
# IoUMetric 的配置：默认会计算每个类的 IoU；mFscore 会被尝试计算，注意 mmseg 版本差异
val_evaluator = dict(type='IoUMetric', iou_metrics=['mIoU', 'mFscore'])
test_evaluator = val_evaluator
