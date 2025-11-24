import os
import shutil
import random
from tqdm import tqdm


DATASETS = [
    ("data/uav_crack/leftImg8bit/train/UAV-CrackX4", "data/uav_crack/gtFine/train/UAV-CrackX4"),
    ("data/uav_crack/leftImg8bit/train/UAV-CrackX8", "data/uav_crack/gtFine/train/UAV-CrackX8"),
    ("data/uav_crack/leftImg8bit/train/UAV-CrackX16", "data/uav_crack/gtFine/train/UAV-CrackX16"),
]


def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)


def collect_all_pairs():
    """
    扫描三种 scale，把所有图像与掩码匹配的样本整合在一起。
    返回一个包含 (img_path, mask_path) 的列表。
    """
    pairs = []

    for img_dir, mask_dir in DATASETS:
        img_files = sorted(os.listdir(img_dir))

        for img_name in img_files:
            basename = os.path.splitext(img_name)[0]
            img_path = os.path.join(img_dir, img_name)
            mask_path = os.path.join(mask_dir, basename + ".png")

            if os.path.exists(mask_path):
                pairs.append((img_path, mask_path))

    return pairs


def split_dataset(val_ratio=0.1, output_dir="dataset"):
    all_pairs = collect_all_pairs()

    print(f"总共找到 {len(all_pairs)} 对图像+掩码")

    # 打乱
    random.shuffle(all_pairs)

    # 划分
    val_count = int(len(all_pairs) * val_ratio)
    val_set = all_pairs[:val_count]
    train_set = all_pairs[val_count:]

    print(f"Train: {len(train_set)}")
    print(f"Val:   {len(val_set)}")

    # 创建输出目录
    train_img_out = os.path.join(output_dir, "images/train")
    train_mask_out = os.path.join(output_dir, "masks/train")
    val_img_out = os.path.join(output_dir, "images/val")
    val_mask_out = os.path.join(output_dir, "masks/val")

    for d in [train_img_out, train_mask_out, val_img_out, val_mask_out]:
        ensure_dir(d)

    # 拷贝 train
    print("\n拷贝 Train...")
    for img_path, mask_path in tqdm(train_set):
        shutil.copy(img_path, train_img_out)
        shutil.copy(mask_path, train_mask_out)

    # 拷贝 val
    print("\n拷贝 Val...")
    for img_path, mask_path in tqdm(val_set):
        shutil.copy(img_path, val_img_out)
        shutil.copy(mask_path, val_mask_out)

    print("\n数据划分完成！输出位置：", output_dir)


if __name__ == "__main__":
    split_dataset(val_ratio=0.1, output_dir="dataset")
