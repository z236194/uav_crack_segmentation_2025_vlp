# UAV Crack Segmentation — Deeplabv3+ (ResNet-50)

### 2025 VLP 挑战赛参赛作品

本项目为 2025 VLP 挑战赛 — 裂缝分割任务 提交作品。
采用 MMSegmentation 框架实现 Deeplabv3+（ResNet50 backbone），并在无人机路面裂缝数据集 UAV-Crack 上进行训练和评估。

## 亮点

- 基于 MMSegmentation 1.0 训练的可复现 Deeplabv3+ 模型
- 完整开源训练代码
- 包含辅助头 FCNHead，提升训练稳定性
- 提供 PyTorch .pth 权重文件 与 ONNX 导出脚本
- 完整评估脚本（支持 mIoU、Dice、IoU、Precision、Recall、F1）

-------------------------------------------------------

# 1. 环境配置

使用 requirements.txt：

pip install -r requirements.txt

或使用 conda（推荐）：

conda create -n mmseg python=3.10 -y
conda activate mmseg
pip install -r requirements.txt

-------------------------------------------------------

# 2. 数据集结构（dataset/）

确保数据目录结构如下：

dataset/
    images/
        train/*.jpg
        val/*.jpg
    masks/
        train/*.png
        val/*.png

所有图像尺寸均为 378 × 672（比赛官方格式）。

-------------------------------------------------------

# 3. 模型与配置

训练配置文件位于：

configs/uav_crack_deeplabv3plus.py

数据集定义位于：

mmseg/datasets/uav_crack_dataset.py

-------------------------------------------------------

# 4. 训练方式

python tools/train.py configs/uav_crack_deeplabv3plus.py

- batch size：2
- crop size：378×672
- 优化器与学习率可在配置文件中调整

-------------------------------------------------------

# 5. 推理（预测）

python tools/test.py configs/uav_crack_deeplabv3plus.py \
    work_dirs/uav_crack/latest.pth \
    --show-dir results/

-------------------------------------------------------

# 6. 模型评估（包含 mIoU）

使用仓库中的脚本：

python tools/eval_uav_miou.py \
    --pred_dir results/ \
    --gt_dir dataset/masks/val

输出包含：

- Accuracy
- Precision
- Recall
- F1
- IoU（Crack）
- Dice
- mIoU（双类平均）

-------------------------------------------------------

# 7. 模型权重（提交比赛用）

请将你的 .pth 和 ONNX 模型上传到网盘，将链接粘贴到此处。

- PyTorch 模型下载：
  [百度网盘链接]（待补充）

- ONNX 模型下载：
  [百度网盘链接]（待补充）

-------------------------------------------------------

# 8. 导出 ONNX

python tools/pytorch2onnx.py \
    configs/uav_crack_deeplabv3plus.py \
    work_dirs/uav_crack/latest.pth \
    --output-file deeplabv3plus_uav.onnx \
    --shape 378 672

-------------------------------------------------------

# License

本项目采用 MIT License  
详见仓库根目录的 LICENSE 文件。

-------------------------------------------------------

# Acknowledgement

本项目基于：

- MMSegmentation
- PyTorch
- 2025 VLP 挑战赛数据与规则

-------------------------------------------------------

# 2025 VLP 挑战赛参赛作品

本仓库为 2025 VLP Challenge 官方参赛作品，所有代码已开源并可复现。

