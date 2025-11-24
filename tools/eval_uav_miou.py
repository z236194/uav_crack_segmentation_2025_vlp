# tools/eval_uav_miou.py
import os
import numpy as np
from glob import glob
from tqdm import tqdm
from sklearn.metrics import confusion_matrix

def compute_metrics(pred, gt):
    pred = pred.flatten()
    gt = gt.flatten()
    
    # 混淆矩阵
    tn, fp, fn, tp = confusion_matrix(gt, pred, labels=[0,1]).ravel()
    
    # 基本指标
    accuracy = (tp + tn) / (tp + tn + fp + fn)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    iou_crack = tp / (tp + fp + fn) if (tp + fp + fn) > 0 else 0.0
    dice_crack = 2*tp / (2*tp + fp + fn) if (2*tp + fp + fn) > 0 else 0.0

    # 计算背景类 IoU
    intersection_bg = np.logical_and(gt==0, pred==0).sum()
    union_bg = np.logical_or(gt==0, pred==0).sum()
    iou_bg = intersection_bg / union_bg if union_bg > 0 else 1.0

    # 双类平均 mIoU
    miou = (iou_bg + iou_crack) / 2

    return accuracy, precision, recall, f1, iou_crack, dice_crack, miou

def load_mask(path):
    mask = np.array(np.load(path) if path.endswith('.npy') else np.array(Image.open(path)))
    # 如果是 0/255，转成 0/1
    mask = (mask > 0).astype(np.uint8)
    return mask

def main(pred_dir, gt_dir):
    pred_files = sorted(glob(os.path.join(pred_dir, '*.png')))
    gt_files = sorted(glob(os.path.join(gt_dir, '*.png')))
    assert len(pred_files) == len(gt_files), "预测文件和真实掩码数量不一致"

    acc_list, prec_list, rec_list, f1_list, iou_list, dice_list, miou_list = [], [], [], [], [], [], []

    from PIL import Image
    for pred_path, gt_path in tqdm(zip(pred_files, gt_files), total=len(pred_files)):
        pred_mask = np.array(Image.open(pred_path))
        gt_mask = np.array(Image.open(gt_path))

        # 转成 0/1
        pred_mask = (pred_mask > 0).astype(np.uint8)
        gt_mask = (gt_mask > 0).astype(np.uint8)

        acc, prec, rec, f1, iou_crack, dice_crack, miou = compute_metrics(pred_mask, gt_mask)

        acc_list.append(acc)
        prec_list.append(prec)
        rec_list.append(rec)
        f1_list.append(f1)
        iou_list.append(iou_crack)
        dice_list.append(dice_crack)
        miou_list.append(miou)

    print("=== UAV Crack Evaluation (with mIoU) ===")
    print("Average Accuracy:", np.mean(acc_list))
    print("Average Precision:", np.mean(prec_list))
    print("Average Recall:", np.mean(rec_list))
    print("Average F1:", np.mean(f1_list))
    print("Average Crack IoU:", np.mean(iou_list))
    print("Average Dice:", np.mean(dice_list))
    print("Average mIoU:", np.mean(miou_list))

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--pred_dir', type=str, required=True, help='预测掩码目录')
    parser.add_argument('--gt_dir', type=str, required=True, help='真实掩码目录')
    args = parser.parse_args()

    main(args.pred_dir, args.gt_dir)
