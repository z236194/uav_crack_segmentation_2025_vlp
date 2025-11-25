#!/usr/bin/env python3

import os
import argparse
from pathlib import Path
import numpy as np
import cv2
from tqdm import tqdm
import torch

from mmengine.config import Config
from mmengine.runner import Runner
from mmengine.registry import init_default_scope
from mmseg.structures import SegDataSample
from mmcv.transforms import LoadImageFromFile

def postprocess(prob, threshold=0.35):
    mask = (prob >= threshold).astype(np.uint8) * 255
    return mask


def build_model(cfg_path, ckpt_path, device='cuda:0'):
    cfg = Config.fromfile(cfg_path)


    if 'work_dir' not in cfg:
        cfg.work_dir = './tmp_infer'   # <<< ADD THIS LINE

    init_default_scope(cfg.default_scope)
    runner = Runner.from_cfg(cfg)
    runner.load_checkpoint(ckpt_path)

    model = runner.model.to(device)
    model.eval()
    return model, cfg



def infer_image(model, cfg, img_path):
    import cv2
    import torch
    from mmseg.structures import SegDataSample

    img = cv2.imread(img_path)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    H, W = img_rgb.shape[:2]

    # device compatible with mmseg 1.x
    device = next(model.parameters()).device

    #to tensor
    img_tensor = torch.from_numpy(img_rgb).permute(2, 0, 1).float()
    img_tensor = img_tensor.unsqueeze(0).to(device)

    # build SegDataSample
    data_sample = SegDataSample()
    data_sample.set_metainfo({
        'ori_shape': (H, W),
        'img_shape': (H, W),
        'pad_shape': (H, W),
        'scale_factor': 1.0,
        'flip': False,
        'flip_direction': None,
    })

    data = {
        'inputs': img_tensor,
        'data_samples': [data_sample]
    }

    # forward
    with torch.no_grad():
        pred = model.test_step(data)[0]

    # probability map
    seg_logits = pred._seg_logits.data  # shape: (2,H,W)
    prob = seg_logits.softmax(dim=0)[1].cpu().numpy()

    return prob


def process_folder(model, cfg, input_dir, out_dir, threshold):
    os.makedirs(out_dir, exist_ok=True)
    imgs = sorted([p for p in Path(input_dir).glob("*") 
                   if p.suffix.lower() in [".jpg",".png",".jpeg",".bmp"]])

    for img_path in tqdm(imgs, desc="Inference"):
        prob = infer_image(model, cfg, str(img_path))
        mask = postprocess(prob, threshold)
        
        # --- force png output ---
        out_name = img_path.stem + ".png"
        out_path = os.path.join(out_dir, out_name)

        cv2.imwrite(out_path, mask)
        #cv2.imwrite(os.path.join(out_dir, img_path.name), mask)

    print(f"Done. Results saved to: {out_dir}")



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--input-dir", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--threshold", type=float, default=0.35)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()

    model, cfg = build_model(args.config, args.checkpoint, args.device)
    process_folder(model, cfg, args.input_dir, args.out_dir, args.threshold)


if __name__ == "__main__":
    main()
