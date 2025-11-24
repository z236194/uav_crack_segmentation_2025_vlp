import os
import argparse
import mmcv
from mmseg.apis import init_model, inference_model

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--img_dir', required=True, help='dataset/images/val')
    parser.add_argument('--out_dir', required=True)
    return parser.parse_args()

def main():
    args = parse_args()

    model = init_model(args.config, args.checkpoint, device='cuda:0')
    os.makedirs(args.out_dir, exist_ok=True)

    img_files = sorted([f for f in os.listdir(args.img_dir)
                        if f.lower().endswith(('.png','.jpg','.jpeg'))])

    print(f"Total images: {len(img_files)}")

    for img_name in img_files:
        img_path = os.path.join(args.img_dir, img_name)

        result = inference_model(model, img_path)
        pred_mask = result.pred_sem_seg.data.cpu().numpy().astype('uint8')

        out_path = os.path.join(args.out_dir, img_name.replace('.jpg','.png').replace('.jpeg','.png'))

        mmcv.imwrite(pred_mask.squeeze()*1, out_path)            #255改为1

    print("Inference finished. Masks saved to:", args.out_dir)

if __name__ == '__main__':
    main()
