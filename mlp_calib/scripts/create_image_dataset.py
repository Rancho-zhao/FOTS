"""
Creates color+normal datasets based on annotation file.
The datasets can be used to train MLP, CycleGAN, Pix2Pix models.
"""
import os

import cv2
import hydra
import imageio
from PIL import Image
import numpy as np
import torch
import sys
sys.path.append('/home/r404/mujoco_pybinding/FOTS/mlp_calib/src')
from src.dataio.create_csv import (combine_csv, create_pixel_csv,
                                   create_train_test_csv)
from src.dataio.data_loader import data_loader
from src.dataio.generate_sphere_gt_normals import generate_sphere_gt_normals
from src.third_party import data_utils

base_path = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))

@hydra.main(config_path=f"{base_path}/config", config_name="normal_to_rgb.yaml", version_base=None)
def main(cfg):
    normal_dataloader, normal_dataset = data_loader(
        dir_dataset=os.path.join(base_path, "gelsight"), params=cfg.dataloader
    )
    dirs = [
        f"{base_path}/datasets/A/imgs",
        f"{base_path}/datasets/B/imgs",
        f"{base_path}/datasets/A/csv",
        f"{base_path}/datasets/B/csv",
        f"{base_path}/datasets/train_test_split",
    ]
    for dir in dirs:
        print(f"Creating directory: {dir}")
        os.makedirs(f"{dir}", exist_ok=True)
    # iterate over images
    img_idx = 0
    radius_bearing = np.int32(0.5 * 4.7 * cfg.mm_to_pixel) #6.0

    while img_idx < len(normal_dataset):
        # read img + annotations
        data = normal_dataset[img_idx]
        if cfg.dataloader.annot_flag:
            img, annot = data
            if annot.shape[0] == 0:
                img_idx = img_idx + 1
                continue
        else:
            img = data

        # get annotation circle params
        if cfg.dataloader.annot_flag:
            annot_np = annot.cpu().detach().numpy()
            center_y, center_x, radius_annot = (
                annot_np[0][1],
                annot_np[0][0],
                annot_np[0][2],
            )
        else:
            center_y, center_x, radius_annot = 0, 0, 0

        img_color_np = (img.permute(2,1,0).cpu().detach().numpy()) # (3,320,240) -> (240,320,3)


        # apply foreground mask
        fg_mask = np.zeros(img_color_np.shape[:2], dtype="uint8")
        fg_mask = cv2.circle(fg_mask, (center_x, center_y), radius_annot, 255, -1)

        # 1. rgb -> normal (generate gt surface normals)
        img_mask = cv2.bitwise_and(img_color_np, img_color_np, mask=fg_mask)
        img_normal_np = generate_sphere_gt_normals(
            img_mask, center_x, center_y, radius=radius_bearing
        )

        # 2. downsample and convert to NumPy: (320,240,3) -> (160,120,3)
        img_normal_np = data_utils.interpolate_img(
            img=torch.tensor(img_normal_np).permute(2, 0, 1), rows=160, cols=120)
        img_normal_np = img_normal_np.permute(1, 2, 0).cpu().detach().numpy()
        img_color_ds = data_utils.interpolate_img(
            img=torch.tensor(img_color_np).permute(2, 0, 1), rows=160, cols=120)
        img_color_np = img_color_ds.permute(1, 2, 0).cpu().detach().numpy()

        # 3. save csv files for color and normal images

        if cfg.dataset.save_dataset:
            imageio.imwrite(
                f"{dirs[0]}/{img_idx:04d}.png", (img_color_np * 255).astype(np.uint8)
            )
            imageio.imwrite(f"{dirs[1]}/{img_idx:04d}.png", (img_normal_np*255).astype(np.uint8))
            print(f"Saved image {img_idx:04d}")
        img_idx += 1

    # post-process CSV files and create train/test split
    create_pixel_csv( img_dir=dirs[0], save_dir=dirs[2], img_type="color")
    create_pixel_csv(img_dir=dirs[1], save_dir=dirs[3], img_type="normal")
    combine_csv(dirs[2],img_type="color")
    combine_csv(dirs[3],img_type="normal")
    create_train_test_csv(color_path=f'{dirs[2]}/combined.csv',
                          normal_path=f'{dirs[3]}/combined.csv',
                          save_dir=dirs[4])


if __name__ == "__main__":
    main()
