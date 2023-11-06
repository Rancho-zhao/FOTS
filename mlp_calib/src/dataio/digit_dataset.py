import glob
import numpy as np
import pandas as pd
import torch
import cv2
from PIL import Image
from torch.utils.data import Dataset


class DigitRealImageAnnotDataset(Dataset):
    def __init__( self, dir_dataset, annot_file, transform=None, annot_flag=True, img_type="png" ):
        self.dir_dataset = dir_dataset
        print(f"Loading dataset from {dir_dataset}")
        self.transform = transform
        self.annot_flag = annot_flag

        # a list of image paths sorted. dir_dataset is the root dir of the datasets (color)
        self.img_files = sorted(glob.glob(f"{self.dir_dataset}/*.{img_type}"))
        print(f"Found {len(self.img_files)} images")
        if self.annot_flag:
            self.annot_dataframe = pd.read_csv(annot_file, sep=",")

    def __getitem__(self, idx):
        """Returns a tuple of (img, annot) where annot is a tensor of shape (3,1)"""

        # read in image
        img0 = Image.open(self.img_files[idx])
        img0 = np.array(img0)
        img0 = cv2.cvtColor(img0,cv2.COLOR_BGR2RGB)
        cv2.imshow("img0",img0)
        cv2.waitKey(1)
        img_bg = Image.open("/home/r404/Digit_Test/digit-depth/scripts/0.png")
        img_bg = np.array(img_bg)
        img_bg = cv2.cvtColor(img_bg,cv2.COLOR_BGR2RGB)
        img = img0 - img_bg
        img = Image.fromarray(img)

        img = self.transform(img)
        img = img.permute(0, 2, 1)  # (3,240,320) -> (3,320,240)
        # read in region annotations
        if self.annot_flag:
            img_name = self.img_files[idx]
            print(self.annot_dataframe["img_names"])
            row_filter = self.annot_dataframe["img_names"] == img_name
            region_attr = self.annot_dataframe.loc[
                row_filter, ["center_x", "center_y", "radius"]
            ]
            annot = (torch.tensor(region_attr.values, dtype=torch.int32) if (len(region_attr) > 0) else torch.tensor([]))
        data = img
        if self.annot_flag:
            data = (img, annot)
        return data

    def __len__(self):
        return len(self.img_files)
