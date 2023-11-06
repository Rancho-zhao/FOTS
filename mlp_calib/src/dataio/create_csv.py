import glob

import numpy as np
import pandas as pd
from PIL import Image
from sklearn.model_selection import train_test_split


def create_pixel_csv(img_dir: str, save_dir: str, img_type: str):
    """
    Creates a CSV file with the pixel coordinates of the image
    :param img_dir: image directory containing the images
    :param save_dir: directory for saving the CSV file per image
    :param img_type: color or normal
    :return: saves the CSV file per image0
    """
    imgs = sorted(glob.glob(f"{img_dir}/*.png"))
    print(f"Found {len(imgs)} {img_type} images")
    for idx, img_path in enumerate(imgs):
        img = Image.open(img_path)
        img_np = np.array(img)
        xy_coords = np.flip(
            np.column_stack(np.where(np.all(img_np >= 0, axis=2))), axis=1
        )
        rgb = np.reshape(img_np, (np.prod(img_np.shape[:2]), 3))
        # Add pixel numbers in front
        pixel_numbers = np.expand_dims(np.arange(1, xy_coords.shape[0] + 1), axis=1)
        value = np.hstack([pixel_numbers, xy_coords, rgb])
        # Properly save as CSV
        if img_type == "color":
            df = pd.DataFrame(value, columns=["pixel_number", "X", "Y", "R", "G", "B"])
            del df["pixel_number"]
            df.to_csv(f"{save_dir}/image_{idx}.csv", sep=",", index=False)
        elif img_type == "normal":
            df = pd.DataFrame(value, columns=["pixel_number", "X", "Y", "Nx", "Ny", "Nz"])
            del df["pixel_number"]
            df.to_csv(f"{save_dir}/image_{idx}.csv", sep=",", index=False)
        print(f"{img_type} image CSV written for image {idx}")


def combine_csv(csv_dir: str, img_type: str):
    """
    Combines all the CSV files in the directory into one CSV file for later use in training

    :param csv_dir: directory containing the CSV files
    :param img_type: color or normal
    """
    csv_files=(glob.glob(f"{csv_dir}/*.csv"))
    print(f"Found {len(csv_files)} {img_type} CSV files")
    df = pd.concat([pd.read_csv(f, sep=",") for f in (glob.glob(f"{csv_dir}/*.csv"))])
    df.to_csv(f"{csv_dir}/combined.csv", sep=",", index=False)
    print(f"Combined CSV written for {img_type} images")
    check_nans(f"{csv_dir}/combined.csv")
    print("----------------------------------------------------")


def create_train_test_csv(save_dir: str, normal_path: str, color_path: str):
    """
    Creates a CSV file with the pixel coordinates of the image. Samples 4% from zeros.
    Splits the cleaned dataset into train and test (80%/20%)
    :param save_dir: path for train_test_split folder
    :param normal_path: path to combined normal CSV file
    :param color_path: path to combined color CSV file
    """
    seed=42
    rgb_df = pd.read_csv(color_path, sep=",")
    normal_df = pd.read_csv(normal_path, sep=",")

    rgb_df['Nx'] = normal_df['Nx']
    rgb_df['Ny'] = normal_df['Ny']
    rgb_df['Nz'] = normal_df['Nz']

    zeros_df = rgb_df[(rgb_df['Nx'] == 127) & (rgb_df['Ny'] == 127) ^ (rgb_df['Nz'] == 127)]
    print(f"Found {len(zeros_df)} zeros in the dataset")
    non_zeros_df = rgb_df[(rgb_df['Nx'] != 127) | (rgb_df['Ny'] != 127) & (rgb_df['Nz'] != 127)]
    print(f"Found {len(non_zeros_df)} non-zeros in the dataset")
    zeros_df.to_csv(f"{save_dir}/zeros.csv", sep=",", index=False)
    non_zeros_df.to_csv(f"{save_dir}/non_zeros.csv", sep=",", index=False)
    print("----------------------------------------------------")
    # sampling 4% of zeros
    clean_df = zeros_df.sample(frac=0.04, random_state=seed)
    clean_df = pd.concat([clean_df, non_zeros_df])
    clean_df.to_csv(f"{save_dir}/clean-data.csv", sep=",", index=False)
    print(f"Clean data CSV of {len(clean_df)} written")

    # split into train and test
    train_df,test_df=train_test_split(clean_df,test_size=0.2,random_state=seed)
    train_df.to_csv(f"{save_dir}/train.csv", sep=",", index=False)
    test_df.to_csv(f"{save_dir}/test.csv", sep=",", index=False)
    print(f"Train set of size {len(train_df)} and Test set of size {len(test_df)} written")


def check_nans(csv_path: str):
    """
    Checks if there are any NaNs in the CSV file
    :param csv_path: path to the CSV file
    :return: True if there are no NaNs, False otherwise
    """
    df = pd.read_csv(csv_path, sep=",")
    if df.isnull().values.any():
        nans=df.isnull().sum().sum()
        print(f"Found {nans} NaNs in {csv_path}")
        print("Replacing NaNs with mean")
        df.fillna(df.mean(), inplace=True)
        df.to_csv(csv_path, sep=",", index=False)
        print(f"NaNs replaced in {csv_path}")
    else:
        print("No NaNs found.Perfect!")

