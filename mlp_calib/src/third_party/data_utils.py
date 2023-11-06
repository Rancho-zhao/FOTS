import numpy as np
import torch

"""
dataloader, logger helpers
"""


def pandas_col_to_numpy(df_col):

    df_col = df_col.apply(lambda x: np.fromstring(x.replace("\n", "").replace("[", "").replace("]", "").replace("  ", " "), sep=", "))
    df_col = np.stack(df_col)
    return df_col


def pandas_string_to_numpy(arr_str):
    arr_npy = np.fromstring(arr_str.replace("\n", "").replace("[", "").replace("]", "").replace("  ", " "),sep=", ")
    return arr_npy


def interpolate_img(img, rows, cols):
    """
    img: C x H x W
    """

    img = torch.nn.functional.interpolate(img, size=cols)
    img = torch.nn.functional.interpolate(img.permute(0, 2, 1), size=rows)
    img = img.permute(0, 2, 1)

    return img
