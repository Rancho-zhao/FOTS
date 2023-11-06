
import cv2
import torch
import numpy as np
import pandas as pd
import copy
seed = 42
torch.seed = seed
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def preproc_mlp(normal) -> torch.Tensor:
    """ Preprocess image for input to model.

    Args: image: OpenCV image in BGR format
    Return: tensor of shape (R*C,5) where R=320 and C=240 for DIGIT images
    5-columns are: X,Y,R,G,B

    """
    xy_coords = np.flip(np.column_stack(np.where(np.all(normal>=-100, axis=2))), axis=1)
    nxyz = np.reshape(normal, (np.prod(normal.shape[:2]), 3))
    # pixel_numbers = np.expand_dims(np.arange(1, xy_coords.shape[0] + 1), axis=1)
    value_base = np.hstack([ xy_coords, nxyz])
    df_base = pd.DataFrame(value_base, columns=['X', 'Y', 'Nx', 'Ny', 'Nz'])
    df_base['X'] = df_base['X'] / 240
    df_base['Y'] = df_base['Y'] / 320
    # df_base.to_csv("data.csv")
    # del df_base['pixel_number']
    test_tensor = torch.tensor(df_base[['X', 'Y', 'Nx', 'Ny', 'Nz']].values, dtype=torch.float32).to(device)
    return test_tensor


def post_proc_mlp(model_output: torch.Tensor):
    """ Postprocess model output to get normal map.

    Args: model_output: torch.Tensor of shape (1,3)
    Return: two torch.Tensor of shape (1,3)

    """
    test_np = model_output.reshape(320, 240, 3)
    normal = copy.deepcopy(test_np)  # surface normal image
    test_np = torch.tensor(test_np,
                           dtype=torch.float32)  # convert to torch tensor for later processing in gradient computation
    test_np = test_np.permute(2, 0, 1)  # swap axes to (3,320,240)
    test_np = test_np # convert to uint8 for visualization
    return test_np, normal


