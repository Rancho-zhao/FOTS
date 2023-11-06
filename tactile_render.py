import sys
import cv2
import numpy as np
import torch
seed = 42
torch.seed = seed
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

from utils.fots_render import FOTSRender
from mlp_calib.src.train.mlp_model import MLP

def get_simapproach():
    # load the image taken from a real digit sensor
    model_path      = './assets/gel/gelsight_bg.npy' #digit_bg.npy'
    background_img  = np.load(model_path)

    # FOTS render params
    # bg_ini_depth = np.load('./utils/utils_data/ini_depth_curve.npy')
    bg_ini_depth = np.load('./utils/utils_data/ini_depth_extent_gelsight.npy')
    ini_bg_mlp = np.load('./utils/utils_data/ini_bg_fots_gelsight.npy')
    # load mlp model
    model = MLP().to(device)
    model.load_state_dict(torch.load("./mlp_calib/models/mlp_n2c_gelsight.pth"))
    model.to(device)

    simulation = FOTSRender(
        background_img      = background_img,
        bg_depth            = bg_ini_depth,
        bg_render           = ini_bg_mlp,
        model               = model,
    )

    return simulation