import sys
import cv2
import numpy as np
import torch
seed = 42
torch.seed = seed
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

from utils.mlp_render import MLPRender
from src.train.mlp_model import MLP

def get_simapproach():
    # load the image taken from a real digit sensor
    model_path      = './assets/gel/digit_bg.npy'
    background_img  = np.load(model_path)

    # FOTS render params
    bg_ini_depth = np.load('./utils/utils_data/ini_depth_extent.npy')
    ini_bg_mlp = np.load('./utils/utils_data/ini_bg_mlp.npy')
    model = MLP().to(device)
    model.load_state_dict(torch.load("./models/mlp_n2c_r.pth"))
    model.to(device)

    simulation = MLPRender(
        background_img      = background_img,
        bg_depth            = bg_ini_depth,
        bg_render           = ini_bg_mlp,
        model               = model,
    )

    return simulation