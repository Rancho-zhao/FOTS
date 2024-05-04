import cv2
import os
import imageio
import numpy as np
from scipy.ndimage import gaussian_filter
from scipy.ndimage import correlate
import scipy.ndimage as ndimage

import params as pr
from shadow_calib.planar_shadow import planar_shadow
from utils.prepost_mlp import preproc_mlp
from mlp_calib.src.train.mlp_model import MLP

w,h = pr.sensor_w, pr.sensor_h

def padding(img):
    # pad one row & one col on each side
    if len(img.shape) == 2:
        return np.pad(img, ((1, 1), (1, 1)), 'symmetric')
    elif len(img.shape) == 3:
        return np.pad(img, ((1, 1), (1, 1), (0, 0)), 'symmetric')


def generate_normals(height_map):
    [h, w] = height_map.shape
    center = height_map[1:h - 1, 1:w - 1]  # z(x,y)
    top = height_map[0:h - 2, 1:w - 1]  # z(x-1,y)
    bot = height_map[2:h, 1:w - 1]  # z(x+1,y)
    left = height_map[1:h - 1, 0:w - 2]  # z(x,y-1)
    right = height_map[1:h - 1, 2:w]  # z(x,y+1)
    dzdx = (bot - top) / 2.0
    dzdy = (right - left) / 2.0
    direction = np.ones((h - 2, w - 2, 3))
    direction[:, :, 0] = dzdy
    direction[:, :, 1] = -dzdx

    magnitude = np.sqrt(direction[:, :, 0] ** 2 + direction[:, :, 1] ** 2 + direction[:, :, 2] ** 2)
    normal = direction / magnitude[:, :, np.newaxis]  # unit norm

    normal = padding(normal)

    normal = (normal+1.0) * 0.5

    return normal


class CalibData:
    def __init__(self, data):
        data = data

        self.numBins = data['bins']
        self.grad_r = data['grad_r']
        self.grad_g = data['grad_g']
        self.grad_b = data['grad_b']


class FOTSRender:

    def __init__(self, **config):
        self.background = config['background_img']
        self.bg_render = config['bg_render']
        self.model = config['model']

    def smooth_heightMap(self, height_map, contact_mask):

        zq_back = height_map.copy()

        kernel_size = pr.pyramid_kernel_size
        for i in range(len(kernel_size)):
            height_map = cv2.GaussianBlur(height_map.astype(np.float32), (kernel_size[i], kernel_size[i]), 0)
            height_map[contact_mask] = zq_back[contact_mask]
        height_map = cv2.GaussianBlur(height_map.astype(np.float32), (pr.kernel_size, pr.kernel_size), 0)
        return height_map

    def generate(self, heightMap, contact_mask, shadow = False):

        heightMap = self.smooth_heightMap(heightMap, contact_mask)
        normal = generate_normals(heightMap)
        img_n = preproc_mlp(normal)
        self.model.eval()
        sim_img_r = self.model(img_n).cpu().detach().numpy()

        # write tactile image
        sim_img = sim_img_r.reshape(320,240,3)-self.bg_render
        # np.save("ini_bg_fots_gelsight3.npy", sim_img)
        sim_img *= (1*255.0)
        # sim_img = cv2.cvtColor(sim_img, cv2.COLOR_BGR2RGB)
        sim_img += self.background
        # cv2.imwrite("sim_img.jpg", sim_img.astype(np.uint8))
        if not shadow:
            # sim_img = cv2.GaussianBlur(sim_img.astype(np.float32),(pr.kernel_size,pr.kernel_size),0)
            sim_img[sim_img<0.0] = 0.0
            sim_img[sim_img>255.0] = 255.0
            return sim_img.astype(np.uint8)

        # light positions in pixel coordinate
        light_type = "spot"
        if light_type=="spot":
            light_r = [-40, -120, 130.0]
            light_g = [-40, 360, 130.0]
            light_b = [500, 120, 100.0]
        else:
            light_r = [-0.01, 0, 0.1]
            light_g = [0.01, 0, 0.1]
            light_b = [0, 0.5, 0.1]
        # generate shadow from rgb channel respectively
        shadow_g = 1 - (1-planar_shadow(light_g, heightMap, light_type)) * (1-contact_mask)
        shadow_b = 1 - (1-planar_shadow(light_b, heightMap, light_type)) * (1-contact_mask)
        shadow_r = 1 - (1-planar_shadow(light_r, heightMap, light_type)) * (1-contact_mask)

        shadow_sim_img = sim_img.copy()
        shadow_sim_img[:,:,0] *= np.clip(shadow_b+0.65,0,1)
        shadow_sim_img[:,:,1] *= np.clip(shadow_r+0.65,0,1)
        shadow_sim_img[:,:,2] *= np.clip(shadow_g+0.65,0,1)
        # # add shadow
        # shadow_sim_img = cv2.GaussianBlur(shadow_sim_img.astype(np.float32),(pr.kernel_size,pr.kernel_size),0)
        shadow_sim_img[sim_img<0.0] = 0.0
        shadow_sim_img[sim_img>255.0] = 255.0
        return shadow_sim_img.astype(np.uint8)
