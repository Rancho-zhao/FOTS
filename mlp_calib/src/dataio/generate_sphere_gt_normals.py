"""
  Script for generating sphere ground truth normal images.
"""
import math

import numpy as np


def generate_sphere_gt_normals(img_mask, center_x, center_y, radius):
    """
    Generates sphere ground truth normal images for an image.
    Args:
      img_mask: a numpy array of shape [H, W, 3]
      center_x: x coordinate of the center of the sphere
      center_y: y coordinate of the center of the sphere
      radius: the radius of the sphere
    Returns:
      img_normal: a numpy array of shape [H, W, 3]
    """
    img_normal = np.zeros(img_mask.shape, dtype="float64")

    for y in range(img_mask.shape[0]):
        for x in range(img_mask.shape[1]):

            img_normal[y, x, 0] = 0.0
            img_normal[y, x, 1] = 0.0
            img_normal[y, x, 2] = 1.0

            if np.sum(img_mask[y, x, :]) > 0:
                dist = np.sqrt((x - center_x) ** 2 + (y - center_y) ** 2)
                ang_xz = math.acos(dist / radius)
                ang_xy = math.atan2(y - center_y, x - center_x)

                nx = math.cos(ang_xz) * math.cos(ang_xy)
                ny = math.cos(ang_xz) * math.sin(ang_xy)
                nz = math.sin(ang_xz)

                img_normal[y, x, 0] = nx
                img_normal[y, x, 1] = -ny
                img_normal[y, x, 2] = nz

            norm_val = np.linalg.norm(img_normal[y, x, :])
            img_normal[y, x, :] = img_normal[y, x, :] / norm_val

    # img_normal between [-1., 1.], converting to [0., 1.]
    img_normal = (img_normal + 1.0) * 0.5
    return img_normal
