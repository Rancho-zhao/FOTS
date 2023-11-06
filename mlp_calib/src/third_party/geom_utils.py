# Copyright (c) Facebook, Inc. and its affiliates.

import copy

import numpy as np
import open3d as o3d
import torch
from scipy import ndimage

from src.third_party.poisson import poisson_reconstruct
from src.third_party.vis_utils import visualize_inlier_outlier

"""
Common functions
"""


def flip(x):
    return torch.flip(x, dims=[0])


def min_clip(x, min_val):
    return torch.max(x, min_val)


def max_clip(x, max_val):
    return torch.min(x, max_val)


def normalize(x, min_val, max_val):
    return (x - torch.min(x)) * (max_val - min_val) / (
        torch.max(x) - torch.min(x)
    ) + min_val


def mask_background(x, bg_mask, bg_val=0.0):
    if bg_mask is not None:
        x[bg_mask] = bg_val

    return x


def remove_background_pts(pts, bg_mask=None):
    if bg_mask is not None:
        fg_mask_pts = ~bg_mask.view(-1)
        points3d_x = pts[0, fg_mask_pts].view(1, -1)
        points3d_y = pts[1, fg_mask_pts].view(1, -1)
        points3d_z = pts[2, fg_mask_pts].view(1, -1)
        pts_fg = torch.cat((points3d_x, points3d_y, points3d_z), dim=0)
    else:
        pts_fg = pts

    return pts_fg


"""
3D transform helper functions
"""


def Rt_to_T(R=None, t=None, device=None):
    """
    :param R: rotation, (B, 3, 3) or (3, 3)
    :param t: translation, (B, 3) or  (3)
    :return: T, (B, 4, 4) or  (4, 4)
    """

    T = torch.eye(4, device=device)

    if (len(R.shape) > 2) & (len(t.shape) > 1):  # batch version
        B = R.shape[0]
        T = T.repeat(B, 1, 1)
        if R is not None:
            T[:, 0:3, 0:3] = R
        if t is not None:
            T[:, 0:3, -1] = t

    else:
        if R is not None:
            T[0:3, 0:3] = R
        if t is not None:
            T[0:3, -1] = t

    return T


def transform_pts3d(T, pts):
    """
    T: 4x4
    pts: 3xN

    returns 3xN
    """
    D, N = pts.shape

    if D == 3:
        pts_tf = (
            torch.cat((pts, torch.ones(1, N)), dim=0)
            if torch.is_tensor(pts)
            else np.concatenate((pts, torch.ones(1, N)), axis=0)
        )

    pts_tf = torch.matmul(T, pts_tf) if torch.is_tensor(pts) else np.matmul(T, pts_tf)
    pts_tf = pts_tf[0:3, :]

    return pts_tf


"""
3D-2D projection / conversion functions
OpenGL transforms reference: http://www.songho.ca/opengl/gl_transform.html
"""


def _vectorize_pixel_coords(rows, cols, device=None):
    y_range = torch.arange(rows, device=device)
    x_range = torch.arange(cols, device=device)
    grid_x, grid_y = torch.meshgrid(x_range, y_range)
    pixel_pos = torch.stack([grid_x.reshape(-1), grid_y.reshape(-1)], dim=0)  # 2 x N

    return pixel_pos


def _clip_to_pixel(clip_pos, img_shape, params):
    H, W = img_shape

    # clip -> ndc coords
    x_ndc = clip_pos[0, :] / clip_pos[3, :]
    y_ndc = clip_pos[1, :] / clip_pos[3, :]
    # z_ndc = clip_pos[2, :] / clip_pos[3, :]

    # ndc -> pixel coords
    x_pix = (W - 1) / 2 * (x_ndc + 1)  # [-1, 1] -> [0, W-1]
    y_pix = (H - 1) / 2 * (y_ndc + 1)  # [-1, 1] -> [0, H-1]
    # z_pix = (f-n) / 2 *  z_ndc + (f+n) / 2

    pixel_pos = torch.stack((x_pix, y_pix), dim=0)

    return pixel_pos


def _pixel_to_clip(pix_pos, depth_map, params):
    """
    :param pix_pos: position in pixel space, (2, N)
    :param depth_map: depth map, (H, W)
    :return: clip_pos position in clip space, (4, N)
    """

    x_pix = pix_pos[0, :]
    y_pix = pix_pos[1, :]

    H, W = depth_map.shape
    f = params.z_far
    n = params.z_near

    # pixel -> ndc coords
    x_ndc = 2 / (W - 1) * x_pix - 1  # [0, W-1] -> [-1, 1]
    y_ndc = 2 / (H - 1) * y_pix - 1  # [0, H-1] -> [-1, 1]
    z_buf = depth_map[y_pix, x_pix]

    # ndc -> clip coords
    z_eye = -z_buf
    w_c = -z_eye
    x_c = x_ndc * w_c
    y_c = y_ndc * w_c
    z_c = -(f + n) / (f - n) * z_eye - 2 * f * n / (f - n) * 1.0

    clip_pos = torch.stack([x_c, y_c, z_c, w_c], dim=0)

    return clip_pos


def _clip_to_eye(clip_pos, P):
    P_inv = torch.inverse(P)
    eye_pos = torch.matmul(P_inv, clip_pos)

    return eye_pos


def _eye_to_clip(eye_pos, P):
    clip_pos = torch.matmul(P, eye_pos)

    return clip_pos


def _eye_to_world(eye_pos, V):
    V_inv = torch.inverse(V)
    world_pos = torch.matmul(V_inv, eye_pos)

    world_pos = world_pos / world_pos[3, :]

    return world_pos


def _world_to_eye(world_pos, V):
    eye_pos = torch.matmul(V, world_pos)

    return eye_pos


def _world_to_object(world_pos, M):
    M_inv = torch.inverse(M)
    obj_pos = torch.matmul(M_inv, world_pos)

    obj_pos = obj_pos / obj_pos[3, :]

    return obj_pos


def _object_to_world(obj_pos, M):
    world_pos = torch.matmul(M, obj_pos)

    world_pos = world_pos / world_pos[3, :]

    return world_pos


def depth_to_pts3d(depth, P, V, params=None, ordered_pts=False):
    """
    :param depth: depth map, (C, H, W) or (H, W)
    :param P: projection matrix, (4, 4)
    :param V: view matrix, (4, 4)
    :return: world_pos position in 3d world coordinates, (3, H, W) or (3, N)
    """

    assert 2 <= len(depth.shape) <= 3
    assert P.shape == (4, 4)
    assert V.shape == (4, 4)

    depth_map = depth.squeeze(0) if (len(depth.shape) == 3) else depth
    H, W = depth_map.shape
    pixel_pos = _vectorize_pixel_coords(rows=H, cols=W)

    clip_pos = _pixel_to_clip(pixel_pos, depth_map, params)
    eye_pos = _clip_to_eye(clip_pos, P)
    world_pos = _eye_to_world(eye_pos, V)

    world_pos = world_pos[0:3, :] / world_pos[3, :]

    if ordered_pts:
        H, W = depth_map.shape
        world_pos = world_pos.reshape(world_pos.shape[0], H, W)

    return world_pos


"""
Optical flow functions
"""


def analytic_flow(img1, depth1, P, V1, V2, M1, M2, gel_depth, params):
    C, H, W = img1.shape
    depth_map = depth1.squeeze(0) if (len(depth1.shape) == 3) else depth1
    pixel_pos = _vectorize_pixel_coords(rows=H, cols=W, device=img1.device)

    clip_pos = _pixel_to_clip(pixel_pos, depth_map, params)
    eye_pos = _clip_to_eye(clip_pos, P)
    world_pos = _eye_to_world(eye_pos, V1)
    obj_pos = _world_to_object(world_pos, M1)

    world_pos = _object_to_world(obj_pos, M2)
    eye_pos = _world_to_eye(world_pos, V2)
    clip_pos = _eye_to_clip(eye_pos, P)
    pixel_pos_proj = _clip_to_pixel(clip_pos, (H, W), params)

    pixel_flow = pixel_pos - pixel_pos_proj
    flow_map = pixel_flow.reshape(pixel_flow.shape[0], H, W)

    # mask out background gel pixels
    mask_idxs = depth_map >= gel_depth
    flow_map[:, mask_idxs] = 0.0

    return flow_map


"""
Normal to depth conversion / integration functions
"""


def _preproc_depth(img_depth, bg_mask=None):
    if bg_mask is not None:
        img_depth = mask_background(img_depth, bg_mask=bg_mask, bg_val=0.0)

    return img_depth


def _preproc_normal(img_normal, bg_mask=None):
    """
    img_normal: lies in range [0, 1]
    """

    # 0.5 corresponds to 0
    img_normal = img_normal - 0.5

    # normalize
    img_normal = img_normal / torch.linalg.norm(img_normal, dim=0)

    # set background to have only z normals (flat, facing camera)
    if bg_mask is not None:
        img_normal[0:2, bg_mask] = 0.0
        img_normal[2, bg_mask] = 1.0

    return img_normal


def _depth_to_grad_depth(img_depth, bg_mask=None):
    gradx = ndimage.sobel(img_depth.cpu().detach().numpy(), axis=1, mode="constant")
    grady = ndimage.sobel(img_depth.cpu().detach().numpy(), axis=0, mode="constant")

    gradx = torch.FloatTensor(gradx, device=img_depth.device)
    grady = torch.FloatTensor(grady, device=img_depth.device)

    if bg_mask is not None:
        gradx = mask_background(gradx, bg_mask=bg_mask, bg_val=0.0)
        grady = mask_background(grady, bg_mask=bg_mask, bg_val=0.0)

    return gradx, grady


def _normal_to_grad_depth(img_normal, gel_width=1.0, gel_height=1.0, bg_mask=None):
    # Ref: https://stackoverflow.com/questions/34644101/calculate-surface-normals-from-depth-image-using-neighboring-pixels-cross-produc/34644939#34644939

    EPS = 1e-1
    nz = torch.max(torch.tensor(EPS), img_normal[2, :])

    dzdx = -(img_normal[0, :] / nz).squeeze()
    dzdy = -(img_normal[1, :] / nz).squeeze()

    # taking out negative sign as we are computing gradient of depth not z
    # since z is pointed towards sensor, increase in z corresponds to decrease in depth
    # i.e., dz/dx = -ddepth/dx
    ddepthdx = -dzdx
    ddepthdy = -dzdy

    # sim: pixel axis v points in opposite dxn of camera axis y
    ddepthdu = ddepthdx
    ddepthdv = -ddepthdy

    gradx = ddepthdu  # cols
    grady = ddepthdv  # rows

    # convert units from pixel to meters
    C, H, W = img_normal.shape
    gradx = gradx * (gel_width / W)
    grady = grady * (gel_height / H)

    if bg_mask is not None:
        gradx = mask_background(gradx, bg_mask=bg_mask, bg_val=0.0)
        grady = mask_background(grady, bg_mask=bg_mask, bg_val=0.0)

    return gradx, grady


def _integrate_grad_depth(gradx, grady, boundary=None, bg_mask=None, max_depth=0.0):
    if boundary is None:
        boundary = torch.zeros((gradx.shape[0], gradx.shape[1]))

    img_depth_recon = poisson_reconstruct(
        grady.cpu().detach().numpy(),
        gradx.cpu().detach().numpy(),
        boundary.cpu().detach().numpy(),
    )
    img_depth_recon = torch.tensor(
        img_depth_recon, device=gradx.device, dtype=torch.float32
    )

    if bg_mask is not None:
        img_depth_recon = mask_background(img_depth_recon, bg_mask)

    # after integration, img_depth_recon lies between 0. (bdry) and a -ve val (obj depth)
    # rescale to make max depth as gel depth and obj depth as +ve values
    img_depth_recon = max_clip(img_depth_recon, max_val=torch.tensor(0.0)) + max_depth

    return img_depth_recon


def depth_to_depth(img_depth, bg_mask=None, boundary=None, params=None):
    # preproc depth img
    img_depth = _preproc_depth(img_depth=img_depth.squeeze(), bg_mask=bg_mask)

    # get grad depth
    gradx, grady = _depth_to_grad_depth(img_depth=img_depth.squeeze(), bg_mask=bg_mask)

    # integrate grad depth
    img_depth_recon = _integrate_grad_depth(
        gradx, grady, boundary=boundary, bg_mask=bg_mask
    )

    return img_depth_recon


def normal_to_depth(
    img_normal,
    bg_mask=None,
    boundary=None,
    gel_width=0.02,
    gel_height=0.03,
    max_depth=0.02,
):
    # preproc normal img
    img_normal = _preproc_normal(img_normal=img_normal, bg_mask=bg_mask)

    # get grad depth
    gradx, grady = _normal_to_grad_depth(
        img_normal=img_normal,
        gel_width=gel_width,
        gel_height=gel_height,
        bg_mask=bg_mask,
    )

    # integrate grad depth
    img_depth_recon = _integrate_grad_depth(
        gradx, grady, boundary=boundary, bg_mask=bg_mask, max_depth=max_depth
    )

    return img_depth_recon


"""
3D registration functions
"""


def _fpfh(pcd, normals):
    pcd.normals = o3d.utility.Vector3dVector(normals)
    pcd_fpfh = o3d.pipelines.registration.compute_fpfh_feature(
        pcd, o3d.geometry.KDTreeSearchParamHybrid(radius=0.3, max_nn=64)
    )
    return pcd_fpfh


def _fast_global_registration(source, target, source_fpfh, target_fpfh):
    distance_threshold = 0.01
    result = o3d.pipelines.registration.registration_fast_based_on_feature_matching(
        source,
        target,
        source_fpfh,
        target_fpfh,
        o3d.pipelines.registration.FastGlobalRegistrationOption(
            maximum_correspondence_distance=distance_threshold
        ),
    )

    transformation = result.transformation
    metrics = [result.fitness, result.inlier_rmse, result.correspondence_set]

    return transformation, metrics


def fgr(source, target, src_normals, tgt_normals):
    source_fpfh = _fpfh(source, src_normals)
    target_fpfh = _fpfh(target, tgt_normals)
    transformation, metrics = _fast_global_registration(
        source=source, target=target, source_fpfh=source_fpfh, target_fpfh=target_fpfh
    )
    return transformation, metrics


def icp(source, target, T_init=np.eye(4), mcd=0.1, max_iter=50, type="point_to_plane"):
    if type == "point_to_point":
        result = o3d.pipelines.registration.registration_icp(
            source=source,
            target=target,
            max_correspondence_distance=mcd,
            init=T_init,
            estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPoint(),
            criteria=o3d.pipelines.registration.ICPConvergenceCriteria(
                max_iteration=max_iter
            ),
        )
    else:
        result = o3d.pipelines.registration.registration_icp(
            source=source,
            target=target,
            max_correspondence_distance=mcd,
            init=T_init,
            estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPlane(),
            criteria=o3d.pipelines.registration.ICPConvergenceCriteria(
                max_iteration=max_iter
            ),
        )

    transformation = result.transformation
    metrics = [result.fitness, result.inlier_rmse, result.correspondence_set]

    return transformation, metrics


"""
Open3D helper functions
"""


def remove_outlier_pts(points3d, nb_neighbors=20, std_ratio=10.0, vis=False):
    points3d_np = (
        points3d.cpu().detach().numpy() if torch.is_tensor(points3d) else points3d
    )

    cloud = o3d.geometry.PointCloud()
    cloud.points = copy.deepcopy(o3d.utility.Vector3dVector(points3d_np.transpose()))
    cloud_filt, ind_filt = cloud.remove_statistical_outlier(
        nb_neighbors=nb_neighbors, std_ratio=std_ratio
    )

    if vis:
        visualize_inlier_outlier(cloud, ind_filt)

    points3d_filt = np.asarray(cloud_filt.points).transpose()
    points3d_filt = (
        torch.tensor(points3d_filt) if torch.is_tensor(points3d) else points3d_filt
    )

    return points3d_filt


def init_points_to_clouds(clouds, points3d, colors=None):
    for idx, cloud in enumerate(clouds):
        points3d_np = (
            points3d[idx].cpu().detach().numpy()
            if torch.is_tensor(points3d[idx])
            else points3d[idx]
        )
        cloud.points = copy.deepcopy(
            o3d.utility.Vector3dVector(points3d_np.transpose())
        )
        if colors is not None:
            cloud.paint_uniform_color(colors[idx])

    return clouds
