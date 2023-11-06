# Copyright (c) Facebook, Inc. and its affiliates.

import copy
import logging
import time
import types

import cv2
import matplotlib.pyplot as plt
import numpy as np
import open3d as o3d
import torch
from attrdict import AttrDict
from matplotlib.patches import Circle, Rectangle

log = logging.getLogger(__name__)

"""
Open3d visualization functions
"""


class Visualizer3d:
    def __init__(self, base_path="", view_params=None, tsleep=0.01):
        self.vis = o3d.visualization.VisualizerWithKeyCallback()
        self.vis.create_window(top=0, left=750, width=1080, height=1080)
        self.tsleep = tsleep

        self.base_path = base_path

        self.view_params = AttrDict(
            {
                "fov": 0,
                "front": [0.0, 0.0, 0.0],
                "lookat": [0.0, 0.0, 0.0],
                "up": [0.0, 0.0, 0.0],
                "zoom": 0.5,
            }
        )
        if view_params is not None:
            self.view_params = view_params

        self._init_options()

    def _init_options(self):
        opt = self.vis.get_render_option()
        opt.show_coordinate_frame = True
        opt.background_color = np.asarray([0.6, 0.6, 0.6])

        # pause option
        self.paused = types.SimpleNamespace()
        self.paused.value = False
        self.vis.register_key_action_callback(
            ord("P"),
            lambda a, b, c: b == 1
            or setattr(self.paused, "value", not self.paused.value),
        )

    def _init_geom_cloud(self):
        return o3d.geometry.PointCloud()

    def _init_geom_frame(self, frame_size=0.01, frame_origin=[0, 0, 0]):
        return o3d.geometry.TriangleMesh.create_coordinate_frame(
            size=frame_size, origin=frame_origin
        )

    def _init_geom_mesh(self, mesh_name, color=None, wireframe=False):
        mesh = o3d.io.read_triangle_mesh(
            f"{self.base_path}/local/resources/meshes/{mesh_name}"
        )
        mesh.compute_vertex_normals()

        if wireframe:
            mesh = o3d.geometry.LineSet.create_from_triangle_mesh(mesh)
        if color is not None:
            mesh.paint_uniform_color(color)

        return mesh

    def init_geometry(
        self,
        geom_type,
        num_items=1,
        sizes=None,
        file_names=None,
        colors=None,
        wireframes=None,
    ):
        geom_list = []
        for i in range(0, num_items):

            if geom_type == "cloud":
                geom = self._init_geom_cloud()
            elif geom_type == "frame":
                frame_size = sizes[i] if sizes is not None else 0.001
                geom = self._init_geom_frame(frame_size=frame_size)
            elif geom_type == "mesh":
                color = colors[i] if colors is not None else None
                wireframe = wireframes[i] if wireframes is not None else False
                geom = self._init_geom_mesh(file_names[i], color, wireframe)
            else:
                log.error(
                    f"[Visualizer3d::init_geometry] geom_type {geom_type} not found."
                )

            geom_list.append(geom)

        return geom_list

    def add_geometry(self, geom_list):

        if geom_list is None:
            return

        for geom in geom_list:
            self.vis.add_geometry(geom)

    def remove_geometry(self, geom_list, reset_bounding_box=False):

        if geom_list is None:
            return

        for geom in geom_list:
            self.vis.remove_geometry(geom, reset_bounding_box=reset_bounding_box)

    def update_geometry(self, geom_list):
        for geom in geom_list:
            self.vis.update_geometry(geom)

    def set_view(self):
        ctr = self.vis.get_view_control()
        ctr.change_field_of_view(self.view_params.fov)
        ctr.set_front(self.view_params.front)
        ctr.set_lookat(self.view_params.lookat)
        ctr.set_up(self.view_params.up)
        ctr.set_zoom(self.view_params.zoom)

    def set_view_cam(self, T):
        ctr = self.vis.get_view_control()
        cam = ctr.convert_to_pinhole_camera_parameters()
        cam.extrinsic = T
        ctr.convert_from_pinhole_camera_parameters(cam)

    def set_zoom(self):
        ctr = self.vis.get_view_control()
        ctr.set_zoom(1.5)

    def rotate_view(self):
        ctr = self.vis.get_view_control()
        ctr.rotate(10.0, -0.0)

    def pan_scene(self, max=300):
        for i in range(0, max):
            self.rotate_view()
            self.render()

    def render(self, T=None):

        if T is not None:
            self.set_view_cam(T)
        else:
            self.set_view()

        self.vis.poll_events()
        self.vis.update_renderer()
        time.sleep(self.tsleep)

    def transform_geometry_absolute(self, transform_list, geom_list):
        for idx, geom in enumerate(geom_list):
            T = transform_list[idx]
            geom.transform(T)

    def transform_geometry_relative(
        self, transform_prev_list, transform_curr_list, geom_list
    ):
        for idx, geom in enumerate(geom_list):
            T_prev = transform_prev_list[idx]
            T_curr = transform_curr_list[idx]

            # a. rotate R1^{-1}*R2 about center t1
            geom.rotate(
                torch.matmul(torch.inverse(T_prev[0:3, 0:3]), T_curr[0:3, 0:3]),
                center=(T_prev[0, -1], T_prev[1, -1], T_prev[2, -1]),
            )

            # b. translate by t2 - t1
            geom.translate(
                (
                    T_curr[0, -1] - T_prev[0, -1],
                    T_curr[1, -1] - T_prev[1, -1],
                    T_curr[2, -1] - T_prev[2, -1],
                )
            )

    def clear_geometries(self):
        self.vis.clear_geometries()

    def destroy(self):
        self.vis.destroy_window()


def visualize_registration(source, target, transformation, vis3d=None, colors=None):
    source_copy = copy.deepcopy(source)
    target_copy = copy.deepcopy(target)

    source_copy.transform(transformation)

    clouds = [target_copy, source_copy]

    if colors is not None:
        clouds[0].paint_uniform_color(colors[1])
        clouds[1].paint_uniform_color(colors[0])

    vis3d.add_geometry(clouds)
    vis3d.render()
    vis3d.remove_geometry(clouds)


def visualize_geometries_o3d(
    vis3d, clouds=None, frames=None, meshes=None, transforms=None
):
    if meshes is not None:
        meshes = [copy.deepcopy(mesh) for mesh in meshes]
        if transforms is not None:
            vis3d.transform_geometry_absolute(transforms, meshes)

    if frames is not None:
        frames = [copy.deepcopy(frame) for frame in frames]
        if transforms is not None:
            vis3d.transform_geometry_absolute(transforms, frames)

    if clouds is not None:
        vis3d.add_geometry(clouds)
    if meshes is not None:
        vis3d.add_geometry(meshes)
    if frames is not None:
        vis3d.add_geometry(frames)

    vis3d.render()

    if clouds is not None:
        vis3d.remove_geometry(clouds)
    if meshes is not None:
        vis3d.remove_geometry(meshes)
    if frames is not None:
        vis3d.remove_geometry(frames)


def visualize_inlier_outlier(cloud, ind):
    inlier_cloud = cloud.select_by_index(ind)
    outlier_cloud = cloud.select_by_index(ind, invert=True)

    print("Showing outliers (red) and inliers (gray): ")
    outlier_cloud.paint_uniform_color([1, 0, 0])
    inlier_cloud.paint_uniform_color([0.8, 0.8, 0.8])
    o3d.visualization.draw_geometries(
        [inlier_cloud, outlier_cloud],
        zoom=0.3412,
        front=[0.4257, -0.2125, -0.8795],
        lookat=[2.6172, 2.0475, 1.532],
        up=[-0.0694, -0.9768, 0.2024],
    )


"""
Optical flow visualization functions
"""


def flow_to_color(flow_uv, cvt=cv2.COLOR_HSV2BGR):
    hsv = np.zeros((flow_uv.shape[0], flow_uv.shape[1], 3), dtype=np.uint8)
    hsv[:, :, 0] = 255
    hsv[:, :, 1] = 255
    mag, ang = cv2.cartToPolar(flow_uv[..., 0], flow_uv[..., 1])
    hsv[..., 0] = ang * 180 / np.pi / 2
    hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
    flow_color = cv2.cvtColor(hsv, cvt)

    return flow_color


def flow_to_arrows(img, flow, step=8):
    img = copy.deepcopy(img)
    # img = (255 * img).astype(np.uint8)

    h, w = img.shape[:2]
    y, x = np.mgrid[step / 2 : h : step, step / 2 : w : step].reshape(2, -1).astype(int)
    fx, fy = 5.0 * flow[y, x].T
    lines = np.vstack([x, y, x + fx, y + fy]).T.reshape(-1, 2, 2)
    lines = np.int32(lines + 0.5)
    cv2.polylines(img, lines, 0, color=(0, 255, 0), thickness=1)
    for (x1, y1), (x2, y2) in lines:
        cv2.circle(img, (x1, y1), 1, (0, 255, 0), -1)

    return img


def depth_to_color(depth):
    gray = (
        np.clip((depth - depth.min()) / (depth.max() - depth.min()), 0, 1) * 255
    ).astype(np.uint8)
    return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)


def visualize_flow_cv2(
    img1, img2, flow_arrow=None, flow_color=None, win_size=(360, 360)
):
    img_disp1 = np.concatenate([img1, img2], axis=1)
    cv2.namedWindow("img1, img2", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("img1, img2", 2 * win_size[0], win_size[1])
    cv2.imshow("img1, img2", img_disp1)

    if flow_arrow is not None:
        cv2.namedWindow("flow_arrow", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("flow_arrow", win_size[0], win_size[1])
        cv2.imshow("flow_arrow", flow_arrow)

    if flow_color is not None:
        cv2.namedWindow("flow_color", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("flow_color", win_size[0], win_size[1])
        cv2.imshow("flow_color", flow_color)

    cv2.waitKey(300)


"""
General visualization functions
"""


def draw_rectangle(
    center_x,
    center_y,
    size_x,
    size_y,
    ang=0.0,
    edgecolor="dimgray",
    facecolor=None,
    linewidth=2,
):
    R = np.array([[np.cos(ang), -np.sin(ang)], [np.sin(ang), np.cos(ang)]])
    offset = np.matmul(R, np.array([[0.5 * size_x], [0.5 * size_y]]))
    anchor_x = center_x - offset[0]
    anchor_y = center_y - offset[1]
    rect = Rectangle(
        (anchor_x, anchor_y),
        size_x,
        size_y,
        angle=(np.rad2deg(ang)),
        facecolor=facecolor,
        edgecolor=edgecolor,
        linewidth=linewidth,
    )
    plt.gca().add_patch(rect)


def draw_circle(center_x, center_y, radius):
    circle = Circle((center_x, center_y), color="dimgray", radius=radius)
    plt.gca().add_patch(circle)


def visualize_imgs(fig, axs, img_list, titles=None, cmap=None):
    for idx, img in enumerate(img_list):

        if img is None:
            continue

        im = axs[idx].imshow(img, cmap=cmap)
        if cmap is not None:
            fig.colorbar(im, ax=axs[idx])
        if titles is not None:
            axs[idx].set_title(titles[idx])
