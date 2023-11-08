import cv2
import time
import numpy as np
import open3d as o3d

from tactile_render import get_simapproach
from utils.marker_motion import MarkerMotion
from utils import rotations
import params as pr

mm_to_pixel = pr.mm_to_pixel
psp_w, psp_h = pr.sensor_w, pr.sensor_h
s_type = pr.sensor_type

# Translation and Rotation (mm & rad)
trans = [0, 0, 0.1]
rot = [0, 0, 0]

if __name__=="__main__":
    # Read the stl file as a mesh
    mesh = o3d.io.read_triangle_mesh("assets/daniel/cylinder.stl")
    gel_pad = np.load("utils/utils_data/"+s_type+"_pad.npy")
    model_center = mesh.get_center()

    R = mesh.get_rotation_matrix_from_xyz(rot) 
    mesh.rotate(R, center=[0,0,0]) 
    mesh.compute_vertex_normals 

    # Sample points from the mesh
    # pcd = mesh.sample_points_poisson_disk(number_of_points=1000000, init_factor=1)
    pcd = mesh.sample_points_uniformly(number_of_points=1000000)
    # o3d.visualization.draw_geometries([pcd]) 
    vertices = np.asarray(pcd.points)

    # gel_map = np.ones((320,240)) * 0.0
    heightMap = np.zeros((psp_h,psp_w))

    # centralize the points
    cx = 0.0#model_center[0]#np.mean(vertices[:,0])
    cy = 0.0#model_center[1]#np.mean(vertices[:,1])

    # add the shifting and change to the pix coordinate
    uu = ((vertices[:,0] - cx + trans[0])*mm_to_pixel + psp_w//2).astype(int)
    vv = ((vertices[:,1] - cy + trans[1])*mm_to_pixel + psp_h//2).astype(int)
    # check boundary of the image
    mask_u = np.logical_and(uu > 0, uu < psp_w)
    mask_v = np.logical_and(vv > 0, vv < psp_h)
    # check the depth
    mask_z = vertices[:,2] > 10
    mask_map = mask_u & mask_v & mask_z
    heightMap[vv[mask_map],uu[mask_map]] = vertices[mask_map][:,2]*mm_to_pixel

    heightMap += gel_pad*mm_to_pixel

    max_o = np.max(heightMap)
    gel_map = np.ones((psp_h, psp_w)) * max_o
    # pressing depth in pixel
    pressing_height_pix = trans[2]*mm_to_pixel

    # shift the gelpad to interact with the object
    gel_map -= pressing_height_pix

    # get the contact area
    contact_mask = heightMap > gel_map

    # combine contact area of object shape with non contact area of gelpad shape
    zq = np.zeros((psp_h,psp_w))

    zq[contact_mask]  = heightMap[contact_mask]
    zq[~contact_mask] = gel_map[~contact_mask]
    zq -= gel_map
    # render height map to tactile img
    simulation = get_simapproach()
    tact_img = simulation.generate(zq, contact_mask)

    # obtain object's relative pose
    relative_pos = []
    if np.max(trans[2])>0.0:
        relative_pos.append([0, 0, 0])
        relative_pos.append([trans[0], trans[1], rot[2]])
    # obtain markers' motion according to depth and object geometry info
    marker = MarkerMotion(frame0_blur=tact_img,depth=zq/mm_to_pixel,mask=contact_mask,traj=relative_pos,
                          lamb=[0.00125,0.00021,0.00038])
    marker_img = marker._marker_motion()

    cv2.imshow("tact_img",tact_img)
    cv2.imshow("mask_img", contact_mask.astype(np.uint8)*255)
    cv2.imshow("marker_img", marker_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()