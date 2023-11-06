"""
Labels images for calibrating shadow.
Specify the image folder containing the circle images.
The image datasets should include the rolling of a sphere with a known radius.

Directions:
-- Click left mouse button to select the center of the sphere.
-- Click right mouse button clockwise to select three shadows of the sphere.
-- Double click ESC to move to the next image.
"""

import argparse
import csv
import glob
import math
import os

import cv2
import numpy as np

import params as pr

base_path = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))

def intersection_of_multi_lines(strt_points, directions):  
    '''
    strt_points: line start points; numpy array, nxdim
    directions: list dierctions; numpy array, nxdim  

    return: the nearest points to n lines 
    '''
    
    n, dim = strt_points.shape

    G_left = np.tile(np.eye(dim), (n, 1))  
    G_right = np.zeros((dim*n, n))  

    for i in range(n):
        G_right[i*dim:(i+1)*dim, i] = -directions[i, :]

    G = np.concatenate([G_left, G_right], axis=1)  
    d = strt_points.reshape((-1, 1)) 

    m = np.linalg.inv(np.dot(G.T, G)).dot(G.T).dot(d)   

    # return m[0:dim]  
    return m

def click_and_store(event, x, y, flags, param):
    global count
    global points
    if event == cv2.EVENT_LBUTTONDOWN:
        points[count,:2] = y, x
        cv2.circle(image, (x, y), 3, (0, 0, 255), -1)
        cv2.imshow("image", image)
    elif event == cv2.EVENT_RBUTTONDOWN:
        points[count,2:] = y, x, 0
        cv2.circle(image, (x, y), 3, (255, 0, 0), -1)
        cv2.imshow("image", image)


if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument("--folder", type=str, default="shadow_data", help="folder containing images")
    args = argparser.parse_args()
    img_folder = os.path.join(base_path, args.folder)
    img_files = sorted(glob.glob(f"{img_folder}/*.png"))
    ball_r = 3 / pr.pixmm
    count = 0
    points = np.zeros((len(img_files), 5))
    for img in img_files:
        image = cv2.imread(img)
        img_name = img
        cv2.imshow("image", image)
        cv2.setMouseCallback("image", click_and_store, image)
        cv2.waitKey(0)
        count += 1
    cv2.destroyAllWindows()
    points = points[~np.all(points==0,axis=1)]
    print("points:\n",points)

    strt_point = points[:,2:]
    
    dis = points[:,:2] - points[:,2:4]
    theta = np.arcsin(ball_r / np.sqrt(dis[:,0]**2+dis[:,1]**2))
    z = np.sqrt(dis[:,0]**2+dis[:,1]**2) * np.tan(theta.T)

    directions = np.zeros((len(points), 3))
    directions[:, :2] = dis
    directions[:, 2] = z*0.5
    # print("directions:\n",directions)

    inters = intersection_of_multi_lines(strt_point, directions)   
    print('[DEBUG] intersection {}'.format(inters))
