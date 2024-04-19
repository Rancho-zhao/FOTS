import argparse
import csv
import glob
import math
import os

import cv2
import numpy as np
import pandas as pd
import lmfit
import matplotlib.pyplot as plt

def click_and_store(event, x, y, flags, param):
    global points
    if event == cv2.EVENT_LBUTTONDOWN:
        points.append([y, x])
        cv2.circle(image, (x, y), 3, (0, 0, 255), -1)
        cv2.imshow("image", image)
    if event == cv2.EVENT_RBUTTONDOWN:
        points.append([0, 0])

def split_array(a):
  result = []
  current = []
  for row in a:
    if (row == [0,0]).all():
      if len(current) > 0:
        result.append(current)
        current = []
    else:
      current.append(row)
  if len(current) > 0:
    result.append(current)
  return result

# define model functions
def f_dilate(x, lam_d):
    # x = [M, C, h]
    # M: markers
    # C: contact points
    # h: the height of contact points
    # calculate the distance between markers and contact points
    d = []
    for j in range((len(x[0])-2)//3):
        dx, dy = 0.0, 0.0
        i = 0
        while x[i,4] != 0.:
            g = np.exp(-(((x[:,0] - x[i,2+3*j]) ** 2 + (x[:,1] - x[i,3+3*j]) ** 2)) * lam_d)

            dx += x[i,4+3*j] * (x[:,0] - x[i,2+3*j]) * g
            dy += x[i,4+3*j] * (x[:,1] - x[i,3+3*j]) * g
            i+=1
        if j==0:
            d = np.hstack((dx,dy))
        else:
            d = np.hstack((d, np.hstack((dx,dy))))
    
    return d

def f_shear(x, lam_s):
    # x = [M, G, s]
    # M: markers
    # G: origin of contact area
    # s: displacement of object
    d = []
    for j in range((len(x[0])-2)//2):
        # calculate displacement
        g = np.exp(-(((x[:,0] - x[0,2+2*j]) ** 2 + (x[:,1] - x[0,3+2*j]) ** 2)) * lam_s)

        dx, dy = x[1,2+2*j] * g, x[1,3+2*j] * g
        if j==0:
            d = np.hstack((dx,dy))
        else:
            d = np.hstack((d, np.hstack((dx,dy))))

    return d

def f_twist(x, lam_t):
    # x = [M, G, theta]
    # M: markers
    # G: origin of contact area
    # theta: twist degree of object
    d = []
    for j in range((len(x[0])-2)//2):
        theta = x[1,2+2*j]

        g = np.exp(-(((x[:,0] - x[0,2+2*j]) ** 2 + (x[:,1] - x[0,3+2*j]) ** 2)) * lam_t)

        d_x = x[:,0] - x[0,2+2*j]
        d_y = x[:,1] - x[0,3+2*j]

        rotx = d_x * np.cos(theta) - d_y * np.sin(theta)
        roty = d_x * np.sin(theta) + d_y * np.cos(theta)  

        dx, dy =  (rotx - d_x) * g, (roty - d_y) * g

        if j==0:
            d = np.hstack((dx,dy))
        else:
            d = np.hstack((d, np.hstack((dx,dy))))

    return d

if __name__ == "__main__":

    calib_type = 0 # 0: dilate, 1: shear, 2: twist

    argparser = argparse.ArgumentParser()
    argparser.add_argument("--folder", type=str, default="data/", help="folder containing images")
    args = argparser.parse_args()

    base_path = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))
    img_folder = os.path.join(base_path, args.folder)
    img_files = sorted(glob.glob(f"{img_folder}/*.png"))
    ball_r = 3 / (0.0266*2)
    points = []

    img_path = "/home/r404/Digit_Test/marker_calib/data/"+str(calib_type)+".png"
    csv_path = "/home/r404/Digit_Test/marker_calib/data/"+str(calib_type)+".csv"
    M = []
    d_dx, d_dy = [], []
    s_dx, s_dy = [], []
    t_dx, t_dy = [], []
    while img_path in img_files:
        image = cv2.imread(img_path)
        cv2.imshow("image", image)
        cv2.setMouseCallback("image", click_and_store, image)
        cv2.waitKey(0)
        print(points)
        with open(csv_path,'r') as f:
            reader = csv.reader(f)
            next(reader)
            for row in reader:
                if calib_type % 3 == 0:
                    if calib_type == 0:
                        M.append([int(row[0]), int(row[1])])
                    d_dx.append(float(row[2]))
                    d_dy.append(float(row[3]))
                elif calib_type % 3 == 1:
                    s_dx.append(float(row[2]))
                    s_dy.append(float(row[3]))
                else:
                    t_dx.append(float(row[2]))
                    t_dy.append(float(row[3]))
        calib_type += 1
        img_path = "/home/r404/Digit_Test/marker_calib/data/"+str(calib_type)+".png"
        csv_path = "/home/r404/Digit_Test/marker_calib/data/"+str(calib_type)+".csv"

    cv2.destroyAllWindows()

    # obtian marker's real displacement under different loads
    d_d = np.array(np.hstack((d_dx,d_dy)))
    d_s = np.array(np.hstack((s_dx,s_dy))) - d_d
    d_t = np.array(np.hstack((t_dx,t_dy))) - d_d

    points = np.array(points)
    all_points = split_array(points)

    Cd, Cd_p = [], []
    Cs, Cs_p = [], []
    Ct, Ct_p = [], []

    count = 0
    for p in all_points:
        # the center pos of contact circle
        print(p)
        O = np.array(p[0])
        O_1 = np.array(p[-2])
        # calculate shear displacement
        s = O_1 - O
        # the contact edge point
        E = np.array(p[1])
        E_1 = np.array(p[-1])
        # calculate twist degree
        a2 = (E-O)[0]**2+(E-O)[1]**2
        b2 = (E_1-O)[0]**2+(E_1-O)[1]**2
        c2 = (E-E_1)[0]**2+(E-E-1)[1]**2
        theta = math.acos((a2+b2-c2)/(2*math.sqrt(a2)*math.sqrt(b2)))
        # contact points
        C = np.array(p[2:-2])
        # calculate height of contact points
        h = []
        for c in C:
            h_square = (E-O)[0]**2+(E-O)[1]**2-((c-O)[0]**2+(c-O)[1]**2)
            h.append([int(math.sqrt(h_square))])
        # concatenate contact points and corresponding height
        Cd = np.concatenate((C,h),axis=1)
        Cd_zero = np.zeros((len(M)-len(h), 3))
        Cd = np.concatenate((Cd,Cd_zero),axis=0)

        # shear
        Cs = [O,s]
        Cs_zero = np.zeros((len(M)-len(Cs), 2))
        Cs = np.concatenate((Cs,Cs_zero),axis=0)

        # twist
        Ct = [O,[theta, 0]]
        Ct_zero = np.zeros((len(M)-len(Ct), 2))
        Ct = np.concatenate((Ct,Ct_zero),axis=0)

        if count==0:
            Cd_p = Cd.copy()
            Cs_p = Cs.copy()
            Ct_p = Ct.copy()
        else: 
            Cd_p = np.hstack((Cd_p, Cd))
            Cs_p = np.hstack((Cs_p, Cd))
            Ct_p = np.hstack((Ct_p, Cd))
        
        count += 1

    # fit data using lmfit to obtian optimal lambda
    f_d = np.concatenate((M, Cd_p),axis=1)
    model_d = lmfit.Model(f_dilate, independent_vars=['x'], param_names=['lam_d'])
    params_d = lmfit.Parameters()
    params_d.add('lam_d', value=0.1, min=0, max=1)
    results_d = model_d.fit(d_d, params=params_d, x=f_d)
    print("lam_d: ", results_d.params['lam_d'].value)
     
    f_s = np.concatenate((M, Cs_p),axis=1)
    model_s = lmfit.Model(f_shear, independent_vars=['x'], param_names=['lam_s'])
    params_s = lmfit.Parameters()
    params_s.add('lam_s', value=0.1, min=0, max=1)
    results_s = model_s.fit(d_s, params=params_s, x=f_s)
    print("lam_s: ", results_s.params['lam_s'].value)

    # twist concatenate
    f_t = np.concatenate((M, Ct_p),axis=1)
    model_t = lmfit.Model(f_twist, independent_vars=['x'], param_names=['lam_t'])
    params_t = lmfit.Parameters()
    params_t.add('lam_t', value=0.1, min=0, max=1)
    results_t = model_t.fit(d_t, params=params_t, x=f_t)
    print("lam_t: ", results_t.params['lam_t'].value)
