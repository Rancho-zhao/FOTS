import cv2
import math
import numpy as np
import params as pr

mm2pix = pr.mm_to_pixel

class MarkerMotion():
    def __init__(self,
                frame0_blur,
                depth,
                mask,
                traj,
                lamb,
                N=pr.N,
                M=pr.M,
                W=pr.sensor_w,
                H=pr.sensor_h,
                is_flow=True):

        # self.model = model
        # self.data = data
        self.frame0_blur = frame0_blur
        self.depth = depth
        self.mask = mask
        self.lamb = lamb

        self.N = N
        self.M = M
        self.W = W
        self.H = H
        self.is_flow = is_flow

        self.traj = traj
        self.contact = []
        self.moving = False
        self.rotation = False

        self.mkr_rng = 0.5

        self.x = np.arange(0, self.W, 1)
        self.y = np.arange(0, self.H, 1)
        self.xx, self.yy = np.meshgrid(self.x, self.y)

    def _shear(self, center_x, center_y, lamb, shear_x, shear_y, xx, yy):
        # TODO: add force and torque effect
        g = np.exp(-(((xx - center_x) ** 2 + (yy - center_y) ** 2)) * lamb)

        dx, dy = shear_x * g, shear_y * g

        xx_ = xx + dx
        yy_ = yy + dy
        return xx_, yy_

    def _twist(self, center_x, center_y, lamb, theta, xx, yy):

        g = np.exp(-(((xx - center_x) ** 2 + (yy - center_y) ** 2)) * lamb)

        dx = xx - center_x
        dy = yy - center_y

        rotx = dx * np.cos(theta) - dy * np.sin(theta)
        roty = dx * np.sin(theta) + dy * np.cos(theta)  

        xx_ = xx + (rotx - dx) * g
        yy_ = yy + (roty - dy) * g
        return xx_, yy_

    def _dilate(self, lamb, xx, yy):
        dx, dy = 0.0, 0.0
        for i in range(len(self.contact)):
            g = np.exp(-(((xx - self.contact[i][1]) ** 2 + (yy - self.contact[i][0]) ** 2)) * lamb)

            dx += self.contact[i][2] * (xx - self.contact[i][1]) * g
            dy += self.contact[i][2] * (yy - self.contact[i][0]) * g

        xx_ = xx + dx
        yy_ = yy + dy
        return xx_, yy_

    def _generate(self,xx,yy):
        img = np.zeros_like(self.frame0_blur.copy())#

        for i in range(self.N):
            for j in range(self.M):
                ini_r = int(self.yy[j,i])
                ini_c = int(self.xx[j,i])
                r = int(yy[j, i])
                c = int(xx[j, i])
                if r >= self.H or r < 0 or c >= self.W or c < 0:
                    continue
                # shape = img[r - 1 : r + 2, c - 1 : c + 2, :].shape
                cv2.circle(img,(c,r), 3, (20,20,20),4)

                img[r, c, :] = self.frame0_blur[r, c, :] * 0
                k = 3
                if self.is_flow:
                    pt1 = (ini_c, ini_r)
                    pt2 = (c+k*(c-ini_c), r+k*(r-ini_r))
                    color = (0, 255, 0)
                    cv2.arrowedLine(img, pt1, pt2, color, 2,  tipLength=0.2)


        # img = img[:self.W, :self.H]
        return img

    def _motion_callback(self,xx,yy):
        for i in range(self.N):
            for j in range(self.M):
                r = int(yy[j, i])
                c = int(xx[j, i])
                if self.mask[r,c] == 1.0:
                    h = self.depth[r,c]
                    self.contact.append([r,c,h])
        
        if not self.contact:
            xx,yy = self.xx,self.yy

        xx_,yy_ = self._dilate(self.lamb[0], xx ,yy)
        if len(self.traj) >= 2:
            xx_,yy_ = self._shear(int(self.traj[0][0]*mm2pix + 120), 
                                int(self.traj[0][1]*mm2pix + 160),
                                self.lamb[1],
                                int((self.traj[-1][0]-self.traj[0][0])*mm2pix),
                                int((self.traj[-1][1]-self.traj[0][1])*mm2pix),
                                xx_,
                                yy_)

            theta = max(min(self.traj[-1][2]-self.traj[0][2], 50 / 180.0 * math.pi), -50 / 180.0 * math.pi)
            xx_,yy_ = self._twist(int(self.traj[-1][0]*mm2pix + 120), 
                                int(self.traj[-1][1]*mm2pix + 160),
                                self.lamb[2],
                                theta,
                                xx_,
                                yy_)

        return xx_,yy_


    def _marker_motion(self):
        xind = (np.random.random(self.N * self.M) * self.W).astype(np.int16)
        yind = (np.random.random(self.N * self.M) * self.H).astype(np.int16)

        x = np.arange(pr.x0, 320, pr.dx)[:self.N]
        y = np.arange(pr.y0, 240, pr.dy)[:self.M]
        
        xind, yind = np.meshgrid(x, y)
        xind = (xind.reshape([1, -1])[0]).astype(np.int16)
        yind = (yind.reshape([1, -1])[0]).astype(np.int16)

        xx_marker, yy_marker = self.xx[xind, yind].reshape([self.M, self.N]), self.yy[xind, yind].reshape([self.M, self.N])
        self.xx,self.yy = xx_marker, yy_marker

        img = self._generate(xx_marker, yy_marker)
        # if self.contact:
        xx_marker_, yy_marker_ = self._motion_callback(xx_marker, yy_marker)
        img = self._generate(xx_marker_, yy_marker_)
        self.contact = []

        return img