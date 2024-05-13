import mujoco
import cv2
import numpy as np

from tactile_render import get_simapproach
from marker_motion import MarkerMotion
from utils import mujoco_utils
from utils.mocap_utils import MyViewer

gel_width, gel_height = 240,320

# define callback func for trackbar
def callback(x):
    pass

if __name__=="__main__":
    filepath = "./assets/gel/fots_mujoco.xml"

    model = mujoco.MjModel.from_xml_path(filepath)
    data = mujoco.MjData(model)
    viewer = MyViewer(model,data)
    img=np.zeros((100,100),dtype='uint8')
    # create a new window
    cv2.namedWindow('img')
    relative_pos = []

    while True:
        img=np.zeros((100,100),dtype='uint8')     

        mujoco.mj_step(model,data)
        viewer.render_new()
        pressKey = cv2.waitKey(1) & 0xFF
        if pressKey == ord('g'):
            # obtain depth and mask
            viewer.render(gel_width,gel_height,camera_id=0)
            depth = viewer.read_pixels(gel_width, gel_height, depth=True)[1].copy()
            depth = np.fliplr(depth)
            ini_depth = np.load("./utils/utils_data/ini_depth_extent.npy")
            depth_diff = ini_depth - depth
            mask = depth_diff > 0.0

            # render depth to tactile img
            simulation = get_simapproach()
            tact_img = simulation.generate(depth)

            # obtain object's relative pose
            digit_xpos = mujoco_utils.get_site_xpos(model, data, "digit:site")
            digit_xpos = digit_xpos.reshape(digit_xpos.shape[0],1)
            digit_xmat = mujoco_utils.get_site_xmat(model, data, "digit:site")
            digit_mat = np.vstack((np.hstack((digit_xmat, digit_xpos)),np.array([0,0,0,1])))

            object_xpos = mujoco_utils.get_site_xpos(model, data, "object1:site")
            object_xpos = object_xpos.reshape(object_xpos.shape[0],1)
            object_xmat = mujoco_utils.get_site_xmat(model, data, "object1:site")
            object_mat = np.vstack((np.hstack((object_xmat, object_xpos)),np.array([0,0,0,1])))

            relative_mat = np.dot(np.linalg.inv(digit_mat), object_mat)
            relative_xpos = relative_mat[:3,3]
            relative_xmat = relative_mat[:3,:3]
            relative_xrot = cv2.Rodrigues(relative_xmat)[0]

            if np.max(depth_diff)>0.0:
                relative_pos.append([-relative_xpos[1], relative_xpos[0], -relative_xrot[2,0]])
            else:
                relative_pos = []

            # # obtain markers' motion according to depth and object geometry info
            marker = MarkerMotion(frame0_blur=tact_img,depth=depth_diff,mask=mask,traj=relative_pos,lamb=[0.00125,0.00021,0.00038])
            marker_img = marker._marker_motion()

            cv2.imshow("tact_img",tact_img)
            cv2.imshow("marker_img", marker_img)

        elif pressKey == ord('s'):
            cv2.imwrite("tact.png",tact_img)
            cv2.imwrite("marker.png", marker_img)
        elif pressKey == ord('q'): 
            cv2.destroyAllWindows()
            continue