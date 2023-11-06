import cv2
import numpy as np

def shadow_matrix(S, light_type):
    if light_type=="spot":
        m = np.mat([[S[2], 0, -S[0], 0],\
                    [0, S[2], -S[1], 0],\
                    [0, 0, 0 ,0],\
                    [0, 0, -1, S[2]]])
    else:
        m = np.mat([[1, 0, -S[0]/S[2], 0],\
                    [0, 1, -S[1]/S[2], 0],\
                    [0, 0, 0, 0]])
    return m

def planar_shadow(light, depth, light_type):

    m = shadow_matrix(light,light_type)
    idx = np.nonzero(depth>5)

    P = np.mat([idx[0], idx[1], depth[idx], np.ones_like(idx[0])])
    Q = np.dot(m, P)
    if light_type=="spot":
        Q = Q / Q[3]
    shadow = Q[:2].astype(np.uint16)
    # limit x,y in [320,240]
    shadow_x = np.asarray(shadow[0])
    shadow_y = np.asarray(shadow[1])
    shadow_x[shadow_x<0]=0
    shadow_x[shadow_x>319]=319
    shadow_y[shadow_y<0]=0
    shadow_y[shadow_y>239]=239
    # generate shadow mask
    img = np.zeros((320,240))
    img[shadow_x,shadow_y] = 1.0
    # erode and dilate
    kernel = np.ones((2,2),np.uint8)
    # mask = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)
    # mask = cv2.erode(img, kernel, iterations=1)
    mask = cv2.dilate(img, kernel, iterations=2)

    return 1.0-mask

if __name__=="__main__":
    # light positions
    light = [80, 0, 100.0]
    height_map = np.load("heightmap.npy")
    cv2.imshow("h", height_map-10)
    mask = planar_shadow(light, height_map)
    cv2.imshow("hh", mask)
    if cv2.waitKey(0) & 0xFF == ord('q'):
        cv2.destroyAllWindows()