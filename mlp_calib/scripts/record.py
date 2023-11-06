"""
Script for capturing individual frames while the camera output is displayed.
-- Press SPACEBAR to capture
-- Press ESC to terminate the program.
"""
import argparse
import os
import os.path

import cv2
import numpy as np

from src.digit.digit_sensor import DigitSensor

base_path = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))


def record_frame(digit_sensor, dir_name: str):
    img_counter = len(os.listdir(dir_name))
    digit_call = digit_sensor()
    while True:
        frame = digit_call.get_frame()
        cv2.imshow("Capture Frame", frame)
        k = cv2.waitKey(1)
        if k % 256 == 27:
            # ESC hit
            print("Escape hit, closing...")
            break
        elif k % 256 == 32:
            # SPACEBAR hit
            img_name = "{}/{:0>4}.png".format(dir_name, img_counter)
            cv2.imwrite(img_name, frame)
            print("{} written!".format(img_name))
            img_counter += 1
    cv2.destroyAllWindows()


def takeimg(dir, num, camnum = 0, filename = None):
    outputfolder = dir + '/'
    if not os.path.exists(outputfolder):
        os.makedirs(outputfolder)

    camera = cv2.VideoCapture(camnum)
    camera.set(cv2.CAP_PROP_EXPOSURE,-10)
    print("曝光度是：",camera.get(cv2.CAP_PROP_EXPOSURE))
    i = 0
    while (i < num+1):
        return_value, img = camera.read()
        image0 = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
        image = cv2.resize(image0, (240,320))
        cv2.imshow("tst",image)
        key = cv2.waitKey(1)
        if key & 0xFF == 32:
            print(i, "done")
            if filename is None:
                cv2.imwrite("{}/{:0>4}.png".format(outputfolder, i), image)
                np.save("gelsight_bg.npy", image)
            else:
                cv2.imwrite("{}/{:0>4}.png".format(outputfolder, i), image)
            i = i+1

    camera.release()

if __name__ == "__main__":
    # argparser = argparse.ArgumentParser()
    # argparser.add_argument("--fps", type=int, default=30)
    # argparser.add_argument("--resolution", type=str, default="QVGA")
    # argparser.add_argument("--serial_num", type=str, default="D20368")
    # args = argparser.parse_args()
    # if not os.path.exists(os.path.join(base_path, "images")):
    #     os.makedirs(os.path.join(base_path, "images"), exist_ok=True)
    #     # print("Directory {} created for saving images".format(args.dir_name))
    # digit = DigitSensor(args.fps, args.resolution, args.serial_num)

    # record_frame(digit, os.path.join(base_path, "images"))
    takeimg('gelsight', 42)