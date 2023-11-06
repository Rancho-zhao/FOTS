"""
Labels images for training MLP depth reconstruction model.
Specify the image folder containing the circle images and csv folder to store the labels ( img_names, center_x, center_y, radius ).
The image datasets should include the rolling of a sphere with a known radius.

Directions:
-- Click left mouse button to select the center of the sphere.
-- Click right mouse button to select the circumference of the sphere.
-- Double click ESC to move to the next image.
"""

import argparse
import csv
import glob
import math
import os

import cv2

base_path = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))


def click_and_store(event, x, y, flags, param):
    global count
    global center_x, center_y, circumference_x, circumference_y, radii
    if event == cv2.EVENT_LBUTTONDOWN:
        center_x = x
        center_y = y
        print("center_x: ", x)
        print("center_y: ", y)
        cv2.circle(image, (x, y), 3, (0, 0, 255), -1)
        cv2.imshow("image", image)
    elif event == cv2.EVENT_RBUTTONDOWN:
        circumference_x = x
        circumference_y = y
        print("circumference_x: ", x)
        print("circumference_y: ", y)
        cv2.circle(image, (x, y), 3, (0, 0, 255), -1)
        cv2.imshow("image", image)
        radius = math.sqrt(
            (center_x - circumference_x) ** 2 + (center_y - circumference_y) ** 2
        )
        print("radius: ", int(radius))
        radii.append(int(radius))
        with open(filename, "a") as csvfile:
            writer = csv.writer(csvfile)
            print("Writing>>")
            count += 1
            writer.writerow([img_name, center_x, center_y, int(radius)])


if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument("--folder", type=str, default="gelsight", help="folder containing images")
    argparser.add_argument("--csv", type=str, default=f"{base_path}/csv/annotate.csv", help="csv file to store results")
    args = argparser.parse_args()
    filename = args.csv
    img_folder = os.path.join(base_path, args.folder)
    img_files = sorted(glob.glob(f"{img_folder}/*.png"))
    os.makedirs(os.path.join(base_path, "csv"), exist_ok=True)
    center_x, center_y, circumference_x, circumference_y, radii = [], [], [], [], []
    count = 0
    for img in img_files:
        image = cv2.imread(img)
        img_name = img
        cv2.imshow("image", image)
        cv2.setMouseCallback("image", click_and_store, image)
        cv2.waitKey(0)
    cv2.destroyAllWindows()
