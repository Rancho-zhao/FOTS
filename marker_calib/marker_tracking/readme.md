## Usage

- `cd marker_tracking`
    - `python optical_flow_marker_tracking.py` : Press SPACEBAR to start recording tactile flow images and corresponding markers' position and displacement. Please record in groups of three in order of dilate, shear and twist.
- `cd ..`
    - `python marker_calib.py` : Press LEFTMOUSE to label center, circumference, and contact points of dilate image, label translated center of shear image, and label rotated circumference of twist image, then RIGHTMOUSE to start the next group.


Some of this part refers to [gsrobotics](https://github.com/gelsightinc/gsrobotics).



