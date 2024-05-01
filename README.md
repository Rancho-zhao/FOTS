# FOTS: A Fast Optical Tactile Simulator for Sim2Real Learning of Tactile-motor Robot Manipulation Skills
This work has been accepted by IEEE Robotics and Automation Letters (RA-L).

FOTS is suitable for GelSight tactile sensors and its variations (like DIGIT sensors). 

In addition, FOTS does not rely on any simulation platform, you can integrate it to any platform (like [MuJoCo](https://github.com/google-deepmind/mujoco)).

## Installation

To install dependencies: `pip install -r requirements.txt`

## Optical Calibration (optional)
1. MLP Calib: Connect a GelSight/DIGIT sensor, and collect ~50 tactile images of a sphere indenter on different locations.
2. Shadow Calib: Choose ~5 tactile images from last step.

*Note: Specific steps refers to respective readme file.*

## Marker Calibration (optional)
1. Collect ~45 tactile flow images of a sphere indenter on different locations (in order of dilate, shear, and twist, 15 images for each type).

*Note: Specific steps refers to respective readme file.*

## Usage Directly
We provide a set of calibration files and you can work with them directly. 

- `python fots_test.py`: you can obtain simulated tactile image, mask, and tactile flow image.



## License
FOTS is licensed under [MIT license](LICENSE).

## Citing FOTS
If you use FOTS in your research, please cite:
```BibTeX
@article{zhao2024fots,
  title={FOTS: A Fast Optical Tactile Simulator for Sim2Real Learning of Tactile-motor Robot Manipulation Skills},
  author={Zhao, Yongqiang and Qian, Kun and Duan, Boyi and Luo, Shan},
  journal={IEEE Robotics and Automation Letters},
  year={2024}
}
```
[ArXiv Paper Link](https://arxiv.org/abs/2404.19217)
