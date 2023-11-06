## Usage
Change **gel height,gel width, mm_to_pix, base_img_path, sensor :serial_num ** values in normal_to_rgb.yaml file in config folder.
- `pip install . `
- `cd scripts`
    - `python record.py` : Press SPACEBAR to start recording.
    - `python label_data.py` : Press LEFTMOUSE to label center and RIGHTMOUSE to label circumference.
    - `python create_image_dataset.py` : Create a dataset of images and save it to a csv file.
    - `python train_mlp.py` : Train an MLP model for Normal to Color mapping.

normal2color model will be saved to a separate folder "models" in the same directory as this file.

Some of this part refers to [digit-depth](https://github.com/vocdex/digit-depth).
