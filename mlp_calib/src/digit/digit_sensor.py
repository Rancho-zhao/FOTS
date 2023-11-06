import time

from digit_interface import Digit
from digit_interface.digit import DigitDefaults


class DigitSensor:
    def __init__(self, fps: int, resolution: str, serial_num: str):
        self.fps = fps
        self.resolution = resolution
        self.serial_num = serial_num

    def __call__(self, *args, **kwargs):
        """Calls the digit sensor."""
        digit = self.setup_digit()
        return digit

    def __str__(self):
        return f"DigitSensor(fps={self.fps}, resolution={self.resolution}, serial_num={self.serial_num})"

    def setup_digit(self,):
        "Sets up the digit sensor and returns it."

        digit = Digit(self.serial_num)
        digit.connect()

        rgb_list = [(15, 0, 0), (0, 15, 0), (0, 0, 15)]

        for rgb in rgb_list:
            digit.set_intensity_rgb(*rgb)
            time.sleep(1)
        digit.set_intensity(15)
        resolution = DigitDefaults.STREAMS[self.resolution]
        digit.set_resolution(resolution)
        fps = Digit.STREAMS[self.resolution]["fps"][str(self.fps) + "fps"]
        digit.set_fps(fps)
        return digit
