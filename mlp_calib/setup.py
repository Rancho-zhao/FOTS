from setuptools import setup

setup(
    name="digit-depth",
    version="0.1",
    description="Digit Depth Reconstruction",
    url="https://github.com/vocdex/digit-depth",
    author="Shukrullo Nazirjonov",
    author_email="nazirjonovsh2000@gmail.com",
    license="MIT",
    packages=["src/digit", "src/third_party", "src/train", "src","scripts"],
    zip_safe=False,
)
