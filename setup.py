# setup.py
# setup script for aopy

import setuptools

with open("README.md", "r") as f:
    long_description = f.read()

setuptools.setup(
    name="aopy",
    version="0.0.1",
    author="aoLab",
    author_email="mnolan@uw.edu", # I am not eternal, please replace ~ MN
    description="python code repository for aoLab @UW",
    long_description=long_description,
    url="https://github.com/m-nolan/aopy_dev",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent"
    ],
)