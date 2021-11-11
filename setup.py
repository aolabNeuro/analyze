# setup.py
# setup script for aopy

import setuptools

with open("README.md", "r") as f:
    long_description = f.read()

install_requires = [
    'numpy',
    'pandas',
    'psutil',
    'h5py',
    'tables',
    'scikit-learn',
    'nitime',
    'xlrd',
    'openpyxl',
    'matplotlib',
    'scipy',
    'seaborn',
    'pyyaml'
]

setuptools.setup(
    name="aopy",
    version="0.2.0",
    author="aoLab",
    author_email="mnolan@uw.edu", # I am not eternal, please replace ~ MN
    description="python code repository for aoLab @UW",
    long_description=long_description,
    url="https://github.com/aolabNeuro/analyze",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent"
    ],
    install_requires=install_requires,
)
