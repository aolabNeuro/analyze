# setup.py
# setup script for aopy

import setuptools

with open("README.md", "r") as f:
    long_description = f.read()

install_requires = [
    'numpy',
    'xarray',
    'pandas',
    'psutil',
    'h5py',
    'tables',
    'scikit-learn',
    'statsmodels',
    'nitime',
    'xlrd',
    'openpyxl',
    'matplotlib>=3.6',
    'scipy',
    'PyWavelets',
    'Pillow',
    'seaborn',
    'pyyaml',
    'tqdm',
    'open-ephys-python-tools',
    'aolab-bmi3d',
]

setuptools.setup(
    name="aolab-aopy",
    version="0.6.5",
    author="aoLab",
    author_email="aorsborn@uw.edu",
    description="python code repository for aoLab @UW",
    long_description=long_description,
    url="https://github.com/aolabNeuro/analyze",
    packages=setuptools.find_namespace_packages(),
    include_package_data=True,
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent"
    ],
    install_requires=install_requires,
)
