import os
from setuptools import setup, find_packages

# Read version
with open("version.txt") as f:
    version = f.read().strip()

setup(
    name="warpgbm",
    version=version,
    packages=find_packages(),
    ext_modules=[], 
    install_requires=[
        "torch",
        "taichi", #NEW
        "numpy",
        "scikit-learn",
        "tqdm",
    ],
    include_package_data=True,
    zip_safe=False,
)