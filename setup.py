from setuptools import find_packages, setup

setup(
    name="cell_segmentation",
    packages=find_packages(),
    version="0.1.0",
    description="sartorius cell instance segmentation kaggle competition",
    author="Leon Truong",
    license="MIT",
    install_requires=[
        "black",
        "click==8.0.3",
        "numpy==1.21.3",
        "pandas==1.3.4",
        "torch@https://download.pytorch.org/whl/cu113/torch-1.10.0%2Bcu113-cp39-cp39-linux_x86_64.whl",
        "torchvision@https://download.pytorch.org/whl/cu113/torchvision-0.11.1%2Bcu113-cp39-cp39-linux_x86_64.whl",
        # "detectron2@https://dl.fbaipublicfiles.com/detectron2/wheels/cu113/torch1.10/index.html",
    ],
)
