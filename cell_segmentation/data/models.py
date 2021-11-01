from dataclasses import dataclass
from typing import List, Optional

import numpy as np


@dataclass
class BBox:
    xmin: float
    ymin: float
    width: float
    height: float


@dataclass
class Image:
    height: int
    width: int
    array: np.ndarray


@dataclass
class Mask:
    array: np.ndarray


@dataclass
class Segmentation:
    annotation: List[int]
    label: str


@dataclass
class MicroscopyImage(Image):
    """
    Microscopy image data model

    Attributes:
        file_id: unique file name of image
        annotations: list of Segmentation annotations
        image_path: path to image file
        mask_path: path to mask file
    """

    file_id: str
    image_path: str
    annotations: Optional[List[Segmentation]] = None
    mask_path: str = None
