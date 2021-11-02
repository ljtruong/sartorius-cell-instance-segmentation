from dataclasses import dataclass
from typing import List, Optional

import numpy as np
import cv2


@dataclass
class BBox:
    xmin: float
    ymin: float
    width: float
    height: float

    def to_row(self, format: str = "XYWH") -> List[float]:
        """
        Convert bounding box to a row list for coco input format.
        """
        if format == "XYXY":
            return [
                self.xmin,
                self.ymin,
                self.xmin + self.width,
                self.ymin + self.height,
            ]
        else:
            return [self.xmin, self.ymin, self.width, self.height]


@dataclass
class Image:
    image_path: str
    height: int
    width: int

    def array(self) -> np.ndarray:
        return cv2.imread(self.image_path)


@dataclass
class Segmentation:
    annotation: List[int]
    label: str
    area: int


@dataclass
class Annotations:
    segmentation: Optional[List[Segmentation]]
    bbox: Optional[BBox] = None


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
    annotations: Optional[Annotations] = None
