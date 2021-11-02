from dataclasses import dataclass
from typing import List, Optional

import numpy as np
import cv2


@dataclass
class BBox:
    """
    Bounding box data model.

    Attributes:
        xmin: x coordinate of the top left corner of the bounding box.
        ymin: y coordinate of the top left corner of the bounding box.
        width: width of the bounding box.
        height: height of the bounding box.
    """

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
    """
    Image data model

    Attributes:
        image_path: Path to image file.
        width: Image width.
        height: Image height.
    """

    image_path: str
    height: int
    width: int

    def array(self) -> np.ndarray:
        """
        Read image as numpy array.
        """
        return cv2.imread(self.image_path)


@dataclass
class Segmentation:
    """
    Segmentation data model.

    Attributes:
        annotation: List of segmentation polygons.
        label: Label of the segmented cell type.
        area: Area of the segmentation.

    """

    annotation: List[int]
    label: str
    area: int


@dataclass
class Annotations:
    """
    Annotations data model.

    Attributes:
        segmentations: List of segmentations.
        bbox: Bounding box.
    """

    segmentation: Optional[List[Segmentation]]
    bbox: Optional[BBox] = None


@dataclass
class MicroscopyImage(Image):
    """
    Microscopy image data model

    Attributes:
        file_id: unique file name of image
        annotations: list of Segmentation annotations
    """

    file_id: str
    annotations: Optional[Annotations] = None
