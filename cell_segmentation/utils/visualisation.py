import os
from typing import Dict

import cv2
import numpy as np
from detectron2.utils.visualizer import ColorMode, Visualizer
from pycocotools import mask as maskUtils


def visualise_predictions(predictor, image_path: str, classes: Dict):
    """
    Visualise the predictions of a predictor on a single image.
    """
    image_path = os.path.join(os.path.dirname(__file__), "..", "..", image_path)
    im = cv2.imread(image_path)
    outputs = predictor(im)
    v = Visualizer(
        im[:, :, ::-1],
        metadata=classes,
        scale=1,
        instance_mode=ColorMode.IMAGE_BW,  # remove the colors of unsegmented pixels. This option is only available for segmentation models
    )
    out = v.draw_instance_predictions(outputs["instances"].to("cpu"))
    return out.get_image()[:, :, ::-1]


def visualise_coco_mask(
    annotations: Dict, height: int, width: int, channels: int
) -> np.ndarray:
    """
    visualise a coco annotation as a mask.

    Parameters:
    -----------
        annotation: Dict
            A dictionary containing the annotations of a single image.
        height: int
            The height of the image.
        width: int
            The width of the image.
        channels: int
            The number of channels of the image.

    Returns:
    --------
        mask: np.ndarray
    """
    mask = np.zeros((height, width, channels), dtype=np.float32)

    for i in range(len(annotations)):
        segm = annotations[i]["segmentation"]
        rles = maskUtils.frPyObjects(segm, height, width)
        rle = maskUtils.merge(rles)

        segment = maskUtils.decode(rle)
        segment = segment[:, :, np.newaxis]
        mask[np.where(segment == 1)] = segment[np.where(segment == 1)]

    return mask
