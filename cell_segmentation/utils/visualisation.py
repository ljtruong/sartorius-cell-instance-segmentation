import os
from typing import Dict

import cv2
from detectron2.utils.visualizer import ColorMode, Visualizer


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
