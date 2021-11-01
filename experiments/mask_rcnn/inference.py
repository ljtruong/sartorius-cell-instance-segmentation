import os
from typing import List

import click
import cv2
import numpy as np
import torch
import pandas as pd
from detectron2.engine import DefaultPredictor

from cell_segmentation.data.models import MicroscopyImage
from cell_segmentation.models.mask_rcnn.default_configs.config import CfgNode
from cell_segmentation.utils.configs import load_config

# TODO revise these constants
THRESHOLDS = [0.15, 0.35, 0.55]
MIN_PIXELS = [75, 150, 75]

# From https://www.kaggle.com/stainsby/fast-tested-rle
def rle_encode(img):
    """
    img: numpy array, 1 - mask, 0 - background
    Returns run length as string formated
    """
    pixels = img.flatten()
    pixels = np.concatenate([[0], pixels, [0]])
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    runs[1::2] -= runs[::2]
    return " ".join(str(x) for x in runs)


def get_masks(image_path: str, predictor):
    im = cv2.imread(image_path)
    pred = predictor(im)
    pred_class = torch.mode(pred["instances"].pred_classes)[0]
    take = pred["instances"].scores >= THRESHOLDS[pred_class]
    pred_masks = pred["instances"].pred_masks[take]
    pred_masks = pred_masks.cpu().numpy()
    res = []
    used = np.zeros(im.shape[:2], dtype=int)
    for mask in pred_masks:
        mask = mask * (1 - used)
        if mask.sum() >= MIN_PIXELS[pred_class]:  # skip predictions with small area
            used += mask
            res.append(rle_encode(mask))
    return res


def load_data(cfg: CfgNode) -> List[MicroscopyImage]:

    microscopy_images = []
    directory = os.path.join(cfg.DATASETS.TEST_DIR)
    for dirs, _, files in os.walk(directory):
        for file in files:
            if file.endswith(".png"):
                image_path = os.path.join(dirs, file)
                file_id = file.split(".")[0]

                microscopy_image = MicroscopyImage(
                    file_id=file_id,
                    image_path=image_path,
                    height=None,
                    width=None,
                    array=None,
                )

                microscopy_images.append(microscopy_image)
    return microscopy_images


def load_model(cfg: CfgNode):
    return DefaultPredictor(cfg)


def main(config: str):
    cfg = load_config(config)
    cfg.MODEL.WEIGHTS = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "..", "..", cfg.MODEL.WEIGHTS
    )
    microscopy_images = load_data(cfg)
    predictor = load_model(cfg)

    submission = pd.DataFrame(columns=["ids", "predicted"])
    for microscopy_image in microscopy_images:
        encoded_masks = get_masks(microscopy_image.image_path, predictor)
        for enc in encoded_masks:
            submission.loc[len(submission)] = [microscopy_image.file_id, enc]

    submission.to_csv("submission.csv", index=False)


@click.command()
@click.option(
    "--config",
    type=str,
    default="experiments/mask_rcnn/configs/inference.yaml",
)
def inference(config: str):
    main(config)


if __name__ == "__main__":
    inference()
