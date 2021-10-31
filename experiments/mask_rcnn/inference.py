import click
from detectron2.engine import DefaultPredictor

from cell_segmentation.utils.configs import load_config


def load_model(cfg):
    return DefaultPredictor(cfg)


def main(config):
    cfg = load_config(config)
    predictor = load_model(cfg)


@click.command()
@click.option(
    "--config",
    type=str,
    default="cell_segmentation/models/mask_rcnn/default_configs/mask_rcnn_R_50_FPN_3x.yaml",
)
def inference(config: str):
    main(config)


if __name__ == "__main__":
    inference()
