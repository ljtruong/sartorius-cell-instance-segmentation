import os

from detectron2.config import get_cfg
import click

from cell_segmentation.data.converter.satorus2coco import Satorus2COCO
from cell_segmentation.data.datasets import Datasets
from cell_segmentation.data.loader import Loader
from cell_segmentation.models.mask_rcnn.sartorius_trainer import SartoriusTrainer
from cell_segmentation.utils.configs import load_config


def load_dataset(cfg):
    data_loader = Loader()
    satorus_converter = Satorus2COCO()
    df = data_loader.load_static_dataset(filepath=cfg.DATASETS.TRAIN_STATIC_FILE)
    df = data_loader.preprocess_static_dataset(df)

    if cfg.DATASETS.TRAIN_STATIC_FILE_ROWS:
        df = df[: cfg.DATASETS.TRAIN_STATIC_FILE_ROWS]  # Debug purposes

    train_df, val_df = Datasets.generate_examples(
        df, train_split=cfg.DATASETS.TRAIN_SPLIT, test_split=cfg.DATASETS.TEST_SPLIT
    )
    train_dataset = data_loader.build_microscopyimage_from_dataframe(train_df)
    validation_dataset = data_loader.build_microscopyimage_from_dataframe(val_df)

    satorus_converter.register_instances("train", train_dataset)
    satorus_converter.register_instances("validation", validation_dataset)


def main(config: str, resume: bool):

    cfg = load_config(config)
    load_dataset(cfg)
    trainer = SartoriusTrainer(cfg)
    trainer.resume_or_load(resume=resume)
    trainer.train()


@click.command()
@click.option(
    "--config",
    type=str,
    default="cell_segmentation/models/mask_rcnn/default_configs/mask_rcnn_R_50_FPN_3x.yaml",
)
@click.option(
    "--resume",
    type=bool,
    default=False,
)
def train(config: str, resume: bool):
    main(config, resume)


if __name__ == "__main__":
    train()
