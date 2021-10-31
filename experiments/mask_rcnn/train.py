import os

from detectron2.config import get_cfg
import click

from cell_segmentation.data.converter.satorus2coco import Satorus2COCO
from cell_segmentation.data.datasets import Datasets
from cell_segmentation.data.loader import Loader
from cell_segmentation.models.mask_rcnn.sartorius_trainer import SartoriusTrainer
from cell_segmentation.utils.configs import load_config


def load_dataset():
    data_loader = Loader()
    satorus_converter = Satorus2COCO()
    df = data_loader.load_static_dataset(filepath="data/train.csv")
    df = data_loader.preprocess_static_dataset(df)

    df = df[:1000]  # Temporary

    train_df, val_df = Datasets.generate_examples(df, train_split=0.8, test_split=0.2)
    train_dataset = data_loader.build_microscopyimage_from_dataframe(train_df)
    validation_dataset = data_loader.build_microscopyimage_from_dataframe(val_df)

    satorus_converter.register_instances("train", train_dataset)
    satorus_converter.register_instances("validation", validation_dataset)


def main(config: str):

    cfg = load_config(config)
    load_dataset()
    trainer = SartoriusTrainer(cfg)
    trainer.resume_or_load(resume=False)
    trainer.train()


@click.command()
@click.option(
    "--config",
    type=str,
    default="cell_segmentation/models/mask_rcnn/default_configs/mask_rcnn_R_50_FPN_3x.yaml",
)
def train(config: str):
    main(config)


if __name__ == "__main__":
    train()
