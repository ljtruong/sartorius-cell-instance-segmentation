from detectron2.data import build_detection_test_loader
from detectron2.evaluation import COCOEvaluator, inference_on_dataset
import click

from cell_segmentation.data.converter.satorius2coco import Satorius2COCO
from cell_segmentation.data.datasets import Datasets
from cell_segmentation.data.loader import Loader
from cell_segmentation.models.mask_rcnn.sartorius_trainer import SartoriusTrainer
from cell_segmentation.utils.configs import load_config


def load_dataset(cfg):
    data_loader = Loader()
    satorus_converter = Satorius2COCO()
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
    load_dataset()

    evaluator = COCOEvaluator(
        cfg.DATASETS.TEST[0], cfg, False, output_dir=cfg.OUTPUT_DIR
    )
    val_loader = build_detection_test_loader(cfg, cfg.DATASETS.TEST)
    trainer = SartoriusTrainer(cfg)
    trainer.resume_or_load(resume=resume)

    # puts the model in evaluation model
    results = inference_on_dataset(trainer.model, val_loader, evaluator)


@click.command()
@click.option(
    "--config",
    type=str,
    default="experiments/mask_rcnn/configs/inference.yaml",
)
@click.option(
    "--resume",
    type=bool,
    default=False,
)
def evaluate(config: str, resume: bool):
    main(config, resume)


if __name__ == "__main__":
    evaluate()
