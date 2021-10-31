from detectron2.data import build_detection_test_loader
from detectron2.evaluation import COCOEvaluator, inference_on_dataset
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

    df = df[:1000]

    train_df, val_df = Datasets.generate_examples(df, train_split=0.8, test_split=0.2)
    train_dataset = data_loader.build_microscopyimage_from_dataframe(train_df)
    validation_dataset = data_loader.build_microscopyimage_from_dataframe(val_df)

    satorus_converter.register_instances("train", train_dataset)
    satorus_converter.register_instances("validation", validation_dataset)


def main(config):
    cfg = load_config(config)
    load_dataset()

    evaluator = COCOEvaluator(cfg.DATASETS.TEST[0], cfg, False, output_dir="./output/")
    val_loader = build_detection_test_loader(cfg, cfg.DATASETS.TEST)
    trainer = SartoriusTrainer(cfg)
    trainer.resume_or_load(resume=False)

    # puts the model in evaluation model
    results = inference_on_dataset(trainer.model, val_loader, evaluator)


@click.command()
@click.option(
    "--config",
    type=str,
    default="experiments/mask_rcnn/configs/inference.yaml",
)
def evaluate(config: str):
    main(config)


if __name__ == "__main__":
    evaluate()
