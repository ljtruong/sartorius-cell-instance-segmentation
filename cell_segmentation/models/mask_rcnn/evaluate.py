from detectron2.evaluation import COCOEvaluator
from detectron2.data import DatasetCatalog, MetadataCatalog, build_detection_test_loader
from detectron2.evaluation import COCOEvaluator, inference_on_dataset
from detectron2.config import get_cfg
from cell_segmentation.data.loader import Loader
from cell_segmentation.data.converter.satorus2coco import Satorus2COCO
from cell_segmentation.models.mask_rcnn.sartorius_trainer import SartoriusTrainer
import os


def main():

    data_loader = Loader()
    satorus_converter = Satorus2COCO()
    df = data_loader.load_static_dataset(filepath="data/train.csv")
    df = data_loader.preprocess_static_dataset(df)

    # TODO split train and val
    df = df[:1000]
    config_path = os.path.join(
        os.getcwd(),
        "cell_segmentation/models/mask_rcnn/configs/mask_rcnn_R_50_FPN_3x.yaml",
    )

    cfg = get_cfg()
    cfg.merge_from_file(config_path)
    cfg.DATASETS.TEST = ("validation",)

    validation_dataset = data_loader.build_microscopyimage_from_dataframe(df)

    satorus_converter.register_instances("train", validation_dataset)
    satorus_converter.register_instances("validation", validation_dataset)

    evaluator = COCOEvaluator("validation", cfg, False, output_dir="./output/")
    cfg.MODEL.WEIGHTS = "/home/letruong/projects/sartorius-cell-instance-segmentation-d2/output/model_final.pth"
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.001  # set a custom

    val_loader = build_detection_test_loader(cfg, cfg.DATASETS.TEST)

    # TODO load model in evalu mode

    trainer = SartoriusTrainer(cfg)
    trainer.resume_or_load(resume=False)
    results = inference_on_dataset(trainer.model, val_loader, evaluator)


if __name__ == "__main__":
    main()
