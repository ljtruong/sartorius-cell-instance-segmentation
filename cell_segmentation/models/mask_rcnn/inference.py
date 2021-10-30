import os
import random

import cv2
import matplotlib.pyplot as plt
from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor
from detectron2.evaluation import COCOEvaluator, inference_on_dataset
from detectron2.utils.visualizer import Visualizer
from cell_segmentation.data.loader import Loader
from cell_segmentation.data.converter.satorus2coco import Satorus2COCO
from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.utils.visualizer import ColorMode


def main():

    data_loader = Loader()
    satorus_converter = Satorus2COCO()
    df = data_loader.load_static_dataset(filepath="data/train.csv")
    df = data_loader.preprocess_static_dataset(df)

    validation_dataset = data_loader.build_microscopyimage_from_dataframe(df)
    satorus_converter.register_instances("test", validation_dataset)

    # TODO split train and val
    df = df[:10]

    cfg = get_cfg()
    config_path = os.path.join(
        os.getcwd(),
        "cell_segmentation/models/mask_rcnn/configs/mask_rcnn_R_50_FPN_3x.yaml",
    )

    cfg.merge_from_file(config_path)

    cfg.DATASETS.TEST = ("test",)

    cfg.MODEL.WEIGHTS = "/home/letruong/projects/sartorius-cell-instance-segmentation-d2/output/model_final.pth"
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.001  # set a custom
    cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_final.pth")
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = (
        0.5  # set a custom testing threshold for this model
    )
    predictor = DefaultPredictor(cfg)

    dataset_dicts_val = DatasetCatalog.get("test")
    metadata_dicts = MetadataCatalog.get("test")

    fig, ax = plt.subplots(4, 1, figsize=(20, 50))
    indices = [ax[0], ax[1], ax[2], ax[3]]
    i = -1
    for d in random.sample(dataset_dicts_val, 4):
        i = i + 1
        im = cv2.imread(d["file_name"])
        outputs = predictor(im)
        v = Visualizer(
            im[:, :, ::-1],
            metadata=metadata_dicts,
            scale=1,
            instance_mode=ColorMode.IMAGE_BW,  # remove the colors of unsegmented pixels. This option is only available for segmentation models
        )
        out = v.draw_instance_predictions(outputs["instances"].to("cpu"))
        indices[i].grid(False)
        indices[i].imshow(out.get_image()[:, :, ::-1])

        fig.savefig("test.png")


if __name__ == "__main__":
    main()
