from cell_segmentation.models.mask_rcnn.sartorius_trainer import SartoriusTrainer
from detectron2.config import get_cfg
from cell_segmentation.data.loader import Loader
from cell_segmentation.data.converter.satorus2coco import Satorus2COCO
import os


def main():

    cfg = get_cfg()

    # trainer.load_config(
    #     config_path="cell_segmentation/models/mask_rcnn/configs/sartorius/sartorius_mask_rcnn_R_50_FPN_3x.yaml"
    # )
    config_path = os.path.join(
        os.getcwd(),
        "cell_segmentation/models/mask_rcnn/configs/mask_rcnn_R_50_FPN_3x.yaml",
    )
    cfg.merge_from_file(config_path)

    trainer = SartoriusTrainer(cfg)
    trainer.resume_or_load(resume=False)
    trainer.train()


if __name__ == "__main__":

    data_loader = Loader()
    satorus_converter = Satorus2COCO()
    df = data_loader.load_static_dataset(filepath="data/train.csv")
    df = data_loader.preprocess_static_dataset(df)

    # TODO split train and val
    df = df[:1000]
    train_dataset = data_loader.build_microscopyimage_from_dataframe(df)
    satorus_converter.register_instances("train", train_dataset)

    main()
    print("here")
