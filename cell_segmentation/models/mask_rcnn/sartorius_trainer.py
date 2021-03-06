import copy
from typing import Dict

import detectron2.data.transforms as T
import torch
from detectron2.data import (
    DatasetMapper,
    build_detection_test_loader,
    build_detection_train_loader,
)
from detectron2.data import detection_utils as utils
from detectron2.engine import DefaultTrainer

from cell_segmentation.models.mask_rcnn.default_configs.config import CfgNode


class TransformsMapper:
    """
    This implements a transformation class for feature engineering or data augmentation of the dataset.
    """

    def __init__(self, cfg: CfgNode, is_train: bool = True):
        self.cfg = cfg
        self.is_train = is_train

        # TODO configurable transforms
        self.transform_list = [
            T.RandomFlip(prob=0, horizontal=True, vertical=False),
        ]

    def __call__(self, dataset_dict: Dict) -> Dict:
        dataset_dict = copy.deepcopy(dataset_dict)
        image = utils.read_image(dataset_dict["file_name"], format="BGR")
        image, transforms = T.apply_transform_gens(self.transform_list, image)
        dataset_dict["image"] = torch.as_tensor(
            image.transpose(2, 0, 1).astype("float32")
        )

        annos = [
            utils.transform_instance_annotations(obj, transforms, image.shape[:2])
            for obj in dataset_dict.pop("annotations")
            if obj.get("iscrowd", 0) == 0
        ]
        instances = utils.annotations_to_instances(annos, image.shape[:2])
        dataset_dict["instances"] = utils.filter_empty_instances(instances)
        return dataset_dict


class SartoriusTrainer(DefaultTrainer):
    """
    This implements a SartorusTrainer for cell segmentation.
    """

    @classmethod
    def build_train_loader(self, cfg: CfgNode):
        """
        Augment the dataset with the transforms and build the train loader.
        """
        mapper = TransformsMapper(cfg)
        return build_detection_train_loader(cfg, mapper=mapper)

    # @classmethod
    # def build_test_loader(cls, cfg, dataset_name):
    #     """
    #     augment the test split too
    #     """
    #     mapper = TransformsMapper(cfg, is_train=False)
    #     return build_detection_test_loader(cfg, dataset_name, mapper=mapper)
