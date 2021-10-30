from detectron2.engine import DefaultTrainer
from detectron2.data import (
    DatasetMapper,
    build_detection_test_loader,
    build_detection_train_loader,
)


class TransformsMapper:
    pass


class SartoriusTrainer(DefaultTrainer):
    pass
    # TODO add custom transforms
    # @classmethod
    # def build_train_loader(self, cfg):
    #     """ """
    #     mapper = TransformsMapper(cfg, is_train=True)
    #     return build_detection_train_loader(cfg, mapper=mapper)

    # @classmethod
    # def build_test_loader(cls, cfg, dataset_name):
    #     """
    #     augment the test split too
    #     """
    #     mapper = TransformsMapper(cfg, is_train=False)
    #     return build_detection_test_loader(cfg, dataset_name, mapper=mapper)
