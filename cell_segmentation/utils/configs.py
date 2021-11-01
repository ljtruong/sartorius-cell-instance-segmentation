import os

from cell_segmentation.models.mask_rcnn.default_configs.config import get_cfg


def load_config(config):
    cfg = get_cfg()
    config_path = os.path.join(
        os.path.dirname(__file__),
        "..",
        "..",
        config,
    )
    cfg.merge_from_file(config_path)
    return cfg
