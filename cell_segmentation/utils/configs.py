import os
from detectron2.config import get_cfg


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
