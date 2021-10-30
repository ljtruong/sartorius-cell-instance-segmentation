from typing import DefaultDict
from detectron2.engine import DefaultTrainer
from detectron2.config import get_cfg
from detectron2 import model_zoo


class SartoriusTrainer(DefaultTrainer):
    def load_config(self, config_path: str) -> None:
        """
        Load a config file and merge it with the default config.
        """
        cfg = get_cfg()
        cfg.merge_from_file(config_path)
        cfg.freeze()
        self.cfg = cfg
