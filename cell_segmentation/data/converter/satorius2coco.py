import json
import multiprocessing as mp
import os
from typing import Dict, List

from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.structures import BoxMode

from cell_segmentation.data.models import MicroscopyImage


class Satorius2COCO:
    """
    This implements Satorus convertor class to transform Satorus dataset to COCO format.
    """

    def __init__(self):
        self._string2int = self.load_class_mapping(
            "models/mask_rcnn/default_configs/class_mapping.json"
        )

    def load_class_mapping(self, filepath: str) -> Dict:
        """
        Parameters:
            filepath (str): path to class mapping json file

        Returns:
            class_mapping (dict): dictionary of class mapping
        """

        relative_filepath = os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
            filepath,
        )
        with open(relative_filepath, "r") as f:
            class_mapping = json.load(f)
        return class_mapping

    def _record_structure(self) -> Dict:
        return {
            "file_name": "",
            "image_id": "",
            "height": 0,
            "width": 0,
            "annotations": [],
        }

    def _ann_structure(self) -> Dict:
        return {
            "bbox": [],
            "bbox_mode": BoxMode.XYWH_ABS,
            "segmentation": [],
            "area": 0,
            "category_id": 0,
        }

    def _get_annotations(self, data: List[Dict]) -> Dict:
        """
        Convert data model to COCO annotation format.

        Parameters:
            data (List[Dict]): Segmentation annotations

        Returns:
            annotations (Dict): COCO annotation format
        """
        objs = []
        for ann in data.annotations:
            obj = self._ann_structure()
            obj["bbox"] = ann.bbox.to_row()
            obj["segmentation"] = ann.segmentation.annotation
            obj["category_id"] = self._string2int[ann.segmentation.label]
            obj["area"] = ann.segmentation.area
            objs.append(obj)
        return objs

    def get_record(self, data: MicroscopyImage) -> Dict:
        """
        Convert MicroscopyImage model to COCO format.

        Parameters:
            data (MicroscopyImage): MicroscopyImage model

        Returns:
            record (Dict): COCO format
        """
        record = self._record_structure()
        record["file_name"] = data.image_path
        record["image_id"] = data.file_id
        record["width"] = data.width
        record["height"] = data.height
        record["annotations"] = self._get_annotations(data)
        return record

    def get_dataset(
        self, data: List[MicroscopyImage], num_processes: int = os.cpu_count()
    ) -> List[Dict]:
        """
        Parameters:
            data (List[MicroscopyImage]): MicroscopyImage model
            num_processes (int): number of processes

        Returns:
            dataset (List[Dict]): COCO format
        """
        with mp.Pool(num_processes) as pool:
            return pool.starmap(self.get_record, zip(data))

    def register_instances(self, name: str, data: List[Dict]) -> None:
        """
        Register dataset in detectron data catalog.

        Parameters:
            name (str): name of dataset
            data (List[Dict]): COCO format
        """
        DatasetCatalog.register(name, lambda: self.get_dataset(data))
        MetadataCatalog.get(name).set(thing_classes=list(self._string2int.keys()))
