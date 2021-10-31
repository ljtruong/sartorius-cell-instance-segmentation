import json
import multiprocessing as mp
import os
from typing import Dict, List, Tuple

import numpy as np
from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.structures import BoxMode
from pycocotools import mask as mutils
from skimage import measure

from cell_segmentation.data.models import MicroscopyImage


class Satorus2COCO:
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
            "category_id": 0,
        }

    def rle_decode(
        self, mask_rle: List[int], shape: Tuple[int, int, int], color: int = 1
    ) -> np.ndarray:
        """
        Convert RLE encoded mask to numpy array.

        source: https://www.kaggle.com/ammarnassanalhajali/sartorius-segmentation-detectron2-training#Loading-Dataset

        Parameters:
            mask_rle (List[int]): RLE encoded pixels
            shape (Tuple[int, int, int]): shape of image
            color (int): color of mask

        Returns:
            mask (np.ndarray): pixel segmentation mask
        """
        s = mask_rle.split()

        starts = list(map(lambda x: int(x) - 1, s[0::2]))
        lengths = list(map(int, s[1::2]))

        ends = [x + y for x, y in zip(starts, lengths)]
        img = np.zeros((shape[0] * shape[1], shape[2]), dtype=np.float32)

        for start, end in zip(starts, ends):
            img[start:end] = color

        return img.reshape(shape)

    def get_mask_rle(
        self, ann: List[int], image_height: int, image_width: int, channels: int = 1
    ) -> np.ndarray:
        """
        Convert annotation run length encoded(RLE) pixels to pixel mask.

        Run-length encoding(RLE) is a form of lossless data compression.

        source: https://www.kaggle.com/ammarnassanalhajali/sartorius-segmentation-detectron2-training#Loading-Dataset

        Parameters:
            ann (List[int]): RLE encoded pixels
            image_height (int): height of image
            image_width (int): width of image
            channels (int): number of channels

        Returns:
            mask (np.ndarray): pixel segmentation mask
        """
        decoded_mask = self.rle_decode(
            ann, shape=(image_height, image_width, channels), color=1
        )
        mask = decoded_mask[:, :, 0]
        mask = np.array(mask, dtype=np.uint8)
        return mask

    def decode_rle(self, mask_rle: np.ndarray):
        """
        Convert RLE encoded mask to numpy array and extract bounding box of mask.

        source: https://www.kaggle.com/ammarnassanalhajali/sartorius-segmentation-detectron2-training#Loading-Dataset

        Parameters:
            mask_rle (List[int]): RLE encoded pixels

        Returns:
            mask (np.ndarray): pixel segmentation mask
            bbox (List[int]): bounding box of mask
        """
        ground_truth_binary_mask = mask_rle
        fortran_ground_truth_binary_mask = np.asfortranarray(ground_truth_binary_mask)
        encoded_ground_truth = mutils.encode(fortran_ground_truth_binary_mask)
        ground_truth_bounding_box = mutils.toBbox(encoded_ground_truth)
        contours = measure.find_contours(ground_truth_binary_mask, 0.5)

        segmentations = []
        for contour in contours:
            contour = np.flip(contour, axis=1)
            segmentations.append(contour.ravel().tolist())

        return segmentations, ground_truth_bounding_box.tolist()

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
            segmenation, bbox = self.decode_rle(
                self.get_mask_rle(ann.annotation, data.height, data.width)
            )
            obj = self._ann_structure()
            obj["bbox"] = bbox
            obj["segmentation"] = segmenation
            obj["category_id"] = self._string2int[ann.label]
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
