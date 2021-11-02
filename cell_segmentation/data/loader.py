import os
from typing import List
import multiprocessing as mp

import pandas as pd

from cell_segmentation.utils.rle_parser import (
    get_mask_rle,
    decode_rle,
    validate_segmentation_length,
    validate_bbox,
    get_segmentation_area,
)
from cell_segmentation.data.models import (
    MicroscopyImage,
    Segmentation,
    Annotations,
    BBox,
)


class Loader:
    """
    This implements the data loader for the satorus dataset.

    Usage:
    ----------
    from cell_segmentation.data.loader import Loader
    data_loader = Loader(cfg)
    train_df = data_loader.load_static_dataset("train.csv")
    train_df = data_loader.preprocess_static_dataset(train_df)
    train_df = data_loader.build_microscopyimage_from_dataframe(train_df)

    """

    def __init__(self, cfg):
        self.cfg = cfg

        if os.path.isabs(cfg.DATASETS.DATA_DIR):
            self.DATA_DIR = cfg.DATASETS.DATA_DIR
        else:
            self.DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "data")

    def load_static_dataset(self, filepath: str) -> pd.DataFrame:
        """
        Load a static dataset from a csv file in the root data directory folder.

        Parameters:
        -----------
            filepath: str
                The path to the csv file.

        Returns:
        -----------
            pd.DataFrame: The dataframe containing the static dataset.
        """
        df = pd.read_csv(os.path.join(self.DATA_DIR, filepath))
        return df

    def preprocess_static_dataset(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Preprcess the static dataset.

        Parameters:
        -----------
            df: pd.DataFrame
                The dataframe containing the static dataset.

        Returns:
        -----------
            pd.DataFrame: The preprocessed dataframe.
        """
        df["image_path"] = self.DATA_DIR + "/train/" + df["id"] + ".png"
        return df

    def _build_segmentation_annotations(
        self, segmentation: List[str], label: str, height: int, width: int
    ) -> Segmentation:
        """
        Build segmentation annotations from a dataframe into Segmentation data model.

        Parameters:
        -----------
            segmentation: List[str]
                the segmentation annotations.
            label: str
                the label of the cell type.
            height: int
                the height of the image.
            width: int
                the width of the image.

        Returns:
        -----------
            Segmentation: The segmentation annotations.
        """
        return Segmentation(
            annotation=segmentation,
            label=label,
            area=get_segmentation_area(segmentation, height, width),
        )

    def _build_bbox_annotations(self, bbox: List[int]) -> BBox:
        """
        Build bbox annotations from a dataframe into BBox data model.

        Parameters:
        ----------
            bbox (List[int]): The bounding box coordinates.

        Returns:
        ----------
            BBox: The bbox annotations.
        """
        return BBox(
            xmin=bbox[0],
            ymin=bbox[1],
            width=bbox[2],
            height=bbox[3],
        )

    def build_single_annotation(
        self,
        annotation: List[str],
        cell_type: str,
        height: int,
        width: int,
    ) -> Annotations:
        """
        Build a single annotation into Annotations data model.

        Parameters:
        ----------
            annotation: List[str]
                the segmentation annotations.
            cell_type: str
                the label of the cell type.
            height: int
                the height of the image.
            width: int
                the width of the image.

        Returns:
        ----------
            Annotations: The annotation storing segmentation and bbox.
        """

        segmentation, bbox = decode_rle(get_mask_rle(annotation, height, width))
        if validate_segmentation_length(segmentation) and validate_bbox(
            bbox, height, width
        ):
            return Annotations(
                segmentation=self._build_segmentation_annotations(
                    segmentation=segmentation,
                    label=cell_type,
                    height=height,
                    width=width,
                ),
                bbox=self._build_bbox_annotations(bbox=bbox),
            )

    def _build_annotations(
        self, annotations: pd.DataFrame, num_processes=os.cpu_count()
    ) -> List[Annotations]:
        """
        Build annotations from a dataframe into Annotations data model.

        Parameters:
        ----------
            annotations (pd.DataFrame): The dataframe containing the annotations.

        Returns:
        ----------
            List[Annotations]: The annotations.
        """
        with mp.Pool(num_processes) as pool:
            return pool.starmap(
                self.build_single_annotation,
                annotations.itertuples(name=None, index=False),
            )

    def build_microscopyimage_from_dataframe(
        self, df: pd.DataFrame, num_processes: int = os.cpu_count()
    ) -> List[MicroscopyImage]:
        """
        Build microscopy images from a dataframe into MicroscopyImage data model.

        Parameters:
            df (pd.DataFrame): The dataframe containing the microscopy image annotations.

        Returns:
            List[MicroscopyImage]: The microscopy image data model.
        """

        microscopyimages_list = []
        grouped = df.groupby("id")
        for image_id, subset in grouped:
            microscopyimage = MicroscopyImage(
                file_id=image_id,
                annotations=self._build_annotations(
                    subset[["annotation", "cell_type", "height", "width"]],
                    num_processes=num_processes,
                ),
                image_path=subset["image_path"].values[0],
                height=subset["height"].values[0],
                width=subset["width"].values[0],
            )
            microscopyimages_list.append(microscopyimage)

        return microscopyimages_list
