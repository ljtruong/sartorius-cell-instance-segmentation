import os
from typing import List

import pandas as pd

from cell_segmentation.data.models import MicroscopyImage, Segmentation


class Loader:
    """
    This implements the data loader for the satorus dataset.
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
            filepath (str): The path to the csv file.

        Returns:
            pd.DataFrame: The dataframe containing the static dataset.
        """
        df = pd.read_csv(os.path.join(self.DATA_DIR, filepath))
        return df

    def preprocess_static_dataset(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Preprcess the static dataset.

        Parameters:
            df (pd.DataFrame): The dataframe containing the static dataset.

        Returns:
            pd.DataFrame: The preprocessed dataframe.
        """
        df["image_path"] = self.DATA_DIR + "/train/" + df["id"] + ".png"
        return df

    def _build_segmentation_annotations(
        self, annotations: pd.DataFrame
    ) -> Segmentation:
        """
        Build segmentation annotations from a dataframe into Segmentation data model.

        Parameters:
            annotations (pd.DataFrame): The dataframe containing the annotations.

        Returns:
            Segmentation: The segmentation annotations.
        """
        segmentation_annotations = []
        for ix, row in enumerate(annotations.itertuples()):
            segmentation_annotations.append(
                Segmentation(annotation=row.annotation, label=row.cell_type)
            )
        return segmentation_annotations

    def build_microscopyimage_from_dataframe(
        self, df: pd.DataFrame
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
                annotations=self._build_segmentation_annotations(
                    subset[["annotation", "cell_type"]]
                ),
                image_path=subset["image_path"].values[0],
                mask_path=None,
                array=None,
                height=subset["height"].values[0],
                width=subset["width"].values[0],
            )
            microscopyimages_list.append(microscopyimage)

        return microscopyimages_list
