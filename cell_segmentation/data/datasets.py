from typing import Tuple

import numpy as np
import pandas as pd


class Datasets:
    @staticmethod
    def generate_examples(
        df: pd.DataFrame,
        seed: int = 42,
        shuffle: bool = True,
        train_split: float = 0.8,
        test_split: float = 0.2,
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Generate train and test examples from a dataframe.

        Parameters:
        -----------
            df: pd.DataFrame
                Dataframe containing the data.
            seed: int
                Seed for the random number generator.
            shuffle: bool
                Whether to shuffle the data.
            train_split: float
                Percentage of the data to be used for training.
            test_split: float
                Percentage of the data to be used for testing.

        Returns:
        --------
            train_df: pd.DataFrame
                Dataframe containing the training examples.
            test_df: pd.DataFrame
                Dataframe containing the test examples.
        """

        if shuffle:
            df.sample(frac=1, random_state=seed).reset_index(drop=True)

        train, test, _ = np.split(
            df,
            [int(len(df) * train_split), int(len(df) * (test_split + train_split))],
        )

        return train, test
