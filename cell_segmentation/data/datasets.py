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

        if shuffle:
            df.sample(frac=1, random_state=seed).reset_index(drop=True)

        train, test, _ = np.split(
            df,
            [int(len(df) * train_split), int(len(df) * (test_split + train_split))],
        )

        return train, test
