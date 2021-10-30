import unittest

import pandas as pd
import numpy as np

from cell_segmentation.data.datasets import Datasets


class TestDataset(unittest.TestCase):
    def test_generate_examples(self):
        """
        Test the generate_examples function of the Datasets class.
        """

        df = pd.DataFrame(np.random.random(100))
        train_df, test_df = Datasets.generate_examples(df)

        self.assertEqual(len(train_df), 80)
        self.assertEqual(len(test_df), 20)


if __name__ == "__main__":
    unittest.main()
