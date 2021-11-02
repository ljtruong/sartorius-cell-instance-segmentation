from os import stat
import pandas as pd


class GenerateStatistics:
    @staticmethod
    def get_statistics(
        df: pd.DataFrame,
        numeric_columns=[],
        categorical_columns=[],
    ):
        """
        Generate statistics for the given dataframe.

        Parameters
        ----------
        df : pd.DataFrame
            The dataframe to generate statistics for.

        Returns
        -------
        pd.DataFrame
            The dataframe with statistics.
        """
        # Get the statistics
        numeric_statistics = df[numeric_columns].describe()
        categorical_statistics = df[categorical_columns].value_counts()

        # Return the statistics
        return categorical_statistics, numeric_statistics
