import pandas as pd
import numpy as np
from typing import Union


def normalise_data(data: Union[pd.Series, np.array, list[float]],
                   max_value: float = None, min_value: float = None) -> pd.Series:
    """
        Normalises the given data to a range between specified minimum and maximum values.

        This function normalises a series, array, or list of numerical values to a scale between the provided
        maximum and minimum values. If the max or min values are not provided, the function uses the maximum and
        minimum values from the data itself. The normalisation formula used is (data - min) / (max - min).

        Parameters:
        data (Union[pd.Series, np.array, list[float]]): The data to be normalised. Can be a pandas Series,
                                                        numpy array, or a list of floats.
        max_value (float, optional): The maximum value for normalisation. Defaults to the max value in the data
                                     if not provided.
        min_value (float, optional): The minimum value for normalisation. Defaults to the min value in the data
                                     if not provided.

        Returns:
        pd.Series: A pandas Series containing the normalised data.
    """
    data = pd.Series(data)
    if max_value is None:
        max_value = data.max()
    if min_value is None:
        min_value = data.min()
    return (data - min_value) / (max_value - min_value)


if __name__ == '__main__':
    data_ = pd.Series([1, 2, 3, 4, 5])
    print(normalise_data(data_))
