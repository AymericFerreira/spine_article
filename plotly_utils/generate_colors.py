from typing import Any
import numpy as np


def create_color_dict(subpop_df: dict[str, Any]) -> dict[str, Any]:
    """
    Determine a widespread range of color for x population and y cluster in each population based on HSV color scheme.
    h : corresponds to the number of populations
    s : corresponds to the number of clusters in this population

    Parameters :
     subpop_df: a list of Pandas DataFrame, each dataframe is an independent dataframe focus on one population

    Return :
     color_dict: a color dictionary with the key a combination of population and the cluster and the hsv color as value
    """
    color_dict = {}
    for i, (key, df) in enumerate(subpop_df.items()):
        print(subpop_df.keys())
        for j, cluster in enumerate(np.unique(df["Clusters"])):
            color_dict[f"{key} {cluster}"] = \
                f"hsv({(i + 1) * 360 / len(subpop_df.items())},100%,{(j + 1) * 100 / len(np.unique(df['Clusters']))}%)"
    return color_dict
