from typing import Any

import pandas as pd

from plotly_utils.themes import general_fig_layout
from plotly_utils.generate_colors import create_color_dict
import numpy as np
import plotly.graph_objects as go
import plotly.colors
from analysis_utils.normalisation import normalise_data


def generate_subfeatures_df(df: pd.DataFrame, col: str):
    """
    Generates a dictionary of sub-DataFrames based on unique values in a specified column.

    Parameters:
    df (pd.DataFrame): The original DataFrame.
    col (str): The column in the DataFrame based on which to segregate sub-DataFrames.

    Returns:
    dict: A dictionary where each key is a unique value from the specified column and each value
    is a DataFrame containing only rows corresponding to that key.
    """
    return {
        population: df[df[col] == population]
        for population in df[col].unique()
    }


def general_population_violin(subfeatures_df: dict[str, Any], seed: int, cols: list[str], pca_cols: list[str],
                              show_violin: bool, save_violin: bool):
    """
    Generates violin plots for different populations based on the provided DataFrames.

    Parameters:
    subfeatures_df (dict[str, Any]): A dictionary of DataFrames, each corresponding to a unique population.
    seed (int): Seed used for the analysis.
    cols (list[str]): List of feature columns to be plotted.
    pca_cols (list[str]): List of PCA columns to be plotted.
    show_violin (bool): Flag to determine whether to display the violin plot.
    save_violin (bool): Flag to determine whether to save the violin plot.

    Returns:
    list[go.Figure]: A list of Plotly figure objects, each representing a violin plot for a given feature.
    """
    color_dict = create_color_dict(subfeatures_df)
    fig_list = []
    for feature in cols + pca_cols:
        fig = general_fig_layout(go.Figure())
        for key, df in subfeatures_df.items():
            for cluster in np.unique(df["Clusters"]):
                df2 = df[df["Clusters"] == cluster]
                violin_y = df2[feature]
                fig.add_trace(
                    go.Violin(y=violin_y,
                              x=[key] * len(df2),
                              name=f"{key} {cluster}", box={"visible": True}, points="all", meanline={"visible": True},
                              line_color=color_dict[f"{key} {cluster}"], opacity=0.6))
                fig.update_layout(title=feature)
        if show_violin:
            fig.show()
        if save_violin:
            fig.write_image(f"saved_analysis/{seed}/{feature}.svg")
        fig_list.append(fig)
    return fig_list


def plot_violin(df: pd.DataFrame, comparison_column: str, cols: [str]):
    """
    Generates violin plots for each specified column in a DataFrame compared against another column.

    Parameters:
    df (pd.DataFrame): The DataFrame containing data to plot.
    comparison_column (str): The column used for comparison in the violin plot.
    cols ([str]): List of columns for which to generate violin plots.

    Returns:
    list[go.Figure]: A list of Plotly figure objects, each representing a violin plot for a given column.
    """
    fig_list = []
    for col in cols:
        fig = go.Figure(go.Violin(y=df[col], x=df[comparison_column].astype(str), points="all"))
        fig.update_layout(title=col)
        fig_list.append(general_fig_layout(fig))
    return fig_list


def better_plot_violin(df: pd.DataFrame, comparison_column: str, cols: [str],
                       normalise: bool = True, colors: dict[str, str] = None, **kwargs):
    """
    Generates enhanced violin plots for each specified column in a DataFrame compared against another column.

    Parameters:
    df (pd.DataFrame): The DataFrame containing data to plot.
    comparison_column (str): The column used for comparison in the violin plot.
    cols ([str]): List of columns for which to generate violin plots.
    normalise (bool, optional): Flag to normalise data before plotting. Defaults to True.
    colors (dict[str, str], optional): Dictionary mapping groups to colors. Defaults to None.
    **kwargs: Additional keyword arguments for the violin plot.

    Returns:
    go.Figure: A Plotly figure object containing the generated violin plot.
    """
    fig = go.Figure()
    if not colors:
        colors_list = plotly.colors.qualitative.Plotly
        for i, group in enumerate(df[comparison_column].unique()):
            colors[group] = colors_list[i]
    for group in df[comparison_column].unique():
        df2 = df[df[comparison_column] == group]
        for col in cols:
            show_legend = col == cols[0]
            if normalise:
                max_data = df[col].max()
                min_data = df[col].min()
                y = normalise_data(df2[col], max_value=max_data, min_value=min_data)
            else:
                y = df2[col]
            fig.add_trace(go.Violin(y=y, x=df2.shape[0] * [col], points="outliers", line_color=colors[group],
                                    name=group, legendgroup=group, showlegend=show_legend, pointpos=0, opacity=0.5,
                                    **kwargs))
    return fig


def group_violin(df, seed: int, cols: list[str], pca_cols: list[str], col: str, color_dict,
                 show_violin: bool, save_violin: bool, unique_col_values=None):
    """
    Generates violin plots for specified features and PCA components, grouped by a specific column.

    Parameters:
    df (pd.DataFrame): DataFrame containing the data.
    seed (int): Seed used for the analysis.
    cols (list[str]): List of feature columns to be plotted.
    pca_cols (list[str]): List of PCA columns to be plotted.
    col (str): The column used to group data in the violin plot.
    color_dict (dict): Dictionary mapping groups to colors for the plot.
    show_violin (bool): Whether to display the violin plot.
    save_violin (bool): Whether to save the violin plot.
    unique_col_values ([Any], optional): Unique values to consider from the grouping column. Defaults to None.

    Returns:
    list[go.Figure]: A list of Plotly figure objects, each representing a violin plot for a given feature or PCA component.
    """
    fig_list = []
    for feature in cols + pca_cols:
        fig = general_fig_layout(go.Figure())
        if unique_col_values is None:
            unique_col_values = np.unique(df[col])
        for cluster in unique_col_values:
            df2 = df[df[col] == cluster]
            violin_y = df2[feature]
            fig.add_trace(
                go.Violin(y=violin_y,
                          x=[cluster] * len(df2),
                          name=f"{cluster}", box={"visible": True}, points="all", meanline={"visible": True},
                          line_color=color_dict[f"{cluster}"], opacity=0.6))
            fig.update_layout(title=feature)
        if show_violin:
            fig.show()
        if save_violin:
            fig.write_image(f"saved_analysis/{seed}/{feature}.svg")
        fig_list.append(fig)
    return fig_list


def darker_general_population_violin(sub_features_df: dict[str, Any], seed: int, cols: list[str], pca_cols: list[str],
                                     show_violin: bool, save_violin: bool, color_dict=None):
    """
    Generates violin plots for different populations with a darker color scheme.

    Parameters:
    sub_features_df (dict[str, Any]): Dictionary of DataFrames, each for a specific population.
    seed (int): Seed used for the analysis.
    cols (list[str]): List of feature columns to be plotted.
    pca_cols (list[str]): List of PCA columns to be plotted.
    show_violin (bool): Whether to display the violin plot.
    save_violin (bool): Whether to save the violin plot.
    color_dict (dict[str, str], optional): Dictionary mapping clusters to colors. Defaults to None.

    Returns:
    list[go.Figure]: A list of Plotly figure objects, each representing a violin plot for a given feature or PCA component.
    """
    if not color_dict:
        color_dict = create_color_dict(sub_features_df)
    fig_list = []
    for feature in cols + pca_cols:
        fig = general_fig_layout(go.Figure())
        for key, df in sub_features_df.items():
            for cluster in np.unique(df["Clusters"]):
                df2 = df[df["Clusters"] == cluster]
                violin_y = df2[feature]
                color = color_dict[f"{key} {cluster}"]
                color = color.replace("180", "240")
                fig.add_trace(
                    go.Violin(y=violin_y,
                              x=[key] * len(df2),
                              name=f"{key} {cluster}", box={"visible": True}, points="all", meanline={"visible": True},
                              line_color=color, opacity=0.6))
                fig.update_layout(title=feature)
        if show_violin:
            fig.show()
        if save_violin:
            fig.write_image(f"saved_analysis/{seed}/{feature}.svg")
        fig_list.append(fig)
    return fig_list
