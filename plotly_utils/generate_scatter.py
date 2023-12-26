from typing import Dict
from plotly.subplots import make_subplots
import plotly.express as px
from plotly_utils.themes import general_fig_layout
import plotly.graph_objects as go
import pandas as pd
import itertools


def generate_2D_PCA_markers(df: pd.DataFrame, column: str, prefix: str = ""):
    """
    Generates a 2D scatter plot of PCA components for different groups in a DataFrame.

    This function creates a scatter plot for each unique value in a specified column of the DataFrame.
    It uses the first two principal components for the x and y axes of the scatter plot.

    Parameters:
    df (pd.DataFrame): DataFrame containing the principal component data.
    column (str): Column name in the DataFrame to differentiate groups for scatter plots.
    prefix (str, optional): Prefix to add before each group's name in the legend. Defaults to an empty string.

    Returns:
    go.Figure: A Plotly figure object with the generated scatter plot.
    """
    fig = make_subplots(rows=2, cols=1)
    colors = px.colors.qualitative.Plotly

    for i, uniqueValue in enumerate(df[column].unique()):
        df2 = df[df[column] == uniqueValue]
        fig.add_trace(go.Scatter(x=df2["Principal component 1"],
                                 y=df2["Principal component 2"],
                                 mode="markers",
                                 marker=dict(size=7, symbol="circle",
                                             color=colors[i],
                                             line=dict(width=2,
                                                       color='DarkSlateGrey')
                                             ),
                                 # hovertext=df["MeshName"],
                                 legendgroup=f"Cluster {uniqueValue}",
                                 name=f"{prefix}{uniqueValue}"),
                      row=1, col=1)
        fig.update_xaxes(title="Principal Component 1", row=1, col=1)
        fig.update_yaxes(title="Principal Component 2", row=1, col=1)
        fig.add_trace(go.Scatter(x=df2["Principal component 2"],
                                 y=df2["Principal component 3"],
                                 mode="markers",
                                 marker=dict(size=7, symbol="circle",
                                             # color=colorList,
                                             color=colors[i],
                                             line=dict(width=2,
                                                       color='DarkSlateGrey')
                                             ),
                                 # hovertext=df["MeshName"],
                                 legendgroup=f"Cluster {uniqueValue}",
                                 showlegend=False,
                                 name=f"{prefix}{uniqueValue}"),
                      row=2, col=1)
        fig.update_xaxes(title="Principal Component 2", row=2, col=1)
        fig.update_yaxes(title="Principal Component 3", row=2, col=1)
    return general_fig_layout(fig)


def new_generate_2D_markers(df: pd.DataFrame,
                            column: str,
                            reduced_dimension_cols: [str],
                            prefix: str = "",
                            title: str = "",
                            height: int = 500,
                            width: int = 800):
    """
    Generates 2D scatter plots for each pair of dimensions in reduced dimension columns.

    Parameters:
    df (pd.DataFrame): DataFrame containing the data for scatter plots.
    column (str): Column name used to differentiate groups in the scatter plot.
    reduced_dimension_cols ([str]): List of column names representing reduced dimensions.
    prefix (str, optional): Prefix to add before each group's name in the legend. Defaults to an empty string.
    title (str, optional): Title of the plot. Defaults to an empty string.
    height (int, optional): Height of the plot. Defaults to 500.
    width (int, optional): Width of the plot. Defaults to 800.

    Returns:
    go.Figure: A Plotly figure object with the generated scatter plots.
    """
    iterations = list(itertools.combinations(reduced_dimension_cols, 2))

    fig = make_subplots(rows=len(iterations) + 1, cols=1)
    colors = px.colors.qualitative.Plotly

    for i, uniqueValue in enumerate(df[column].unique()):
        for graphNumber, (dimension1, dimension2) in enumerate(iterations):
            show_legend = graphNumber == int(i / len(iterations)) + 1
            df2 = df[df[column] == uniqueValue]
            fig.add_trace(go.Scatter(x=df2[dimension1],
                                     y=df2[dimension2],
                                     mode="markers",
                                     marker=dict(size=7, symbol="circle",
                                                 color=colors[i],
                                                 line=dict(width=2,
                                                           color='DarkSlateGrey')
                                                 ),
                                     legendgroup=f"Cluster {uniqueValue}",
                                     name=f"{prefix}{uniqueValue}",
                                     showlegend=show_legend),
                          row=graphNumber + 1, col=1)
            fig.update_xaxes(title=dimension1, row=graphNumber + 1, col=1)
            fig.update_yaxes(title=dimension2, row=graphNumber + 1, col=1)

    fig['layout'].update(height=height * len(iterations), width=width)

    fig.update_layout(title=title)
    return general_fig_layout(fig)


def generate_3D_markers(df: pd.DataFrame, column: str, prefix: str = "") -> go.Figure:
    """
    Generates a 3D scatter plot of PCA components for different groups in a DataFrame.

    This function creates a 3D scatter plot for each unique value in a specified column of the DataFrame.
    It uses the first three principal components for the x, y, and z axes of the scatter plot.

    Parameters:
    df (pd.DataFrame): DataFrame containing the principal component data.
    column (str): Column name in the DataFrame to differentiate groups for the scatter plot.
    prefix (str, optional): Prefix to add before each group's name in the legend. Defaults to an empty string.

    Returns:
    go.Figure: A Plotly figure object with the generated 3D scatter plot.
    """
    fig = go.Figure()
    for uniqueValue in df[column].unique():
        df2 = df[df[column] == uniqueValue]
        fig.add_trace(go.Scatter3d(x=df2["Principal component 1"],
                                   y=df2["Principal component 2"],
                                   z=df2["Principal component 3"],
                                   mode="markers",
                                   marker=dict(size=7, symbol="circle",
                                               line=dict(width=2,
                                                         color='DarkSlateGrey')
                                               ),
                                   name=f"{prefix}{uniqueValue}"))
    fig.update_xaxes(title="Principal component 1")
    fig.update_yaxes(title="Principal component 2")
    return general_fig_layout(fig)


def generate_2D_markers(df: pd.DataFrame, column: str, prefix: str = ""):
    """
    Generates a 2D scatter plot of PCA components for different groups in a DataFrame.

    This function creates a 2D scatter plot for each unique value in a specified column of the DataFrame.
    It uses the first two principal components for the x and y axes of the scatter plot.

    Parameters:
    df (pd.DataFrame): DataFrame containing the principal component data.
    column (str): Column name in the DataFrame to differentiate groups for scatter plots.
    prefix (str, optional): Prefix to add before each group's name in the legend. Defaults to an empty string.

    Returns:
    go.Figure: A Plotly figure object with the generated scatter plot.
    """
    fig = go.Figure()
    for uniqueValue in df[column].unique():
        df2 = df[df[column] == uniqueValue]
        fig.add_trace(go.Scatter(x=df2["Principal component 1"],
                                 y=df2["Principal component 2"],
                                 mode="markers",
                                 marker=dict(size=7, symbol="circle",
                                             line=dict(width=2,
                                                       color='DarkSlateGrey')
                                             ),
                                 name=f"{prefix}{uniqueValue}"))
    return general_fig_layout(fig)


def generate_line_plot(df: pd.DataFrame, x_col: str, y_cols: [str]):
    """
    Generates a line plot for specified columns in a DataFrame.

    Parameters:
    df (pd.DataFrame): DataFrame containing the data for the line plot.
    x_col (str): Column name to be used for the x-axis.
    y_cols ([str]): List of column names to be plotted on the y-axis.

    Returns:
    go.Figure: A Plotly figure object with the generated line plot.
    """
    fig = go.Figure()
    for y_col in y_cols:
        fig.add_trace(go.Scatter(x=df[x_col], y=df[y_col],
                                 mode="lines+markers", name=y_col))
    fig.update_yaxes(range=[df[y_cols].min(), 100])
    return general_fig_layout(fig)
