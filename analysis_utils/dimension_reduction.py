import pandas as pd
import plotly.express as px
import sklearn.pipeline
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from plotly_utils.themes import general_fig_layout
import numpy as np
import itertools
from plotly.subplots import make_subplots
import plotly.graph_objects as go


def biplot(df: pd.DataFrame, features: [str], n_components: int = 3, seed: int = 42,
           x_title: str = None, y_title: str = None):
    """
    Creates a biplot using PCA for the given DataFrame.

    The biplot visualizes the loadings (principal component coefficients) along with the scores
    (projected data) of the first two principal components. It's useful for understanding the
    contribution of original features to the principal components.

    Parameters:
    df (pd.DataFrame): DataFrame containing the data to be analyzed.
    features ([str]): List of column names in df to be included in the PCA.
    n_components (int, optional): Number of principal components to compute. Defaults to 3.
    seed (int, optional): Random state seed for PCA reproducibility. Defaults to 42.
    x_title (str, optional): Title for the x-axis. Defaults to None.
    y_title (str, optional): Title for the y-axis. Defaults to None.

    Returns:
    plotly.graph_objs._figure.Figure: A Plotly figure object representing the biplot.
    """
    pipeline = sklearn.pipeline.make_pipeline(StandardScaler(), PCA(n_components=n_components, random_state=seed))
    components = pipeline.fit_transform(df[features].astype(float))
    pca = pipeline["pca"]

    loadings = pca.components_.T * np.sqrt(pca.explained_variance_)

    fig = px.scatter(components, x=0, y=1)

    for i, feature in enumerate(features):
        fig.add_shape(
            type='line',
            x0=0, y0=0,
            x1=loadings[i, 0],
            y1=loadings[i, 1]
        )
        fig.add_annotation(
            x=loadings[i, 0],
            y=loadings[i, 1],
            ax=0, ay=0,
            xanchor="center",
            yanchor="bottom",
            text=feature,
        )
    fig.update_xaxes(title=x_title)
    fig.update_yaxes(title=y_title)
    return general_fig_layout(fig)


def new_biplot(x: [float], y: [float], show_legend: bool = False):
    """
    Creates a scatter plot (biplot) using Plotly.

    Parameters:
    x ([float]): List of x-coordinates for the scatter plot.
    y ([float]): List of y-coordinates for the scatter plot.
    show_legend (bool, optional): Flag to control the display of legend. Defaults to False.

    Returns:
    plotly.graph_objs.Scatter: A Plotly Scatter object for the biplot.
    """
    return go.Scatter(
        x=x,
        y=y,
        mode="markers",
        marker=dict(
            size=7,
            symbol="circle",
            color="gray",
            line=dict(width=2, color='DarkSlateGrey'),
        ),
        showlegend=show_legend,
    )


def biplots(df: pd.DataFrame,
            column: str,
            reduced_dimension_cols: [str],
            pca: PCA,
            title: str = "",
            height: int = 500,
            width: int = 800):
    """
    Creates multiple biplots in a subplot layout using PCA results.

    Parameters:
    df (pd.DataFrame): DataFrame containing the data.
    column (str): Name of the column in df.
    reduced_dimension_cols ([str]): List of column names representing reduced dimensions.
    pca (PCA): A fitted PCA object.
    title (str, optional): Title for the entire subplot. Defaults to an empty string.
    height (int, optional): Height of the entire subplot figure. Defaults to 500.
    width (int, optional): Width of the entire subplot figure. Defaults to 800.

    Returns:
    plotly.graph_objs.Figure: A Plotly figure object containing the subplots.
    """
    dim_col_identifier = list(range(len(reduced_dimension_cols)))
    iterations = list(itertools.combinations(reduced_dimension_cols, 2))
    iterations_id = list(itertools.combinations(dim_col_identifier, 2))
    fig = make_subplots(rows=len(iterations) + 1, cols=1)
    colors = px.colors.qualitative.Plotly
    loadings = pca.components_.T * np.sqrt(pca.explained_variance_)

    for graphNumber, (dimension1, dimension2) in enumerate(iterations):
        iteration_id = iterations_id[graphNumber]
        show_legend = graphNumber == int(graphNumber / len(iterations)) + 1
        subplot_fig = new_biplot(df[dimension1], df[dimension2], show_legend=show_legend)
        fig.add_trace(subplot_fig, row=graphNumber + 1, col=1)
        for i, feature in enumerate(column):
            fig.add_shape(
                type="line",
                x0=0, y0=0,
                x1=loadings[i, iteration_id[0]] * np.max(df[dimension1]),
                y1=loadings[i, iteration_id[1]] * np.max(df[dimension2]),
                line=dict(
                    color=colors[i],
                    width=4,
                ), row=graphNumber + 1, col=1
            )
            fig.add_annotation(
                x=loadings[i, iteration_id[0]] * np.max(df[dimension1]),
                y=loadings[i, iteration_id[1]] * np.max(df[dimension2]),
                ax=0, ay=0,
                xanchor="center",
                yanchor="bottom",
                text=feature,
                row=graphNumber + 1, col=1
            )
        fig.update_xaxes(title=dimension1, row=graphNumber + 1, col=1)
        fig.update_yaxes(title=dimension2, row=graphNumber + 1, col=1)

    fig['layout'].update(height=height * len(iterations), width=width)
    fig.update_layout(title=title)
    return general_fig_layout(fig)


def separated_biplots(df: pd.DataFrame,
                      column: str,
                      reduced_dimension_cols: [str],
                      pca: PCA,
                      title: str = "",
                      height: int = 500,
                      width: int = 800):
    """
    Creates separate biplot figures for each combination of reduced dimensions.

    Parameters:
    df (pd.DataFrame): DataFrame containing the data.
    column (str): Name of the column in df.
    reduced_dimension_cols ([str]): List of column names representing reduced dimensions.
    pca (PCA): A fitted PCA object.
    title (str, optional): Title for the figures. Defaults to an empty string.
    height (int, optional): Height of each figure. Defaults to 500.
    width (int, optional): Width of each figure. Defaults to 800.

    Returns:
    [plotly.graph_objs.Figure]: A list of Plotly figure objects, each representing a biplot.
    """
    dim_col_identifier = list(range(len(reduced_dimension_cols)))
    iterations = list(itertools.combinations(reduced_dimension_cols, 2))
    iterations_id = list(itertools.combinations(dim_col_identifier, 2))
    colors = px.colors.qualitative.Plotly
    loadings = pca.components_.T * np.sqrt(pca.explained_variance_)
    fig_list = []
    for graphNumber, (dimension1, dimension2) in enumerate(iterations):
        fig = go.Figure()
        iteration_id = iterations_id[graphNumber]
        show_legend = graphNumber == int(graphNumber / len(iterations)) + 1
        fig.add_trace(new_biplot(df[dimension1], df[dimension2], show_legend=show_legend))
        for i, feature in enumerate(column):
            fig.add_shape(
                type="line",
                x0=0, y0=0,
                x1=loadings[i, iteration_id[0]] * np.max(df[dimension1]),
                y1=loadings[i, iteration_id[1]] * np.max(df[dimension2]),
                line=dict(
                    color=colors[i],
                    width=4,
                )
            )
            fig.add_annotation(
                x=loadings[i, iteration_id[0]] * np.max(df[dimension1]),
                y=loadings[i, iteration_id[1]] * np.max(df[dimension2]),
                ax=0, ay=0,
                xanchor="center",
                yanchor="bottom",
                text=feature,
            )
        fig.update_xaxes(title=f"{dimension1}")
        fig.update_yaxes(title=f"{dimension2}")

        fig['layout'].update(height=height * len(iterations), width=width)
        fig.update_layout(title=title)

        fig_list.append(general_fig_layout(fig))
    return fig_list


def color_biplot(x: [float], y: [float], color: str, group: str, show_legend: bool = False):
    """
    Creates a colored scatter plot (biplot) using Plotly, with points colored by group.

    Parameters:
    x ([float]): List of x-coordinates for the scatter plot.
    y ([float]): List of y-coordinates for the scatter plot.
    color (str): Color for the scatter points.
    group (str): Group name associated with the scatter points.
    show_legend (bool, optional): Flag to control the display of legend. Defaults to False.

    Returns:
    plotly.graph_objs.Scatter: A Plotly Scatter object for the biplot.
    """
    return go.Scatter(
        x=x,
        y=y,
        mode="markers",
        marker=dict(
            size=7,
            symbol="circle",
            color=color,
            opacity=0.7,
            line=dict(width=2, color='DarkSlateGrey'),
        ),
        name=group,
        showlegend=show_legend,
    )


def group_biplots(df: pd.DataFrame, column: str, reduced_dimension_cols: [str], pca: PCA,
                  group_column: str, colors: {str} = None, prefix: str = "", title: str = "",
                  height: int = 500, width: int = 800):
    """
    Creates a set of biplots grouped by a specified column, displayed in a subplot layout.

    Parameters:
    df (pd.DataFrame): DataFrame containing the data.
    column (str): Name of the column in df.
    reduced_dimension_cols ([str]): List of column names representing reduced dimensions.
    pca (PCA): A fitted PCA object.
    group_column (str): Name of the column used for grouping data points.
    colors ({str}): Optional dictionary mapping group names to colors.
    prefix (str, optional): Prefix for the plot titles. Defaults to an empty string.
    title (str, optional): Title for the entire subplot. Defaults to an empty string.
    height (int, optional): Height of the entire subplot figure. Defaults to 500.
    width (int, optional): Width of the entire subplot figure. Defaults to 800.

    Returns:
    plotly.graph_objs.Figure: A Plotly figure object containing the subplots.
    """
    dim_col_identifier = list(range(len(reduced_dimension_cols)))
    iterations = list(itertools.combinations(reduced_dimension_cols, 2))
    iterations_id = list(itertools.combinations(dim_col_identifier, 2))
    fig = make_subplots(rows=len(iterations) + 1, cols=1)
    if not colors:
        colors_list = px.colors.qualitative.Plotly
        for i, group in enumerate(df[group_column].unique()):
            colors[group] = colors_list[i]
    loadings = pca.components_.T * np.sqrt(pca.explained_variance_)

    color_list = [colors[df.iloc[i][group_column]] for i in range(len(df))]
    pca_arrow_colors = px.colors.qualitative.Plotly

    for graphNumber, (dimension1, dimension2) in enumerate(iterations):
        iteration_id = iterations_id[graphNumber]
        show_legend = graphNumber == int(graphNumber / len(iterations)) + 1
        for group in df[group_column].unique():
            df2 = df[df[group_column] == group]
            subplot_fig = color_biplot(df2[dimension1], df2[dimension2], color=colors[group], group=group,
                                       show_legend=show_legend)
            fig.add_trace(subplot_fig, row=graphNumber + 1, col=1)
        for i, feature in enumerate(column):
            fig.add_shape(
                type="line",
                x0=0, y0=0,
                x1=loadings[i, iteration_id[0]] * np.max(df[dimension1]),
                y1=loadings[i, iteration_id[1]] * np.max(df[dimension2]),
                line=dict(
                    color=pca_arrow_colors[i],
                    width=4,
                ), row=graphNumber + 1, col=1
            )
            fig.add_annotation(
                x=loadings[i, iteration_id[0]] * np.max(df[dimension1]),
                y=loadings[i, iteration_id[1]] * np.max(df[dimension2]),
                ax=0, ay=0,
                xanchor="center",
                yanchor="bottom",
                text=feature,
                row=graphNumber + 1, col=1
            )
        fig.update_xaxes(title=dimension1, row=graphNumber + 1, col=1)
        fig.update_yaxes(title=dimension2, row=graphNumber + 1, col=1)

    fig['layout'].update(height=height * len(iterations), width=width)
    fig.update_layout(title=title)
    return general_fig_layout(fig)
