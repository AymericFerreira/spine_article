import pandas as pd
import plotly.colors
import plotly.graph_objects as go


def generate_all_pie_charts(df: pd.DataFrame, column: str, cluster_column: str, title: str = "", colors_dict=None):
    """
    Generates a list of pie chart figures based on the grouping of a DataFrame.

    This function groups the DataFrame based on specified columns and creates a pie chart for each unique
    value in the specified 'column'. The pie charts represent the size of each cluster within the groups.

    Parameters:
    df (pd.DataFrame): The DataFrame containing the data.
    column (str): The name of the column to be used for grouping and generating individual pie charts.
    cluster_column (str): The name of the column representing the cluster data.
    title (str, optional): A base title for the pie charts. Each chart's title will be appended with the group value. Defaults to an empty string.
    colors_dict (dict, optional): A dictionary mapping group values to color lists for the pie charts. Defaults to None.

    Returns:
    list of go.Figure: A list containing the generated pie chart figures.
    """
    df2 = df.groupby([column, cluster_column]).size().to_frame(name='Count')
    df2 = df2.reset_index()
    fig_list = []
    for cond in df2[column].unique():
        df3 = df2[df2[column] == cond]
        if colors_dict:
            fig_list.append(generate_pie_figure(labels=list(df3[cluster_column]), values=list(df3["Count"]),
                                                title=f"{title} {cond}", color_table=colors_dict[cond]))
        else:
            fig_list.append(generate_pie_figure(labels=list(df3[cluster_column]), values=list(df3["Count"]),
                                                title=f"{title} {cond}"))
    return fig_list


def generate_all_pie_charts_dict(df: pd.DataFrame, column: str, cluster_column: str, title: str = "", colors_dict=None):
    """
    Generates a list of pie chart figures based on the grouping of a DataFrame.

    This function groups the DataFrame based on specified columns and creates a pie chart for each unique
    value in the specified 'column'. The pie charts represent the size of each cluster within the groups.

    Parameters:
    df (pd.DataFrame): The DataFrame containing the data.
    column (str): The name of the column to be used for grouping and generating individual pie charts.
    cluster_column (str): The name of the column representing the cluster data.
    title (str, optional): A base title for the pie charts. Each chart's title will be appended with the group value. Defaults to an empty string.
    colors_dict (dict, optional): A dictionary mapping group values to color lists for the pie charts. Defaults to None.

    Returns:
    list of go.Figure: A list containing the generated pie chart figures.
    """
    df2 = df.groupby([column, cluster_column]).size().to_frame(name='Count')
    df2 = df2.reset_index()
    fig_dict = {}
    for cond in df2[column].unique():
        df3 = df2[df2[column] == cond]
        if colors_dict:
            fig_dict[cond] = generate_pie_figure(labels=list(df3[cluster_column]), values=list(df3["Count"]),
                                                 title=f"{title} {cond}", color_table=colors_dict[cond])
        else:
            fig_dict[cond] = generate_pie_figure(labels=list(df3[cluster_column]), values=list(df3["Count"]),
                                                 title=f"{title} {cond}")
    return fig_dict


def generate_pie_figure(labels: [], values: [], title: str, color_table: [str] = None):
    """
    Generates a pie chart figure with specified labels, values, and coloring.

    This function creates a pie chart figure using Plotly, with the labels and values provided.
    It allows customization of the color scheme for the pie segments.

    Parameters:
    labels ([]): A list of labels for each segment of the pie chart.
    values ([]): A list of values corresponding to each label of the pie chart.
    title (str): The title of the pie chart.
    color_table ([str], optional): A list of colors for the pie segments. If not provided, default Plotly colors are used.

    Returns:
    go.Figure: The generated Plotly figure object representing the pie chart.
    """
    colors = color_table or plotly.colors.qualitative.Plotly
    fig = go.Figure(data=[go.Pie(
        labels=[f"Cluster {int(i)}" for i in labels],
        values=values,
        marker_colors=[colors[int(i) - 1] for i in labels],
        sort=False)])
    fig.update_layout(title=f"{title}")
    fig.update_layout(font=dict(size=20, color="black", family="Arial"))
    return fig
