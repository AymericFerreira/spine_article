import pandas as pd
from plotly_utils.generate_piecharts import generate_all_pie_charts_dict
from analysis_utils.statistics import generate_populations, compare_all_pairs
import plotly.graph_objects as go


def generate_pie_charts(df: pd.DataFrame, group_col: str, cluster_col: str, title: str = "Cluster distribution",
                        display: bool = True, save_prefix: str = None, statistical_test: bool = True) -> [go.Figure]:
    """
    Generates and optionally displays and saves a list of pie charts for each group and cluster in a DataFrame.

    This function creates pie charts representing the distribution of clusters within each group in the DataFrame.
    Each pie chart is generated for a unique group and displays the proportion of each cluster within that group.
    The function can also perform a statistical test on the data, and it returns a list of the generated pie chart figures.

    Parameters:
    df (pd.DataFrame): The DataFrame containing the data.
    group_col (str): The column name in the DataFrame that contains the group labels.
    cluster_col (str): The column name in the DataFrame that contains the cluster labels.
    title (str, optional): The title for the pie charts. Defaults to "Cluster distribution".
    display (bool, optional): If True, displays the pie charts. Defaults to True.
    save_prefix (str, optional): The file prefix for saving the pie charts. If provided, charts are saved as images. Defaults to None.
    statistical_test (bool, optional): If True, performs a statistical test on the data. Defaults to True.

    Returns:
    [go.Figure]: A list of Plotly graph objects (go.Figure) for each generated pie chart.
    """
    if display or save_prefix:
        fig_dict = generate_all_pie_charts_dict(df, group_col, cluster_col, title=title)
        for cond, fig in fig_dict.items():
            fig.update_layout(title=cond)
            fig.update_layout(font=dict(size=20, color="black", family="Arial"))
            if display:
                fig.show()
            if save_prefix:
                fig.write_image(f"{save_prefix}_{cond}.png")
    if statistical_test:
        pops = generate_populations(df, ["Group", "Clusters"])
        print(compare_all_pairs(pops))


# if __name__ == '__main__':
#     df = pd.read_csv(r"G:\Documents\PycharmProjects\newSpineClassifier\datasets\dataset_for_article\good_dataset_from_september.csv")
#     generate_pie_charts(df, "Group", "Clusters", title="Cluster distribution", statistical_test=True)