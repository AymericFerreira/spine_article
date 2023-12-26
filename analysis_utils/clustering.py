import numpy as np
from sklearn.cluster import KMeans
from yellowbrick.cluster import KElbowVisualizer
import plotly.express as px
import pandas as pd
from sklearn.tree import DecisionTreeClassifier

try:
    from dtreeviz.trees import dtreeviz
except ImportError:
    print("dtreeviz not imported")
    print("Note that dtreeviz API changed, and code here could not reflect changes.")


def decision_tree(data: np.ndarray,
                  cluster_list: np.ndarray,
                  number_of_clusters: int,
                  seed: int = 42,
                  max_depth: int = 6,
                  display=True):
    """
    Constructs and visualizes a decision tree classifier for the given data and cluster labels.

    This function builds a decision tree classifier using scikit-learn's DecisionTreeClassifier,
    trained on the provided data and cluster labels. It then visualizes the tree using the dtreeviz library.

    Parameters:
    data (np.ndarray): The input data for the decision tree, typically principal component data.
    cluster_list (np.ndarray): The cluster labels for each data point.
    number_of_clusters (int): The number of distinct clusters.
    seed (int, optional): The random state seed for reproducibility. Defaults to 42.
    max_depth (int, optional): The maximum depth of the tree. Defaults to 6.
    display (bool, optional): If True, displays the tree visualization. Defaults to True.

    Returns:
    dtreeviz.trees.DTreeViz: The visualization of the decision tree.
    """
    classifier = DecisionTreeClassifier(random_state=seed, max_depth=max_depth)  # limit depth of tree
    classifier.fit(data, cluster_list)

    cluster_name = [f"Cluster {i}" for i in range(number_of_clusters)]

    if np.size(data, 1) == 3:
        feature_names = ["PC1", "PC2", "PC3"]
    else:
        feature_names = ["PC1", "PC2"]

    viz = dtreeviz(classifier,
                   data,
                   cluster_list,
                   target_name='variety',
                   feature_names=feature_names,
                   class_names=cluster_name,
                   )
    if display:
        viz.view()
    return viz


def simple_elbow(X: np.ndarray,
                 min_k: int = 2,
                 max_k: int = 12,
                 metric="distortion",
                 display=True,
                 save=""):
    """
    Computes and displays an elbow plot to determine the optimal number of clusters.

    This function uses the KElbowVisualizer from the Yellowbrick library to compute the elbow plot
    for KMeans clustering. The elbow plot helps in determining the optimal number of clusters for KMeans.

    Parameters:
    X (np.ndarray): The input data for clustering.
    min_k (int, optional): The minimum number of clusters to try. Defaults to 2.
    max_k (int, optional): The maximum number of clusters to try. Defaults to 12.
    metric (str, optional): The metric used to evaluate the clustering. Defaults to "distortion".
    display (bool, optional): If True, displays the elbow plot. Defaults to True.
    save (str, optional): Path to save the elbow plot. If empty, the plot is not saved. Defaults to an empty string.

    Returns:
    None
    """
    model = KMeans()
    visualizer = KElbowVisualizer(model, k=(min_k, max_k), timings=False, metric=metric)
    visualizer.fit(X)
    if display:
        visualizer.show(outpath=save)
    # return visualizer
    # Possible improvement : here jupyter notebook sometimes return an error if it receives visualizer, try to fix


def simple_corr_matrix(df: pd.DataFrame,
                       display=True):
    """
    Computes and displays a correlation matrix using Plotly Express.

    This function calculates the correlation matrix for the given DataFrame and visualizes it using
    an interactive heatmap created with Plotly Express.

    Parameters:
    df (pd.DataFrame): The DataFrame for which to compute the correlation matrix.
    display (bool, optional): If True, displays the correlation matrix heatmap. Defaults to True.

    Returns:
    plotly.graph_objs._figure.Figure: The Plotly figure object for the correlation matrix heatmap.
    """
    fig = px.imshow(df.corr(), text_auto=True, x=df.columns, y=df.columns)
    if display:
        fig.show()
    return fig
