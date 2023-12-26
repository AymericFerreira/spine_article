import pandas as pd
import numpy as np
from scipy import stats
from sklearn.neighbors import KernelDensity
import plotly.graph_objects as go
import seaborn as sns
import matplotlib.pyplot as plt


def determine_center_of_mass(xx, yy, f):
    """
    Determines the coordinates of the center of mass for a given density function.

    This function identifies the grid point(s) where the density function 'f' reaches its maximum value.
    It then extracts the corresponding x and y coordinates from the grid arrays 'xx' and 'yy'. The
    function assumes that the center of mass is located at these maximum value points.

    Parameters:
    xx (numpy.ndarray): A 2D array representing the grid's x-coordinates.
    yy (numpy.ndarray): A 2D array representing the grid's y-coordinates.
    f (numpy.ndarray): A 2D array representing the density values on the grid.

    Returns:
    tuple: Two arrays containing the x and y coordinates of the center of mass. If there are multiple
           maximum points, the function returns arrays of coordinates for all these points.
    """
    x, y = np.where(f == np.amax(f))
    x = xx[x, 0]
    y = yy[0, y]
    return x, y


class Vector:
    """
    Class representing a vector.

    Args:
        point1: The starting point of the vector.
        point2: The ending point of the vector.

    Methods:
        get_norm(): Calculates the norm of the vector.
        extend_vector(norm): Extends the vector by a given norm.

    """

    def __init__(self, point1, point2):
        self.A = point1
        self.B = point2

    def get_norm(self):
        return np.linalg.norm(self.A.get_nparray() - self.B.get_nparray())

    def extend_vector(self, norm):
        len_ab = np.sqrt(pow(self.A.x - self.B.x, 2.0) + pow(self.A.y - self.B.y, 2.0))
        x = self.B.x + (self.B.x - self.A.x) / len_ab * norm
        y = self.B.y + (self.B.y - self.A.y) / len_ab * norm
        return Point(x, y)


class Point:
    """
    Class representing a point.

    Args:
        x: The x-coordinate of the point.
        y: The y-coordinate of the point.

    Methods:
        get_tuple(): Returns the coordinates of the point as a tuple.
        get_nparray(): Returns the coordinates of the point as a NumPy array.
        __repr__(): Returns a string representation of the point.

    """

    def __init__(self, x, y):
        self.x = x
        self.y = y

    def get_tuple(self):
        return self.x, self.y

    def get_nparray(self):
        return np.array([self.x, self.y])

    def __repr__(self):
        return f"X: {self.x}, Y: {self.y}"


def calculate_kernel_density(component1, component2, xx, yy, mode="stats"):
    """
    Calculates the kernel density estimate for given data components.

    This function computes the kernel density estimate (KDE) for two data components using either
    the Scikit-learn's KernelDensity (when mode is 'stats') or scipy's gaussian_kde. It returns
    the KDE values reshaped to match the grid defined by 'xx' and 'yy'.

    Parameters:
    component1 (numpy.ndarray): The first component of the data.
    component2 (numpy.ndarray): The second component of the data.
    xx (numpy.ndarray): The grid coordinates for the x-axis.
    yy (numpy.ndarray): The grid coordinates for the y-axis.
    mode (str, optional): The mode of KDE calculation. 'stats' for Scikit-learn's method or
                          any other value for scipy's method. Defaults to 'stats'.

    Returns:
    tuple: A tuple containing the KDE values reshaped to the grid's shape, and the grid arrays 'xx' and 'yy'.
    """
    x = component1
    y = component2
    positions = np.vstack([xx.ravel(), yy.ravel()])
    values = np.vstack([x, y])
    d = values.shape[0]
    n = values.shape[1]
    if mode == "stats":
        bw = (n * (d + 2) / 4.) ** (-1. / (d + 4))  # silverman
        kde = KernelDensity(kernel="gaussian", bandwidth=bw).fit(values.T)
        z = np.reshape(np.exp(kde.score_samples(positions.T)), xx.shape)
        z = np.rot90(z)
    else:
        kernel = stats.gaussian_kde(values)
        z = np.reshape(kernel(positions), xx.shape)

    return z, xx, yy


def sns_kd(df: pd.DataFrame):
    """
        Creates kernel density plots for different clusters and groups in a dataset.

        This function reads a dataset, calculates kernel density estimates for different clusters and groups,
        and plots these estimates. It also calculates the center of mass for each group and draws arrows
        from the center to the mass center. The plots can optionally be saved to SVG files.

        Returns:
        None

        Notes:
            This function expects Principal component 1, Principal component 2, Group and Cluster columns in
            the dataset.
            The quality of the resulting plots depends on your data, feel free to adjust the different parameters
    """
    xmin = df["Principal component 1"].min()
    xmax = df["Principal component 1"].max()
    ymin = df["Principal component 2"].min()
    ymax = df["Principal component 2"].max()
    xx, yy = np.mgrid[xmin:xmax:100j, ymin:ymax:100j]

    group_cl_df = pd.DataFrame(columns=["Type", "Group", "Cluster", "X", "Y"])
    for group in sorted(df["Group"].unique()):
        df_group = df[df["Group"] == group]
        fig = go.Figure()
        for cluster in sorted(df_group["Clusters"].unique()):
            df_cl_group = df_group[df_group["Clusters"] == cluster]
            x = df_cl_group["Principal component 1"].mean()
            y = df_cl_group["Principal component 2"].mean()
            f, xx, yy = calculate_kernel_density(df_cl_group["Principal component 1"],
                                                 df_cl_group["Principal component 2"],
                                                 xx, yy, "gauss")
            group_cl_df = pd.concat([group_cl_df,
                                     pd.DataFrame([["center", group, cluster, x, y]],
                                                  columns=["Type", "Group", "Cluster", "X", "Y"])],
                                    axis=0, ignore_index=True)
            fig.add_trace(go.Contour(x=xx[:, 0], y=yy[0, :], z=f,
                                     connectgaps=False, contours_coloring='lines',
                                     contours=dict(start=0, end=0.22, size=0.03)))
            fig.add_trace(go.Scatter(x=[x], y=[y], mode="markers", marker=dict(size=8)))
            fig.update_xaxes(range=(xmin, xmax))
            fig.update_yaxes(range=(ymin, ymax))
            x, y = determine_center_of_mass(xx, yy, f)
            fig.add_trace(go.Scatter(x=x, y=y, mode="markers", marker=dict(size=12)))

            x = x[0]
            y = y[0]
            group_cl_df = pd.concat([group_cl_df,
                                     pd.DataFrame([["mass", group, cluster, x, y]],
                                                  columns=["Type", "Group", "Cluster", "X", "Y"])],
                                    axis=0, ignore_index=True)
    for group in sorted(df["Group"].unique()):
        df_group = df[df["Group"] == group]
        col = "Reds" if "Learners" in group else "Blues"
        for cluster in sorted(df_group["Clusters"].unique()):
            df_group_cluster = df_group[df_group["Clusters"] == cluster]
            ax = sns.kdeplot(x='Principal component 1', y='Principal component 2',
                             cmap=col,
                             data=df_group_cluster,
                             levels=4,
                             common_norm=True,
                             joint_kws={"colors": "black", "cmap": None, "linewidths": 0.5}
                             )
            arrow_x_base = \
                group_cl_df[(group_cl_df["Type"] == "center") & (group_cl_df["Cluster"] == cluster)]["X"].iloc[0]
            arrow_x_end = \
                group_cl_df[(group_cl_df["Type"] == "center") & (group_cl_df["Cluster"] == cluster)]["X"].iloc[1]
            arrow_y_base = \
                group_cl_df[(group_cl_df["Type"] == "center") & (group_cl_df["Cluster"] == cluster)]["Y"].iloc[0]
            arrow_y_end = \
                group_cl_df[(group_cl_df["Type"] == "center") & (group_cl_df["Cluster"] == cluster)]["Y"].iloc[1]

            ax.arrow(arrow_x_base, arrow_y_base, arrow_x_end - arrow_x_base, arrow_y_end - arrow_y_base,
                     head_width=0.1, length_includes_head=True, zorder=10, facecolor='black')

            center_group_cl_df = group_cl_df[(group_cl_df["Type"] == "center") & (group_cl_df["Cluster"] == cluster)]
            f_center = center_group_cl_df[center_group_cl_df["Group"] == group]
            col1 = "Blue" if "Control" in group else "Red"
            ax.scatter(x=f_center["X"], y=f_center["Y"], c=col1, s=20)

        plt.xlim([xmin, xmax])
        plt.ylim([ymin, ymax])

        plt.show()