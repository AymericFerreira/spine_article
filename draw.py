import matplotlib
from matplotlib import pyplot as plt
import numpy as np

matplotlib.use("TkAgg")


def get_frequency(array):
    """
    Calculates the frequency of each unique element in an array.

    This function computes the count of each unique value in the provided array and returns these counts
    alongside the unique values.

    Parameters:
    array (numpy.ndarray): The array for which the frequencies of its unique elements are calculated.

    Returns:
    numpy.ndarray: A 2D array where each row contains a unique value and its frequency in the original array.
    """
    unique, counts = np.unique(array, return_counts=True)
    return np.column_stack((unique, counts))


def plot_numpy_bar(array, x_label, y_label, title):
    """
        Plots a bar chart using a numpy array.

        This function takes a 2D numpy array where the first column is treated as the x-axis and the second column
        as the y-axis, and plots a bar chart.

        Parameters:
        array (numpy.ndarray): A 2D array to plot.
        x_label (str): The label for the x-axis.
        y_label (str): The label for the y-axis.
        title (str): The title of the plot.

    """
    plt.bar(array[:, 0], array[:, 1])  # arguments are passed to np.histogram
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(title)
    plt.show()


def plot_frequency(array, x_label, y_label, title):
    """
        Plots the frequency of elements in an array as a bar chart.

        This function first calculates the frequency of each unique element in the provided array and then plots
        these frequencies as a bar chart.

        Parameters:
        array (numpy.ndarray): The array for which to plot the frequency.
        x_label (str): The label for the x-axis.
        y_label (str): The label for the y-axis.
        title (str): The title of the plot.

    """
    np_frequency = get_frequency(array)
    plot_numpy_bar(np_frequency, x_label, y_label, title)


def plot_3d_scatter_with_color(array, x_label, y_label, z_label, title):
    """
        Plots a 3D scatter plot with color coding from a 4D array.

        This function creates a 3D scatter plot where the first three dimensions of the array are used for the
        x, y, and z coordinates and the fourth dimension for color coding.

        Parameters:
        array (numpy.ndarray): A 4D array where the first three columns are coordinates and the fourth is the color.
        x_label (str): The label for the x-axis.
        y_label (str): The label for the y-axis.
        z_label (str): The label for the z-axis.
        title (str): The title of the plot.

    """
    fig = plt.figure()
    scatter_plot = fig.add_subplot(111, projection='3d')

    p = scatter_plot.scatter(array[:, 0], array[:, 1], array[:, 2], c=array[:, 3], cmap='Set1', alpha=0.75, vmin=2,
                             vmax=15)
    scatter_plot.set_xlabel(x_label)
    scatter_plot.set_ylabel(y_label)
    scatter_plot.set_zlabel(z_label)
    scatter_plot.set_title(title)

    fig.colorbar(p, label='number of neighbors')
    plt.show()


def plot_3d_scatter_with_color_and_gravity_center(array, x_label, y_label, z_label, title, gravity_center):
    """
       Plots a 3D scatter plot with color coding and highlights the gravity center.

       This function creates a 3D scatter plot where the first three dimensions of the array are used for x, y,
       and z coordinates, and the fourth dimension for color coding. Additionally, it highlights the gravity center
       of the plot.

       Parameters:
       array (numpy.ndarray): A 4D array where the first three columns are coordinates and the fourth is color.
       x_label (str): The label for the x-axis.
       y_label (str): The label for the y-axis.
       z_label (str): The label for the z-axis.
       title (str): The title of the plot.
       gravity_center (tuple/list): The coordinates of the gravity center to highlight on the plot.

       """
    fig = plt.figure()
    scatter_plot = fig.add_subplot(111, projection='3d')

    p = scatter_plot.scatter(array[:, 0], array[:, 1], array[:, 2], c=array[:, 3], cmap='Set1', alpha=0.75)
    scatter_plot.set_xlabel(x_label)
    scatter_plot.set_ylabel(y_label)
    scatter_plot.set_zlabel(z_label)
    scatter_plot.set_title(title)
    scatter_plot.scatter(gravity_center[0], gravity_center[1], gravity_center[2], c='black')

    fig.colorbar(p, label='number of neighbors')
    plt.show()


def plot_3d_scatter_fixed(array, fixed_list, x_label, y_label, z_label, title):
    """
    Plots a 3D scatter plot highlighting specific points from a given list.

    This function creates a 3D scatter plot where the coordinates are taken from the array. It additionally
    highlights specific points, specified by 'fixed_list', in a different color.

    Parameters:
    array (numpy.ndarray): A 2D array where the first three columns are coordinates.
    fixed_list (list): A list of indices from the array to highlight on the plot.
    x_label (str): The label for the x-axis.
    y_label (str): The label for the y-axis.
    z_label (str): The label for the z-axis.
    title (str): The title of the plot.

    """
    fig = plt.figure()
    scatter_plot = fig.add_subplot(111, projection='3d')

    scatter_plot.scatter(array[:, 0], array[:, 1], array[:, 2], c='grey', cmap='Set1', alpha=0.5)
    scatter_plot.scatter(array[fixed_list, 0], array[fixed_list, 1], array[fixed_list, 2], color='red')
    scatter_plot.set_xlabel(x_label)
    scatter_plot.set_ylabel(y_label)
    scatter_plot.set_zlabel(z_label)
    scatter_plot.set_title(title)

    plt.show()


def plot_3d_scatter_with_color_and_gravity_center_and_gravity_median(array, x_label, y_label, z_label, title,
                                                                     gravity_center, gravity_median):
    """
       Plots a 3D scatter plot and highlights both the gravity center and gravity median.

       This function creates a 3D scatter plot from a given array. It also highlights two specific points:
       the gravity center and the gravity median, using different colors for each.

       Parameters:
       array (numpy.ndarray): A 2D array where the first three columns are coordinates.
       x_label (str): The label for the x-axis.
       y_label (str): The label for the y-axis.
       z_label (str): The label for the z-axis.
       title (str): The title of the plot.
       gravity_center (tuple/list): The coordinates of the gravity center to highlight on the plot.
       gravity_median (tuple/list): The coordinates of the gravity median to highlight on the plot.

   """
    fig = plt.figure()
    scatter_plot = fig.add_subplot(111, projection='3d')

    scatter_plot.scatter(array[:, 0], array[:, 1], array[:, 2], alpha=0.75)
    scatter_plot.set_xlabel(x_label)
    scatter_plot.set_ylabel(y_label)
    scatter_plot.set_zlabel(z_label)
    scatter_plot.set_title(title)
    scatter_plot.scatter(gravity_center[0], gravity_center[1], gravity_center[2], c='black')
    scatter_plot.scatter(gravity_median[0], gravity_median[1], gravity_median[2], c='blue')

    plt.show()
