import plotly.graph_objects as go
from plotly_utils.themes import general_fig_layout


def generate_histogram(groups):
    """
    Generates and displays a histogram using Plotly, with bars representing different groups.

    This function takes a list of groups, each represented by a dictionary containing bar properties,
    and creates a bar chart (histogram) using Plotly. The function utilizes the 'go.Bar' class from
    Plotly to create individual bars for each group and adds them to a single figure.

    Parameters:
    groups (list of dict): A list where each element is a dictionary representing the properties of a bar.
                           Each dictionary should contain the data and visual properties for each group's bar.

    Returns:
    None: The function displays the histogram using Plotly but does not return any value.

    Example of 'groups' parameter:
    groups = [
        {'x': ['A', 'B', 'C'], 'y': [10, 15, 20], 'name': 'Group 1'},
        {'x': ['A', 'B', 'C'], 'y': [5, 7, 12], 'name': 'Group 2'}
    ]
    """
    fig = go.Figure()
    for group in groups:
        fig.add_trace(go.Bar(group))
    general_fig_layout(fig).show()