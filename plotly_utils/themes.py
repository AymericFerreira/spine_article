import plotly.graph_objects as go


def general_fig_layout(fig: go.Figure) -> go.Figure:
    """
    Applies a general layout and styling to a Plotly figure.

    This function takes a Plotly figure object and updates its layout and styling to provide a
    consistent and visually appealing appearance. It sets the figure template, updates the axes
    with specific line properties, and adjusts font settings.

    Parameters:
    fig (go.Figure): The Plotly figure object to be styled.

    Returns:
    go.Figure: The updated Plotly figure object with applied styling.

    The function performs the following updates:
    - Sets the figure template to 'plotly_white' for a clean and clear background.
    - Updates both x and y axes to show lines with specified thickness, color, and tick properties.
    - Sets the autosize property to True, allowing the figure to adjust its size dynamically.
    - Updates the font to 'Arial' with a specified size and color for text elements in the figure.
    """
    fig.update_layout(template="plotly_white")
    fig.update_xaxes(showline=True, linewidth=2, linecolor='black', ticks="outside", tickwidth=1, tickcolor='black',
                     ticklen=10)
    fig.update_yaxes(showline=True, linewidth=2, linecolor='black', ticks="outside", tickwidth=1, tickcolor='black',
                     ticklen=10)
    fig.update_layout(
        autosize=True,
        font=dict(family="Arial", size=20, color="black")
    )
    return fig
