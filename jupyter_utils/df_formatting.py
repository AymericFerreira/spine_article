import pandas as pd
import pandas.io.formats.style
import numpy as np


def show_df_info(df: pd.DataFrame,
                 columns_to_show: [str],
                 columns_to_groupby: [str],
                 ) -> pandas.io.formats.style.Styler:
    """
    Formats and styles a DataFrame based on group-by operation for visualization.

    This function groups the DataFrame by specified columns and computes aggregated statistics
    (like mean) for specified columns to show. It then returns a styled DataFrame object with
    custom formatting, including bar visualization for counts and a background gradient.

    Parameters:
    df (pd.DataFrame): The DataFrame to be processed.
    columns_to_show ([str]): List of column names whose statistics are to be displayed.
    columns_to_groupby ([str]): List of column names to group the DataFrame by.

    Returns:
    pandas.io.formats.style.Styler: The styled DataFrame object for display.
    """

    data_df = df_info(df, columns_to_show, columns_to_groupby)

    return data_df.style \
        .format("{:.3f}", subset=columns_to_show) \
        .bar(subset="Count", color="Teal") \
        .background_gradient(subset=columns_to_show, axis=0, cmap="RdYlGn")


def show_variance(df: pd.DataFrame,
                  columns_to_show: [str],
                  columns_to_groupby: [str],
                  ) -> pandas.io.formats.style.Styler:
    """
    Formats and styles a DataFrame based on group-by operation to show variance.

    This function groups the DataFrame by specified columns and computes the standard deviation
    for specified columns to show. It then returns a styled DataFrame object with custom formatting,
    including bar visualization for counts and a background gradient.

    Parameters:
    df (pd.DataFrame): The DataFrame to be processed.
    columns_to_show ([str]): List of column names whose variance is to be displayed.
    columns_to_groupby ([str]): List of column names to group the DataFrame by.

    Returns:
    pandas.io.formats.style.Styler: The styled DataFrame object for display.
    """

    data_df = df_variance(df, columns_to_show, columns_to_groupby)

    return data_df.style \
        .format("{:.3f}", subset=columns_to_show) \
        .bar(subset="Count", color="Teal") \
        .background_gradient(subset=columns_to_show, axis=0, cmap="RdYlGn")


def dataset_style(df: pd.DataFrame,
                  columns_to_show: [str]
                  ):
    """
    Applies styling to a DataFrame for enhanced visualization.

    This function applies custom formatting to the given DataFrame, including number formatting,
    bar visualization for counts, and background gradient. It also handles special formatting
    for percentage columns.

    Parameters:
    df (pd.DataFrame): The DataFrame to be styled.
    columns_to_show ([str]): List of column names to apply specific styling.

    Returns:
    pandas.io.formats.style.Styler: The styled DataFrame object for display.
    """
    percent_column = None
    for col in df.columns:
        if "Percent" in col:
            percent_column = col
            # previous color for percent column was #05386B
    if percent_column:
        return df.style \
            .format("{:.3f}", subset=columns_to_show) \
            .format("{:.2%}", subset=percent_column) \
            .bar(subset="Count", color="Teal") \
            .bar(subset=percent_column, color="#557A95") \
            .background_gradient(subset=columns_to_show, axis=0, cmap="RdYlGn")
    return df.style \
        .format("{:.3f}", subset=columns_to_show) \
        .bar(subset="Count", color="Teal") \
        .background_gradient(subset=columns_to_show, axis=0, cmap="RdYlGn")


def df_info(df: pd.DataFrame,
            columns_to_show: [str],
            columns_to_groupby: [str],
            ) -> pd.DataFrame:
    """
    Groups a DataFrame by specified columns and computes mean values for other specified columns.

    This function is used to aggregate data in a DataFrame by grouping based on specified columns
    and then computing the mean for each group for other specified columns. It also calculates the
    count of each group.

    Parameters:
    df (pd.DataFrame): The DataFrame to be processed.
    columns_to_show ([str]): List of column names to calculate mean values for.
    columns_to_groupby ([str]): List of column names to group the DataFrame by.

    Returns:
    pd.DataFrame: The resulting aggregated DataFrame.
    """
    mean_df = df.groupby(columns_to_groupby, sort=False)[columns_to_show] \
        .apply("mean") \
        .sort_index(level=0, ascending=True)
    data_df = mean_df

    count_df = df.groupby(columns_to_groupby, sort=False)[columns_to_show] \
        .apply("count")[columns_to_show[0]]
    data_df["Count"] = count_df
    new_col_order = [data_df.columns[-1]] + list(data_df.columns[:-1])
    return data_df[new_col_order]


def df_variance(df: pd.DataFrame,
                columns_to_show: [str],
                columns_to_groupby: [str],
                ) -> pd.DataFrame:
    """
    Groups a DataFrame by specified columns and computes standard deviation for other specified columns.

    This function is used to aggregate data in a DataFrame by grouping based on specified columns
    and then computing the standard deviation for each group for other specified columns.

    Parameters:
    df (pd.DataFrame): The DataFrame to be processed.
    columns_to_show ([str]): List of column names to calculate standard deviation for.
    columns_to_groupby ([str]): List of column names to group the DataFrame by.

    Returns:
    pd.DataFrame: The resulting aggregated DataFrame with standard deviation.
    """
    std_df = df.groupby(columns_to_groupby, sort=False)[columns_to_show] \
        .apply("std") \
        .sort_index(level=0, ascending=True)
    data_df = std_df
    new_col_order = [data_df.columns[-1]] + list(data_df.columns[:-1])
    return data_df[new_col_order]


def percent_of_specific_pop(df: pd.DataFrame,
                            columns_to_show: [str],
                            columns_to_groupby: [str],
                            column_to_compare: str,
                            ) -> pd.DataFrame:
    """
    Computes and returns the percentage of a specific population within each group of a DataFrame.

    This function first aggregates the DataFrame by specified group-by columns and then computes
    the percentage of each unique value in the 'column_to_compare' within each group.

    Parameters:
    df (pd.DataFrame): The DataFrame to be processed.
    columns_to_show ([str]): List of column names for which to compute the aggregated statistics.
    columns_to_groupby ([str]): List of column names to group the DataFrame by.
    column_to_compare (str): The column name to compute percentages for each unique value.

    Returns:
    pd.DataFrame: The resulting DataFrame with additional percentage columns.
    """
    df2 = df_info(df, columns_to_show, columns_to_groupby)
    number_of_pop = len(df[column_to_compare].unique())
    pops = [np.arange(pop, len(df2), number_of_pop) for pop in range(number_of_pop)]
    data = [df2.iloc[pop, [0]] for pop in pops]
    data /= np.sum(data, axis=0)
    percent_pop = []
    for percent_pops in zip(*data):
        list(map(percent_pop.append, list(np.array(list(percent_pops)).flatten())))
    df2[f"Percent {column_to_compare}"] = percent_pop
    return df2


def show_percent_of_specific_pop(df: pd.DataFrame,
                                 columns_to_show: [str],
                                 columns_to_groupby: [str],
                                 column_to_compare: str,
                                 ) -> pandas.io.formats.style.Styler:
    """
    Formats and styles a DataFrame to show the percentage of a specific population within each group.

    This function computes the percentage of each unique value in 'column_to_compare' within each group
    defined by 'columns_to_groupby' and applies custom styling to the resulting DataFrame.

    Parameters:
    df (pd.DataFrame): The DataFrame to be processed.
    columns_to_show ([str]): List of column names for which to compute the aggregated statistics.
    columns_to_groupby ([str]): List of column names to group the DataFrame by.
    column_to_compare (str): The column name to compute percentages for each unique value.

    Returns:
    pandas.io.formats.style.Styler: The styled DataFrame object for display.
    """
    df_ = percent_of_specific_pop(df, columns_to_show, columns_to_groupby, column_to_compare)
    return dataset_style(df_, columns_to_show)


def show_corr(df: pd.DataFrame,
              columns_to_show: [str],
              cmap="viridis"):
    """
    Displays a styled correlation matrix for specified columns of a DataFrame.

    This function calculates the correlation matrix for the specified columns and applies a
    background gradient styling for better visualization. The color map for the gradient can be
    customized.

    Parameters:
    df (pd.DataFrame): The DataFrame containing the data.
    columns_to_show ([str]): List of column names to include in the correlation matrix.
    cmap (str, optional): The color map to use for background gradient. Defaults to 'viridis'.

    Returns:
    pandas.io.formats.style.Styler: The styled correlation matrix for display.
    """
    return df[columns_to_show].corr().style.background_gradient(cmap=cmap)


if __name__ == '__main__':
    cols = ["Length", "Surface", "Hull Ratio", "CVD", "Open Angle"]
    pca_cols = ["Principal component 1", "Principal component 2", "Principal component 3"]
