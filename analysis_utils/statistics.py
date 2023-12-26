import pandas as pd
from statsmodels.stats.proportion import test_proportions_2indep
import itertools


def generate_populations(df: pd.DataFrame, cols: [str]) -> {str: [int]}:
    """
        Generates a dictionary of population counts grouped by specified columns in a DataFrame.

        This function groups the data in the provided DataFrame based on the specified columns and counts the
        number of occurrences in each group. It returns a dictionary where each key corresponds to a unique
        value from the first column in 'cols' and the value is a list of counts for each group.

        Parameters:
        df (pd.DataFrame): The DataFrame to process.
        cols ([str]): A list of column names to group the DataFrame by.

        Returns:
        dict: A dictionary with keys as unique values from the first column in 'cols' and values as lists of counts for each group.
    """
    data_dict = {}
    df2 = df.groupby(cols).size().to_frame(name='Count')
    df2.reset_index(inplace=True)
    for cond in df[cols[0]].unique():
        df3 = df2[df2[cols[0]] == cond]
        data_dict[cond] = list(df3["Count"])

    return data_dict


def compare_populations(data_dict: {str: [int]}, pair: [str, str]):
    """
        Compares two populations and computes statistical significance between them.

        This function takes two population groups, identified by a pair of keys from the data dictionary, and
        performs a statistical test to determine the significance of differences between them. It returns a
        dictionary with the pair as a key and a list of p-values as the value.

        Parameters:
        data_dict ({str: [int]}): A dictionary with population data.
        pair ([str, str]): A pair of keys from the data dictionary to compare.

        Returns:
        dict: A dictionary with the pair as a key and a list of p-values as the value.
    """
    a1 = data_dict[pair[0]]
    a2 = data_dict[pair[1]]
    result = []
    for i, (el1, el2) in enumerate(zip(a1, a2)):
        if el1 == 0 or el2 == 0:
            result.append(None)
        result.append(test_proportions_2indep(el1, sum(a1), el2, sum(a2))[1])
    return {f"{pair[0]}, {pair[1]}": result}


def compare_all_pairs(data_dict: {str: [int]}):
    """
        Compares all possible pairs of populations in the given data dictionary.

        This function iterates over all possible pairs of population groups in the data dictionary and
        performs a comparison on each pair using the 'compare_populations' function. It returns a list
        of dictionaries, each containing the comparison results for a pair of populations.

        Parameters:
        data_dict ({str: [int]}): A dictionary with population data.

        Returns:
        list: A list of dictionaries, each containing the comparison results for a pair of populations.
    """
    combinations = itertools.combinations(data_dict.keys(), 2)
    return [compare_populations(data_dict, pair) for pair in combinations]
