import re


def find_first(sequence: re, main_string: str):
    """
    Searches for the first occurrence of a specified regex pattern within a string.

    Parameters:
    sequence (re.Pattern): The compiled regular expression pattern to search for.
    main_string (str): The string in which to search for the pattern.

    Returns:
    str or None: The first occurrence of the matching string if found, otherwise None.
    """
    a = re.search(sequence, main_string)
    return None if a is None else main_string[a.regs[0][0]:a.regs[0][1]]
