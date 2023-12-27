import re
import pathlib
import os
from shutil import copyfile
import pandas as pd


def regroup_files(folder, extension=".czi"):
    """
    Renames and reorganizes files with a specific extension within a given folder.

    This function iterates through files in the specified folder with a given extension (default .czi).
    Files not containing 'colloc' in their name are renamed (spaces replaced with underscores) and moved
    to a predefined directory.

    Parameters:
    folder (str): The folder containing the files to be regrouped.
    extension (str, optional): The file extension to filter by. Defaults to '.czi'.

    Returns:
    None: This function does not return any value. It renames and moves files.
    """
    filenames = get_filepaths(folder, extension)
    for filename in filenames:
        if "colloc" not in filename.__str__():
            # Assure that there is no space
            try:
                filename.rename(pathlib.Path(f"{folder}/{filename.name.replace(' ', '_')}"))
            except FileExistsError:
                print("File already exists")


def get_filepaths(directory, condition=None, pathlib_bool=True):
    """
        Generates a list of file paths in a directory tree.

        This function walks through a directory tree, either top-down or bottom-up, to generate file paths.
        It can optionally filter files based on a condition, such as a substring in the filename.

        Parameters:
        directory (str): The directory to walk through.
        condition (str, optional): A condition to filter the files. Default is None.
        pathlibBool (bool, optional): Whether to return paths as pathlib.Path objects. Default is True.

        Returns:
        list: A list of file paths, either as strings or pathlib.Path objects, based on 'pathlibBool'.
    """
    file_paths = []

    for root, directories, files in os.walk(directory):
        for filename in files:
            filepath = os.path.join(root, filename)
            if (
                    condition in filename
                    and pathlib_bool
                    or not condition
                    and pathlib_bool
            ):
                file_paths.append(pathlib.Path(filepath))
            elif condition in filename or not condition:
                file_paths.append(filepath)
    return file_paths


def count_spines(folder: str):
    """
    Counts the number of spine files in a specific directory.

    This function searches for files with a naming pattern indicating spine data (e.g., 'spine1.ply') in
    a predefined directory and counts them.

    Returns:
    None: This function prints the count but does not return any value.
    """
    filepaths = get_filepaths(folder, ".ply")
    number = sum(bool(re.search("spine(\d+).ply", file.__str__()))
                 for file in filepaths)
    print(f"{number} spines found in {folder}")


def copy_spines(base_folder, destination_folder):
    """
        Copies spine-related files from one folder to another.

        This function scans a base folder for files matching a spine naming pattern and copies them to a
        destination folder. The file names are preserved during copying.

        Parameters:
        base_folder (str): The folder to copy files from.
        destination_folder (str): The folder to copy files to.

        Returns:
        None: This function copies files and prints the count of copied files but does not return any value.
        """
    filepaths = get_filepaths(base_folder, ".ply")
    spine_number = 0
    for file in filepaths:
        if re.search("spine(\d+)", file.stem.__str__().split("_")[0]):
            copyfile(f"{file.as_posix()}", f"{destination_folder}/{file.parents._parts[-2]}_{file.name}")
            spine_number += 1
    print(f"{spine_number} spines copied to {destination_folder}")


def parser(filename):
    """
        Parses a filename to extract biological data identifiers.

        This function extracts identifiers like animal number, slice number, cell number, etc., from a given
        filename using regular expressions. It handles different naming conventions and special cases within the filename.

        Parameters:
        filename (str): The filename to be parsed.

        Returns:
        tuple: A tuple containing parsed elements like animal number, slice number, etc.
        """
    filename = filename.__str__()
    # animal_parser = re_parser("Deconvolved_5_(.+?)_", filename)
    animal_parser = re_parser("Animal(.+?)_", filename)
    if not animal_parser:
        print("No animal", filename)
    slice_parser = re_parser("slice(.+?)_", filename)
    cell_parser = re_parser("cell(.+?)_", filename)
    dendrite_parser = re_parser("dendrite(.+?)_", filename) or \
                      re_parser("dendritic_(.+?)_", filename)
    shf_parser = "No"
    if "shf" in filename.split("_")[-1]:
        shf_parser = "shf"
    if "SHF" in filename.split("_")[-1]:
        shf_parser = "SHF"
    if shf_parser != "No":
        spine_parser = re_parser("spine(.+?)", filename.split('_')[-2])
    else:
        spine_parser = re_parser("spine(.+?)", filename.split('_')[-1])
    return animal_parser, slice_parser, cell_parser, dendrite_parser, spine_parser, shf_parser


def re_parser(reg_expression, text, case_sensitive=False):
    """
    Parses a string using a regular expression to extract specific information.

    This function applies a regular expression to a given text to extract a specific part of the string.
    It can operate in either case-sensitive or case-insensitive mode.

    Parameters:
    reg_expression (str): The regular expression used for parsing.
    text (str): The text to apply the regular expression to.
    case_sensitive (bool, optional): Whether the search should be case-sensitive. Default is False.

    Returns:
    str or None: The extracted part of the string if a match is found, otherwise None.
    """
    if not case_sensitive:
        reg_parser = re.search(reg_expression, text, flags=re.IGNORECASE)
    else:
        reg_parser = re.search(reg_expression, text)
    if reg_parser:
        return (text[reg_parser.regs[1][0]:reg_parser.regs[1][1]]).replace("_", "")


def create_data_file():
    """
    Creates a data file summarizing the spine data from a set of files in a directory.

    This function iterates through a directory of files (matching a specific pattern) and compiles
    information about each file into a DataFrame. The resulting DataFrame is then saved as a CSV file.
    This is used for organizing and summarizing spine data.

    Returns:
    None: This function creates a CSV file but does not return any value.
    """
    df = pd.DataFrame(columns=["Animal", "Slice", "Cell", "Dendrite", "SpineNumber"])
    for file in get_filepaths("D:/Downloads/spines2", ".ply"):
        if re.search("spine(\d+).ply", file.__str__()):
            animal_parser, slice_parser, cell_parser, dendrite_parser, spine_parser, shf_parser = parser(file.stem)
            idx = ((df["Animal"] == animal_parser) & (df["Slice"] == slice_parser) & (df["Cell"] == cell_parser) &
                   (df["Dendrite"] == dendrite_parser))
            if idx.any():
                df["SpineNumber"].iloc[len(idx.index.tolist()) - 1] = df["SpineNumber"].iloc[
                                                                          len(idx.index.tolist()) - 1] + 1
            else:
                df2 = pd.DataFrame([[animal_parser, slice_parser, cell_parser, dendrite_parser, 0]],
                                   columns=["Animal", "Slice", "Cell", "Dendrite", "SpineNumber"])
                df = df.append(df2)
    df.to_csv("spine_number_new.csv", index=False)
