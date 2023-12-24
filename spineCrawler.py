import re
import pathlib
import os
import subprocess
from shutil import copyfile
import pandas as pd


def regroup_files(folder, extension=".czi"):
    filenames = get_filepaths(folder, extension)
    for filename in filenames:
        if "colloc" not in filename.__str__():
            # Assure that there is no space
            try:
                filename.rename(pathlib.Path(f"D:\Downloads\\complex\\{filename.name.replace(' ', '_')}"))
            except FileExistsError:
                pass


def get_filepaths(directory, condition=None, pathlibBool=True):
    """
    This function will generate the file names in a directory
    tree by walking the tree either top-down or bottom-up. For each
    directory in the tree rooted at directory top (including top itself),
    it yields a 3-tuple (dirpath, dirnames, filenames).
    """
    file_paths = []

    for root, directories, files in os.walk(directory):
        for filename in files:
            filepath = os.path.join(root, filename)
            if (
                    condition in filename
                    and pathlibBool
                    or not condition
                    and pathlibBool
            ):
                file_paths.append(pathlib.Path(filepath))
            elif condition in filename or not condition:
                file_paths.append(filepath)
    return file_paths


def count_spines():
    filepaths = get_filepaths("D:/Downloads/temp", ".ply")
    number = sum(
        1 for file in filepaths if re.search("spine(\d+).ply", file.__str__())
    )


def copy_spines(base_folder, destination_folder):
    filepaths = get_filepaths(base_folder, ".ply")
    # print(filepaths)
    # exit()
    spineNumber = 0
    for file in filepaths:
        # print(file)
        # print(file)
        # if re.search("spine(\d+).ply", file.__str__().split("_")[0]):
        if re.search("spine(\d+)", file.stem.__str__().split("_")[0]):
            # if "finalMesh" in file.__str__():
            #     print(file)
            #     exit()
            # print(["cp", f"{file}", f"D:/Downloads/spines/{file.parents._parts[-2]}_{file.name}"])
            # subprocess.call(["cp", f"{file.as_posix()}", f"D:/Downloads/spines/{file.parents._parts[-2]}_{file.name}"])
            copyfile(f"{file.as_posix()}", f"{destination_folder}/{file.parents._parts[-2]}_{file.name}")
            spineNumber += 1
    print(f"{spineNumber} spines copied to {destination_folder}")


def parser(filename):
    filename = filename.__str__()
    # print(filename)
    animalParser = re_parser("Deconvolved_5_(.+?)_", filename)
    if not animalParser:
        print("No animal", filename)
    sliceParser = re_parser("slice(.+?)_", filename)
    cellParser = re_parser("cell(.+?)_", filename)
    dendriteParser = re_parser("dendrite(.+?)_", filename)
    if not dendriteParser:
        dendriteParser = re_parser("dendritic_(.+?)_", filename)
    shfParser = "No"
    if "shf" in filename.split("_")[-1]:
        shfParser = "shf"
    if "SHF" in filename.split("_")[-1]:
        shfParser = "SHF"
    if shfParser != "No":
        spineParser = re_parser("spine(.+?)", filename.split('_')[-2])
    else:
        spineParser = re_parser("spine(.+?)", filename.split('_')[-1])
    return animalParser, sliceParser, cellParser, dendriteParser, spineParser, shfParser
    # print(animalParser, sliceParser, cellParser, dendriteParser, spineParser)


def re_parser(regexpression, text, case_sensitive=False):
    if not case_sensitive:
        regParser = re.search(regexpression, text, flags=re.IGNORECASE)
    else:
        regParser = re.search(regexpression, text)
    if regParser:
        return (text[regParser.regs[1][0]:regParser.regs[1][1]]).replace("_", "")


def create_data_file():
    """"
    This function count in order, it's working because after filepaths is used, the logic is always the same,
    (example : cell1_spine1, cell1_spine2, cell2_spine1, cell2_spine2 ...)
    although, it will not work if we have cell1_spine1 then cell2_spine1 and then cell1_spine2 (it will count 2 cell2)
    """
    df = pd.DataFrame(columns=["Animal", "Slice", "Cell", "Dendrite", "SpineNumber"])
    for file in get_filepaths("D:/Downloads/spines2", ".ply"):
        if re.search("spine(\d+).ply", file.__str__()):
            # print(file.stem)
            # animalParser, sliceParser, cellParser, dendriteParser, _ = parser(file.stem)
            animalParser, sliceParser, cellParser, dendriteParser, spineParser, shfParser = parser(file.stem)
            # print((df["Animal"] == animalParser))
            idx = ((df["Animal"] == animalParser) & (df["Slice"] == sliceParser) & (df["Cell"] == cellParser) &
                   (df["Dendrite"] == dendriteParser))
            # idx = ((df["Animal"] == animalParser) & (df["Slice"] == sliceParser) & (df["Cell"] == cellParser) &
            #        (df["Dendrite"] == dendriteParser)).index.tolist()
            # print(idx.index.tolist(), animalParser, sliceParser, cellParser, dendriteParser)
            if idx.any():
                df["SpineNumber"].iloc[len(idx.index.tolist()) - 1] = df["SpineNumber"].iloc[
                                                                          len(idx.index.tolist()) - 1] + 1
            else:
                df2 = pd.DataFrame([[animalParser, sliceParser, cellParser, dendriteParser, 0]],
                                   columns=["Animal", "Slice", "Cell", "Dendrite", "SpineNumber"])
                df = df.append(df2)
    df.to_csv("spine_number_new.csv", index=False)

    # df2 = pd.DataFrame([[]], columns=["Animal", "Slice", "Cell", "Dendrite", "SpineNumber"])
    # print(["cp", f"{file}", f"D:/Downloads/spines/{file.parents._parts[-2]}_{file.name}"])
    # subprocess.call(["cp", f"{file.as_posix()}", f"D:/Downloads/spines/{file.parents._parts[-2]}_{file.name}"])


if __name__ == '__main__':
    # regroup_files("D:\Downloads\\complex\\")
    # exit()

    # def regroup_files(folder, extension=".czi"):
    #     filenames = get_filepaths(folder, extension)
    #     for filename in filenames:
    #         if "colloc" not in filename.__str__():
    #             Assure that there is no space
    # try:
    #     filename.rename(pathlib.Path(f"D:\Downloads\\complex\\{filename.name.replace(' ', '_')}"))
    # except FileExistsError:
    #     pass
    # for folder in ["D:/Downloads/temp/11966_1", "D:/Downloads/temp/11957_2", "D:/Downloads/temp/Animal1"]:
    # for folder in ["D:/Downloads/complex/", "D:/Downloads/temp/"]:
    # for folder in ["D:/Downloads/new_spines/17030", "D:/Downloads/new_spines/17228"]:
    # for folder in ["D:\Downloads/new_spines\Baseline/17228", "D:\Downloads/new_spines\Baseline/17030"]:
    #     copy_spines(folder)
    # copy_spines(r"D:\Downloads\new_spines", r"D:\Downloads\spines_2022")
    copy_spines(r"D:\Downloads\new_spines", r"D:\Downloads\spines_2022")
        # count_spines()
        # create_data_file()
