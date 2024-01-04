import os
import pathlib
import sys
import czifile
import numpy as np
import tifffile
import morphsnakes as ms

import threading

try:
    import imagej
except ImportError:
    print("ImageJ not found. Deconvolution will not work.")

np.set_printoptions(threshold=sys.maxsize)  # variable output


def flatten_image(img: np.ndarray) -> np.ndarray:
    """
       Flattens a multidimensional image by taking the maximum value over the last axis.

       This function processes an image array by collapsing the last axis (axis=3). It computes the maximum value
       along this axis for each position in the earlier dimensions. This is commonly used in image processing
       to reduce a multichannel image to a single channel.

       Parameters:
       img (numpy.ndarray): The multidimensional image array to be flattened.

       Returns:
       numpy.ndarray: The flattened image array.
   """
    return np.amax(img, axis=3)


def get_filepaths(directory, condition="", pathlib_bool=True):
    """
    This function will generate the file names in a directory
    tree by walking the tree either top-down or bottom-up. For each
    directory in the tree rooted at directory top (including top itself),
    it yields a 3-tuple (dirpath, dirnames, filenames).
    """
    file_paths = []

    for root, directories, files in os.walk(directory):
        for filename in files:
            file_path = os.path.join(root, filename)
            if (
                    condition in filename
                    and pathlib_bool
            ):
                file_paths.append(pathlib.Path(file_path))
            elif condition in filename or not condition:
                file_paths.append(file_path)
    return file_paths


def process_folder(folder):
    """
       Processes a folder of TIFF images for segmentation using the Morphological Chan-Vese algorithm.

       This function iterates over each TIFF file in the specified folder, applies Morphological Chan-Vese
       segmentation, and saves the segmented images in a subfolder named 'Segmented' within the parent directory
       of the provided folder.

       Parameters:
       folder (str or pathlib.Path): The path to the folder containing TIFF images to be segmented.

       Returns:
       None: The function performs segmentation and saves the segmented images, but does not return any value.
       """
    folder = pathlib.Path(folder)
    segmentation_path = folder.parent.joinpath("Segmented/")
    segmentation_path.mkdir(exist_ok=True, parents=True)
    for filename in get_filepaths(folder, ".tif"):
        morphologic_chan_vese_segmentation(filename, segmentation_path)


def multi_process_folder(folder, n_threads=5):
    """
        Processes a folder of TIFF images for segmentation in parallel using multiple threads.

        This function divides the task of segmenting TIFF images using the Morphological Chan-Vese algorithm
        across multiple threads for parallel processing. The segmented images are saved in a subfolder named
        'Segmented' within the parent directory of the provided folder.

        Parameters:
        folder (str or pathlib.Path): The path to the folder containing TIFF images to be segmented.
        n_threads (int, optional): The number of threads to use for parallel processing. Defaults to 5.

        Returns:
        None: The function performs parallel segmentation and saves the segmented images, but does not return any value.
        """
    folder = pathlib.Path(folder)
    segmentation_path = folder.parent.joinpath("Segmented/")
    segmentation_path.mkdir(exist_ok=True, parents=True)

    file_list = get_filepaths(folder, ".tif")
    # Splitting the items into chunks equal to number of threads
    array_chunk = np.array_split(file_list, n_threads)
    thread_list = []
    for thr in range(n_threads):
        thread = threading.Thread(target=threaded_process, args=(array_chunk[thr], segmentation_path), )
        thread_list.append(thread)
        thread_list[thr].start()

    for thread in thread_list:
        thread.join()
    print(f"Images are segmented and saved in {segmentation_path} folder")


def threaded_process(items_chunk, segmentation_path):
    """
       Processes a chunk of TIFF images for segmentation in a separate thread.

       This function is designed to be run in a separate thread, processing a chunk of TIFF images for segmentation
       using the Morphological Chan-Vese algorithm. The segmented images are saved in the specified segmentation path.

       Parameters:
       items_chunk (list): A list of TIFF image file paths to be processed.
       segmentation_path (pathlib.Path): The path where segmented images will be saved.

       Returns:
       None: The function processes and saves segmented images, but does not return any value.
       """
    for item in items_chunk:
        morphologic_chan_vese_segmentation(item, segmentation_path)


def morphologic_chan_vese_segmentation(filename, segmentation_path, mu=0, lambda1=1, lambda2=4, iterations=100):
    """
        Applies Morphological Chan-Vese segmentation to a single TIFF image and saves the result.

        This function reads a TIFF image, applies Morphological Chan-Vese segmentation with specified parameters,
        and saves the segmented image in the provided segmentation path. The saved filename incorporates the
        segmentation parameters for reference.

        Parameters:
        filename (str or pathlib.Path): The path to the TIFF image file to be segmented.
        segmentationPath (pathlib.Path): The path where the segmented image will be saved.
        mu (float, optional): The 'smoothing' parameter for segmentation. Defaults to 1.
        lambda1 (float, optional): The 'lambda1' parameter for foreground segmentation. Defaults to 1.
        lambda2 (float, optional): The 'lambda2' parameter for background segmentation. Defaults to 4.
        iterations (float, optional): The "iterations" corresponds to the number of iterations to run. Defaults to 100.

        Returns:
        None: The function performs segmentation and saves the segmented image
    """
    img = tifffile.imread(filename)

    segmented_image = ms.morphological_chan_vese(img, iterations=iterations,
                                                 init_level_set='checkerboard',
                                                 smoothing=mu, lambda1=lambda1, lambda2=lambda2)

    image_stack = img * segmented_image

    segmented_image_name = f"{segmentation_path}/{filename.stem}_{mu}_{lambda1}_{lambda2}_{iterations}_{filename.suffix}"
    tifffile.imwrite(segmented_image_name, image_stack)


def get_list_of_files_pattern(dir_name: str, pattern=None):
    """
    Get a list of files in folder dirName ONLY
    :param
    dirName: relative or absolute path to the directory
    pattern: a string that needs to be contained in the filename
    :return: List of files
    """

    list_of_file = os.listdir(dir_name)
    all_files = []

    for entry in list_of_file:
        full_path = os.path.join(dir_name, entry)
        filename = full_path.split('/')[-1]
        if not pattern or pattern in filename:
            all_files.append(full_path)
    return all_files


def deconvolve_folder(dir_path: str, fiji_path):
    """
    Deconvolves all the image files in a specified directory using Fiji.

    This function iterates over all image files in the specified directory,
    identified by `dirpath`, and applies a deconvolution process to each image.
    The deconvolution process is carried out using the Fiji software, the path to
    which is specified by `fiji_path`.

    Parameters:
    dir_path (str): The path to the directory containing the image files to be deconvolve.
    fiji_path (str): The path to the Fiji software used for deconvolution.

    Returns:
    None: The function performs deconvolution on the images but does not return any value.
    """
    for imageName in get_list_of_files_pattern(dir_path):
        deconvolution(imageName, fiji_path)


def deconvolution(filename: str, fiji_path: str):
    """
    Performs image deconvolution on files with specific formats using Fiji's ImageJ.

    This function reads an image file, checks its format (.czi, .tif, or .lsm), and performs
    image deconvolution using Fiji's ImageJ software. The deconvolved image is saved in a
    'deconvolved' directory within the same directory as the input file.

    Parameters:
    filename (str): Path to the image file to be deconvolved.
    fiji_path (str): Path to the Fiji/ImageJ installation directory.

    The function performs the following operations:
    1. Reads the image file and adjusts the dimensions if necessary, based on the file format.
    2. Initializes Fiji/ImageJ and creates temporary and deconvolved directories if they don't exist.
    3. Writes the image to a temporary file and then runs a Fiji macro for deconvolution.
    4. The result is saved in the 'deconvolved' directory with a modified file name.

    Note:
    - Supported file formats are .czi, .tif, and .lsm. Other formats will result in a notification
      that the file format is not supported.
    - The function assumes Fiji/ImageJ is installed and accessible at the given path.

    Example Usage:
    >>> deconvolution("path/to/image.czi", "/path/to/Fiji.app")
    """
    image_path = pathlib.Path(filename)
    if image_path.suffix == '.czi':
        image = czifile.imread(image_path)
        if len(image.shape) == 7:
            image = image[0, 0, 0, :, :, :, 0]
    elif image_path.suffix == '.tif':
        image = tifffile.imread(image_path)
    elif image_path.suffix == '.lsm':
        image = tifffile.imread(image_path)
        if len(image.shape) == 5:
            image = image[0, 0, :, :, :]
    else:
        print(f"Not supported file format {image_path.suffix}")
        return

    ij = imagej.init(fiji_path)
    ij.getContext().dispose()
    ij = imagej.init(fiji_path, new_instance=True)

    if not os.path.exists(f"{pathlib.Path(image_path.drive, image_path.parent, 'temp')}"):
        os.mkdir(f"{pathlib.Path(image_path.drive, image_path.parent, 'temp')}")

    if not os.path.exists(f"{pathlib.Path(image_path.drive, image_path.parent, 'deconvolved')}"):
        os.mkdir(f"{pathlib.Path(image_path.drive, image_path.parent, 'deconvolved')}")

    tifffile.imwrite(
        f"{pathlib.Path(image_path.drive, image_path.parent, 'temp', f'{image_path.stem}.tif').as_posix()}",
        image,
    )

    macro = f"""
    open("{pathlib.Path(image_path.drive, image_path.parent, 'temp', f'{image_path.stem}.tif').as_posix()}")
    run("Iterative Deconvolve 3D", "image={image_path.stem} point={image_path.stem} output=Deconvolved show perform wiener=0.000 low=1 z_direction=1 maximum=1 terminate=0.000")
    selectWindow("Deconvolved_1");
    saveAs("Tiff", "{pathlib.Path(image_path.drive, image_path.parent, 'deconvolved', image_path.stem).as_posix()}_Deconvolved.tif");
    """

    args = {}

    ij.py.run_macro(macro, args)
    print(f"Saved : {pathlib.Path(image_path.drive, image_path.parent, 'deconvolved', image_path.stem)}_Deconvolved.tif")


def convert_to_8bit_folder(dirpath: str):
    """
        Converts all image files in a specified directory to 8-bit format.

        This function iterates over all image files in a given directory and converts each one to
        8-bit format. The conversion is performed by the `convert_to_8bit` function, which needs to
        be defined elsewhere in the code. The function uses `get_list_of_files_pattern` to retrieve
        a list of image file names from the directory.

        Parameters:
        dirpath (str): Path to the directory containing the image files to be converted.

        The function performs the following operations:
        1. Retrieves a list of image file names in the specified directory.
        2. Iterates over each file name and calls the `convert_to_8bit` function on it.

        Example Usage:
        >>> convert_to_8bit_folder("/path/to/image/directory")
        """
    for imageName in get_list_of_files_pattern(dirpath):
        convert_to_8bit(imageName)


def convert_to_8bit(imagepath: str):
    """
        Converts an image to 8-bit format and overwrites the original file.

        This function reads an image from the specified path, normalizes it, scales it to 255 to fit
        the 8-bit format, converts it to an unsigned 8-bit integer type, and then overwrites the original
        image file with this 8-bit version.

        Parameters:
        imagepath (str): The file path of the image to be converted.

        The function performs the following operations:
        1. Reads the image from the given file path.
        2. Normalizes the image using the `normalize_image` function (needs to be defined elsewhere).
        3. Scales the normalized image to a maximum value of 255.
        4. Converts the scaled image to an 8-bit unsigned integer format.
        5. Overwrites the original image file with the converted 8-bit image.

        Note:
        - The original image file is overwritten with the new 8-bit image. Ensure to create backups if
          the original data needs to be preserved.

        Example Usage:
        >>> convert_to_8bit("/path/to/image.tif")
        """
    image = tifffile.imread(imagepath)
    image = normalize_image(image)
    image = image * 255
    image = image.astype('uint8')
    tifffile.imwrite(imagepath, image)


def normalize_image(image: np.ndarray) -> np.ndarray:
    """
        Normalizes an image array to a range between 0 and 1.

         Example Usage:
    >>> image_array = np.array([[0, 127, 255], [63, 191, 127]])
    >>> normalized_image = normalize_image(image_array)
    >>> print(normalized_image)
    array([[0.  , 0.5 , 1.  ],
           [0.25, 0.75, 0.5 ]])
    """
    return (image - image.min()) / (image.max() - image.min())


def czi_to_tif(folder: str):
    """
        Converts CZI image files in a specified folder to TIFF format.

        This function reads CZI files (.czi extension) from the given folder, processes each
        image, and then saves it in TIFF (.tif) format in a subfolder named 'Images' within
        the same directory.

        Parameters:
        folder (str): The directory path where the CZI files are located.

        The function performs the following operations:
        1. Retrieves a list of CZI file paths from the specified folder.
        2. For each file, reads the image using `czifile.imread`, squeezes the image dimensions
           to remove singleton dimensions, and saves it as a TIFF file in the 'Images' subfolder.
           The TIFF file is named after the original CZI file's stem (base name without extension).

        - The 'Images' subfolder is expected to exist within the specified folder. If it does not,
          the function will fail to save the TIFF files.

        Example Usage:
        >>> czi_to_tif("/path/to/folder")
        """
    filenames = get_filepaths(folder, ".czi")
    for filename in filenames:
        czi_img = czifile.imread(filename)
        tiff_img = np.squeeze(czi_img)
        tifffile.imwrite(f"{folder}/Images/{filename.stem}.tif", tiff_img)


def regroup_files(folder, extension=".czi"):
    """
       Converts image files in a specified folder to .tif format and organizes them into subfolders.

       This function reads image files with a specified extension (default is .czi) from a given folder,
       converts each image to .tif format, and saves the .tif files into a subfolder named 'Images'.
       It also creates a 'Deconvolved' subfolder for later use.

       Parameters:
       folder (str): The directory path where the image files are located.
       extension (str, optional): The file extension of the images to be processed (default is '.czi').
    """
    folder_path = pathlib.Path(folder)
    image_path = folder_path.joinpath("Images/")
    image_path.mkdir(exist_ok=True, parents=True)
    filenames = get_filepaths(folder, extension)
    count = 0
    for filename in filenames:
        if "colloc" not in filename.__str__():
            czi_img = czifile.imread(filename)
            tiff_img = np.squeeze(czi_img)
            tifffile.imwrite(f"{folder}/Images/{filename.stem}.tif", tiff_img)
            count += 1
    deconvolve_path = folder_path.joinpath("Deconvolved/")
    deconvolve_path.mkdir(exist_ok=True, parents=True)


def remove_space(folder: str):
    """
        Renames all .tif files in a specified folder by removing spaces from their filenames.

        This function iterates over all .tif files in the given folder and renames each file by
        removing any spaces in its filename. The renaming is done only if the original filename
        contains spaces.

        Parameters:
        folder (str): The directory path where the .tif files are located.
    """
    for filename in get_filepaths(folder, ".tif", pathlib_bool=False):
        filename_without_space = filename.replace(" ", "")
        if filename != filename_without_space:
            os.rename(filename, filename_without_space)


if __name__ == "__main__":
    # deconvolve_folder("Images", "Fiji.app")
    multi_process_folder("Images")
