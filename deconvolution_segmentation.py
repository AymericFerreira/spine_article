import contextlib
import os
import pathlib
import sys

import imagej
import scyjava_config
import czifile

# matplotlib.use('TkAgg')
# matplotlib.use('Qt5Agg')
scyjava_config.add_options('-Xmx24g')
import czifile

import matplotlib.pyplot as plt
import numpy as np
import skimage
# from skimage import util
from matplotlib.widgets import Slider, Button
from skimage.segmentation import mark_boundaries, chan_vese
from skimage import io
import tifffile
import morphsnakes as ms

import threading

counter = -1

np.set_printoptions(threshold=sys.maxsize)  # variable output

def flatten_image(img):
    return np.amax(img, axis=3)


def morphologic_chain_segmentation(dirpath, filename):
    img = tifffile.imread(f'{dirpath}/{filename}')
    callback = non_visual_callback_3d(plot_each=1)

    segmentedImage = ms.morphological_chan_vese(img, iterations=100,
                                                init_level_set='checkerboard',
                                                smoothing=0, lambda1=0, lambda2=4,
                                                iter_callback=callback)

    imageStack = img * segmentedImage
    segmentedImageName = f'segmentedImages/{(filename.split(".")[:-1])}_segmentedImage.tif'
    with skimage.external.tifffile.TiffWriter(segmentedImageName) as tif:
        for image in range(imageStack.shape[0]):
            tif.save(imageStack[image], compress=0)


def for_article_loop_over_chan_vese(dirpath, filename):
    img = tifffile.imread(f'{dirpath}/{filename}')
    # callback = visual_callback_3d(plot_each=1)

    callback = real_visual_callback_3d()
    segmentedImage = ms.morphological_chan_vese(img, iterations=100,
                                                init_level_set='checkerboard',
                                                smoothing=0, lambda1=0, lambda2=4,
                                                iter_callback=callback)
    # imageStack = img * segmentedImage
    # segmentedImageName = f'article_test/segmentedImage_{i}.tif'
    # with tifffile.TiffWriter(segmentedImageName) as tif:
    #     for image in range(imageStack.shape[0]):
    #         tif.save(imageStack[image], compress=0)


def get_filepaths(directory, condition="", pathlibBool=True):
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
            ):
                file_paths.append(pathlib.Path(filepath))
            elif condition in filename or not condition:
                file_paths.append(filepath)
    return file_paths


def process_folder(folder):
    folder = pathlib.Path(folder)
    segmentationPath = folder.parent.joinpath("Segmented/")
    segmentationPath.mkdir(exist_ok=True, parents=True)
    for filename in get_filepaths(folder, ".tif"):
        morphologic_chan_vese_segmentation(filename, segmentationPath)


def multi_process_folder(folder, n_threads=5):
    folder = pathlib.Path(folder)
    segmentationPath = folder.parent.joinpath("Segmented/")
    segmentationPath.mkdir(exist_ok=True, parents=True)

    fileList = get_filepaths(folder, ".tif")
    # Splitting the items into chunks equal to number of threads
    array_chunk = np.array_split(fileList, n_threads)
    thread_list = []
    for thr in range(n_threads):
        thread = threading.Thread(target=threaded_process, args=(array_chunk[thr], segmentationPath), )
        thread_list.append(thread)
        thread_list[thr].start()

    for thread in thread_list:
        thread.join()


def threaded_process(items_chunk, segmentationPath):
    for item in items_chunk:
        print(item)
        try:
            morphologic_chan_vese_segmentation(item, segmentationPath)
        except Exception:
            print('error with item')


def morphologic_chan_vese_segmentation(filename, segmentationPath, mu=1, lambda1=1, lambda2=9):
    img = tifffile.imread(filename)

    segmentedImage = ms.morphological_chan_vese(img, iterations=100,
                                                init_level_set='checkerboard',
                                                smoothing=mu, lambda1=lambda1, lambda2=lambda2)

    imageStack = img * segmentedImage

    segmentedImageName = f"{segmentationPath}/{filename.stem}_{mu}_{lambda1}_{lambda2}{filename.suffix}"
    tifffile.imsave(segmentedImageName, imageStack)


def real_visual_callback_3d(fig=None, plot_each=1):
    """
    Returns a callback than can be passed as the argument `iter_callback`
    of `morphological_geodesic_active_contour` and
    `morphological_chan_vese` for visualizing the evolution
    of the levelsets. Only works for 3D images.
    Parameters
    ----------
    fig : matplotlib.figure.Figure
        Figure where results will be drawn. If not given, a new figure
        will be created.
    plot_each : positive integer
        The plot will be updated once every `plot_each` calls to the callback
        function.
    Returns
    -------
    callback : Python function
        A function that receives a levelset and updates the current plot
        accordingly. This can be passed as the `iter_callback` argument of
        `morphological_geodesic_active_contour` and
        `morphological_chan_vese`.
    """

    from mpl_toolkits.mplot3d import Axes3D
    # PyMCubes package is required for `visual_callback_3d`
    try:
        import mcubes
    except ImportError:
        raise ImportError("PyMCubes is required for 3D `visual_callback_3d`")

    # Prepare the visual environment.
    if fig is None:
        fig = plt.figure()
    fig.clf()
    ax = fig.add_subplot(111, projection='3d')
    plt.pause(1)

    counter = [-1]

    def callback(levelset):

        counter[0] += 1
        if (counter[0] % plot_each) != 0:
            return

        if ax.collections:
            del ax.collections[0]
        coords, triangles = mcubes.marching_cubes(np.array(levelset), 0.5)
        ax.plot_trisurf(coords[:, 0], coords[:, 1], coords[:, 2],
                        triangles=triangles)
        plt.pause(0.1)

    return callback


def example_confocal3d():
    PATH_ARRAY_CONFOCAL = 'images/confocal.npy'

    # logging.info('Running: example_confocal3d (MorphACWE)...')

    # Load the image.
    img = np.load(PATH_ARRAY_CONFOCAL)

    # Initialization of the level-set.
    init_ls = ms.circle_level_set(img.shape, (30, 50, 80), 25)

    # Callback for visual plotting
    callback = real_visual_callback_3d(plot_each=20)

    # Morphological Chan-Vese (or ACWE)
    ms.morphological_chan_vese(img, iterations=150,
                               init_level_set=init_ls,
                               smoothing=1, lambda1=1, lambda2=2,
                               iter_callback=callback)


def visual_callback_3d(fig=None, plot_each=1):
    """
    Returns a callback than can be passed as the argument `iter_callback`
    of `morphological_geodesic_active_contour` and
    `morphological_chan_vese` for visualizing the evolution
    of the levelsets. Only works for 3D images.

    Parameters
    ----------
    fig : matplotlib.figure.Figure
        Figure where results will be drawn. If not given, a new figure
        will be created.
    plot_each : positive integer
        The plot will be updated once every `plot_each` calls to the callback
        function.

    Returns
    -------
    callback : Python function
        A function that receives a levelset and updates the current plot
        accordingly. This can be passed as the `iter_callback` argument of
        `morphological_geodesic_active_contour` and
        `morphological_chan_vese`.

    """
    # PyMCubes package is required for `visual_callback_3d`
    try:
        import mcubes
    except ImportError:
        raise ImportError("PyMCubes is required for 3D `visual_callback_3d`")

    # Prepare the visual environment.
    if fig is None:
        fig = plt.figure()
    fig.clf()
    ax = fig.add_subplot(111, projection='3d')
    plt.pause(0.001)

    def callback(levelset):
        img = levelset[1]
        levelset = levelset[0]

        if ax.collections:
            del ax.collections[0]

        volume = np.ascontiguousarray(img, np.float32)

        coords, triangles = mcubes.marching_cubes(img, volume.max() / 50)
        ax.plot_trisurf(coords[:, 0], coords[:, 1], coords[:, 2],
                        triangles=triangles)
        plt.show()
        plt.pause(0.1)

    return callback


def non_visual_callback_3d(plot_each=1):
    """
    Returns a callback than can be passed as the argument `iter_callback`
    of `morphological_geodesic_active_contour` and
    `morphological_chan_vese` for visualizing the evolution
    of the levelsets. Only works for 3D images.

    Parameters
    ----------
    plot_each : positive integer
        The plot will be updated once every `plot_each` calls to the callback
        function.

    Returns
    -------
    callback : Python function
        A function that receives a levelset and updates the current plot
        accordingly. This can be passed as the `iter_callback` argument of
        `morphological_geodesic_active_contour` and
        `morphological_chan_vese`.

    """
    # PyMCubes package is required for `visual_callback_3d`
    try:
        import mcubes
    except ImportError:
        raise ImportError("PyMCubes is required for 3D `visual_callback_3d`")

    counter = [-1]

    def callback(levelset):
        counter[0] += 1
        if (counter[0] % plot_each) != 0:
            return

        coords, triangles = mcubes.marching_cubes(levelset, 0.5)

    return callback


def export_mesh(vertices, faces, iteration):
    """
        Save vertices and faces into an obj file. This function is used during visual callback events

    :param vertices: Collection of vertices with three coordinates
    :param faces: Collection of triangular faces
    :param iteration: The actual iteration in visual callback function

    :return: None
    """
    f = open(f'mesh_{iteration}.obj', 'w')
    f.write(f'# Number of vertices : {vertices.shape[0]}')
    f.write(f'# Number of faces : {faces.shape[0]}')
    for vertice in vertices:
        f.write(f'v {vertice[0]} {vertice[1]} {vertice[2]}')
    for face in faces:
        f.write(f'v {face[0]} {face[1]} {face[2]}')


def get_list_of_files_pattern(dirName, pattern=None):
    """
    Get a list of files in folder dirName ONLY
    :param
	dirName: relative or absolute path to the directory
	pattern: a string that needs to be contain in the filename
	:return: List of files
    """

    listOfFile = os.listdir(dirName)
    allFiles = []

    for entry in listOfFile:
        fullPath = os.path.join(dirName, entry)
        filename = fullPath.split('/')[-1]
        if not pattern:
            allFiles.append(fullPath)
        elif pattern in filename:
            allFiles.append(fullPath)

    return allFiles


def deconvolve_folder(dirpath: str, fiji_path):
    for imageName in get_list_of_files_pattern(dirpath):
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
    imagePath = pathlib.Path(filename)
    if imagePath.suffix == '.czi':
        image = czifile.imread(imagePath)
        if len(image.shape) == 7:
            image = image[0, 0, 0, :, :, :, 0]
    elif imagePath.suffix == '.tif':
        image = tifffile.imread(imagePath)
    elif imagePath.suffix == '.lsm':
        image = tifffile.imread(imagePath)
        if len(image.shape) == 5:
            image = image[0, 0, :, :, :]
    else:
        print(f"Not supported file format {imagePath.suffix}")
        return

    ij = imagej.init(fiji_path)
    ij.getContext().dispose()
    ij = imagej.init(fiji_path, new_instance=True)

    if not os.path.exists(f"{pathlib.Path(imagePath.drive, imagePath.parent, 'temp')}"):
        os.mkdir(f"{pathlib.Path(imagePath.drive, imagePath.parent, 'temp')}")

    if not os.path.exists(f"{pathlib.Path(imagePath.drive, imagePath.parent, 'deconvolved')}"):
        os.mkdir(f"{pathlib.Path(imagePath.drive, imagePath.parent, 'deconvolved')}")

    tifffile.imwrite(f"{pathlib.Path(imagePath.drive, imagePath.parent, 'temp', imagePath.stem + '.tif').as_posix()}",
                     image)

    macro = f"""
    open("{pathlib.Path(imagePath.drive, imagePath.parent, 'temp', imagePath.stem + '.tif').as_posix()}")
    run("Iterative Deconvolve 3D", "image={imagePath.stem} point={imagePath.stem} output=Deconvolved show perform wiener=0.000 low=1 z_direction=1 maximum=1 terminate=0.000")
    selectWindow("Deconvolved_1");
    saveAs("Tiff", "{pathlib.Path(imagePath.drive, imagePath.parent, 'deconvolved', imagePath.stem).as_posix()}_Deconvolved.tif");
    """

    args = {}

    ij.py.run_macro(macro, args)
    print(f"Saved : {pathlib.Path(imagePath.drive, imagePath.parent, 'deconvolved', imagePath.stem)}_Deconvolved.tif")


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
        cziImg = czifile.imread(filename)
        tiffImg = np.squeeze(cziImg)
        tifffile.imsave(f"{folder}/Images/{filename.stem}.tif", tiffImg)


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
    folderPath = pathlib.Path(folder)
    imagePath = folderPath.joinpath("Images/")
    imagePath.mkdir(exist_ok=True, parents=True)
    filenames = get_filepaths(folder, extension)
    count = 0
    for filename in filenames:
        if "colloc" not in filename.__str__():
            cziImg = czifile.imread(filename)
            tiffImg = np.squeeze(cziImg)
            tifffile.imsave(f"{folder}/Images/{filename.stem}.tif", tiffImg)
            count += 1
    deconvolvePath = folderPath.joinpath("Deconvolved/")
    deconvolvePath.mkdir(exist_ok=True, parents=True)


def remove_space(folder: str):
    """
        Renames all .tif files in a specified folder by removing spaces from their filenames.

        This function iterates over all .tif files in the given folder and renames each file by
        removing any spaces in its filename. The renaming is done only if the original filename
        contains spaces.

        Parameters:
        folder (str): The directory path where the .tif files are located.
    """
    for filename in get_filepaths(folder, ".tif", pathlibBool=False):
        filename_without_space = filename.replace(" ", "")
        if filename != filename_without_space:
            os.rename(filename, filename_without_space)


if __name__ == "__main__":
    # remove_space("D:/Downloads/new_spines/")
    # regroup_files("D:/Downloads/new_spines/11968/")

    example_confocal3d()
    exit()
    for_article_loop_over_chan_vese(r"D:\Documents\PycharmProjects\spineReconstruction3D\article_test",
                                    "modif_deconvolved_Deconvolved_5.tif")
    # multi_process_folder(r"D:\Documents\PycharmProjects\spineReconstruction3D\article_test", n_threads=10)

    exit()
    multi_process_folder(r"D:\Documents\PycharmProjects\mesh_article/Deconvolved", n_threads=10)
    exit()
    deconvolve_folder('D:\Documents\PycharmProjects\spineReconstruction3D\\newAnalyse\Animal3',
                      'D:/Documents/Fiji.app')
    exit()

    # exit()
    # regroup_files("D:/Downloads/new_spines/12163/")
    # regroup_files("D:/Downloads/new_spines/12185/")
    # regroup_files("D:/Downloads/new_spines/17030/")
    # regroup_files("D:/Downloads/new_spines/17228/")
    # exit()

    # regroup_files("D:/Downloads/new_spines/11968/")
    # regroup_files("D:/Downloads/new_spines/SD_new/")
    # regroup_files("D:/Downloads/new_spines/12186/")
    # exit()
    # process_folder("D:/Downloads/new_spines/11968/")
    # multi_process_folder("D:/Downloads/temp/11964/deconvolved")
    # multi_process_folder("alinaImages/", 2)
    # exit()

    multi_process_folder("D:/Downloads/new_spines/12163/Deconvolved", n_threads=10)
    # multi_process_folder("D:/Downloads/new_spines/17030/Deconvolved", n_threads=10)
    # multi_process_folder("D:/Downloads/new_spines/12163/Deconvolved", n_threads=10)
    exit()
    # multi_process_folder("D:/Downloads/complex/11973/Deconvolved", n_threads=10)
    # multi_process_folder("D:/Downloads/complex/11974/Deconvolved", n_threads=10)
    # multi_process_folder("D:/Downloads/complex/12186/Deconvolved", n_threads=10)
    # multi_process_folder("D:/Downloads/temp/11966/Deconvolved", n_threads=10)
    # compare_segmented_and_not()
    # for (dirpath, _, filenames) in os.walk("images/"):
    #     for filename in filenames:
    # filename_plan_segmentation(dirpath, filename)
    # deconvolve_folder('D:\Documents\PycharmProjects\spineReconstruction3D\\newAnalyse\Animal 3', 'D:/Documents/Fiji.app')

    # convert_to_8bit_folder('D:\Documents\PycharmProjects\spineReconstruction3D\\newAnalyse\Animal 3\deconvolved')

    # filename = 'FITC-t.tif'
    # filename_plan_segmentation('/mnt/4EB2FF89256EC207/PycharmProjects/spineProcessingVlad/', filename)

    # img = tifffile.imread(f"{'/mnt/4EB2FF89256EC207/PycharmProjects/spineProcessingVlad/'}/{filename}")

    # max_image_segmentation('/mnt/4EB2FF89256EC207/PycharmProjects/spineProcessingVlad/', filename)
    # morphologic_chain_segmentation('D:\Documents\PycharmProjects\spineReconstruction3D\Output_Images',
    #                                '11957_slice2_cell7_dendrite1_spines_63x_z2_1024px_avg2_0.35um-stack_SHF.czi')

    # morphologic_chain_segmentation('D:\Documents\PycharmProjects\spineReconstruction3D', 'test_small.tif')

    # filename = '/mnt/4EB2FF89256EC207/PycharmProjects/spineReconstruction/images/Deconvolved_3.tif'
    # image = io.imread(filename)
    # filename = 'deconvolved'
    # filename_plan_segmentation(image, filename)
