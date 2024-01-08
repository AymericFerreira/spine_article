import sys
import os

import numpy as np
from skimage import measure
import tifffile
import trimesh
import pathlib
from numba import njit
import optimise_trimesh
import czifile

NUMBA_DEBUG = 0

np.set_printoptions(threshold=sys.maxsize)  # variable output


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


def construct_mesh_from_Lewiner_trimesh(image_stack, spacing_data, level_threshold):
    """
        Lewiner marching cubes algorithm using skimage.measure.marching_cubes_lewiner to find
        surfaces in 3d volumetric data.

        Convert an imagestack to a trimesh Mesh object
        The objective of this function is to replace pymesh one

        :param image_stack: (M, N, P) array
        Volume data aka each stack of an  image to find isosurfaces
        :param spacing_data: (3) list
        Information about the image properties, size of voxel z, x, y
        Note that you should correct Z-axis depending of medium refraction index
        :param level_threshold: float
        Contour value to search for isosurfaces

        :return:
        A trimesh Mesh object
    """
    mesh = None
    try:
        if level_threshold is None:
            vertices, faces, normals, values = measure.marching_cubes(image_stack, method='lewiner',
                                                                      spacing=spacing_data,
                                                                      allow_degenerate=False)
        else:
            vertices, faces, normals, values = measure.marching_cubes(image_stack, method='lewiner',
                                                                      level=float(level_threshold),
                                                                      spacing=spacing_data,
                                                                      allow_degenerate=False)
        if mesh := trimesh.Trimesh(vertices=vertices, faces=faces):
            return mesh
    except RuntimeError or MemoryError:
        print(f"Error with image {image_stack.shape} ")
        return None


@njit
def numba_segmentation(image: np.ndarray):
    """
       Accelerated image segmentation correction using Numba.

       This function fills in missing pixels in a segmented image using a 3D neighborhood approach.
       It checks each pixel's 3D neighbors and, if the current pixel is missing (value 0), it calculates
       an average value from the neighbors. This function is accelerated using Numba for performance improvement.

       Parameters:
       image (numpy.ndarray): A 3D array representing the segmented image with potential missing pixels.

       Returns:
       numpy.ndarray: The corrected image with missing pixels filled in.
   """
    image_copy = image.copy()
    for z in range(1, image.shape[0] - 2):
        for x in range(1, image.shape[1] - 2):
            for y in range(1, image.shape[2] - 2):
                if image[z, x, y] == 0:
                    number = 0
                    value = 0
                    number, value = pixel_numba(image[z, x - 1, y - 1], number, np.uint8(value))
                    number, value = pixel_numba(image[z, x - 1, y], number, np.uint8(value))
                    number, value = pixel_numba(image[z, x - 1, y + 1], number, np.uint8(value))
                    number, value = pixel_numba(image[z, x, y - 1], number, np.uint8(value))
                    number, value = pixel_numba(image[z, x, y + 1], number, np.uint8(value))
                    number, value = pixel_numba(image[z, x + 1, y - 1], number, np.uint8(value))
                    number, value = pixel_numba(image[z, x + 1, y], number, np.uint8(value))
                    number, value = pixel_numba(image[z, x + 1, y + 1], number, np.uint8(value))
                    if number >= 6:
                        image_copy[z, x, y] = value / number
    return image_copy


@njit
def pixel_numba(pixel, number, value):
    """
        Helper function for numba_segmentation, accumulates pixel values and counts.

        It accumulates the value of a pixel and increments the counter if the pixel is relevant for averaging.

        Parameters:
        pixel (int/float): The pixel value to be added.
        number (int): The current count of relevant pixels.
        value (int/float): The current accumulated value of relevant pixels.

        Returns:
        tuple: A tuple containing the updated number and value.
    """
    number += 1
    value += pixel
    return number, value


def correct_segmentation(image):
    """
        Corrects segmentation of an image by filling missing pixels.

        This function applies a simple algorithm to correct segmented images. It fills in missing pixels (value 0)
        by averaging the values of neighboring pixels. This process is done through a 3D neighborhood approach.

        Parameters:
        image (numpy.ndarray): A 3D array representing the segmented image with potential missing pixels.

        Returns:
        numpy.ndarray: The corrected image with missing pixels filled in.
    """
    image_copy = image.copy()
    for z in range(1, image.shape[0] - 2):
        for x in range(1, image.shape[1] - 2):
            for y in range(1, image.shape[2] - 2):
                if image[z, x, y] == 0:
                    number = 0
                    value = 0
                    number, value = pixel_compare(image[z, x - 1, y - 1], number, value)
                    number, value = pixel_compare(image[z, x - 1, y], number, value)
                    number, value = pixel_compare(image[z, x - 1, y + 1], number, value)
                    number, value = pixel_compare(image[z, x, y - 1], number, value)
                    number, value = pixel_compare(image[z, x, y + 1], number, value)
                    number, value = pixel_compare(image[z, x + 1, y - 1], number, value)
                    number, value = pixel_compare(image[z, x + 1, y], number, value)
                    number, value = pixel_compare(image[z, x + 1, y + 1], number, value)
                    if number >= 6:
                        image_copy[z, x, y] = value / number
    return image_copy


def pixel_compare(pixel, number, value):
    """
    Helper function for correct_segmentation, accumulates pixel values and counts.

    It accumulates the value of a pixel and increments the counter if the pixel is relevant for averaging.

    Parameters:
    pixel (int/float): The pixel value to be added.
    number (int): The current count of relevant pixels.
    value (int/float): The current accumulated value of relevant pixels.

    Returns:
    tuple: A tuple containing the updated number and value.
    """
    number += 1
    value += pixel

    return number, value


def get_param_list(image_stack):
    """
        Generates a list of parameters based on the min and max values in an image stack.

        This function calculates the minimum and maximum values in the provided image stack and then
        generates a list of parameters. These parameters are a series of fractions of the maximum value,
        along with the minimum, maximum, and mean values.

        Parameters:
        image_stack (numpy.ndarray): The image stack from which to calculate parameters.

        Returns:
        list: A list of calculated parameters based on the min and max values of the image stack.
    """
    volume = np.ascontiguousarray(image_stack, np.float32)
    min_volume = np.amin(volume)
    max_volume = np.amax(volume)
    return [min_volume, max_volume / 100, max_volume / 80, max_volume / 60, max_volume / 40, max_volume / 20,
            max_volume / 15, max_volume / 10, max_volume / 5, max_volume, 0.5 * (min_volume + max_volume)]


def reduce_image(image_stack, margin=1):
    """
        Reduces the size of an image stack by trimming zero-padding while keeping a specified margin.

        This function initially expands the image stack by a margin of zeroes and then trims the zeros
        from the edges while maintaining the specified margin around the non-zero regions of the image stack.

        Parameters:
        image_stack (numpy.ndarray): A 3D array representing the image stack to be reduced.
        margin (int, optional): The margin of zeros to retain around the image stack. Defaults to 1.

        Returns:
        numpy.ndarray: The reduced image stack with the specified margin retained.
    """
    z, x, y = image_stack.shape
    # Image is increased to try to close the mesh, in any case all non useful zero will be remove to keep margin value
    new_image = np.zeros((z + 2 * margin, x + 2 * margin, y + 2 * margin))
    new_image[1:-1, 1:-1, 1:-1] = image_stack
    new_image = trim_zeros(new_image)
    return new_image[0]


def trim_zeros(arr, margin=1):
    """
        Trims zero-padding from an array while keeping a specified margin around the non-zero regions.

        This function iteratively trims zero-padding from each dimension of the array, retaining a specified
        margin around the non-zero regions of the array.

        Parameters:
        arr (numpy.ndarray): The array from which zero-padding should be trimmed.
        margin (int, optional): The margin of zeros to retain around the non-zero regions of the array. Defaults to 1.

        Returns:
        tuple: A tuple containing the trimmed array and the slicing information.
    """
    s = []
    for dim in range(arr.ndim):
        start = 0
        end = -1
        slice_ = [slice(None)] * arr.ndim

        go = True
        while go:
            slice_[dim] = start
            go = not np.any(arr[tuple(slice_)])
            start += 1
        start = max(start - 1 - margin, 0)

        go = True
        while go:
            slice_[dim] = end
            go = not np.any(arr[tuple(slice_)])
            end -= 1
        end = arr.shape[dim] + min(-1, end + 1 + margin) + 1

        s.append(slice(start, end))
    return arr[tuple(s)], tuple(s)


def try_reconstruction(image_stack, spacing_data, mesh_path, name_list, division=True, loop=0):
    """
        Attempts to reconstruct a mesh from an image stack with different parameters.

        This function iteratively tries to reconstruct a mesh from an image stack using varying parameters.
        If successful, the mesh is saved, and an optimization process is applied. In case of large meshes,
        the function can optionally divide the image stack and attempt reconstruction on the smaller segments.

        Parameters:
        image_stack (numpy.ndarray): The stack of images for mesh reconstruction.
        spacing_data (tuple): The spacing data for mesh reconstruction.
        mesh_path (str): The path where the reconstructed mesh will be saved.
        name_list (list): A list of names corresponding to different reconstruction parameters.
        division (bool, optional): Whether to divide the image stack for large meshes. Defaults to True.
        loop (int, optional): The current loop iteration for recursive calls. Defaults to 0.

    """
    param_list = get_param_list(image_stack)
    print(f"List of parameters {param_list}")
    for num, param in enumerate(param_list):
        print(f"Computing {param}")
        if param_list[num] >= param_list[0]:
            if mesh := construct_mesh_from_Lewiner_trimesh(image_stack, spacing_data, param):
                if mesh.vertices.shape[0] < 4.5e6:
                    mesh.export(f"{mesh_path}/{name_list[num]}.ply")
                    optimise_trimesh.mesh_remesh(mesh, mesh_path, param, f"{name_list[num]}_{loop}")
                elif division:
                    print("Divided mesh")
                    z, x, y = image_stack.shape
                    try_reconstruction(image_stack[:, 0:int(x / 2), 0:int(y / 2)], spacing_data, mesh_path,
                                       name_list, loop=2 * loop + 1)
                    try_reconstruction(image_stack[:, int(x / 2):, int(y / 2):], spacing_data, mesh_path,
                                       name_list, loop=2 * loop + 2)

                del mesh


def folder_reconstruction(folder, z_spacing: float = 1, pixel_size_x: float = 1, pixel_size_y: float = 1):
    """
        Performs mesh reconstruction for all TIFF images in a specified folder.

        This function walks through a specified folder, reads each TIFF image, preprocesses it, and then
        attempts mesh reconstruction using the `try_reconstruction` function. The reconstructed meshes
        are saved in a subfolder within the original image folder.

        Parameters:
        folder (str): The path to the folder containing TIFF images for reconstruction.
        z_spacing (float, optional): The z-spacing between two stacks. Defaults to 1.
        pixel_size_x (float, optional): The pixel size in the x-direction. Defaults to 1.
        pixel_size_y (float, optional): The pixel size in the y-direction. Defaults to 1.

    """
    folder = pathlib.Path(folder)
    folder = pathlib.Path(folder)
    filename_bar = get_filepaths(folder, ".tif")
    for filename in filename_bar:
        image_stack = tifffile.imread(filename)

        image_stack = numba_segmentation(image_stack)
        image_stack = (image_stack - image_stack.min()) / (image_stack.max() - image_stack.min()) * 255
        image_stack = image_stack.astype(np.int8)

        image_stack = reduce_image(image_stack)

        name_list = ["min", "40", "35", "30", "25", "20", "15", "10", "5", "max", "mean"]
        mesh_path = pathlib.Path(f"{filename.parent.parent}/Mesh/{filename.stem}/")
        mesh_path.mkdir(parents=True, exist_ok=True)

        try_reconstruction(image_stack, [z_spacing, pixel_size_x, pixel_size_y],
                           mesh_path, name_list, division=False)


def czi_to_tif(folder):
    """
        Converts CZI files to TIFF format in a specified folder.

        This function iterates through all CZI files in a given folder, reads each file, and converts it
        to TIFF format. The TIFF files are saved in the same folder with the same base filename.

        Parameters:
        folder (str): The folder containing CZI files to convert.

        Returns:
        None
    """
    filenames = get_filepaths(folder, ".czi")
    for filename in filenames:
        czi_img = czifile.imread(filename)
        tiff_img = np.squeeze(czi_img)
        tifffile.imwrite(f"{folder}/{filename.stem}.tif", tiff_img)


if __name__ == "__main__":
    folder_reconstruction(r"Segmented", z_spacing=0.35 / 1.518, pixel_size_x=0.05, pixel_size_y=0.05)
