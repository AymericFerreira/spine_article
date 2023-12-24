import sys
import os

# print(os.environ["JAVA_HOME"])
# os.environ["JAVA_HOME"] = "C:\Program Files\Java\jdk-14.0.2\\bin"
# print(os.environ["JAVA_HOME"])

import numpy as np
from skimage import measure, io
import tifffile
import trimesh
import pathlib
from tqdm import tqdm
from numba import njit
import numba
import optimise_trimesh
import imagej
import czifile

NUMBA_DEBUG = 0

try:
    import pymesh
    pymeshVariable = True
except ImportError:
    pymeshVariable = False
if pymeshVariable:
    import optimise

np.set_printoptions(threshold=sys.maxsize)  # variable output


def construct_mesh_from_lewiner(imageStack, spacingData, levelThreshold):
    """
        Lewiner marching cubes algorithm using skimage.measure.marching_cubes_lewiner to find surfaces in 3d volumetric data.

        Convert an imagestack to a pymesh Mesh object

        :param imageStack: (M, N, P) array
        Volume data aka each stack of an  image to find isosurfaces
        :param spacingData: (3) list
        Information about the image properties, size of voxel z, x, y
        Note that you should correct Z-axis depending of medium refraction index
        :param levelThreshold: float
        Contour value to search for isosurfaces

        :return:
        A pymesh Mesh object
    """
    if levelThreshold is None:
        vertices, faces, normals, values = measure.marching_cubes(imageStack,
                                                                  spacing=spacingData,
                                                                  allow_degenerate=False)
    else:
        vertices, faces, normals, values = measure.marching_cubes(imageStack,
                                                                  level=float(levelThreshold),
                                                                  spacing=spacingData,
                                                                  allow_degenerate=False)

    return pymesh.form_mesh(vertices, faces)


def construct_mesh_from_lewiner_trimesh(imageStack, spacingData, levelThreshold):
    """
        Lewiner marching cubes algorithm using skimage.measure.marching_cubes_lewiner to find surfaces in 3d volumetric data.

        Convert an imagestack to a trimesh Mesh object
        The objective of this function is to replace pymesh one

        :param imageStack: (M, N, P) array
        Volume data aka each stack of an  image to find isosurfaces
        :param spacingData: (3) list
        Information about the image properties, size of voxel z, x, y
        Note that you should correct Z-axis depending of medium refraction index
        :param levelThreshold: float
        Contour value to search for isosurfaces

        :return:
        A trimesh Mesh object
    """
    mesh = None
    try:
        if levelThreshold is None:
            vertices, faces, normals, values = measure.marching_cubes(imageStack, method='lewiner', spacing=spacingData,
                                                                      allow_degenerate=False)
        else:
            vertices, faces, normals, values = measure.marching_cubes(imageStack, method='lewiner',
                                                                      level=float(levelThreshold), spacing=spacingData,
                                                                      allow_degenerate=False)
        mesh = trimesh.Trimesh(vertices=vertices, faces=faces)
        if mesh:
            return mesh
    except RuntimeError or MemoryError:
        print(f"Error with image {imageStack.shape} ")
        return None


def construct_and_optimise_from_lewiner(imageStack, spacingData, levelThreshold, tol=5):
    """
        Lewiner marching cubes algorithm using skimage.measure.marching_cubes_lewiner
        to find surfaces in 3d volumetric data.

        Convert an imagestack to a pymesh Mesh object, improve it via an optimisation pipeline and delete 'noise' aka
        small meshes

        :param imageStack: (M, N, P) array
        Volume data aka each stack of an  image to find isosurfaces
        :param spacingData: (3) list
        Information about the image properties, size of voxel z, x, y
        Note that you should correct Z-axis depending of medium refraction index
        :param levelThreshold: float
        Contour value to search for isosurfaces
        :param tol: float
        Percentage of tolerance when looking for small meshes. If the submesh contains less than tol% (5% by default)
        vertices, delete this mesh from output object

        :return:
        An optimised pymesh Mesh object
    """
    mesh = optimise.fix_meshes(construct_mesh_from_lewiner(imageStack, spacingData, levelThreshold))
    mesh = optimise.new_remove_small_meshes(mesh, tolerance=tol)
    return mesh


def verify_mesh_stability(mesh):
    meshList = optimise.get_size_of_meshes(optimise.create_graph(mesh))
    meshList.sort(reverse=True)
    if (
            len(meshList) > 1
            and (meshList[0] + meshList[1]) / np.sum(meshList) > 0.9
            and meshList[0] / np.sum(meshList) < 0.8
    ):
        stability = False
        # print(f'Can"t recontruct this spine {filename}')


def iterative_deconvolve(filename, number=5):
    fijiInstallFolder = 'D:\Documents\Fiji.app'
    # ij = imagej.init(fijiInstallFolder, headless=True)
    ij = imagej.init(fijiInstallFolder)
    deconvolveFolder = pathlib.Path(filename.stem / "iDeconvolve")
    deconvolveFolder.mkdir(parents=True, exist_ok=True)

    args = {}

    imageName = pathlib.Path(filename)

    macro = f"""
    open("{imageName}")
    run("Iterative Deconvolve 3D", "image={imageName} point={imageName} output=Deconvolved show perform wiener=0.000 low=1 z_direction=0.3 maximum={number} terminate=0.010")
    selectWindow("Deconvolved_{number}");
    saveAs("Tiff", "{deconvolveFolder}/{imageName.stem}_Deconvolved{number}.tif");
    """

    print('start')
    ij.py.run_macro(macro, args)
    print('end')
    exit()


@njit
def numba_segmentation(image):
    """
        Fill missing pixels from chan vese segmentation algorithm, accelerated version with numba
        :param image: An image with or without missing pixels
        :return: Corrected image
    """
    imageCopy = image.copy()
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
                        imageCopy[z, x, y] = value / number
    return imageCopy


@njit
def pixel_numba(pixel, number, value):
    number += 1
    value += pixel

    return number, value


def correct_segmentation(image):
    """
        Simple algorithm to correct segmentation created
        :param image:
        :return:
    """
    imageCopy = image.copy()
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
                        imageCopy[z, x, y] = value / number
    return imageCopy


def pixel_compare(pixel, number, value):
    number += 1
    value += pixel

    return number, value


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


def automatic_marching_cube_reconstruction(dirpath, filename):
    """
        An automatic routine to convert image stacks to 3D triangular meshes.
        :param dirpath: The full directory path where the file(s) are stored,
        ex : /home/user/Documents/spineReconstruction/Images/ or C:/Users/Documents/spineReconstruction/Images
        :param filename: The name of the image stack file in tiff, ex : stackimage.tiff or stackimage.tif
        :return:
    """
    print(f'Computing : {filename.strip(".tif").split("/")[-1]}_mesh')
    imageStack = io.imread(f'{dirpath}/{filename}')
    print(imageStack.shape)
    zSpacing = 0.36 / 1.518

    levelThreshold = 0
    stability = True

    try:
        mesh = construct_mesh_from_lewiner(imageStack, (zSpacing, 0.05, 0.05), levelThreshold)

    except ValueError("Surface level must be within volume data range.") as texterror:
        print(f"No reconstruction because of error : {texterror}, skip {filename}")
        exit()

    # Todo : refactoring and cut into more functions
    while stability:
        # if levelThreshold > 200:
        if levelThreshold > 400:
            print('Image seems to be mostly noise, or resolution is super good. Stopping at levelTreshold 400')
            stability = False
            # levelThreshold = 200
            levelThreshold = 395

        try:
            mesh2 = construct_mesh_from_lewiner(imageStack, (zSpacing, 0.05, 0.05), levelThreshold)

        except Exception as texterror:
            stability = False
            # print(texterror)
            # print(texterror.__str__())
            levelThreshold = 1
            # print("Surface level must be within volume data range.")
            try:
                mesh2 = construct_mesh_from_lewiner(imageStack, (zSpacing, 0.05, 0.05), levelThreshold)
            except Exception as texterror:
                print(
                    f"No reconstruction because of error : {texterror} at threshold : {levelThreshold}, skip {filename}")
                return
        meshList = optimise.get_size_of_meshes(optimise.create_graph(mesh2))
        meshList.sort(reverse=True)
        meshList = [x if x > 0.01 * np.sum(meshList) else 0 for x in meshList]
        if len(meshList) > 1:
            if (meshList[0] + meshList[1]) / np.sum(meshList) > 0.9 and meshList[0] / np.sum(meshList) < 0.8:
                stability = False
                stability2 = False
                # neck and head are dissociated
                while not stability2:
                    levelThreshold -= 1
                    levelThreshold = max(levelThreshold, 0)
                    try:
                        mesh2 = construct_mesh_from_lewiner(imageStack, (zSpacing, 0.05, 0.05), levelThreshold)
                    except Exception as texterror:
                        print(
                            f"No reconstruction because of error : {texterror} at threshold : {levelThreshold}, skip {filename}")
                        return

                    meshL = optimise.remove_noise(mesh2)
                    if len(meshL) > 1:
                        if (
                                len(meshL[0].vertices) + len(meshL[1].vertices)
                        ) / len(mesh2.vertices) <= 0.9 or len(
                            meshL[0].vertices
                        ) / len(
                            mesh2.vertices
                        ) >= 0.8:
                            levelThreshold -= levelThreshold / 10
                            break
                    else:
                        levelThreshold -= levelThreshold / 10
                        break
            else:
                levelThreshold += 5
        else:
            levelThreshold += 5
            # Look for narrow reconstruction
    try:
        mesh3 = construct_and_optimise_from_lewiner(imageStack, (zSpacing, 0.05, 0.05), levelThreshold)

    except Exception as texterror:
        levelThreshold *= 0.9
        try:
            mesh3 = construct_and_optimise_from_lewiner(imageStack, (zSpacing, 0.05, 0.05), levelThreshold)
        except Exception as texterror:
            print(f"No reconstruction because of error : {texterror} at threshold : {levelThreshold}, skip {filename}")
            return

    mesh3 = optimise.new_remove_small_meshes(mesh3)
    print(f'Saving mesh with level threshold : {levelThreshold} in optimisedMeshes')
    pymesh.save_mesh(f'optimisedMeshes/{filename.split(".")[0]}_{levelThreshold}_optimised.stl', mesh3)


def get_param_list(imageStack):
    volume = np.ascontiguousarray(imageStack, np.float32)
    minVolume = np.amin(volume)
    maxVolume = np.amax(volume)
    return [minVolume, maxVolume / 100, maxVolume / 80, maxVolume / 60, maxVolume / 40, maxVolume / 20,
            maxVolume / 15, maxVolume / 10, maxVolume / 5, maxVolume, 0.5 * (minVolume + maxVolume)]


def reduce_image(imageStack, margin=1):
    """
    Trim the leading and trailing zeros from a N-D array.

    :param arr: numpy array
    :param margin: how many zeros to leave as a margin
    :returns: trimmed array
    :returns: slice object
    """
    z, x, y = imageStack.shape
    # Image is increased to try to close the mesh, in any case all non useful zero will be remove to keep margin value
    newImage = np.zeros((z + 2 * margin, x + 2 * margin, y + 2 * margin))
    newImage[1:-1, 1:-1, 1:-1] = imageStack
    newImage = trim_zeros(newImage)
    return newImage[0]


def trim_zeros(arr, margin=1):
    """
    Trim the leading and trailing zeros from a N-D array.

    :param arr: numpy array
    :param margin: how many zeros to leave as a margin
    :returns: trimmed array
    :returns: slice object
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


def try_reconstruction(imageStack, spacingData, meshpath, nameList, division=True, loop=0):
    paramList = get_param_list(imageStack)
    print(paramList)
    for num, param in enumerate(paramList):
        print(param)
        if paramList[num] >= paramList[0]:
            if mesh := construct_mesh_from_lewiner_trimesh(imageStack, spacingData, param):
                if mesh.vertices.shape[0] < 4.5e6:
                    mesh.export(f"{meshpath}/{nameList[num]}.ply")
                    optimise_trimesh.mesh_remesh(mesh, meshpath, param, f"{nameList[num]}_{loop}")
                elif division:
                    print("divided")
                    z, x, y = imageStack.shape
                    try_reconstruction(imageStack[:, 0:int(x / 2), 0:int(y / 2)], spacingData, meshpath,
                                       nameList, loop=2 * loop + 1)
                    try_reconstruction(imageStack[:, int(x / 2):, int(y / 2):], spacingData, meshpath,
                                       nameList, loop=2 * loop + 2)

                del mesh


def folder_reconstruction(folder):
    folder = pathlib.Path(folder)
    folder = pathlib.Path(folder)
    # for (dirpath, _, filenames) in tqdm(os.walk(folder)):
    filenameBar = get_filepaths(folder, ".tif")
    for filename in filenameBar:
        imageStack = tifffile.imread(filename)

        dirPath = pathlib.Path(filename.parent)

        imageStack = numba_segmentation(imageStack)
        imageStack = (imageStack - imageStack.min()) / (imageStack.max() - imageStack.min()) * 255
        imageStack = imageStack.astype(np.int)

        imageStack = reduce_image(imageStack)

        nameList = ["min", "40", "35", "30", "25", "20", "15", "10", "5", "max", "mean"]
        meshPath = pathlib.Path(f"{filename.parent.parent}/Mesh/{filename.stem}/")
        meshPath.mkdir(parents=True, exist_ok=True)

        try_reconstruction(imageStack, [0.35 / 1.518, 0.05, 0.05], meshPath, nameList, division=False)


def czi_to_tif(folder):
    filenames = get_filepaths(folder, ".czi")
    for filename in filenames:
        cziImg = czifile.imread(filename)
        tiffImg = np.squeeze(cziImg)
        tifffile.imsave(f"{folder}/{filename.stem}.tif", tiffImg)


def regroup_files(folder, extension=".czi"):
    filenames = get_filepaths(folder, extension)
    for filename in filenames:
        if "colloc" not in filename.__str__():
            # Assure that there is no space
            try:
                filename.rename(pathlib.Path(f"D:\Downloads\\temp\\{filename.name.replace(' ', '_')}"))
            except FileExistsError:
                pass


if __name__ == "__main__":
    folder_reconstruction(r"D:\Documents\PycharmProjects\spineReconstruction3D\article_test")
    exit()
