import tqdm
import trimesh
import trimesh.convex
import numpy as np
import os
import pandas as pd
import re
from spineCrawler import parser, get_filepaths
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from draw import plot_3d_scatter_with_color_and_gravity_center_and_gravity_median, plot_3d_scatter_fixed, plot_frequency

try:
    import pymesh
except ImportError:
    print("Pymesh not installed, some functions will not work")


def pymesh_to_trimesh(mesh):
    """
        A function to convert pymesh object to trimesh.
        Trimesh has some built-in functions very useful to split mesh into submeshes

        :param mesh: Pymesh Mesh object
        :return: mesh : Trimesh Mesh object
    """
    return trimesh.Trimesh(vertices=mesh.vertices, faces=mesh.faces)


def trimesh_to_pymesh(mesh):
    """
        A function to convert trimesh object to pymesh.
        Pymesh has some built-in functions very useful to count total number of vertices and to optimise mesh.

        :param mesh: Trimesh Mesh object
        :return: mesh : Pymesh Mesh object
        """
    return pymesh.form_mesh(vertices=mesh.vertices, faces=mesh.faces)


def center_mesh(mesh):
    """
        Centers a mesh around the origin based on its gravity center.

        This function computes the gravity center of a mesh and then translates the mesh such that
        its gravity center is aligned with the origin.

        Parameters:
        mesh: A mesh object containing vertices and faces.

        Returns:
        A new mesh object with centered vertices.

        Note:
        - It assumes that the input mesh has a 'vertices' and 'faces' attribute.
        """
    centered_vertices = np.zeros(shape=(len(mesh.vertices), 3))

    centered_vertices[:, 0] = mesh.vertices[:, 0] - gravity_center(mesh)[0]
    centered_vertices[:, 1] = mesh.vertices[:, 1] - gravity_center(mesh)[1]
    centered_vertices[:, 2] = mesh.vertices[:, 2] - gravity_center(mesh)[2]
    return pymesh.meshio.form_mesh(centered_vertices, mesh.faces)


def gravity_center(mesh):
    """
    Calculates the center of gravity (mean position) of a mesh.

    This function computes the center of gravity of a mesh based on the average position of
    its vertices.

    Parameters:
    mesh: A mesh object with vertices.

    Returns:
    A numpy array representing the center of gravity of the mesh.
    """
    return np.array([np.mean(mesh.vertices[:, 0]), np.mean(mesh.vertices[:, 1]), np.mean(mesh.vertices[:, 2])])


def gravity_median(mesh):
    """
        Calculates the median position of a mesh's vertices.

        This function computes the median position of the vertices of a mesh along each axis.

        Parameters:
        mesh: A mesh object with vertices.

        Returns:
        A list containing the median position of the mesh's vertices along each axis.
        """
    return [np.median(mesh.vertices[:, 0]), np.median(mesh.vertices[:, 1]), np.median(mesh.vertices[:, 2])]


def calculate_mesh_surface(mesh):
    """
       Calculates the total surface area of a mesh.

       This function computes the sum of areas of all faces in the mesh.

       Parameters:
       mesh: A mesh object with faces and face areas.

       Returns:
       The total surface area of the mesh.

       Note:
       - The mesh must have a 'face_area' attribute.
    """
    mesh.add_attribute("face_area")
    return sum(mesh.get_attribute("face_area"))


def calculate_mesh_volume(mesh):
    """
        Calculates the volume of a mesh using a tetrahedron-based method.

        This function calculates the volume of the mesh by summing up the volumes of tetrahedrons
        formed by each face of the mesh and the gravity center.

        Parameters:
        mesh: A mesh object with vertices and faces.

        Returns:
        The total volume of the mesh.
    """
    return sum(tetrahedron_calc_volume(mesh.vertices[faces[0]], mesh.vertices[faces[1]], mesh.vertices[faces[2]],
                                       gravity_center(mesh)) for faces in mesh.faces)


def mesh_volume_spine_base_center(mesh):
    """
        Calculates the volume of a mesh using a tetrahedron-based method.

        This function calculates the volume of the mesh by summing up the volumes of tetrahedrons
        formed by each face of the mesh and the spine base center.

        Parameters:
        mesh: A mesh object with vertices and faces.

        Returns:
        The total volume of the mesh.

        Note:
        - Similar to `calculate_mesh_volume`, but uses a spine base center instead of gravity center.
    """
    return sum(tetrahedron_calc_volume(mesh.vertices[faces[0]], mesh.vertices[faces[1]], mesh.vertices[faces[2]],
                                       find_spine_base_center(mesh)) for faces in mesh.faces)


def determinant_3x3(m):
    """
        Computes the determinant of a 3x3 matrix.

        Parameters:
        m: A 3x3 matrix represented as a list of lists or a 2D numpy array.

        Returns:
        The determinant of the matrix.
    """
    return (m[0][0] * (m[1][1] * m[2][2] - m[1][2] * m[2][1]) -
            m[1][0] * (m[0][1] * m[2][2] - m[0][2] * m[2][1]) +
            m[2][0] * (m[0][1] * m[1][2] - m[0][2] * m[1][1]))


def subtract(a, b):
    """
        Subtracts two 3-dimensional vectors.

        Parameters:
        a: The first 3D vector.
        b: The second 3D vector.

        Returns:
        A tuple representing the result of the subtraction.
    """
    return (a[0] - b[0],
            a[1] - b[1],
            a[2] - b[2])


def tetrahedron_calc_volume(vertex1, vertex2, vertex3, vertex4):
    """
        Calculates the volume of a tetrahedron defined by four vertices.

        Parameters:
        vertex1, vertex2, vertex3, vertex4: The four vertices of the tetrahedron.

        Returns:
        The volume of the tetrahedron.

        Note:
        - The function uses the determinant of a matrix formed by the vectors of the tetrahedron's edges.
        - Assumes vertices are provided in 3-dimensional space.
    """
    return (abs(determinant_3x3((subtract(vertex1, vertex2),
                                 subtract(vertex2, vertex3),
                                 subtract(vertex3, vertex4),
                                 ))) / 6.0)


def calculate_distance(vertex1, vertex2):
    """
        Calculates the Euclidean distance between two points in 3D space.

        Parameters:
        vertex1: The first vertex as a 3D point.
        vertex2: The second vertex as a 3D point.

        Returns:
        The Euclidean distance between the two vertices.
    """
    return np.sqrt(pow(vertex1[0] - vertex2[0], 2) + pow(vertex1[1] - vertex2[1], 2) + pow(vertex1[2] - vertex2[2], 2))


def calculate_vector(vertex1, vertex2):
    """
        Calculates the vector from one vertex to another in 3D space.

        Parameters:
        vertex1: The start vertex as a 3D point.
        vertex2: The end vertex as a 3D point.

        Returns:
        A numpy array representing the vector from vertex1 to vertex2.
    """
    return np.array([vertex1[0] - vertex2[0], vertex1[1] - vertex2[1], vertex1[2] - vertex2[2]])


def calculate_spine_length(mesh):
    """
        Calculates the average length of the top 5% of distances from vertices to the spine base center in a mesh.

        Parameters:
        mesh: A mesh object with vertices.

        Returns:
        The average length of the top 5% of distances from mesh vertices to the spine base center.

    """
    spine_base_center = find_spine_base_center2(mesh)
    length = [calculate_distance(vertex, spine_base_center) for vertex in mesh.vertices]

    length = np.array(length)
    length[::-1].sort()
    list_of_lengths = range(int(0.05 * np.size(length)))
    return np.mean(length[list_of_lengths])


def calculate_average_distance(mesh):
    """
        Calculates the average distance of all vertices from the spine base center in a mesh.

        Parameters:
        mesh: A mesh object with vertices.

        Returns:
        The average distance of all vertices from the spine base center.

    """
    spine_base_center = find_spine_base_center2(mesh)
    length = [calculate_distance(vertex, spine_base_center) for vertex in mesh.vertices]

    length = np.array(length)
    return np.mean(length)


def coefficient_of_variation_in_distance(mesh):
    """
        Calculates the coefficient of variation of distances from mesh vertices to the spine base center.

        Parameters:
        mesh: A mesh object with vertices.

        Returns:
        The coefficient of variation (standard deviation divided by the mean) of the distances from all
        vertices in the mesh to the spine base center.

    """
    spine_base_center = find_spine_base_center2(mesh)
    length = [calculate_distance(vertex, spine_base_center) for vertex in mesh.vertices]

    length = np.array(length)
    average_distance = np.mean(length)
    std = np.std(length)
    return std / average_distance


def calculate_open_angle(mesh):
    """
        Calculates the average open angle formed between the center of gravity, spine base center, and each vertex in a mesh.

        Parameters:
        mesh: A mesh object with vertices.

        Returns:
        The average of the open angles formed at each vertex in the mesh.

        """
    spine_center = gravity_center(mesh)
    spine_base_center = find_spine_base_center(mesh)

    open_angle = [calculate_angle(spine_center, spine_base_center, vertex) for vertex in mesh.vertices]

    return np.mean(open_angle)


def trimesh_calculate_open_angle(mesh):
    """
        Calculates the average open angle for vertices in a mesh using trimesh, considering the mesh's gravity center and spine base center.

        Parameters:
        mesh: A mesh object with vertices.

        Returns:
        The average open angle formed by the gravity center, spine base center, and each vertex in the mesh

    """
    spine_center = gravity_center(mesh)
    spine_base_center = trimesh_find_spine_base_center(mesh)

    open_angle = [calculate_angle(spine_center, spine_base_center, vertex) for vertex in mesh.vertices]

    return np.mean(open_angle)


def calculate_angle(point1: list[float], point2: list[float], point3: list[float]):
    """
    Calculates the angle formed by three points in 3D space.

    Parameters:
    point1, point2, point3: The three points forming the angle, where point2 is the vertex.

    Returns:
    The angle in radians formed at point2 by the lines joining point1-point2 and point2-point3.

    Note:
    - Points are expected to be provided as sequences (like lists or tuples) of three numerical values.
    """
    point1 = np.array(point1)
    point2 = np.array(point2)
    point3 = np.array(point3)

    vector1 = calculate_vector(point1, point2)
    vector2 = calculate_vector(point3, point2)
    return np.arccos(np.dot(vector1, vector2) / (np.linalg.norm(vector1) * np.linalg.norm(vector2)))


def calculate_edges(mesh):
    """
    Generates a list of edges from the faces of a mesh.

    Parameters:
    mesh: A mesh object with faces.

    Returns:
    A list of edges, where each edge is represented as a tuple of two vertex indices.
    """
    edge_list = []
    for face in mesh.faces:
        edge_list.extend(([face[0], face[1]], [face[0], face[2]], [face[1], face[2]]))
    return edge_list


def calculate_hull_volume(mesh):
    """
        Calculates the volume of the convex hull of a given mesh.

        Parameters:
        mesh: A mesh object.

        Returns:
        The volume of the convex hull of the mesh.
    """
    hull_mesh = pymesh.convex_hull(mesh)
    return calculate_mesh_volume(hull_mesh)


def trimesh_hull_volume(mesh):
    """
    Calculates the volume of the convex hull of a given mesh using trimesh.

    Parameters:
    mesh: A mesh object compatible with trimesh.

    Returns:
    The volume of the convex hull of the mesh.
    """
    hull_mesh = trimesh.convex.convex_hull(mesh)
    return calculate_mesh_volume(hull_mesh)


def calculate_hull_ratio(mesh):
    """
    Calculates the ratio of the difference in volume between the mesh's convex hull and the mesh itself to the volume of the mesh.

    Parameters:
    mesh: A mesh object.

    Returns:
    The ratio indicating how much larger the convex hull is compared to the original mesh volume.
    """
    hull_volume = calculate_hull_volume(mesh)
    mesh_volume = calculate_mesh_volume(mesh)
    return (hull_volume - mesh_volume) / mesh_volume


def trimesh_hull_ratio(mesh):
    """
    Calculates the ratio of the difference in volume between the convex hull and the original mesh to the volume of
    the mesh.

       This function computes the volume of the mesh's convex hull and the mesh itself using trimesh
       and then calculates the ratio of the volume difference to the original mesh volume. This ratio
       can be useful for assessing the 'convexity' of the mesh.

       Parameters:
       mesh: A mesh object compatible with trimesh.

       Returns:
       A float representing the ratio of the volume difference between the convex hull and the mesh
       to the mesh's volume.

       Note:
       - The ratio is defined as (hull_volume - mesh_volume) / mesh_volume.
   """
    hull_volume = trimesh_hull_volume(mesh)
    mesh_volume = calculate_mesh_volume(mesh)
    return (hull_volume - mesh_volume) / mesh_volume


def calculate_gaussian_curvature(mesh):
    """
    Calculates various statistics of Gaussian curvature for a mesh.

    This function computes the Gaussian curvature for each vertex of the mesh and then
    calculates the mean, variance, lower 5% mean, and higher 95% mean of these curvatures.

    Parameters:
    mesh: A mesh object with vertices.

    Returns:
    A list containing the mean, variance, lower 5% mean, and higher 95% mean of Gaussian curvatures.

    Note:
    - Currently, the function relies on external methods to add and retrieve Gaussian curvature
      attributes. To remove these dependencies, implement a method to calculate the Gaussian
      curvature directly from the mesh's geometry.
    """
    mesh.add_attribute('vertex_gaussian_curvature')
    gaussian_curvature = mesh.get_attribute('vertex_gaussian_curvature')

    if np.amax(gaussian_curvature) > 1E308:
        print('mesh is not clean')

    return calculate_bounds(gaussian_curvature)


def calculate_bounds(curvature: [float]):
    """
       Calculates statistical measures for a list of curvature values.

       This function computes the mean, variance, and the mean of the lower 5% and upper 95% of
       curvature values in the provided list.

       Parameters:
       curvature ([float]): A list of curvature values (floats).

       Returns:
       A list containing the mean, variance, mean of the lower 5%, and mean of the upper 95% of
       the provided curvature values.

       The function performs the following calculations:
       1. Mean of all curvature values.
       2. Variance of all curvature values.
       3. Mean of the lowest 5% of curvature values.
       4. Mean of the highest 5% of curvature values.
       """
    return [
        np.mean(curvature),
        np.var(curvature),
        np.mean(curvature[: round(0.05 * len(curvature))]),
        np.mean(curvature[round(0.95 * len(curvature)):])
    ]


def calculate_mean_curvature(mesh):
    """
    Calculates various statistics of mean curvature for a mesh.

    This function computes the mean curvature for each vertex of the mesh and calculates the
    mean, variance, lower 5% mean, and higher 95% mean of these curvatures.

    Parameters:
    mesh: A mesh object with vertices.

    Returns:
    A list containing the mean, variance, lower 5% mean, and higher 95% mean of mean curvatures.
    """
    mesh.add_attribute('vertex_mean_curvature')
    mean_curvature_divide_by_surface = mesh.get_attribute('vertex_mean_curvature')
    mesh.add_attribute('vertex_voronoi_area')

    mean_curvature = mean_curvature_divide_by_surface

    # remove nan value
    mean_curvature = mean_curvature[~np.isnan(mean_curvature)]

    return calculate_bounds(mean_curvature)


def mesh_treatment(mesh):
    """
        Processes a given mesh by removing vertices with zero connectivity.

        This function calculates the connectivity of each vertex in the mesh. It then removes
        any vertices that have zero connectivity, essentially cleaning the mesh of isolated vertices.
        The modified mesh is then reformed and returned.

        Parameters:
        mesh (Mesh): The input mesh to be processed.

        Returns:
        Mesh: The processed mesh with isolated vertices removed.
    """
    vertex_connectivity = calculate_vertex_connectivity(mesh)
    result = np.where(vertex_connectivity == 0)

    mesh_vertices = mesh.vertices
    for node in result:
        mesh_vertices = np.delete(mesh_vertices, node, 0)

    return pymesh.meshio.form_mesh(mesh_vertices, mesh.faces)


def find_spine_base_center2(mesh):
    """
        Finds the center of the base of a spine in a mesh.

        This function calculates the center of the base of a spine by identifying vertices with
        low connectivity (<= 3 connections). It averages the coordinates of these vertices to
        determine the center point.

        Parameters:
        mesh (Mesh): The mesh representing the spine.

        Returns:
        list: A list containing the X, Y, and Z coordinates of the base center of the spine.
        """
    list_of_neighbors = np.bincount(mesh.faces.ravel())

    vertices_and_neighbors = np.column_stack((mesh.vertices, np.transpose(list_of_neighbors)))
    less_connected_vertices = np.where(vertices_and_neighbors[:, 3] <= 3)
    return np.mean(mesh.vertices[less_connected_vertices], axis=0)


def find_spine_base_center(mesh):
    """
        Finds the center of the base of a spine in a mesh using vertex valence.

        Similar to 'find_spine_base_center2', this function identifies the base of the spine
        by locating vertices with low connectivity (<= 3). The function uses vertex valence
        to determine connectivity and calculates the average coordinates of these vertices.

        Parameters:
        mesh (Mesh): The mesh representing the spine.

        Returns:
        list: A list containing the X, Y, and Z coordinates of the base center of the spine.
    """
    valence = get_mesh_valence(mesh)
    vertices_and_neighbors = np.column_stack((mesh.vertices, np.transpose(valence)))
    less_connected_vertices = np.where(vertices_and_neighbors[:, 3] <= 3)
    [x, y, z] = np.mean(mesh.vertices[less_connected_vertices], axis=0)
    return [x, y, z]


def trimesh_find_spine_base_center(mesh):
    """
        Finds the center of the base of a spine in a mesh using Trimesh-specific valence calculation.

        This function computes the center of the base of a spine in a similar manner to
        'find_spine_base_center', but it uses the Trimesh library's method for calculating
        vertex valence. It identifies vertices with low connectivity and averages their coordinates.

        Parameters:
        mesh (Trimesh): The mesh representing the spine, processed using the Trimesh library.

        Returns:
        list: A list containing the X, Y, and Z coordinates of the base center of the spine.
        """
    valence = trimesh_mesh_valence(mesh)
    valence2 = [len(val) for val in valence]
    vertices_and_neighbors = np.column_stack((mesh.vertices, np.transpose(valence2)))
    less_connected_vertices = np.where(vertices_and_neighbors[:, 3] <= 3)
    [x, y, z] = np.mean(mesh.vertices[less_connected_vertices], axis=0)
    return [x, y, z]


def get_mesh_valence(mesh):
    """
        Calculates and returns the valence (number of connected edges) for each vertex in a mesh.

        This function adds the 'vertex_valance' attribute to the mesh and then retrieves this
        attribute, effectively returning the valence for each vertex in the mesh.

        Parameters:
        mesh (Mesh): The mesh for which the vertex valence is to be calculated.

        Returns:
        numpy.ndarray: An array containing the valence for each vertex in the mesh.
    """
    mesh.add_attribute("vertex_valance")
    return mesh.get_attribute("vertex_valance")


def trimesh_mesh_valence(mesh):
    """
        Retrieves the valence of each vertex in a mesh using the Trimesh library.

        Valence refers to the number of edges connected to a vertex in a mesh. This function
        utilizes the Trimesh library to quickly obtain the valence of each vertex in the provided mesh.

        Parameters:
        mesh (Trimesh): The mesh whose vertex valences are to be determined.

        Returns:
        numpy.ndarray: An array of lists, where each list contains the indices of neighboring vertices for each vertex.
    """
    return mesh.vertex_neighbors


def find_x_y_z_length(mesh):
    """
        Calculates the lengths of a mesh along the X, Y, and Z axes.

        This function computes the range (maximum - minimum) along each of the X, Y, and Z axes of the mesh,
        providing a basic measure of the mesh's dimensions.

        Parameters:
        mesh (Mesh): The mesh for which the dimensions are to be calculated.

        Returns:
        list: A list containing the lengths along the X, Y, and Z axes of the mesh.
    """
    x = np.amax(mesh.vertices[0]) - np.amin(mesh.vertices[0])
    y = np.amax(mesh.vertices[1]) - np.amin(mesh.vertices[1])
    z = np.amax(mesh.vertices[2]) - np.amin(mesh.vertices[2])
    return [x, y, z]


def calculate_vertex_connectivity(mesh):
    """
        Calculates the connectivity of each vertex in a mesh.

        This function computes the number of connected vertices for each vertex in the mesh. It uses
        a wire network representation of the mesh to assess connectivity.

        Parameters:
        mesh (Mesh): The mesh for which vertex connectivity is to be calculated.

        Returns:
        numpy.ndarray: An array containing the connectivity count for each vertex in the mesh.
    """
    vertex_connectivity = np.array([])

    wire_network = pymesh.wires.WireNetwork.create_from_data(mesh.vertices, mesh.faces)
    for vertex in range(mesh.num_vertices):
        vertex_connectivity = np.append(vertex_connectivity, wire_network.get_vertex_neighbors(vertex).size)
    return vertex_connectivity


def calculate_metrics(mesh):
    """
        Calculates various geometric and topological metrics of a mesh and writes them to files.

        This function computes a set of metrics for the given mesh, including spine length, surface area, volume,
        hull volume, hull ratio, and curvatures. These metrics are written to two separate text files for easy
        reference and export. The function utilizes several other functions to calculate each individual metric.

        Parameters:
        mesh (Mesh): The mesh for which metrics are to be calculated.

        Returns:
        None: The function writes the calculated metrics to files but does not return any value.
    """
    spine_length = calculate_spine_length(mesh)
    mesh_surface = calculate_mesh_surface(mesh)
    mesh_volume = calculate_mesh_volume(mesh)
    hull_volume = calculate_hull_volume(mesh)
    hull_ratio = calculate_hull_ratio(mesh)
    mesh_length = find_x_y_z_length(mesh)
    average_distance = calculate_average_distance(mesh)
    coefficient_variation_distance = coefficient_of_variation_in_distance(mesh)
    open_angle = calculate_open_angle(mesh)
    gaussian_curvature = calculate_gaussian_curvature(mesh)
    mean_curvature = calculate_mean_curvature(mesh)

    with open('3DImages/newSpines/spineProperties.txt', 'w') as metricsFile:
        metricsFile.write('')

        metricsFile.write(f'Spine Length : {spine_length}\n'
                          f'Mesh surface : {mesh_surface}\n'
                          f'Mesh volume : {mesh_volume}\n'
                          f'Hull Volume : {hull_volume}\n'
                          f'Hull Ratio : {hull_ratio}\n'
                          f'Average distance : {average_distance}\n'
                          f'Coefficient of variation in distance : {coefficient_variation_distance}\n'
                          f'Open angle : {open_angle}\n'
                          f'Average of mean curvature : {mean_curvature[0]}\n'
                          f'Variance of mean curvature : {mean_curvature[1]}\n'
                          f'Average of gaussian curvature : {gaussian_curvature[0]}\n'
                          f'Variance of gaussian curvature : {gaussian_curvature[1]}\n'
                          f'Average of lower 5 percent mean curvature : {mean_curvature[2]}\n'
                          f'Average of higher 5 percent mean curvature : {mean_curvature[3]}\n'
                          f'Average of lower 5 percent gauss curvature : {gaussian_curvature[2]}\n'
                          f'Average of higher 5 percent gauss curvature : {gaussian_curvature[3]}\n'
                          f'Length X Y Z : {mesh_length}\n'
                          f'Gravity center computed with median : {gravity_median(mesh)}\n'
                          f'Gravity center computed with mean : {gravity_center(mesh)}\n')

    with open('3DImages/newSpines/spinePropertiesExport.txt', 'w') as metricsFile2:
        metricsFile2.write('')

        metricsFile2.write(f'{spine_length}\n'
                           f'{mesh_surface}\n'
                           f'{mesh_volume}\n'
                           f'{hull_volume}\n'
                           f'{hull_ratio}\n'
                           f'{average_distance}\n'
                           f'{coefficient_variation_distance}\n'
                           f'{open_angle}\n'
                           f'{mean_curvature[0]}\n'
                           f'{mean_curvature[1]}\n'
                           f'{gaussian_curvature[0]}\n'
                           f'{gaussian_curvature[1]}\n'
                           f'{mean_curvature[2]}\n'
                           f'{mean_curvature[3]}\n'
                           f'{gaussian_curvature[2]}\n'
                           f'{gaussian_curvature[3]}\n')


def find_sequence(text, sequence):
    """
        Finds and extracts a numeric value following a specified text within a given sequence.

        This function searches for a given text in a sequence (such as a filename) and extracts
        the numeric value immediately following this text. The function is particularly useful
        for parsing filenames or strings where a number is prefixed by a specific keyword.

        Parameters:
        text (str): The text to search for in the sequence.
        sequence (str): The sequence (e.g., filename or string) in which to search for the text.

        Returns:
        int: The extracted numeric value as an integer.
    """
    position = sequence.find(text)
    number = sequence[position + len(text):position + len(text) + 2]
    number_extracted = re.findall('[0-9]+', number)
    return int(number_extracted[0])


def parsing(root, dir_path, filename):
    """
        Parses a filename to extract specific numerical identifiers related to biological samples.

        This function extracts numerical identifiers like animal number, slice number, cell number,
        dendrite number, and spine number from a given filename. It also forms a standardized image
        name using these extracted numbers and checks for the existence of this image in a directory.

        Parameters:
        root (str): The root directory path.
        dirpath (str): The directory path where the file is located.
        filename (str): The filename to be parsed.

        Returns:
        tuple: A tuple containing the extracted numerical identifiers and the standardized image name.
    """
    animal_number = find_sequence('Animal', filename)
    slice_number = find_sequence('slice', filename)
    cell_number = find_sequence('cell', filename)
    dendrite_number = find_sequence('dendrite', filename)
    spine_number = find_sequence('spine', filename)
    image_name = f'Animal{animal_number}_slice{slice_number}_cell{cell_number}_dendrite{dendrite_number}_spine{spine_number:02d}.png'
    check_parsing(root, dir_path, image_name)
    return animal_number, slice_number, cell_number, dendrite_number, spine_number, image_name


def check_parsing(root, dir_path, image_name):
    """
        Checks if an image file exists in a specified directory based on a standardized image name.

        This function verifies the existence of an image file in a directory. The image name is expected
        to be in a standardized format, typically generated by a parsing function.

        Parameters:
        root (str): The root directory path.
        dir_path (str): The specific directory path to check.
        image_name (str): The standardized image name to look for.

        Returns:
        None: The function prints a message if the file is not found but does not return any value.
    """
    if not os.path.exists(f'{root}/{dir_path.split("/")[0]}/Images/{image_name}'):
        print(f'{root}/{dir_path.split("/")[0]}/Images/{image_name} not found.')


def check_mesh_and_image_match(folder):
    """
        Checks and reports mismatches between mesh and image files in a directory structure.

        This function iterates through mesh and image files in specified subdirectories of a given
        folder, parsing their filenames to extract biological sample identifiers. It then checks for
        mismatches or missing pairs between mesh and image files based on these identifiers.

        Parameters:
        folder (str): The path to the folder containing subdirectories with mesh and image files.

        Returns:
        None: The function prints a report of mismatches but does not return any value.
    """
    df = pd.DataFrame(columns=['Animal', 'Slice', 'Cell', 'Dendrite', 'Spine', 'Type'])
    for (root, dirs, filenames) in os.walk(f"{folder}/Mesh/", topdown=True):
        for filename in filenames:
            animal_number, slice_number, cell_number, dendrite_number, spine_number, __ = parsing(root, dirs, filename)
            df = df.append({'Animal': animal_number,
                            'Slice': slice_number,
                            'Cell': cell_number,
                            'Dendrite': dendrite_number,
                            'Spine': spine_number,
                            'Type': 'Mesh'
                            }, ignore_index=True)

    for (root, dirs, filenames) in os.walk(f"{folder}/Images/", topdown=True):
        for filename in filenames:
            animal_number, slice_number, cell_number, dendrite_number, spine_number, __ = parsing(root, dirs, filename)
            df = df.append({'Animal': animal_number,
                            'Slice': slice_number,
                            'Cell': cell_number,
                            'Dendrite': dendrite_number,
                            'Spine': spine_number,
                            'Type': 'Image'
                            }, ignore_index=True)

    col_list = df.duplicated(subset=['Animal', 'Slice', 'Cell', 'Dendrite', 'Spine'], keep=False)
    col_list = np.invert(col_list)
    if df[col_list].empty:
        print('No problem detected')
    else:
        print(df[col_list])


def is_mesh(scene_or_mesh):
    """
        Determines if the provided object is a mesh.

        This function checks whether the given object has an attribute typically associated with a mesh,
        such as 'area', to determine if it is a mesh object.

        Parameters:
        scene_or_mesh (object): The object to check.

        Returns:
        bool: Returns True if the object is a mesh, False otherwise.
    """
    return bool(hasattr(scene_or_mesh, 'area'))


def is_pymesh():
    """
       Checks if the PyMesh library is installed and available for use.

       This function attempts to import the PyMesh library. If the import succeeds, it indicates that
       PyMesh is installed and available; otherwise, it is not installed.

       Returns:
       bool: Returns True if PyMesh is installed, False otherwise.
   """
    try:
        import pymesh
    except ImportError:
        return False


def compute_pymesh_metrics(folder):
    """
    Computes and records various geometric and topological metrics for 3D meshes using PyMesh.

    This function processes each mesh file (in .ply format) within the specified folder. It calculates
    metrics such as spine length, surface area, volume, and curvatures using PyMesh. These metrics are
    compiled into a DataFrame and saved to a CSV file within the same folder.

    Parameters:
    folder (str): The path to the folder containing mesh files.

    Returns:
    None: The function outputs the metrics to a CSV file but does not return any value.
    """
    colList = ['Name', 'ImageName', 'Animal', 'Slice', 'Cell', 'Dendrite', 'Spine',
               'Length', 'Surface', 'Volume', 'Hull Volume', 'Hull Ratio',
               'Average Distance', 'CVD', 'Open Angle', 'Mean Curvature', 'Variance Curvature',
               'Mean Gaussian', 'Variance Gaussian', 'Highest Curvature', 'Lowest Curvature',
               'Lowest Gaussian', 'Highest Gaussian']
    df = pd.DataFrame(columns=colList)

    for file in get_filepaths(folder, ".ply"):
        mesh = trimesh.load_mesh(file)

        if is_mesh(mesh):
            mesh = trimesh_to_pymesh(mesh)

            mean_curvature = calculate_mean_curvature(mesh)
            gaussian_curvature = calculate_gaussian_curvature(mesh)
            animal_number, slice_number, cell_number, dendrite_number, spine_number, shf_presence = parser(file)
            image_name = ""
            df2 = pd.DataFrame([{'Name': str(file),
                                 'ImageName': image_name,
                                 'Animal': animal_number,
                                 'Slice': slice_number,
                                 'Cell': cell_number,
                                 'Dendrite': dendrite_number,
                                 'Spine': spine_number,
                                 'SHF': shf_presence,
                                 'Length': calculate_spine_length(mesh),
                                 'Surface': calculate_mesh_surface(mesh),
                                 'Volume': calculate_mesh_volume(mesh),
                                 'Hull Volume': calculate_hull_volume(mesh),
                                 'Hull Ratio': calculate_hull_ratio(mesh),
                                 'Average Distance': calculate_average_distance(mesh),
                                 'CVD': coefficient_of_variation_in_distance(mesh),
                                 'Open Angle': calculate_open_angle(mesh),
                                 'Mean Curvature': mean_curvature[0],
                                 'Variance Curvature': mean_curvature[1],
                                 'Mean Gaussian': gaussian_curvature[0],
                                 'Variance Gaussian': gaussian_curvature[1],
                                 'Highest Curvature': mean_curvature[2],
                                 'Lowest Curvature': mean_curvature[3],
                                 'Lowest Gaussian': gaussian_curvature[2],
                                 'Highest Gaussian': gaussian_curvature[3]}])
            df = df.append(df2, ignore_index=True)
    df = df.fillna(0)
    df.to_csv(f'{folder}/metrics.csv')
    print(f'Created {folder}/metrics.csv')


def compute_trimesh_metrics(folder: str):
    """
    Computes and records various geometric and topological metrics for 3D meshes using Trimesh.

    Similar to `compute_pymesh_metrics`, this function calculates various metrics for each mesh file
    in the specified folder using the Trimesh library. The results are compiled into a DataFrame and
    saved as a CSV file in the folder.

    Parameters:
    folder (str): The path to the folder containing mesh files.

    Returns:
    None: The function outputs the metrics to a CSV file but does not return any value.
    """
    col_list = ['Name', 'ImageName', 'Animal', 'Slice', 'Cell', 'Dendrite', 'Spine',
                'Length', 'Surface', 'Volume', 'Hull Volume', 'Hull Ratio',
                'Average Distance', 'CVD', 'Open Angle']
    df = pd.DataFrame(columns=col_list)

    image_name = ""
    for file in tqdm.tqdm(get_filepaths(folder, ".ply")):
        mesh = trimesh.load_mesh(file)
        animal_number, slice_number, cell_number, dendrite_number, spine_number, shf_presence = parser(file)
        df2 = pd.DataFrame([{'Name': str(file),
                             'ImageName': image_name,
                             'Animal': animal_number,
                             'Slice': slice_number,
                             'Cell': cell_number,
                             'Dendrite': dendrite_number,
                             'Spine': spine_number,
                             'SHF': shf_presence,
                             'Length': calculate_spine_length(mesh),
                             'Surface': mesh.area,
                             'Volume': calculate_mesh_volume(mesh),
                             'Hull Volume': trimesh_hull_volume(mesh),
                             'Hull Ratio': trimesh_hull_ratio(mesh),
                             'Average Distance': calculate_average_distance(mesh),
                             'CVD': coefficient_of_variation_in_distance(mesh),
                             'Open Angle': trimesh_calculate_open_angle(mesh),
                             }])
        df = df2 if df.empty else pd.concat([df, df2])
    df = df.fillna(0)
    df.to_csv(f'{folder}/metrics.csv')
    print(f'Created {folder}/metrics.csv')


def compute_metrics(folder: str):
    """
        Computes and records geometric and topological metrics for 3D meshes in a specified folder.

        This function determines whether to use PyMesh or Trimesh based on library availability and
        computes various metrics for each mesh file within the folder. The calculated metrics are saved
        to a CSV file in the folder.

        Parameters:
        folder (str): The path to the folder containing mesh files to be analyzed.

        Returns:
        None: The function saves the metrics to a CSV file but does not return any value.
    """

    if is_pymesh():
        compute_pymesh_metrics(folder)
    else:
        compute_trimesh_metrics(folder)


def neighbor_calc(mesh):
    """
        Calculates the connectivity of each vertex in a mesh and visualizes the results.

        This function computes the number of neighboring vertices for each vertex in the mesh. It then
        visualizes this data in a 3D scatter plot, including the gravity center and base center of the spine.

        Parameters:
        mesh (Mesh): The mesh for which neighbor calculations are to be performed.

        Returns:
        None: The function visualizes the results but does not return any value.
    """
    neighbor_array = np.zeros(int(mesh.vertices.size / 3))
    for face in mesh.faces:
        neighbor_array[face[0]] = neighbor_array[face[0]] + 1
        neighbor_array[face[1]] = neighbor_array[face[1]] + 1
        neighbor_array[face[2]] = neighbor_array[face[2]] + 1

    result = np.column_stack((mesh.vertices, np.transpose(neighbor_array)))

    plot_3d_scatter_with_color_and_gravity_center_and_gravity_median(result, 'X', 'Y', "Z", 'Spine in pixel',
                                                                     gravity_center(mesh), find_spine_base_center(mesh))
    print(find_spine_base_center(mesh))


def neighbor_calc2(mesh):
    """
        Calculates vertex valence in a mesh and visualizes the results along with frequency distribution.

        This function determines the valence (connectivity) of each vertex in the mesh. It then visualizes
        this data in a 3D scatter plot and a frequency distribution plot, including the gravity center and
        base center of the spine.

        Parameters:
        mesh (Mesh): The mesh for which valence calculations are to be performed.

        Returns:
        None: The function visualizes the results but does not return any value.
    """
    neighbor_array = get_mesh_valence(mesh)

    result = np.column_stack((mesh.vertices, np.transpose(neighbor_array)))

    plot_3d_scatter_with_color_and_gravity_center_and_gravity_median(result, 'X', 'Y', "Z", 'Spine in pixel',
                                                                     gravity_center(mesh),
                                                                     find_spine_base_center2(mesh))
    plot_frequency(np.transpose(neighbor_array), 'frequency', 'neighbor', "node")
    print(find_spine_base_center2(mesh))


def find_fixed(mesh):
    """
        Find the fixed points of a mesh by finding the less connected vertices
        :param mesh:
        :return:
    """
    fixed = calculate_fixed(mesh)
    plot_3d_scatter_fixed(mesh.vertices, fixed, 'X', 'Y', "Z", 'Spine in pixel')


def calculate_fixed(mesh):
    """
        Find the fixed points of a mesh by finding the less connected edges
        :param mesh:
        :return:
    """
    edges = calculate_edges(mesh)
    fixed = []
    for edge in edges:
        if np.sum(np.multiply(np.sum(mesh.faces == edge[0], axis=1), np.sum(mesh.faces == edge[1], axis=1))) < 2:
            fixed.extend((edge[0], edge[1]))
    return fixed


def compare_gravity_and_edges(mesh):
    """
        Compare the results of gravity center computed via mass center (mean of all vertices) and mean
        :param mesh: a pymesh mesh object
        :return:
    """
    edges = calculate_edges(mesh)
    fig = plt.figure()
    scatter_plot = fig.add_subplot(111, projection='3d')

    scatter_plot.scatter(mesh.vertices[:, 0], mesh.vertices[:, 1], mesh.vertices[:, 2], alpha=0.75)
    fixed = []
    for edge in edges:
        if np.sum(np.multiply(np.sum(mesh.faces == edge[0], axis=1), np.sum(mesh.faces == edge[1], axis=1))) < 2:
            fixed.extend((edge[0], edge[1]))
            xs = [mesh.vertices[edge[0], 0], mesh.vertices[edge[1], 0]]
            ys = [mesh.vertices[edge[0], 1], mesh.vertices[edge[1], 1]]
            zs = [mesh.vertices[edge[0], 2], mesh.vertices[edge[1], 2]]
            plt.plot(xs, ys, zs, color='red')

    plt.show()


def plot_number_of_nodes(mesh):
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    i = np.linspace(0, mesh.num_vertices, mesh.num_vertices, endpoint=False)
    x = mesh.vertices[:, 0]
    y = mesh.vertices[:, 1]
    z = mesh.vertices[:, 2]
    for w in range(np.shape(x)[0]):
        ax.text(x[w], y[w], z[w], int(i[w]), None)

    ax.set_xlim(np.min(mesh.vertices[:, 0]), np.max(mesh.vertices[:, 0]))
    ax.set_ylim(np.min(mesh.vertices[:, 1]), np.max(mesh.vertices[:, 1]))
    ax.set_zlim(np.min(mesh.vertices[:, 2]), np.max(mesh.vertices[:, 2]))
    plt.show()


def plotly_number_of_nodes(mesh):
    """
        Plot all vertices of a mesh and show them via plotly with blue dots. Plotly is far better than matplotlib
        when you have a lot of points to deal with.
        :param mesh: a pymesh Mesh object
        :return:
    """
    fig = go.Figure()
    i = np.linspace(0, mesh.num_vertices, mesh.num_vertices, endpoint=False)
    fig.add_trace(go.Scatter3d(
        x=mesh.vertices[:, 0],
        y=mesh.vertices[:, 1],
        z=mesh.vertices[:, 2],
        hovertext=i,
        mode='markers',
    ))
    fig.show()


def plotly_number_of_nodes_and_fixed(mesh):
    """
        Plot all vertices of a mesh and show them via plotly with blue dots. Plotly is far better than matplotlib
        when you have a lot of points to deal with. Also calculate "fixed" points by finding less connected vertices
        and show them in red.
        :param mesh: a pymesh Mesh object
        :return:
    """
    fig = go.Figure()
    fixed = np.unique(calculate_fixed(mesh))
    fixed_vertices = mesh.vertices[fixed]
    i = np.linspace(0, mesh.num_vertices, mesh.num_vertices, endpoint=False)
    fig.add_trace(go.Scatter3d(
        x=mesh.vertices[:, 0],
        y=mesh.vertices[:, 1],
        z=mesh.vertices[:, 2],
        hovertext=i,
        mode='markers',
    ))
    fig.add_trace(go.Scatter3d(
        x=fixed_vertices[:, 0],
        y=fixed_vertices[:, 1],
        z=fixed_vertices[:, 2],
        mode='markers',
        marker=dict(color='red'),
    ))
    fig.show()


if __name__ == "__main__":
    compute_metrics(r'Spines/')
