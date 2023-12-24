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


def center_mesh(mesh):
    centeredVertices = np.zeros(shape=(len(mesh.vertices), 3))

    centeredVertices[:, 0] = mesh.vertices[:, 0] - gravity_center(mesh)[0]
    centeredVertices[:, 1] = mesh.vertices[:, 1] - gravity_center(mesh)[1]
    centeredVertices[:, 2] = mesh.vertices[:, 2] - gravity_center(mesh)[2]
    return pymesh.meshio.form_mesh(centeredVertices, mesh.faces)


def gravity_center(mesh):
    return np.array([np.mean(mesh.vertices[:, 0]), np.mean(mesh.vertices[:, 1]), np.mean(mesh.vertices[:, 2])])


def gravity_median(mesh):
    return [np.median(mesh.vertices[:, 0]), np.median(mesh.vertices[:, 1]), np.median(mesh.vertices[:, 2])]


def mesh_surface(mesh):
    mesh.add_attribute("face_area")
    return sum(mesh.get_attribute("face_area"))


def calculate_mesh_volume(mesh):
    return sum(tetrahedron_calc_volume(mesh.vertices[faces[0]], mesh.vertices[faces[1]], mesh.vertices[faces[2]],
                                       gravity_center(mesh)) for faces in mesh.faces)


def mesh_volume3(mesh):
    return sum(tetrahedron_calc_volume(mesh.vertices[faces[0]], mesh.vertices[faces[1]], mesh.vertices[faces[2]],
                                       find_spine_base_center(mesh)) for faces in mesh.faces)


def determinant_3x3(m):
    return (m[0][0] * (m[1][1] * m[2][2] - m[1][2] * m[2][1]) -
            m[1][0] * (m[0][1] * m[2][2] - m[0][2] * m[2][1]) +
            m[2][0] * (m[0][1] * m[1][2] - m[0][2] * m[1][1]))


def subtract(a, b):
    return (a[0] - b[0],
            a[1] - b[1],
            a[2] - b[2])


def tetrahedron_calc_volume(vertex1, vertex2, vertex3, vertex4):
    return (abs(determinant_3x3((subtract(vertex1, vertex2),
                                 subtract(vertex2, vertex3),
                                 subtract(vertex3, vertex4),
                                 ))) / 6.0)


def calculate_distance(vertex1, vertex2):
    return np.sqrt(pow(vertex1[0] - vertex2[0], 2) + pow(vertex1[1] - vertex2[1], 2) + pow(vertex1[2] - vertex2[2], 2))


def calculate_vector(vertex1, vertex2):
    return np.array([vertex1[0] - vertex2[0], vertex1[1] - vertex2[1], vertex1[2] - vertex2[2]])


def spine_length(mesh):
    spineBaseCenter = find_spine_base_center2(mesh)
    length = [calculate_distance(vertice, spineBaseCenter) for vertice in mesh.vertices]

    length = np.array(length)
    length[::-1].sort()
    listOfLengths = range(int(0.05 * np.size(length)))
    return np.mean(length[listOfLengths])


def average_distance(mesh):
    spineBaseCenter = find_spine_base_center2(mesh)
    length = [calculate_distance(vertice, spineBaseCenter) for vertice in mesh.vertices]

    length = np.array(length)
    return np.mean(length)


def coefficient_of_variation_in_distance(mesh):
    spineBaseCenter = find_spine_base_center2(mesh)
    length = [calculate_distance(vertice, spineBaseCenter) for vertice in mesh.vertices]

    length = np.array(length)
    averageDistance = np.mean(length)
    std = np.std(length)
    return std / averageDistance


def open_angle(mesh):
    spineCenter = gravity_center(mesh)
    spineBaseCenter = find_spine_base_center(mesh)

    openAngle = [calculate_angle(spineCenter, spineBaseCenter, vertice) for vertice in mesh.vertices]

    return np.mean(openAngle)


def trimesh_open_angle(mesh):
    spineCenter = gravity_center(mesh)
    spineBaseCenter = trimesh_find_spine_base_center(mesh)

    openAngle = [calculate_angle(spineCenter, spineBaseCenter, vertice) for vertice in mesh.vertices]

    return np.mean(openAngle)


def calculate_angle(point1, point2, point3):
    point1 = np.array(point1)
    point2 = np.array(point2)
    point3 = np.array(point3)

    vector1 = calculate_vector(point1, point2)
    vector2 = calculate_vector(point3, point2)
    return np.arccos(np.dot(vector1, vector2) / (np.linalg.norm(vector1) * np.linalg.norm(vector2)))


def calculate_edges(mesh):
    edgeList = []
    for face in mesh.faces:
        edgeList.extend(([face[0], face[1]], [face[0], face[2]], [face[1], face[2]]))
    return edgeList


def calculate_hull_volume(mesh):
    hull_mesh = pymesh.convex_hull(mesh)
    return calculate_mesh_volume(hull_mesh)


def trimesh_hull_volume(mesh):
    hull_mesh = trimesh.convex.convex_hull(mesh)
    return calculate_mesh_volume(hull_mesh)


def calculate_hull_ratio(mesh):
    hull_volume = calculate_hull_volume(mesh)
    mesh_volume = calculate_mesh_volume(mesh)
    return (hull_volume - mesh_volume) / mesh_volume


def trimesh_hull_ratio(mesh):
    hull_volume = trimesh_hull_volume(mesh)
    mesh_volume = calculate_mesh_volume(mesh)
    return (hull_volume - mesh_volume) / mesh_volume


def calculate_gaussian_curvature(mesh):
    mesh.add_attribute('vertex_gaussian_curvature')
    gaussianCurvature = mesh.get_attribute('vertex_gaussian_curvature')
    # remove infinite value
    # gaussianCurvature = gaussianCurvature[gaussianCurvature < 1E308]

    if np.amax(gaussianCurvature) > 1E308:
        print('mesh is not clean')

    meanGaussianCurvature = np.mean(gaussianCurvature)
    varianceGaussianCurvature = np.var(gaussianCurvature)
    sortGaussianCurvature = np.sort(gaussianCurvature)
    lowerGaussianCurvature = np.mean(sortGaussianCurvature[: round(0.05 * gaussianCurvature.size)])

    higherGaussianCurvature = np.mean(
        sortGaussianCurvature[round(0.95 * gaussianCurvature.size):gaussianCurvature.size])
    return [meanGaussianCurvature, varianceGaussianCurvature, lowerGaussianCurvature, higherGaussianCurvature]


def calculate_mean_curvature(mesh):
    mesh.add_attribute('vertex_mean_curvature')
    meanCurvatureDivideBySurface = mesh.get_attribute('vertex_mean_curvature')
    mesh.add_attribute('vertex_voronoi_area')

    meanCurvature = meanCurvatureDivideBySurface

    # remove nan value
    meanCurvature = meanCurvature[~np.isnan(meanCurvature)]

    meanMeanCurvature = np.mean(meanCurvature)
    varianceMeanCurvature = np.var(meanCurvature)
    sortMeanCurvature = np.sort(meanCurvature)
    lowerMeanCurvature = np.mean(sortMeanCurvature[: round(0.05 * meanCurvature.size)])

    higherMeanCurvature = np.mean(sortMeanCurvature[round(0.95 * meanCurvature.size):meanCurvature.size])
    return [meanMeanCurvature, varianceMeanCurvature, lowerMeanCurvature, higherMeanCurvature]


def mesh_treatment(mesh):
    vertexConnectivity = calculate_vertex_connectivity(mesh)
    result = np.where(vertexConnectivity == 0)

    meshVertices = mesh.vertices
    for node in result:
        meshVertices = np.delete(meshVertices, node, 0)

    return pymesh.meshio.form_mesh(meshVertices, mesh.faces)


def find_spine_base_center2(mesh):
    listOfNeighbors = np.bincount(mesh.faces.ravel())

    verticesAndNeighbors = np.column_stack((mesh.vertices, np.transpose(listOfNeighbors)))
    lessConnectedVertices = np.where(verticesAndNeighbors[:, 3] <= 3)
    [X, Y, Z] = np.mean(mesh.vertices[lessConnectedVertices], axis=0)
    return [X, Y, Z]


def find_spine_base_center(mesh):
    valence = get_mesh_valence(mesh)
    verticesAndNeighbors = np.column_stack((mesh.vertices, np.transpose(valence)))
    lessConnectedVertices = np.where(verticesAndNeighbors[:, 3] <= 3)
    [x, y, z] = np.mean(mesh.vertices[lessConnectedVertices], axis=0)
    return [x, y, z]


def trimesh_find_spine_base_center(mesh):
    valence = trimesh_mesh_valence(mesh)
    valence2 = [len(val) for val in valence]
    verticesAndNeighbors = np.column_stack((mesh.vertices, np.transpose(valence2)))
    lessConnectedVertices = np.where(verticesAndNeighbors[:, 3] <= 3)
    [x, y, z] = np.mean(mesh.vertices[lessConnectedVertices], axis=0)
    return [x, y, z]


def get_mesh_valence(mesh):
    mesh.add_attribute("vertex_valance")
    return mesh.get_attribute("vertex_valance")


def trimesh_mesh_valence(mesh):
    return mesh.vertex_neighbors


def find_x_y_z_length(mesh):
    x = np.amax(mesh.vertices[0]) - np.amin(mesh.vertices[0])
    y = np.amax(mesh.vertices[1]) - np.amin(mesh.vertices[1])
    z = np.amax(mesh.vertices[2]) - np.amin(mesh.vertices[2])
    return [x, y, z]


def calculate_vertex_connectivity(mesh):
    vertexConnectivity = np.array([])

    wire_network = pymesh.wires.WireNetwork.create_from_data(mesh.vertices, mesh.faces)
    for vertex in range(mesh.num_vertices):
        vertexConnectivity = np.append(vertexConnectivity, wire_network.get_vertex_neighbors(vertex).size)
    return vertexConnectivity


def calculate_metrics(mesh):
    spineLength = spine_length(mesh)
    meshSurface = mesh_surface(mesh)
    meshVolume = calculate_mesh_volume(mesh)
    hullVolume = calculate_hull_volume(mesh)
    hullRatio = calculate_hull_ratio(mesh)
    meshLength = find_x_y_z_length(mesh)
    averageDistance = average_distance(mesh)
    CVD = coefficient_of_variation_in_distance(mesh)
    OA = open_angle(mesh)
    gaussianCurvature = calculate_gaussian_curvature(mesh)
    meanCurvature = calculate_mean_curvature(mesh)

    with open('3DImages/newSpines/spineProperties.txt', 'w') as metricsFile:
        metricsFile.write('')

        metricsFile.write(f'Spine Length : {spineLength}\n'
                          f'Mesh surface : {meshSurface}\n'
                          f'Mesh volume : {meshVolume}\n'
                          f'Hull Volume : {hullVolume}\n'
                          f'Hull Ratio : {hullRatio}\n'
                          f'Average distance : {averageDistance}\n'
                          f'Coefficient of variation in distance : {CVD}\n'
                          f'Open angle : {OA}\n'
                          f'Average of mean curvature : {meanCurvature[0]}\n'
                          f'Variance of mean curvature : {meanCurvature[1]}\n'
                          f'Average of gaussian curvature : {gaussianCurvature[0]}\n'
                          f'Variance of gaussian curvature : {gaussianCurvature[1]}\n'
                          f'Average of lower 5 percent mean curvature : {meanCurvature[2]}\n'
                          f'Average of higher 5 percent mean curvature : {meanCurvature[3]}\n'
                          f'Average of lower 5 percent gauss curvature : {gaussianCurvature[2]}\n'
                          f'Average of higher 5 percent gauss curvature : {gaussianCurvature[3]}\n'
                          f'Length X Y Z : {meshLength}\n'
                          f'Gravity center computed with median : {gravity_median(mesh)}\n'
                          f'Gravity center computed with mean : {gravity_center(mesh)}\n')

    with open('3DImages/newSpines/spinePropertiesExport.txt', 'w') as metricsFile2:
        metricsFile2.write('')

        metricsFile2.write(f'{spineLength}\n'
                           f'{meshSurface}\n'
                           f'{meshVolume}\n'
                           f'{hullVolume}\n'
                           f'{hullRatio}\n'
                           f'{averageDistance}\n'
                           f'{CVD}\n'
                           f'{OA}\n'
                           f'{meanCurvature[0]}\n'
                           f'{meanCurvature[1]}\n'
                           f'{gaussianCurvature[0]}\n'
                           f'{gaussianCurvature[1]}\n'
                           f'{meanCurvature[2]}\n'
                           f'{meanCurvature[3]}\n'
                           f'{gaussianCurvature[2]}\n'
                           f'{gaussianCurvature[3]}\n')


def find_sequence(text, sequence):
    position = sequence.find(text)
    number = sequence[position + len(text):position + len(text) + 2]
    numberExtracted = re.findall('[0-9]+', number)
    return int(numberExtracted[0])


def parsing(root, dirpath, filename):
    animalNumber = find_sequence('Animal', filename)
    sliceNumber = find_sequence('slice', filename)
    cellNumber = find_sequence('cell', filename)
    dendriteNumber = find_sequence('dendrite', filename)
    spineNumber = find_sequence('spine', filename)
    imageName = f'Animal{animalNumber}_slice{sliceNumber}_cell{cellNumber}_dendrite{dendriteNumber}_spine{spineNumber:02d}.png'
    check_parsing(root, dirpath, imageName)
    return animalNumber, sliceNumber, cellNumber, dendriteNumber, spineNumber, imageName


def check_parsing(root, dirpath, imageName):
    if not os.path.exists(f'{root}/{dirpath.split("/")[0]}/Images/{imageName}'):
        print(f'{root}/{dirpath.split("/")[0]}/Images/{imageName} not found.')


def check_mesh_and_image_match(folder):
    df = pd.DataFrame(columns=['Animal', 'Slice', 'Cell', 'Dendrite', 'Spine', 'Type'])
    for (root, dirs, filenames) in os.walk(f"{folder}/Mesh/", topdown=True):
        for filename in filenames:
            animalNumber, sliceNumber, cellNumber, dendriteNumber, spineNumber, __ = parsing(root, dirs, filename)
            df = df.append({'Animal': animalNumber,
                            'Slice': sliceNumber,
                            'Cell': cellNumber,
                            'Dendrite': dendriteNumber,
                            'Spine': spineNumber,
                            'Type': 'Mesh'
                            }, ignore_index=True)

    for (root, dirs, filenames) in os.walk(f"{folder}/Images/", topdown=True):
        for filename in filenames:
            animalNumber, sliceNumber, cellNumber, dendriteNumber, spineNumber, __ = parsing(root, dirs, filename)
            df = df.append({'Animal': animalNumber,
                            'Slice': sliceNumber,
                            'Cell': cellNumber,
                            'Dendrite': dendriteNumber,
                            'Spine': spineNumber,
                            'Type': 'Image'
                            }, ignore_index=True)

    list = df.duplicated(subset=['Animal', 'Slice', 'Cell', 'Dendrite', 'Spine'], keep=False)
    list = np.invert(list)
    if df[list].empty:
        print('No problem detected')
    else:
        print(df[list])


def is_mesh(scene_or_mesh):
    return bool(hasattr(scene_or_mesh, 'area'))


def is_pymesh():
    try:
        import pymesh
    except ImportError:
        return False


def compute_pymesh_metrics(folder):
    colList = ['Name', 'ImageName', 'Animal', 'Slice', 'Cell', 'Dendrite', 'Spine',
               'Length', 'Surface', 'Volume', 'Hull Volume', 'Hull Ratio',
               'Average Distance', 'CVD', 'Open Angle', 'Mean Curvature', 'Variance Curvature',
               'Mean Gaussian', 'Variance Gaussian', 'Highest Curvature', 'Lowest Curvature',
               'Lowest Gaussian', 'Highest Gaussian']
    df = pd.DataFrame(columns=colList)

    for file in get_filepaths(folder, ".ply"):
        # print(f'opening {os.path.join(root, dirs, filename)}')
        mesh = trimesh.load_mesh(file)

        if is_mesh(mesh):
            mesh = trimesh_to_pymesh(mesh)

            meanCurvature = calculate_mean_curvature(mesh)
            gaussianCurvature = calculate_gaussian_curvature(mesh)
            # animalNumber, sliceNumber, cellNumber, dendriteNumber, spineNumber, imageName = parsing(root, dirs, filename)
            # animalNumber, sliceNumber, cellNumber, dendriteNumber, spineNumber = parser(file)
            animalNumber, sliceNumber, cellNumber, dendriteNumber, spineNumber, shfPresence = parser(file)
            imageName = ""
            df2 = pd.DataFrame([{'Name': str(file),
                                 'ImageName': imageName,
                                 'Animal': animalNumber,
                                 'Slice': sliceNumber,
                                 'Cell': cellNumber,
                                 'Dendrite': dendriteNumber,
                                 'Spine': spineNumber,
                                 'SHF': shfPresence,
                                 'Length': spine_length(mesh),
                                 'Surface': mesh_surface(mesh),
                                 'Volume': calculate_mesh_volume(mesh),
                                 'Hull Volume': calculate_hull_volume(mesh),
                                 'Hull Ratio': calculate_hull_ratio(mesh),
                                 'Average Distance': average_distance(mesh),
                                 'CVD': coefficient_of_variation_in_distance(mesh),
                                 'Open Angle': open_angle(mesh),
                                 'Mean Curvature': meanCurvature[0],
                                 'Variance Curvature': meanCurvature[1],
                                 'Mean Gaussian': gaussianCurvature[0],
                                 'Variance Gaussian': gaussianCurvature[1],
                                 'Highest Curvature': meanCurvature[2],
                                 'Lowest Curvature': meanCurvature[3],
                                 'Lowest Gaussian': gaussianCurvature[2],
                                 'Highest Gaussian': gaussianCurvature[3]}])
            df = df.append(df2, ignore_index=True)
    df = df.fillna(0)
    df.to_csv(f'{folder}/metrics.csv')
    print(f'Created {folder}/metrics.csv')


def compute_trimesh_metrics(folder):
    colList = ['Name', 'ImageName', 'Animal', 'Slice', 'Cell', 'Dendrite', 'Spine',
               'Length', 'Surface', 'Volume', 'Hull Volume', 'Hull Ratio',
               'Average Distance', 'CVD', 'Open Angle']
    # 'Mean Curvature', 'Variance Curvature',
    # 'Mean Gaussian', 'Variance Gaussian', 'Highest Curvature', 'Lowest Curvature',
    # 'Lowest Gaussian', 'Highest Gaussian']
    df = pd.DataFrame(columns=colList)

    imageName = ""
    for file in tqdm.tqdm(get_filepaths(folder, ".ply")):
        # print(file)
        # print(file)
        # if not "17228" in file.__str__():
        #     continue
        # print("Okay")
        # print(f'opening {os.path.join(root, dirs, filename)}')
        mesh = trimesh.load_mesh(file)
        animalNumber, sliceNumber, cellNumber, dendriteNumber, spineNumber, shfPresence = parser(file)
        df2 = pd.DataFrame([{'Name': str(file),
                             'ImageName': imageName,
                             'Animal': animalNumber,
                             'Slice': sliceNumber,
                             'Cell': cellNumber,
                             'Dendrite': dendriteNumber,
                             'Spine': spineNumber,
                             'SHF': shfPresence,
                             'Length': spine_length(mesh),
                             'Surface': mesh.area,
                             'Volume': calculate_mesh_volume(mesh),
                             'Hull Volume': trimesh_hull_volume(mesh),
                             'Hull Ratio': trimesh_hull_ratio(mesh),
                             'Average Distance': average_distance(mesh),
                             'CVD': coefficient_of_variation_in_distance(mesh),
                             'Open Angle': trimesh_open_angle(mesh),
                             }])
        df = df.append(df2, ignore_index=True)

    df = df.fillna(0)
    df.to_csv(f'{folder}/metrics.csv')
    print(f'Created {folder}/metrics.csv')


def compute_metrics(folder):
    """
        Calculate all metrics of all meshes in folder toAnalyse/, put them into a dataframes object
         and save it as metrics.csv in toAnalyse folder
        :return:
    """

    if is_pymesh():
        compute_pymesh_metrics(folder)


    else:
        compute_trimesh_metrics(folder)


def neighbor_calc(mesh):
    neighborArray = np.zeros(int(mesh.vertices.size / 3))
    for face in mesh.faces:
        neighborArray[face[0]] = neighborArray[face[0]] + 1
        neighborArray[face[1]] = neighborArray[face[1]] + 1
        neighborArray[face[2]] = neighborArray[face[2]] + 1

    result = np.column_stack((mesh.vertices, np.transpose(neighborArray)))

    plot_3d_scatter_with_color_and_gravity_center_and_gravity_median(result, 'X', 'Y', "Z", 'Spine in pixel',
                                                                     gravity_center(mesh), find_spine_base_center(mesh))
    print(find_spine_base_center(mesh))


def neighbor_calc2(mesh):
    neighborArray = get_mesh_valence(mesh)

    result = np.column_stack((mesh.vertices, np.transpose(neighborArray)))

    plot_3d_scatter_with_color_and_gravity_center_and_gravity_median(result, 'X', 'Y', "Z", 'Spine in pixel',
                                                                     gravity_center(mesh),
                                                                     find_spine_base_center2(mesh))
    plot_frequency(np.transpose(neighborArray), 'frequency', 'neighbor', "node")
    print(find_spine_base_center2(mesh))


def get_frequency2(mesh):
    array = np.zeros(int(mesh.vertices.size / 3))
    for face in mesh.faces:
        array[face[0]] = array[face[0]] + 1
        array[face[1]] = array[face[1]] + 1
        array[face[2]] = array[face[2]] + 1
    unique, counts = np.unique(array, return_counts=True)
    frequencyArray = np.column_stack((unique, counts))
    print(frequencyArray)
    print('\n')


def find_fixed(mesh):
    """
        Find the fixed points of a mesh by finding the less connected vertices
        :param mesh:
        :return:
    """
    edges = calculate_edges(mesh)
    fixed = []
    for edge in edges:
        if np.sum(np.multiply(np.sum(mesh.faces == edge[0], axis=1), np.sum(mesh.faces == edge[1], axis=1))) < 2:
            fixed.extend((edge[0], edge[1]))
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
    scatterPlot = fig.add_subplot(111, projection='3d')

    scatterPlot.scatter(mesh.vertices[:, 0], mesh.vertices[:, 1], mesh.vertices[:, 2], alpha=0.75)
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
        # text=i
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
    fixedVertices = mesh.vertices[fixed]
    i = np.linspace(0, mesh.num_vertices, mesh.num_vertices, endpoint=False)
    # print(i)
    # w = []
    # for j in i:
    #     print(j)
    #     w.append(int(i[int(j)]))
    fig.add_trace(go.Scatter3d(
        x=mesh.vertices[:, 0],
        y=mesh.vertices[:, 1],
        z=mesh.vertices[:, 2],
        hovertext=i,
        mode='markers',
        # text=i
    ))
    fig.add_trace(go.Scatter3d(
        x=fixedVertices[:, 0],
        y=fixedVertices[:, 1],
        z=fixedVertices[:, 2],
        # hovertext=i,
        mode='markers',
        marker=dict(color='red'),
    ))
    fig.show()


if __name__ == "__main__":
    # messy examples of commands
    # filename = askopenfilename()
    # filename = '/mnt/4EB2FF89256EC207/PycharmProjects/Segmentation/Reconstruction/spine255.ply'

    # filename = '/mnt/4EB2FF89256EC207/PycharmProjects/spineReconstruction/optimisedMeshes/' \
    #            'deconvolved_spine_mesh_0_0.stl'

    # filename = '/mnt/4EB2FF89256EC207/PycharmProjects/spineReconstruction/optimisedMeshes/' \
    #            'Slice2_spine6_new_9.0_optimised.stl'
    #
    # mesh = pymesh.load_mesh(filename)
    # plotly_number_of_nodes_and_fixed(mesh)

    # compute_metrics('all_animal_analyse')
    # check_mesh_and_image_match('a')
    # compute_metrics('spines4'
    # )
    compute_metrics(r'D:\Downloads\spines_2022_modified\Done')
    # calculate_PCA()
    # calculate_PCA3D()
    # calculate_umap()

    # meshSpine = pymesh.load_mesh(filename)
    # plotly_number_of_nodes_and_fixed(meshSpine)

    # plot_3d_scatter_with_color_and_gravity_center_and_gravity_median(m)
    # neighbor_calc(meshSpine)
    # compare_gravity_and_edges(mesh)
    # find_fixed(mesh)
    # calculate_fixed(mesh)
    # print(calculate_edges(meshSpine))
    # neighbor_calc(meshSpine)
    # find_spine_base_center(meshSpine)
    # find_spine_base_center(meshSpine)
    # neighbor_calc(meshSpine)
    # neighbor_calc2(meshSpine)
    # calculate_metrics(meshSpine)
    # print(mesh_volume(meshSpine))
    # print(mesh_volume3(meshSpine))
