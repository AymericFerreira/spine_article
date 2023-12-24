import numpy as np
import networkx as nx
import trimesh
import os
import pathlib
from tqdm import tqdm
from subprocess import DEVNULL, STDOUT, check_call
import pymeshlab as ml



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


def remove_small_meshes(mesh):
    # todo : refactoring
    graph = create_graph(mesh)
    meshSizeList = get_size_of_meshes(graph)
    biggestMeshes = np.where(meshSizeList > 0.01 * np.max(meshSizeList))[0]
    meshList = [
        get_list_of_nodes_in_each_meshes(graph)[extractedMesh]
        for extractedMesh in biggestMeshes
    ]

    # convert list of set to np.array
    for meshNumber in range(len(meshList)):
        meshList[meshNumber] = list(meshList[meshNumber])

    return recreate_meshes(meshList, mesh)


# def remove_small_meshes_trimesh(mesh, tolerance='auto'):
#     """
#         Find all submeshes in a Trimesh Mesh object and remove small meshes based on tolerance in tolerance% (default
#         auto parameter is the mean of vertices in all meshes). Split function of trimesh giving not accurate result we
#         will use networkx to isolate biggest meshes based on graph theory. Then we will remove all meshes under the
#         tolerance threshold based on number of vertices. If the submesh contains less than x% of total it will be remove
#         . If not, all submeshes will be concatenate into one object. Then, the mesh will be returned
#         :param mesh: Trimesh Mesh object
#         :param tolerance:
#         :return: Trimesh Mesh object
#     """
#     graph = create_graph(mesh)
#     meshSizeArray = np.array(get_size_of_meshes(graph))
#     tolerance = np.mean(meshSizeArray) if tolerance == 'auto' else tolerance/100
#     biggestMeshes = meshSizeArray[meshSizeArray > tolerance]
#     if len(biggestMeshes) == 1:
#         return biggestMeshes[0]
#     elif len(biggestMeshes) > 1:
#         biggestMeshes.sort(reverse=True)
#         if biggestMeshes[0] > 0.5*np.sum(biggestMeshes[0:2])
#             return biggestMeshes[0]
#         else:
#             return

def new_smoothing():
    folder = "D:/Downloads/temp/11964/Mesh"
    fileList = get_filepaths(folder, [".ply", "remeshed"])
    for file in fileList:
        if "smoothed" not in file.__str__():
            print(file)
            # smoothing_via_meshlabserver(file)
            smoothing_via_pymeshlab(file)


def smoothing_via_pymeshlab(meshFilename, laplacien=1):
    if not meshFilename.exists():
        return
    ms = ml.MeshSet()
    meshDir = meshFilename.parent
    ms.load_new_mesh(meshFilename)
    filter = f"""
    <!DOCTYPE FilterScript>
    <FilterScript>
        <filter name="Laplacian Smooth">
          <Param value="{laplacien}" type="RichInt" description="Smoothing steps" name="stepSmoothNum" tooltip="The number of times that the whole algorithm (normal smoothing + vertex fitting) is iterated."/>
          <Param value="false" type="RichBool" description="1D Boundary Smoothing" name="Boundary" tooltip="Smooth boundary edges only by themselves (e.g. the polyline forming the boundary of the mesh is independently smoothed). This can reduce the shrinking on the border but can have strange effects on very small boundaries."/>
          <Param value="false" type="RichBool" description="Cotangent weighting" name="cotangentWeight" tooltip="Use cotangent weighting scheme for the averaging of the position. Otherwise the simpler umbrella scheme (1 if the edge is present) is used."/>
          <Param value="false" type="RichBool" description="Affect only selection" name="Selected" tooltip="If checked the filter is performed only on the selected area"/>
        </filter>
    </FilterScript>"""

    filterFileName = pathlib.Path(meshDir, 'filter.mlx')
    outputMeshFilename = pathlib.Path(meshDir, meshFilename.stem + f'_smoothed_{laplacien}_false.ply')

    with open(f'{filterFileName}', 'w') as filterFile:
        filterFile.write(filter)
    ms.load_filter_script("filter.mlx")
    ms.apply_filter_script()
    ms.save_current_mesh(outputMeshFilename.__str__())


def remeshing_via_pymeshlab(meshFilename, laplacien=2):
    if not meshFilename.exists():
        return


    ms = ml.MeshSet()

    mesh = trimesh.load_mesh(meshFilename)
    meshPath = pathlib.Path(meshFilename)
    meshFaces = int(mesh.faces.shape[0] / 3)
    meshDir = pathlib.Path(meshPath.parent, 'finalMesh')

    del mesh
    filter = f"""
        <!DOCTYPE FilterScript>
        <FilterScript>
         <filter name="Simplification: Quadric Edge Collapse Decimation">
          <Param isxmlparam="0" type="RichInt" value="{meshFaces}" name="TargetFaceNum" description="Target number of faces" tooltip="The desired final number of faces."/>
          <Param isxmlparam="0" type="RichFloat" value="0" name="TargetPerc" description="Percentage reduction (0..1)" tooltip="If non zero, this parameter specifies the desired final size of the mesh as a percentage of the initial size."/>
          <Param isxmlparam="0" type="RichFloat" value="0.3" name="QualityThr" description="Quality threshold" tooltip="Quality threshold for penalizing bad shaped faces.&lt;br>The value is in the range [0..1]&#xa; 0 accept any kind of face (no penalties),&#xa; 0.5  penalize faces with quality &lt; 0.5, proportionally to their shape&#xa;"/>
          <Param isxmlparam="0" type="RichBool" value="true" name="PreserveBoundary" description="Preserve Boundary of the mesh" tooltip="The simplification process tries to do not affect mesh boundaries during simplification"/>
          <Param isxmlparam="0" type="RichFloat" value="1" name="BoundaryWeight" description="Boundary Preserving Weight" tooltip="The importance of the boundary during simplification. Default (1.0) means that the boundary has the same importance of the rest. Values greater than 1.0 raise boundary importance and has the effect of removing less vertices on the border. Admitted range of values (0,+inf). "/>
          <Param isxmlparam="0" type="RichBool" value="false" name="PreserveNormal" description="Preserve Normal" tooltip="Try to avoid face flipping effects and try to preserve the original orientation of the surface"/>
          <Param isxmlparam="0" type="RichBool" value="false" name="PreserveTopology" description="Preserve Topology" tooltip="Avoid all the collapses that should cause a topology change in the mesh (like closing holes, squeezing handles, etc). If checked the genus of the mesh should stay unchanged."/>
          <Param isxmlparam="0" type="RichBool" value="true" name="OptimalPlacement" description="Optimal position of simplified vertices" tooltip="Each collapsed vertex is placed in the position minimizing the quadric error.&#xa; It can fail (creating bad spikes) in case of very flat areas. &#xa;If disabled edges are collapsed onto one of the two original vertices and the final mesh is composed by a subset of the original vertices. "/>
          <Param isxmlparam="0" type="RichBool" value="false" name="PlanarQuadric" description="Planar Simplification" tooltip="Add additional simplification constraints that improves the quality of the simplification of the planar portion of the mesh, as a side effect, more triangles will be preserved in flat areas (allowing better shaped triangles)."/>
          <Param isxmlparam="0" type="RichFloat" value="0.001" name="PlanarWeight" description="Planar Simp. Weight" tooltip="How much we should try to preserve the triangles in the planar regions. If you lower this value planar areas will be simplified more."/>
          <Param isxmlparam="0" type="RichBool" value="false" name="QualityWeight" description="Weighted Simplification" tooltip="Use the Per-Vertex quality as a weighting factor for the simplification. The weight is used as a error amplification value, so a vertex with a high quality value will not be simplified and a portion of the mesh with low quality values will be aggressively simplified."/>
          <Param isxmlparam="0" type="RichBool" value="true" name="AutoClean" description="Post-simplification cleaning" tooltip="After the simplification an additional set of steps is performed to clean the mesh (unreferenced vertices, bad faces, etc)"/>
          <Param isxmlparam="0" type="RichBool" value="false" name="Selected" description="Simplify only selected faces" tooltip="The simplification is applied only to the selected set of faces.&#xa; Take care of the target number of faces!"/>
         </filter>
         <filter name="Laplacian Smooth">
          <Param value="{laplacien}" type="RichInt" description="Smoothing steps" name="stepSmoothNum" tooltip="The number of times that the whole algorithm (normal smoothing + vertex fitting) is iterated."/>
          <Param value="false" type="RichBool" description="1D Boundary Smoothing" name="Boundary" tooltip="Smooth boundary edges only by themselves (e.g. the polyline forming the boundary of the mesh is independently smoothed). This can reduce the shrinking on the border but can have strange effects on very small boundaries."/>
          <Param value="false" type="RichBool" description="Cotangent weighting" name="cotangentWeight" tooltip="Use cotangent weighting scheme for the averaging of the position. Otherwise the simpler umbrella scheme (1 if the edge is present) is used."/>
          <Param value="false" type="RichBool" description="Affect only selection" name="Selected" tooltip="If checked the filter is performed only on the selected area"/>
         </filter>
        </FilterScript>"""

    meshDir = pathlib.Path(meshPath.parent, 'SimplifiedMesh')
    filterFileName = pathlib.Path(meshDir.parent, 'filter.mlx')
    outputMeshFilename = pathlib.Path(meshDir.parent.parent, 'reMeshed',
                                      meshFilename.stem + 'remeshed_smoothed.ply')

    with open(f'{filterFileName}', 'w') as filterFile:
        filterFile.write(filter)

    ms.load_new_mesh(meshFilename.__str__())
    ms.load_filter_script('filter.mlx')
    ms.apply_filter_script()
    ms.save_current_mesh(outputMeshFilename.__str__())

    newMesh = trimesh.load_mesh(outputMeshFilename)

    newSimplifiedMesh = new_remove_small_meshes(newMesh)
    del newMesh

    if newSimplifiedMesh:
        newSimplifiedMesh.export(
            pathlib.Path(meshDir.parent.parent, 'finalMesh', meshFilename.stem + 'remeshed_smoothed.ply'))

        filter = f"""
            <!DOCTYPE FilterScript>
            <FilterScript>
             <filter name="Simplification: Quadric Edge Collapse Decimation">
              <Param isxmlparam="0" type="RichInt" value="{meshFaces}" name="TargetFaceNum" description="Target number of faces" tooltip="The desired final number of faces."/>
              <Param isxmlparam="0" type="RichFloat" value="0" name="TargetPerc" description="Percentage reduction (0..1)" tooltip="If non zero, this parameter specifies the desired final size of the mesh as a percentage of the initial size."/>
              <Param isxmlparam="0" type="RichFloat" value="0.3" name="QualityThr" description="Quality threshold" tooltip="Quality threshold for penalizing bad shaped faces.&lt;br>The value is in the range [0..1]&#xa; 0 accept any kind of face (no penalties),&#xa; 0.5  penalize faces with quality &lt; 0.5, proportionally to their shape&#xa;"/>
              <Param isxmlparam="0" type="RichBool" value="true" name="PreserveBoundary" description="Preserve Boundary of the mesh" tooltip="The simplification process tries to do not affect mesh boundaries during simplification"/>
              <Param isxmlparam="0" type="RichFloat" value="1" name="BoundaryWeight" description="Boundary Preserving Weight" tooltip="The importance of the boundary during simplification. Default (1.0) means that the boundary has the same importance of the rest. Values greater than 1.0 raise boundary importance and has the effect of removing less vertices on the border. Admitted range of values (0,+inf). "/>
              <Param isxmlparam="0" type="RichBool" value="false" name="PreserveNormal" description="Preserve Normal" tooltip="Try to avoid face flipping effects and try to preserve the original orientation of the surface"/>
              <Param isxmlparam="0" type="RichBool" value="false" name="PreserveTopology" description="Preserve Topology" tooltip="Avoid all the collapses that should cause a topology change in the mesh (like closing holes, squeezing handles, etc). If checked the genus of the mesh should stay unchanged."/>
              <Param isxmlparam="0" type="RichBool" value="true" name="OptimalPlacement" description="Optimal position of simplified vertices" tooltip="Each collapsed vertex is placed in the position minimizing the quadric error.&#xa; It can fail (creating bad spikes) in case of very flat areas. &#xa;If disabled edges are collapsed onto one of the two original vertices and the final mesh is composed by a subset of the original vertices. "/>
              <Param isxmlparam="0" type="RichBool" value="false" name="PlanarQuadric" description="Planar Simplification" tooltip="Add additional simplification constraints that improves the quality of the simplification of the planar portion of the mesh, as a side effect, more triangles will be preserved in flat areas (allowing better shaped triangles)."/>
              <Param isxmlparam="0" type="RichFloat" value="0.001" name="PlanarWeight" description="Planar Simp. Weight" tooltip="How much we should try to preserve the triangles in the planar regions. If you lower this value planar areas will be simplified more."/>
              <Param isxmlparam="0" type="RichBool" value="false" name="QualityWeight" description="Weighted Simplification" tooltip="Use the Per-Vertex quality as a weighting factor for the simplification. The weight is used as a error amplification value, so a vertex with a high quality value will not be simplified and a portion of the mesh with low quality values will be aggressively simplified."/>
              <Param isxmlparam="0" type="RichBool" value="true" name="AutoClean" description="Post-simplification cleaning" tooltip="After the simplification an additional set of steps is performed to clean the mesh (unreferenced vertices, bad faces, etc)"/>
              <Param isxmlparam="0" type="RichBool" value="false" name="Selected" description="Simplify only selected faces" tooltip="The simplification is applied only to the selected set of faces.&#xa; Take care of the target number of faces!"/>
             </filter>
            </FilterScript>"""

        meshDir = pathlib.Path(meshPath.parent, 'SimplifiedMesh')

        filterFileName = pathlib.Path(meshDir.parent, 'filter.mlx')
        outputMeshFilename = pathlib.Path(meshDir.parent.parent, 'reMeshed', meshFilename.stem + 'remeshed.ply')

        with open(f'{filterFileName}', 'w') as filterFile:
            filterFile.write(filter)

        ms = ml.MeshSet()
        ms.load_new_mesh(meshFilename.__str__())
        ms.load_filter_script('filter.mlx')
        ms.apply_filter_script()
        ms.save_current_mesh(outputMeshFilename.__str__())
        # print(f'{meshLabServerPath} -i {meshFilename.as_posix()} -s {filterFileName} -o {outputMeshFilename.as_posix()}')
        # os.system(
        #     f'{meshLabServerPath} -i {meshFilename.as_posix()} -s {filterFileName} -o {outputMeshFilename.as_posix()}')
        # check_call(
        #     f'{meshLabServerPath} -i {meshFilename.as_posix()} -s {filterFileName} -o {outputMeshFilename.as_posix()}',
        #     stdout=DEVNULL, stderr=STDOUT)

        if outputMeshFilename.exists():
            newMesh = trimesh.load_mesh(outputMeshFilename)
            newSimplifiedMesh = new_remove_small_meshes(newMesh)
            del newMesh

            if newSimplifiedMesh:
                newSimplifiedMesh.export(
                    pathlib.Path(meshDir.parent.parent, 'finalMesh', meshFilename.stem + 'remeshed.ply'))


def smoothing_via_meshlabserver(meshFilename, laplacien=1):
    if not meshFilename.exists():
        return
    # if "smoothed" in meshFilename.__str__():
    #     return
    # if "remesh" not in meshFilename.__str__():
    #     return
    meshDir = meshFilename.parent
    mesh = trimesh.load_mesh(meshFilename)
    meshPath = pathlib.Path(meshFilename)
    meshFaces = int(mesh.faces.shape[0] / 3)

    del mesh

    # meshLabServerPath = '/mnt/4EB2FF89256EC207/meshlab/distrib/meshlabserver'
    meshLabServerPath = 'D:/Programs/MeshLab/meshlabserver'

    filter = f"""
    <!DOCTYPE FilterScript>
    <FilterScript>
        <filter name="Laplacian Smooth">
          <Param value="{laplacien}" type="RichInt" description="Smoothing steps" name="stepSmoothNum" tooltip="The number of times that the whole algorithm (normal smoothing + vertex fitting) is iterated."/>
          <Param value="false" type="RichBool" description="1D Boundary Smoothing" name="Boundary" tooltip="Smooth boundary edges only by themselves (e.g. the polyline forming the boundary of the mesh is independently smoothed). This can reduce the shrinking on the border but can have strange effects on very small boundaries."/>
          <Param value="false" type="RichBool" description="Cotangent weighting" name="cotangentWeight" tooltip="Use cotangent weighting scheme for the averaging of the position. Otherwise the simpler umbrella scheme (1 if the edge is present) is used."/>
          <Param value="false" type="RichBool" description="Affect only selection" name="Selected" tooltip="If checked the filter is performed only on the selected area"/>
        </filter>
    </FilterScript>"""

    filterFileName = pathlib.Path(meshDir, 'filter.mlx')
    # outputMeshFilename = meshFilename.split('/')[-1]
    outputMeshFilename = pathlib.Path(meshDir, meshFilename.stem + f'_smoothed_{laplacien}_false.ply')

    with open(f'{filterFileName}', 'w') as filterFile:
        filterFile.write(filter)

    # print(f'{meshLabServerPath} -i {meshFilename.as_posix()} -s {filterFileName} -o {outputMeshFilename.as_posix()}')
    check_call(
        f'{meshLabServerPath} -i {meshFilename.as_posix()} -s {filterFileName} -o {outputMeshFilename.as_posix()}',
        stdout=DEVNULL, stderr=STDOUT)


def remeshing_via_meshlabserver(meshFilename, laplacien=2):
    if not meshFilename.exists():
        return

    mesh = trimesh.load_mesh(meshFilename)
    meshPath = pathlib.Path(meshFilename)
    meshFaces = int(mesh.faces.shape[0] / 3)
    meshDir = pathlib.Path(meshPath.parent, 'finalMesh')

    del mesh

    # meshLabServerPath = '/mnt/4EB2FF89256EC207/meshlab/distrib/meshlabserver'
    meshLabServerPath = 'D:/Programs/MeshLab/meshlabserver'

    filter = f"""
        <!DOCTYPE FilterScript>
        <FilterScript>
         <filter name="Simplification: Quadric Edge Collapse Decimation">
          <Param isxmlparam="0" type="RichInt" value="{meshFaces}" name="TargetFaceNum" description="Target number of faces" tooltip="The desired final number of faces."/>
          <Param isxmlparam="0" type="RichFloat" value="0" name="TargetPerc" description="Percentage reduction (0..1)" tooltip="If non zero, this parameter specifies the desired final size of the mesh as a percentage of the initial size."/>
          <Param isxmlparam="0" type="RichFloat" value="0.3" name="QualityThr" description="Quality threshold" tooltip="Quality threshold for penalizing bad shaped faces.&lt;br>The value is in the range [0..1]&#xa; 0 accept any kind of face (no penalties),&#xa; 0.5  penalize faces with quality &lt; 0.5, proportionally to their shape&#xa;"/>
          <Param isxmlparam="0" type="RichBool" value="true" name="PreserveBoundary" description="Preserve Boundary of the mesh" tooltip="The simplification process tries to do not affect mesh boundaries during simplification"/>
          <Param isxmlparam="0" type="RichFloat" value="1" name="BoundaryWeight" description="Boundary Preserving Weight" tooltip="The importance of the boundary during simplification. Default (1.0) means that the boundary has the same importance of the rest. Values greater than 1.0 raise boundary importance and has the effect of removing less vertices on the border. Admitted range of values (0,+inf). "/>
          <Param isxmlparam="0" type="RichBool" value="false" name="PreserveNormal" description="Preserve Normal" tooltip="Try to avoid face flipping effects and try to preserve the original orientation of the surface"/>
          <Param isxmlparam="0" type="RichBool" value="false" name="PreserveTopology" description="Preserve Topology" tooltip="Avoid all the collapses that should cause a topology change in the mesh (like closing holes, squeezing handles, etc). If checked the genus of the mesh should stay unchanged."/>
          <Param isxmlparam="0" type="RichBool" value="true" name="OptimalPlacement" description="Optimal position of simplified vertices" tooltip="Each collapsed vertex is placed in the position minimizing the quadric error.&#xa; It can fail (creating bad spikes) in case of very flat areas. &#xa;If disabled edges are collapsed onto one of the two original vertices and the final mesh is composed by a subset of the original vertices. "/>
          <Param isxmlparam="0" type="RichBool" value="false" name="PlanarQuadric" description="Planar Simplification" tooltip="Add additional simplification constraints that improves the quality of the simplification of the planar portion of the mesh, as a side effect, more triangles will be preserved in flat areas (allowing better shaped triangles)."/>
          <Param isxmlparam="0" type="RichFloat" value="0.001" name="PlanarWeight" description="Planar Simp. Weight" tooltip="How much we should try to preserve the triangles in the planar regions. If you lower this value planar areas will be simplified more."/>
          <Param isxmlparam="0" type="RichBool" value="false" name="QualityWeight" description="Weighted Simplification" tooltip="Use the Per-Vertex quality as a weighting factor for the simplification. The weight is used as a error amplification value, so a vertex with a high quality value will not be simplified and a portion of the mesh with low quality values will be aggressively simplified."/>
          <Param isxmlparam="0" type="RichBool" value="true" name="AutoClean" description="Post-simplification cleaning" tooltip="After the simplification an additional set of steps is performed to clean the mesh (unreferenced vertices, bad faces, etc)"/>
          <Param isxmlparam="0" type="RichBool" value="false" name="Selected" description="Simplify only selected faces" tooltip="The simplification is applied only to the selected set of faces.&#xa; Take care of the target number of faces!"/>
         </filter>
         <filter name="Laplacian Smooth">
          <Param value="{laplacien}" type="RichInt" description="Smoothing steps" name="stepSmoothNum" tooltip="The number of times that the whole algorithm (normal smoothing + vertex fitting) is iterated."/>
          <Param value="false" type="RichBool" description="1D Boundary Smoothing" name="Boundary" tooltip="Smooth boundary edges only by themselves (e.g. the polyline forming the boundary of the mesh is independently smoothed). This can reduce the shrinking on the border but can have strange effects on very small boundaries."/>
          <Param value="false" type="RichBool" description="Cotangent weighting" name="cotangentWeight" tooltip="Use cotangent weighting scheme for the averaging of the position. Otherwise the simpler umbrella scheme (1 if the edge is present) is used."/>
          <Param value="false" type="RichBool" description="Affect only selection" name="Selected" tooltip="If checked the filter is performed only on the selected area"/>
         </filter>
        </FilterScript>"""

    meshDir = pathlib.Path(meshPath.parent, 'SimplifiedMesh')

    # meshDir = '/'.join(meshFilename.split('/')[:-1])
    filterFileName = pathlib.Path(meshDir.parent, 'filter.mlx')
    # outputMeshFilename = meshFilename.split('/')[-1]
    outputMeshFilename = pathlib.Path(meshDir.parent.parent, 'reMeshed', meshFilename.stem + 'remeshed_smoothed.ply')

    with open(f'{filterFileName}', 'w') as filterFile:
        filterFile.write(filter)

    # print(f'{meshLabServerPath} -i {meshFilename.as_posix()} -s {filterFileName} -o {outputMeshFilename.as_posix()}')
    check_call(
        f'{meshLabServerPath} -i {meshFilename.as_posix()} -s {filterFileName} -o {outputMeshFilename.as_posix()}',
        stdout=DEVNULL, stderr=STDOUT)

    # os.system(
    #     f'{meshLabServerPath} -i {meshFilename.as_posix()} -s {filterFileName} -o {outputMeshFilename.as_posix()}')

    newMesh = trimesh.load_mesh(outputMeshFilename)

    newSimplifiedMesh = new_remove_small_meshes(newMesh)
    del newMesh

    if newSimplifiedMesh:
        newSimplifiedMesh.export(
            pathlib.Path(meshDir.parent.parent, 'finalMesh', meshFilename.stem + 'remeshed_smoothed.ply'))
        # print(f'Final number of vertices : {len(newSimplifiedMesh.vertices)}')

        filter = f"""
            <!DOCTYPE FilterScript>
            <FilterScript>
             <filter name="Simplification: Quadric Edge Collapse Decimation">
              <Param isxmlparam="0" type="RichInt" value="{meshFaces}" name="TargetFaceNum" description="Target number of faces" tooltip="The desired final number of faces."/>
              <Param isxmlparam="0" type="RichFloat" value="0" name="TargetPerc" description="Percentage reduction (0..1)" tooltip="If non zero, this parameter specifies the desired final size of the mesh as a percentage of the initial size."/>
              <Param isxmlparam="0" type="RichFloat" value="0.3" name="QualityThr" description="Quality threshold" tooltip="Quality threshold for penalizing bad shaped faces.&lt;br>The value is in the range [0..1]&#xa; 0 accept any kind of face (no penalties),&#xa; 0.5  penalize faces with quality &lt; 0.5, proportionally to their shape&#xa;"/>
              <Param isxmlparam="0" type="RichBool" value="true" name="PreserveBoundary" description="Preserve Boundary of the mesh" tooltip="The simplification process tries to do not affect mesh boundaries during simplification"/>
              <Param isxmlparam="0" type="RichFloat" value="1" name="BoundaryWeight" description="Boundary Preserving Weight" tooltip="The importance of the boundary during simplification. Default (1.0) means that the boundary has the same importance of the rest. Values greater than 1.0 raise boundary importance and has the effect of removing less vertices on the border. Admitted range of values (0,+inf). "/>
              <Param isxmlparam="0" type="RichBool" value="false" name="PreserveNormal" description="Preserve Normal" tooltip="Try to avoid face flipping effects and try to preserve the original orientation of the surface"/>
              <Param isxmlparam="0" type="RichBool" value="false" name="PreserveTopology" description="Preserve Topology" tooltip="Avoid all the collapses that should cause a topology change in the mesh (like closing holes, squeezing handles, etc). If checked the genus of the mesh should stay unchanged."/>
              <Param isxmlparam="0" type="RichBool" value="true" name="OptimalPlacement" description="Optimal position of simplified vertices" tooltip="Each collapsed vertex is placed in the position minimizing the quadric error.&#xa; It can fail (creating bad spikes) in case of very flat areas. &#xa;If disabled edges are collapsed onto one of the two original vertices and the final mesh is composed by a subset of the original vertices. "/>
              <Param isxmlparam="0" type="RichBool" value="false" name="PlanarQuadric" description="Planar Simplification" tooltip="Add additional simplification constraints that improves the quality of the simplification of the planar portion of the mesh, as a side effect, more triangles will be preserved in flat areas (allowing better shaped triangles)."/>
              <Param isxmlparam="0" type="RichFloat" value="0.001" name="PlanarWeight" description="Planar Simp. Weight" tooltip="How much we should try to preserve the triangles in the planar regions. If you lower this value planar areas will be simplified more."/>
              <Param isxmlparam="0" type="RichBool" value="false" name="QualityWeight" description="Weighted Simplification" tooltip="Use the Per-Vertex quality as a weighting factor for the simplification. The weight is used as a error amplification value, so a vertex with a high quality value will not be simplified and a portion of the mesh with low quality values will be aggressively simplified."/>
              <Param isxmlparam="0" type="RichBool" value="true" name="AutoClean" description="Post-simplification cleaning" tooltip="After the simplification an additional set of steps is performed to clean the mesh (unreferenced vertices, bad faces, etc)"/>
              <Param isxmlparam="0" type="RichBool" value="false" name="Selected" description="Simplify only selected faces" tooltip="The simplification is applied only to the selected set of faces.&#xa; Take care of the target number of faces!"/>
             </filter>
            </FilterScript>"""

        meshDir = pathlib.Path(meshPath.parent, 'SimplifiedMesh')

        # meshDir = '/'.join(meshFilename.split('/')[:-1])
        filterFileName = pathlib.Path(meshDir.parent, 'filter.mlx')
        # outputMeshFilename = meshFilename.split('/')[-1]
        outputMeshFilename = pathlib.Path(meshDir.parent.parent, 'reMeshed', meshFilename.stem + 'remeshed.ply')

        with open(f'{filterFileName}', 'w') as filterFile:
            filterFile.write(filter)

        # print(f'{meshLabServerPath} -i {meshFilename.as_posix()} -s {filterFileName} -o {outputMeshFilename.as_posix()}')
        # os.system(
        #     f'{meshLabServerPath} -i {meshFilename.as_posix()} -s {filterFileName} -o {outputMeshFilename.as_posix()}')
        check_call(
            f'{meshLabServerPath} -i {meshFilename.as_posix()} -s {filterFileName} -o {outputMeshFilename.as_posix()}',
            stdout=DEVNULL, stderr=STDOUT)

        if outputMeshFilename.exists():
            newMesh = trimesh.load_mesh(outputMeshFilename)
            newSimplifiedMesh = new_remove_small_meshes(newMesh)
            del newMesh

            if newSimplifiedMesh:
                newSimplifiedMesh.export(
                    pathlib.Path(meshDir.parent.parent, 'finalMesh', meshFilename.stem + 'remeshed.ply'))


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
            if condition:
                conditionBool = False
                if isinstance(condition, list):
                    conditionBool = False if False not in [subcondition not in filename for subcondition in condition] else True

                elif condition in filename:
                    conditionBool = True

                if conditionBool and pathlibBool:
                    file_paths.append(pathlib.Path(filepath))
                if conditionBool and not pathlibBool:
                    file_paths.append(filepath)
                    #
                    #     if isinstance(condition, list):
                    #         for subcondition in condition:
                    #             if subcondition not in
                    # if (
                    #     condition in filename
                    #     and pathlibBool
                    #     or not condition
                    #     and pathlibBool
                    # ):
                    #     file_paths.append(pathlib.Path(filepath))
                    # elif condition in filename or not condition:
                    #     file_paths.append(filepath)
    return file_paths


def getListOfFiles(dirName):
    """
        Get a list of files in folder and subfolders of dirName
        :param dirName: relative or absolute path to the directory
        :return: List of files
    """

    listOfFile = os.listdir(dirName)
    allFiles = []

    for entry in listOfFile:
        fullPath = os.path.join(dirName, entry)
        if os.path.isdir(fullPath):
            allFiles += getListOfFiles(fullPath)
        else:
            allFiles.append(fullPath)

    return allFiles


def new_remove_small_meshes(mesh, tolerance='auto'):
    """
        Find all submeshes in a Pymesh Mesh object and remove small meshes based on tolerance in tolerance% (default
        5%). The mesh will be convert to a Trimesh Mesh object to use .split() built-in function to find all
        submeshes based on graph theory. Then we will remove all meshes under the tolerance threshold based on number
        of vertices. If the submesh contains less than 5% of total it will be remove. If not, all submeshes will be
        concatenate into one object. Then, the mesh will be converted back to Pymesh Mesh object.
        :param mesh: Pymesh Mesh object
        :param tolerance:
        :return: Pymesh Mesh object
    """
    import trimesh.graph

    try:
        cc = trimesh.graph.connected_components(mesh.face_adjacency, min_len=10)
        if len(cc) < 1:
            return None

        mask = np.zeros(len(mesh.faces), dtype=np.bool)
        mask[np.concatenate(cc)] = True
        mesh.update_faces(mask)
        meshT = mesh.split(only_watertight=False).tolist()
        if type(meshT) is not list:
            return meshT
        if tolerance == 'auto':
            sumM = sum(len(submesh.vertices) for submesh in meshT)
            tolerance = int(sumM / len(meshT))
            print(f'Auto tolerance set to {tolerance} vertices')
        elif isinstance(tolerance, np.float):
            tolerance = tolerance / 100 * len(mesh.vertices)
        biggestMeshesList = []
        maxVertice = 0
        biggerMesh = None
        for subMesh in meshT:
            if len(subMesh.vertices) > tolerance:
                biggestMeshesList.append(subMesh)
            if maxVertice < len(subMesh.vertices):
                maxVertice = len(subMesh.vertices)
                # print(f'max vertices : {maxVertice}')
                biggerMesh = subMesh
        if biggestMeshesList is []:
            print('Mesh seems broken')
            return meshT
        else:
            if len(biggestMeshesList) > 1:
                # print(f'Mesh was simplified from {len(meshT)} bodies to {len(biggestMeshesList)} bodies.')
                print(f'Mesh was simplifed from {len(mesh.vertices)} to {len(biggerMesh.vertices)}')
                # print(f'Keeping only 3 biggest for a total of ')
                # return trimesh.util.concatenate(biggestMeshesList)
                return biggerMesh
            else:
                print(f'{len(biggerMesh.vertices)} vertices in bigger mesh, total : {len(mesh.vertices)}')
            return biggerMesh
    except MemoryError:
        return None


def remove_and_remesh(meshFileName, tolerance='auto', laplacien=5):
    meshFileName = pathlib.Path(meshFileName)
    mesh = trimesh.load_mesh(meshFileName)
    simplifiedMesh = new_remove_small_meshes(mesh, tolerance)
    if simplifiedMesh:
        simplifiedMeshDir = pathlib.Path(meshFileName.parent.parent, 'SimplifiedMesh')
        simplifiedMeshDir.mkdir(exist_ok=True)

        remeshedDir = pathlib.Path(meshFileName.parent.parent, 'reMeshed')
        remeshedDir.mkdir(exist_ok=True)

        remeshedDir = pathlib.Path(meshFileName.parent.parent, 'finalMesh')
        remeshedDir.mkdir(exist_ok=True)

        simplifiedMeshFilename = pathlib.Path(simplifiedMeshDir, meshFileName.stem + '.ply')
        simplifiedMesh.export(simplifiedMeshFilename)
        remeshing_via_meshlabserver(simplifiedMeshFilename, laplacien=laplacien)


def mesh_remesh(mesh, meshPath, meshFileName, name, tolerance='auto', laplacien=5):
    # meshFileName = pathlib.Path(meshFileName)
    # mesh = trimesh.load_mesh(meshFileName)
    simplifiedMesh = new_remove_small_meshes(mesh, tolerance)

    if simplifiedMesh:
        simplifiedMeshDir = pathlib.Path(meshPath, 'SimplifiedMesh')
        simplifiedMeshDir.mkdir(exist_ok=True)

        remeshedDir = pathlib.Path(meshPath, 'reMeshed')
        remeshedDir.mkdir(exist_ok=True)

        remeshedDir = pathlib.Path(meshPath, 'finalMesh')
        remeshedDir.mkdir(exist_ok=True)

        simplifiedMeshFilename = pathlib.Path(simplifiedMeshDir / f"{name}_{meshFileName}.ply")
        simplifiedMesh.export(simplifiedMeshFilename)
        # remeshing_via_meshlabserver(simplifiedMeshFilename, laplacien=laplacien)
        remeshing_via_pymeshlab(simplifiedMeshFilename, laplacien=laplacien)


def remove_noise(mesh, tolerance=5):
    meshT = pymesh_to_trimesh(mesh)
    meshList = meshT.split()
    if len(meshList) > 1:
        biggestMeshesList = [
            subMesh
            for subMesh in meshList
            if len(subMesh.vertices) > tolerance / 100 * len(meshT.vertices)
        ]

        if biggestMeshesList is not None:
            return biggestMeshesList
        print('Mesh seems broken')
        return [mesh]


def recreate_meshes(nodeList, mesh):
    # todo : refactoring

    isolatedMeshesList = []

    for _ in nodeList:
        to_keep = np.ones(mesh.num_vertices, dtype=bool)
        to_keep[nodeList[0]] = False  # all matching value become false
        to_keep = ~np.array(to_keep)  # True become False and vice-versa

        faces_to_keep = mesh.faces[np.all(to_keep[mesh.faces], axis=1)]
        out_mesh = pymesh.form_mesh(mesh.vertices, faces_to_keep)
        isolatedMeshesList.append(out_mesh)
    return isolatedMeshesList


def is_mesh_broken(mesh, meshCopy):
    """
        Easy way to see if the detail settings broke the mesh or not. It can happens with highly details settings.
        :param mesh: Pymesh Mesh object
        Mesh after optimisation
        :param meshCopy: Pymesh Mesh object
        Mesh before optimisation
        :return: boolean
        if True the mesh is broken, if False the mesh isn't broken
    """
    if mesh.vertices.size > 0:
        return np.max(get_size_of_meshes(create_graph(mesh))) < 0.1 * np.max(
            get_size_of_meshes(create_graph(meshCopy))
        )

    else:
        return True


def create_graph(mesh):
    meshGraph = nx.Graph()
    for faces in mesh.faces:
        meshGraph.add_edge(faces[0], faces[1])
        meshGraph.add_edge(faces[1], faces[2])
        meshGraph.add_edge(faces[0], faces[2])
    return meshGraph


def get_size_of_meshes(graph):
    return [len(c) for c in nx.connected_components(graph)]


def get_list_of_nodes_in_each_meshes(graph):
    return [subG for subG in nx.connected_components(graph)]


def count_number_of_meshes(mesh):
    get_size_of_meshes(create_graph(mesh))


def fix_meshes(mesh, detail="normal"):
    """
    A pipeline to optimise and fix mesh based on pymesh Mesh object.

    1. A box is created around the mesh.
    2. A target length is found based on diagonal of the mesh box.
    3. You can choose between 3 level of details, normal details settings seems to be a good compromise between
    final mesh size and sufficient number of vertices. It highly depends on your final goal.
    4. Remove degenerated triangles aka collinear triangles composed of 3 aligned points. The number of iterations is 5
    and should remove all degenerated triangles
    5. Remove isolated vertices, not connected to any faces or edges
    6. Remove self intersection edges and faces which is not realistic
    7. Remove duplicated faces
    8. The removing of duplicated faces can leave some vertices alone, we will removed them
    9. The calculation of outer hull volume is useful to be sure that the mesh is still ok
    10. Remove obtuse triangles > 179 who is not realistic and increase computation time
    11. We will remove potential duplicated faces again
    12. And duplicated vertices again
    13. Finally we will look if the mesh is broken or not. If yes we will try lower settings, if the lowest settings
    broke the mesh we will return the initial mesh. If not, we will return the optimised mesh.

    :param mesh: Pymesh Mesh object to optimise
    :param detail: string 'high', 'normal' or 'low' ('normal' as default), or float/int
    Settings to choose the targeting minimum length of edges
    :return: Pymesh Mesh object
    An optimised mesh or not depending on detail settings and mesh quality
    """
    meshCopy = mesh

    # copy/pasta of pymesh script fix_mesh from qnzhou, see pymesh on GitHub
    bbox_min, bbox_max = mesh.bbox
    diag_len = np.linalg.norm(bbox_max - bbox_min)
    if detail == "normal":
        target_len = diag_len * 5e-3
    elif detail == "high":
        target_len = diag_len * 2.5e-3
    elif detail == "low":
        target_len = diag_len * 1e-2
    elif detail is float or detail is int and detail > 0:
        target_len = diag_len * detail
    else:
        print('Details settings is invalid, must be "low", "normal", "high" or positive int or float')
        quit()

    count = 0
    mesh, __ = pymesh.remove_degenerated_triangles(mesh, 5)
    mesh, __ = pymesh.split_long_edges(mesh, target_len)
    num_vertices = mesh.num_vertices
    while True:
        mesh, __ = pymesh.collapse_short_edges(mesh, target_len,
                                               preserve_feature=True)
        mesh, info = pymesh.remove_obtuse_triangles(mesh, 179.0, 5)
        if mesh.num_vertices == num_vertices:
            break

        num_vertices = mesh.num_vertices
        count += 1
        if count > 10:
            break

    mesh, __ = pymesh.remove_duplicated_vertices(mesh)
    mesh, __ = pymesh.remove_isolated_vertices(mesh)
    mesh = pymesh.resolve_self_intersection(mesh)
    mesh, __ = pymesh.remove_duplicated_faces(mesh)
    mesh, __ = pymesh.remove_duplicated_vertices(mesh)
    mesh = pymesh.compute_outer_hull(mesh)
    mesh, __ = pymesh.remove_obtuse_triangles(mesh, 179.0, 5)
    mesh, __ = pymesh.remove_duplicated_faces(mesh)
    mesh, __ = pymesh.remove_isolated_vertices(mesh)

    if is_mesh_broken(mesh, meshCopy) is not True:
        return mesh
    if detail == "high":
        print(f'The function fix_meshes broke mesh, trying with lower details settings')
        fix_meshes(meshCopy, detail="normal")
        return mesh
    if detail == "normal":
        print(f'The function fix_meshes broke mesh, trying with lower details settings')
        mesh = fix_meshes(meshCopy, detail="low")
        return mesh
    if detail == "low":
        print(f'The function fix_meshes broke mesh, no lower settings can be applied, no fix was done')
        return meshCopy


def optimise():
    filenameList = meshIO.load_folder()

    for file in filenameList:
        mesh = pymesh.load_mesh(file)
        isolatedMeshesList = remove_small_meshes(mesh)
        for subMeshnumber, isolatedMesh in enumerate(isolatedMeshesList):
            optimisedMesh, _ = fix_meshes(isolatedMesh)
            meshIO.save_optimised_mesh(optimisedMesh, subMeshnumber, file.split('/')[-1])


def trimesh_quick_optimisation():
    return


if __name__ == "__main__":
    # optimise()
    # mesh = trimesh.load_mesh('/mnt/4EB2FF89256EC207/PycharmProjects/spineReconstruction3D/Output_Images/Mesh/11957_slice1_cell1_dendrite1_spines_63x_z2_1024px_avg2_0.35um-stack_SHF_0_0_9_1.28e+01.obj')
    # mesh.split(only_watertight=False)
    # mesh2 = new_remove_small_meshes(mesh)
    # print(f'Mesh2 : {len(mesh2.vertices)}')
    # mesh2.export('b.stl')
    # graph = create_graph(mesh)
    # meshSizeList = get_size_of_meshes(graph)
    # biggestMeshes = np.where(meshSizeList > 0.01*np.max(meshSizeList))[0]
    # print(meshSizeList)
    # print(biggestMeshes)
    # for mesh in tqdm(getListOfFiles('D:\Documents\PycharmProjects\spineReconstruction3D\Output_Images\Mesh')):
    #     remove_and_remesh(pathlib.Path(mesh))
    new_smoothing()