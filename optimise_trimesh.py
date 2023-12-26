import numpy as np
import networkx as nx
import trimesh
import os
import pathlib
from subprocess import DEVNULL, STDOUT, check_call
import pymeshlab as ml

def smoothing_via_pymeshlab(mesh_filename, laplacien=1):
    if not mesh_filename.exists():
        return
    ms = ml.MeshSet()
    mesh_dir = mesh_filename.parent
    ms.load_new_mesh(mesh_filename)
    pymeshlab_filter = f"""
    <!DOCTYPE FilterScript>
    <FilterScript>
        <filter name="Laplacian Smooth">
          <Param value="{laplacien}" type="RichInt" description="Smoothing steps" name="stepSmoothNum" tooltip="The number of times that the whole algorithm (normal smoothing + vertex fitting) is iterated."/>
          <Param value="false" type="RichBool" description="1D Boundary Smoothing" name="Boundary" tooltip="Smooth boundary edges only by themselves (e.g. the polyline forming the boundary of the mesh is independently smoothed). This can reduce the shrinking on the border but can have strange effects on very small boundaries."/>
          <Param value="false" type="RichBool" description="Cotangent weighting" name="cotangentWeight" tooltip="Use cotangent weighting scheme for the averaging of the position. Otherwise the simpler umbrella scheme (1 if the edge is present) is used."/>
          <Param value="false" type="RichBool" description="Affect only selection" name="Selected" tooltip="If checked the filter is performed only on the selected area"/>
        </filter>
    </FilterScript>"""

    filter_file_name = pathlib.Path(mesh_dir, 'filter.mlx')
    output_mesh_filename = pathlib.Path(
        mesh_dir, f'{mesh_filename.stem}_smoothed_{laplacien}_false.ply'
    )

    with open(f'{filter_file_name}', 'w') as filterFile:
        filterFile.write(pymeshlab_filter)
    ms.load_filter_script("filter.mlx")
    ms.apply_filter_script()
    ms.save_current_mesh(output_mesh_filename.__str__())


def remeshing_via_pymeshlab(mesh_filename, laplacien=2):
    if not mesh_filename.exists():
        return
    ms = ml.MeshSet()

    mesh = trimesh.load_mesh(mesh_filename)
    mesh_path = pathlib.Path(mesh_filename)
    mesh_faces = int(mesh.faces.shape[0] / 3)
    mesh_dir = pathlib.Path(mesh_path.parent, 'finalMesh')

    del mesh
    pymeshlab_filter = f"""
        <!DOCTYPE FilterScript>
        <FilterScript>
         <filter name="Simplification: Quadric Edge Collapse Decimation">
          <Param isxmlparam="0" type="RichInt" value="{mesh_faces}" name="TargetFaceNum" description="Target number of faces" tooltip="The desired final number of faces."/>
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

    mesh_dir = pathlib.Path(mesh_path.parent, 'SimplifiedMesh')
    filter_file_name = pathlib.Path(mesh_dir.parent, 'filter.mlx')
    output_mesh_filename = pathlib.Path(
        mesh_dir.parent.parent,
        'reMeshed',
        f'{mesh_filename.stem}remeshed_smoothed.ply',
    )

    with open(f'{filter_file_name}', 'w') as filterFile:
        filterFile.write(pymeshlab_filter)

    load_filter_and_apply_to_mesh(
        ms, mesh_filename, output_mesh_filename
    )
    if new_simplified_mesh := remove_small_meshes_from_path(
        output_mesh_filename
    ):
        new_simplified_mesh.export(
            pathlib.Path(
                mesh_dir.parent.parent,
                'finalMesh',
                f'{mesh_filename.stem}remeshed_smoothed.ply',
            )
        )

        filter = f"""
            <!DOCTYPE FilterScript>
            <FilterScript>
             <filter name="Simplification: Quadric Edge Collapse Decimation">
              <Param isxmlparam="0" type="RichInt" value="{mesh_faces}" name="TargetFaceNum" description="Target number of faces" tooltip="The desired final number of faces."/>
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

        mesh_dir = pathlib.Path(mesh_path.parent, 'SimplifiedMesh')

        filter_file_name = pathlib.Path(mesh_dir.parent, 'filter.mlx')
        output_mesh_filename = pathlib.Path(
            mesh_dir.parent.parent,
            'reMeshed',
            f'{mesh_filename.stem}remeshed.ply',
        )

        with open(f'{filter_file_name}', 'w') as filterFile:
            filterFile.write(filter)

        ms = ml.MeshSet()
        load_filter_and_apply_to_mesh(
            ms, mesh_filename, output_mesh_filename
        )
        if output_mesh_filename.exists():
            if new_simplified_mesh := remove_small_meshes_from_path(
                output_mesh_filename
            ):
                new_simplified_mesh.export(
                    pathlib.Path(
                        mesh_dir.parent.parent,
                        'finalMesh',
                        f'{mesh_filename.stem}remeshed.ply',
                    )
                )


def load_filter_and_apply_to_mesh(ms, mesh_filename, output_mesh_filename):
    """
    Loads a mesh, applies a filter script to it, and saves the modified mesh.

    This function loads a mesh from a file, applies a predefined filter script using PyMeshLab, and
    then saves the modified mesh to a specified file.

    Parameters:
    ms (pymeshlab.MeshSet): The PyMeshLab MeshSet instance to work with.
    mesh_filename (pathlib.Path): The path to the original mesh file.
    output_mesh_filename (pathlib.Path): The path where the modified mesh will be saved.
    """
    ms.load_new_mesh(mesh_filename.__str__())
    ms.load_filter_script('filter.mlx')
    ms.apply_filter_script()
    ms.save_current_mesh(output_mesh_filename.__str__())


def remove_small_meshes_from_path(output_mesh_filename):
    """
        Removes small components from a mesh loaded from a file and returns the result.

        This function loads a mesh from a file, removes small components from it, and then returns the
        modified mesh.

        Parameters:
        output_mesh_filename (str): The filename of the mesh from which small components are to be removed.

        Returns:
        trimesh.Trimesh: The modified mesh with small components removed.
    """
    new_mesh = trimesh.load_mesh(output_mesh_filename)
    result = new_remove_small_meshes(new_mesh)
    del new_mesh

    return result


def smoothing_via_mesh_lab_server(mesh_filename, laplacien=1):
    """
        Applies smoothing to a mesh using MeshLab's server.

        This function uses MeshLab's server to apply a Laplacian smoothing filter to a mesh. The smoothed
        mesh is then saved in the same directory as the original mesh.

        Parameters:
        mesh_filename (pathlib.Path): The path to the mesh file to be smoothed.
        laplacien (int, optional): The laplacian smoothing parameter. Defaults to 1.

        Returns:
        None
    """
    if not mesh_filename.exists():
        return
    mesh_dir = mesh_filename.parent
    mesh = trimesh.load_mesh(mesh_filename)
    mesh_path = pathlib.Path(mesh_filename)
    mesh_faces = int(mesh.faces.shape[0] / 3)

    del mesh

    # mesh_lab_server_path = '/mnt/4EB2FF89256EC207/meshlab/distrib/meshlabserver'
    mesh_lab_server_path = 'D:/Programs/MeshLab/meshlabserver'

    pymeshlab_filter = f"""
    <!DOCTYPE FilterScript>
    <FilterScript>
        <filter name="Laplacian Smooth">
          <Param value="{laplacien}" type="RichInt" description="Smoothing steps" name="stepSmoothNum" tooltip="The number of times that the whole algorithm (normal smoothing + vertex fitting) is iterated."/>
          <Param value="false" type="RichBool" description="1D Boundary Smoothing" name="Boundary" tooltip="Smooth boundary edges only by themselves (e.g. the polyline forming the boundary of the mesh is independently smoothed). This can reduce the shrinking on the border but can have strange effects on very small boundaries."/>
          <Param value="false" type="RichBool" description="Cotangent weighting" name="cotangentWeight" tooltip="Use cotangent weighting scheme for the averaging of the position. Otherwise the simpler umbrella scheme (1 if the edge is present) is used."/>
          <Param value="false" type="RichBool" description="Affect only selection" name="Selected" tooltip="If checked the filter is performed only on the selected area"/>
        </filter>
    </FilterScript>"""

    filter_file_name = pathlib.Path(mesh_dir, 'filter.mlx')
    output_mesh_filename = pathlib.Path(
        mesh_dir, f'{mesh_filename.stem}_smoothed_{laplacien}_false.ply'
    )

    with open(f'{filter_file_name}', 'w') as filterFile:
        filterFile.write(pymeshlab_filter)

    check_call(
        f'{mesh_lab_server_path} -i {mesh_filename.as_posix()} -s {filter_file_name} -o {output_mesh_filename.as_posix()}',
        stdout=DEVNULL, stderr=STDOUT)


def remeshing_via_meshlabserver(mesh_filename, laplacien=2):
    """
        Applies remeshing and smoothing to a mesh using MeshLab's server.

        This function remeshes and then applies a Laplacian smoothing filter to a mesh using MeshLab's server.
        The remeshed and smoothed mesh is saved in a 'finalMesh' subdirectory in the parent directory of the
        original mesh file.

        Parameters:
        mesh_filename (pathlib.Path): The path to the mesh file to be remeshed and smoothed.
        laplacien (int, optional): The laplacian smoothing parameter. Defaults to 2.

        Returns:
        None
    """
    if not mesh_filename.exists():
        return

    mesh = trimesh.load_mesh(mesh_filename)
    mesh_path = pathlib.Path(mesh_filename)
    mesh_faces = int(mesh.faces.shape[0] / 3)
    mesh_dir = pathlib.Path(mesh_path.parent, 'finalMesh')

    del mesh  # to free memory

    mesh_lab_server_path = 'D:/Programs/MeshLab/meshlabserver'

    filter = f"""
        <!DOCTYPE FilterScript>
        <FilterScript>
         <filter name="Simplification: Quadric Edge Collapse Decimation">
          <Param isxmlparam="0" type="RichInt" value="{mesh_faces}" name="TargetFaceNum" description="Target number of faces" tooltip="The desired final number of faces."/>
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

    mesh_dir = pathlib.Path(mesh_path.parent, 'SimplifiedMesh')

    # mesh_dir = '/'.join(meshFilename.split('/')[:-1])
    filter_file_name = pathlib.Path(mesh_dir.parent, 'filter.mlx')
    output_mesh_filename = pathlib.Path(
        mesh_dir.parent.parent,
        'reMeshed',
        f'{mesh_filename.stem}remeshed_smoothed.ply',
    )

    with open(f'{filter_file_name}', 'w') as filterFile:
        filterFile.write(filter)

    check_call(
        f'{mesh_lab_server_path} -i {mesh_filename.as_posix()} -s {filter_file_name} -o {output_mesh_filename.as_posix()}',
        stdout=DEVNULL, stderr=STDOUT)

    new_mesh = trimesh.load_mesh(output_mesh_filename)

    new_simplified_mesh = new_remove_small_meshes(new_mesh)
    del new_mesh

    if new_simplified_mesh:
        new_simplified_mesh.export(
            pathlib.Path(
                mesh_dir.parent.parent,
                'finalMesh',
                f'{mesh_filename.stem}remeshed_smoothed.ply',
            )
        )

        filter = f"""
            <!DOCTYPE FilterScript>
            <FilterScript>
             <filter name="Simplification: Quadric Edge Collapse Decimation">
              <Param isxmlparam="0" type="RichInt" value="{mesh_faces}" name="TargetFaceNum" description="Target number of faces" tooltip="The desired final number of faces."/>
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

        mesh_dir = pathlib.Path(mesh_path.parent, 'SimplifiedMesh')

        filter_file_name = pathlib.Path(mesh_dir.parent, 'filter.mlx')
        output_mesh_filename = pathlib.Path(
            mesh_dir.parent.parent,
            'reMeshed',
            f'{mesh_filename.stem}remeshed.ply',
        )

        with open(f'{filter_file_name}', 'w') as filterFile:
            filterFile.write(filter)
        check_call(
            f'{mesh_lab_server_path} -i {mesh_filename.as_posix()} -s {filter_file_name} -o {output_mesh_filename.as_posix()}',
            stdout=DEVNULL, stderr=STDOUT)

        if output_mesh_filename.exists():
            new_mesh = trimesh.load_mesh(output_mesh_filename)
            new_simplified_mesh = new_remove_small_meshes(new_mesh)
            del new_mesh

            if new_simplified_mesh:
                new_simplified_mesh.export(
                    pathlib.Path(
                        mesh_dir.parent.parent,
                        'finalMesh',
                        f'{mesh_filename.stem}remeshed.ply',
                    )
                )


def get_filepaths(directory, condition=None, pathlib_bool=True):
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
            if condition:
                condition_bool = False
                if isinstance(condition, list):
                    condition_bool = False in [
                        subcondition not in filename
                        for subcondition in condition
                    ]

                elif condition in filename:
                    condition_bool = True

                if condition_bool and pathlib_bool:
                    file_paths.append(pathlib.Path(file_path))
                if condition_bool and not pathlib_bool:
                    file_paths.append(file_path)
    return file_paths


def get_list_of_files(dir_name):
    """
        Get a list of files in folder and subfolders of dirName
        :param dir_name: relative or absolute path to the directory
        :return: List of files
    """

    list_of_file = os.listdir(dir_name)
    all_files = []

    for entry in list_of_file:
        fullPath = os.path.join(dir_name, entry)
        if os.path.isdir(fullPath):
            all_files += get_list_of_files(fullPath)
        else:
            all_files.append(fullPath)

    return all_files


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

        mask = np.zeros(len(mesh.faces), dtype=bool)
        mask[np.concatenate(cc)] = True
        mesh.update_faces(mask)
        meshT = mesh.split(only_watertight=False).tolist()
        if type(meshT) is not list:
            return meshT
        if tolerance == 'auto':
            sum_m = sum(len(submesh.vertices) for submesh in meshT)
            tolerance = int(sum_m / len(meshT))
            print(f'Auto tolerance set to {tolerance} vertices')
        elif isinstance(tolerance, np.float):
            tolerance = tolerance / 100 * len(mesh.vertices)
        biggest_meshes_list = []
        max_vertex = 0
        bigger_mesh = None
        for subMesh in meshT:
            if len(subMesh.vertices) > tolerance:
                biggest_meshes_list.append(subMesh)
            if max_vertex < len(subMesh.vertices):
                max_vertex = len(subMesh.vertices)
                # print(f'max vertices : {max_vertex}')
                bigger_mesh = subMesh
        if biggest_meshes_list is []:
            print('Mesh seems broken')
            return meshT
        else:
            if len(biggest_meshes_list) > 1:
                print(f'Mesh was simplifed from {len(mesh.vertices)} to {len(bigger_mesh.vertices)}')
                return bigger_mesh
            else:
                print(f'{len(bigger_mesh.vertices)} vertices in bigger mesh, total : {len(mesh.vertices)}')
            return bigger_mesh
    except MemoryError:
        return None


def remove_and_remesh(mesh_file_name, tolerance='auto', laplacien=5):
    """
        Removes small components from a mesh and applies remeshing.

        This function loads a mesh from a file, removes small mesh components based on a tolerance level, and
        then applies a remeshing process. The simplified and remeshed meshes are saved in designated subdirectories
        within the same parent directory as the original mesh file.

        Parameters:
        mesh_file_name (str or pathlib.Path): The path to the mesh file.
        tolerance (str or float, optional): The tolerance level for removing small meshes. Defaults to 'auto'.
        laplacien (int, optional): The laplacian smoothing parameter for remeshing. Defaults to 5.

        Returns:
        None
    """
    mesh_file_name = pathlib.Path(mesh_file_name)
    mesh = trimesh.load_mesh(mesh_file_name)
    if simplifiedMesh := new_remove_small_meshes(mesh, tolerance):
        simplified_mesh_dir = pathlib.Path(mesh_file_name.parent.parent, 'SimplifiedMesh')
        simplified_mesh_dir.mkdir(exist_ok=True)

        remeshed_dir = pathlib.Path(mesh_file_name.parent.parent, 'reMeshed')
        remeshed_dir.mkdir(exist_ok=True)

        remeshed_dir = pathlib.Path(mesh_file_name.parent.parent, 'finalMesh')
        remeshed_dir.mkdir(exist_ok=True)

        simplified_mesh_filename = pathlib.Path(
            simplified_mesh_dir, f'{mesh_file_name.stem}.ply'
        )
        simplifiedMesh.export(simplified_mesh_filename)
        remeshing_via_meshlabserver(simplified_mesh_filename, laplacien=laplacien)


def mesh_remesh(mesh, mesh_path, mesh_file_name, name, tolerance='auto', laplacien=5):
    """
        Removes small components from a mesh and applies remeshing, then saves the remeshed mesh.

        This function removes small mesh components based on a tolerance level and applies a remeshing
        process to the provided mesh. The remeshed mesh is saved in subdirectories within the specified
        mesh path.

        Parameters:
        mesh (trimesh.base.Trimesh): The mesh to process.
        mesh_path (str or pathlib.Path): The path to the directory where the mesh file is located.
        mesh_file_name (str): The file name of the mesh.
        name (str): The name to use for saving the remeshed mesh.
        tolerance (str or float, optional): The tolerance level for removing small meshes. Defaults to 'auto'.
        laplacien (int, optional): The laplacian smoothing parameter for remeshing. Defaults to 5.

        Returns:
        None
    """
    if simplifiedMesh := new_remove_small_meshes(mesh, tolerance):
        simplified_mesh_dir = pathlib.Path(mesh_path, 'SimplifiedMesh')
        simplified_mesh_dir.mkdir(exist_ok=True)

        remeshed_dir = pathlib.Path(mesh_path, 'reMeshed')
        remeshed_dir.mkdir(exist_ok=True)

        remeshed_dir = pathlib.Path(mesh_path, 'finalMesh')
        remeshed_dir.mkdir(exist_ok=True)

        simplified_mesh_filename = pathlib.Path(simplified_mesh_dir / f"{name}_{mesh_file_name}.ply")
        simplifiedMesh.export(simplified_mesh_filename)
        remeshing_via_pymeshlab(simplified_mesh_filename, laplacien=laplacien)


def is_mesh_broken(mesh, mesh_copy):
    """
        Easy way to see if the detail settings broke the mesh or not. It can happens with highly details settings.
        :param mesh: Pymesh Mesh object
        Mesh after optimisation
        :param mesh_copy: Pymesh Mesh object
        Mesh before optimisation
        :return: boolean
        if True the mesh is broken, if False the mesh isn't broken
    """
    if mesh.vertices.size > 0:
        return np.max(get_size_of_meshes(create_graph(mesh))) < 0.1 * np.max(
            get_size_of_meshes(create_graph(mesh_copy))
        )

    else:
        return True


def create_graph(mesh):
    """
        Creates a graph representation of a mesh.

        This function generates a graph where each vertex represents a vertex in the mesh, and edges are
        added between vertices that share a face in the mesh. The graph is undirected and captures the
        connectivity of the mesh's vertices.

        Parameters:
        mesh (Mesh): The mesh object from which to create the graph. The mesh should have an attribute 'faces'
                     that represents the faces of the mesh as a list of tuples, where each tuple contains the
                     indices of the vertices that form a face.

        Returns:
        networkx.Graph: A graph representation of the mesh.
    """
    mesh_graph = nx.Graph()
    for faces in mesh.faces:
        mesh_graph.add_edge(faces[0], faces[1])
        mesh_graph.add_edge(faces[1], faces[2])
        mesh_graph.add_edge(faces[0], faces[2])
    return mesh_graph


def get_size_of_meshes(graph):
    """
    Calculates the size of each connected component in a graph.

    This function iterates over the connected components of a graph and returns the size (number of nodes)
    of each component.

    Parameters:
    graph (networkx.Graph): The graph to analyze.

    Returns:
    list: A list containing the sizes of each connected component in the graph.
    """
    return [len(c) for c in nx.connected_components(graph)]


def get_list_of_nodes_in_each_meshes(graph):
    """
        Retrieves the list of nodes in each connected component of a graph.

        This function iterates over the connected components of a graph and returns the nodes within each component.

        Parameters:
        graph (networkx.Graph): The graph to analyze.

        Returns:
        list: A list of sets, where each set contains the nodes of a connected component in the graph.
    """
    return list(nx.connected_components(graph))


def count_number_of_meshes(mesh):
    """
        Counts the number of connected components (meshes) in a given mesh.

        This function creates a graph representation of the mesh and uses it to count the number of connected
        components (or sub-meshes).

        Parameters:
        mesh (Mesh): The mesh to analyze.

        Returns:
        int: The number of connected components (sub-meshes) in the mesh.
    """
    get_size_of_meshes(create_graph(mesh))
